"""In-memory job table and the single worker thread (one GPU, one job at a time)."""

from __future__ import annotations

import logging
import queue
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from bsmapper.server.models import (
    JobEvent,
    JobInfo,
    JobOptions,
    JobStatus,
    MapPreview,
    RegenerateRequest,
)
from bsmapper.server.runner import JobInput, JobResult, JobRunner

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = (".mp3", ".ogg", ".egg", ".wav", ".flac", ".m4a", ".aac", ".opus")
TERMINAL = (JobStatus.DONE, JobStatus.FAILED)


class JobBusyError(RuntimeError):
    """The job is queued or running, so it cannot be changed right now."""


@dataclass(slots=True)
class Job:
    info: JobInfo
    input: JobInput
    result: JobResult | None = None
    events: list[JobEvent] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def events_after(self, seq: int) -> list[JobEvent]:
        with self.lock:
            return [e for e in self.events if e.seq > seq]

    def preview(self) -> MapPreview | None:
        r = self.result
        if r is None:
            return None
        return MapPreview(
            job_id=self.info.id,
            revision=self.info.revision,
            duration_s=r.duration_s,
            beatmap=r.beatmap,
            warnings=r.warnings,
            notes=r.notes,
        )


class JobManager:
    def __init__(self, runner: JobRunner, work_dir: Path) -> None:
        self.runner = runner
        self.work_dir = Path(work_dir)
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[tuple[Job, Callable[[], JobResult]] | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    # --- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="bsmapper-jobs", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=5)
        self._worker = None

    # --- API -------------------------------------------------------------------------

    def submit(self, filename: str, data: BinaryIO, options: JobOptions) -> Job:
        suffix = Path(filename).suffix.lower()
        if suffix not in AUDIO_SUFFIXES:
            raise ValueError(f"unsupported audio type {suffix or '(none)'}")
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        # Keep the original name: it is the fallback for song title and artist.
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(filename).stem).strip(" .") or "song"
        audio = job_dir / "upload" / f"{stem[:100]}{suffix}"
        audio.parent.mkdir()
        with audio.open("wb") as f:
            while chunk := data.read(1 << 20):
                f.write(chunk)
        info = JobInfo(
            id=job_id,
            status=JobStatus.QUEUED,
            created_at=datetime.now(UTC),
            filename=Path(filename).name,
            options=options,
        )
        job = Job(info, JobInput(job_id, audio, job_dir, options))
        self.jobs[job_id] = job
        self._emit(job, "status", status=JobStatus.QUEUED, message="Queued")
        self._enqueue(job, lambda: self.runner.run(job.input, self._progress_fn(job)))
        return job

    def regenerate(self, job: Job, request: RegenerateRequest) -> None:
        with job.lock:
            if job.info.status not in TERMINAL or job.result is None:
                raise JobBusyError("job has no finished map to regenerate")
            previous = job.result
        self._emit(
            job,
            "status",
            status=JobStatus.QUEUED,
            message=f"Regenerate {request.characteristic} {request.difficulty} queued",
        )
        self._enqueue(
            job,
            lambda: self.runner.regenerate(job.input, previous, request, self._progress_fn(job)),
        )

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    # --- internals --------------------------------------------------------------------

    def _enqueue(self, job: Job, fn: Callable[[], JobResult]) -> None:
        self.start()
        self._queue.put((job, fn))

    def _work(self) -> None:
        while (item := self._queue.get()) is not None:
            job, fn = item
            self._emit(job, "status", status=JobStatus.RUNNING, message="Started")
            try:
                result = fn()
            except Exception as exc:  # report any runner failure on the job, keep serving
                log.exception("job %s failed", job.info.id)
                with job.lock:
                    job.info.error = str(exc) or type(exc).__name__
                self._emit(job, "status", status=JobStatus.FAILED, message=job.info.error or "")
                continue
            with job.lock:
                job.result = result
                job.info.revision += 1
                job.info.error = None
            self._emit(job, "status", status=JobStatus.DONE, progress=1.0, message="Done")

    def _progress_fn(self, job: Job):
        def progress(stage: str, fraction: float, message: str = "") -> None:
            self._emit(
                job,
                "progress",
                status=JobStatus.RUNNING,
                stage=stage,
                progress=max(0.0, min(1.0, fraction)),
                message=message,
            )

        return progress

    def _emit(
        self,
        job: Job,
        type_: str,
        *,
        status: JobStatus,
        stage: str | None = None,
        progress: float | None = None,
        message: str = "",
    ) -> None:
        with job.lock:
            info = job.info
            info.status = status
            if status == JobStatus.QUEUED:
                info.stage, info.progress = "", 0.0
            if stage is not None:
                info.stage = stage
            if progress is not None:
                info.progress = progress
            info.message = message
            seq = info.last_seq = info.last_seq + 1
            job.events.append(
                JobEvent(
                    seq=seq,
                    type=type_,
                    status=status,
                    stage=info.stage,
                    progress=info.progress,
                    message=message,
                )
            )
