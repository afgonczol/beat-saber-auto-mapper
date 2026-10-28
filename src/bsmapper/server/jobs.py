"""In-memory job table and the single worker thread (one GPU, one job at a time).

Jobs run in the order they were queued; the queue has no size limit, so a batch of songs can
be queued at once. Queued jobs can be cancelled or removed; a running job is cancelled at its
next progress report. Every job is saved in its folder (``store``), so after a server restart
``restore`` finds the finished maps again and queues the songs a restart interrupted.
"""

from __future__ import annotations

import logging
import queue
import re
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from bsmapper.server import store
from bsmapper.server.install import MapNotReady, install_job, uninstall_path
from bsmapper.server.models import (
    InstallResult,
    JobEvent,
    JobInfo,
    JobOptions,
    JobStatus,
    MapPreview,
    RegenerateRequest,
)
from bsmapper.server.runner import JobInput, JobResult, JobRunner
from bsmapper.server.songinfo import retitle

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = (".mp3", ".ogg", ".egg", ".wav", ".flac", ".m4a", ".aac", ".opus")
TERMINAL = (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)

Installer = Callable[[JobInput, JobResult], InstallResult]
Uninstaller = Callable[[str], None]


class JobBusyError(RuntimeError):
    """The job is queued or running, so it cannot be changed right now."""


class JobNotActiveError(RuntimeError):
    """The job is not queued or running, so there is nothing to cancel."""


class JobCancelled(BaseException):  # noqa: N818 - control flow, not an error
    """Raised from a running job's progress callback once it is cancelled.

    A ``BaseException``, like ``asyncio.CancelledError``: the pipeline's ``except Exception``
    fallbacks must not swallow it.
    """


@dataclass(slots=True)
class Job:
    info: JobInfo
    input: JobInput
    result: JobResult | None = None
    events: list[JobEvent] = field(default_factory=list)
    # Reentrant: status changes are emitted while the checks that allow them still hold.
    lock: threading.RLock = field(default_factory=threading.RLock)
    # The queued or running run; a queue item with another run_id was cancelled.
    active_run: int | None = None
    runs: int = 0
    cancel_requested: bool = False

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


@dataclass(slots=True)
class _Run:
    job: Job
    fn: Callable[[], JobResult]
    run_id: int
    first: bool  # the first generation (not a regenerate): install it if asked to


class JobManager:
    def __init__(
        self,
        runner: JobRunner,
        work_dir: Path,
        installer: Installer = install_job,
        uninstaller: Uninstaller = uninstall_path,
    ) -> None:
        self.runner = runner
        self.work_dir = Path(work_dir)
        self.installer = installer
        self.uninstaller = uninstaller
        self.jobs: dict[str, Job] = {}  # in the order they were submitted
        self._jobs_lock = threading.Lock()
        self._queue: queue.Queue[_Run | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()

    # --- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._stopping.clear()
            self._worker = threading.Thread(target=self._work, name="bsmapper-jobs", daemon=True)
            self._worker.start()

    def stop(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._stopping.set()  # finish the running job, leave the rest of the queue
            self._queue.put(None)
            self._worker.join(timeout=5)
        self._worker = None

    def restore(self) -> int:
        """Load the jobs saved in ``work_dir``, oldest first; returns how many.

        Finished jobs come back as they were. A job a restart interrupted is queued again,
        unless it already had a map (an interrupted regenerate): then it keeps that map.
        """
        if not self.work_dir.is_dir():
            return 0
        saved = []
        for job_dir in self.work_dir.iterdir():
            if job_dir.is_dir() and job_dir.name not in self.jobs:
                loaded = store.load(job_dir)
                if loaded is not None:
                    saved.append((job_dir, *loaded))
        saved.sort(key=lambda item: item[1].created_at)
        for job_dir, info, audio, result in saved:
            if info.id in self.jobs:
                continue
            job = Job(info, JobInput(info.id, audio, job_dir, info.options), result)
            with self._jobs_lock:
                self.jobs[info.id] = job
            with job.lock:
                interrupted = info.status not in TERMINAL
                if info.status == JobStatus.DONE and result is None:
                    info.error = "The map files are missing."
                    self._emit(job, "status", status=JobStatus.FAILED, message=info.error)
                elif interrupted and result is not None:
                    info.started_at = None
                    message = "Interrupted by a server restart; the earlier map is kept"
                    self._emit(job, "status", status=JobStatus.DONE, progress=1.0, message=message)
                elif interrupted and not audio.is_file():
                    info.error = "The uploaded song is missing."
                    self._emit(job, "status", status=JobStatus.FAILED, message=info.error)
                elif interrupted:
                    info.started_at = None
                    self._queue_first_run(job, "Queued again after a server restart")
        return len(saved)

    # --- API -------------------------------------------------------------------------

    def submit(
        self,
        filename: str,
        data: BinaryIO,
        options: JobOptions,
        install_when_done: bool = False,
        source_id: str | None = None,
    ) -> Job:
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
            install_when_done=install_when_done,
            source_id=source_id,
        )
        job = Job(info, JobInput(job_id, audio, job_dir, options))
        with self._jobs_lock:
            self.jobs[job_id] = job
        self._queue_first_run(job, "Queued")
        return job

    def rerun(
        self, job: Job, options: JobOptions | None = None, install_when_done: bool = False
    ) -> Job:
        """Queue the same song again as a new job (same options unless given)."""
        with job.lock:
            source, filename = job.input.audio_path, job.info.filename
            options = options or job.info.options.model_copy(deep=True)
        if not source.is_file():
            raise FileNotFoundError("the uploaded song is gone")
        with source.open("rb") as f:
            new = self.submit(filename, f, options, install_when_done, source_id=job.info.id)
        return new

    def set_song_info(self, job: Job, title: str, artist: str) -> None:
        """Rename the song in the finished map (Info.dat and zip); reinstall it if installed."""
        with job.lock:
            if job.info.status not in TERMINAL:
                raise JobBusyError("job is busy")
            if job.result is None:
                raise MapNotReady("map not ready")
            title, artist = title.strip(), artist.strip()
            result = retitle(job.input, job.result, title, artist)
            job.result = result
            job.info.title, job.info.artist = title, artist
            job.info.revision += 1
            previous = job.info.installed
            job.info.installed = None
            store.save_result(job.input.work_dir, result)
            self._save(job)
        if previous is None:
            return
        try:
            installed = self._install(job, result)
            if installed.path != previous.path:
                self.uninstaller(previous.path)
        except Exception as exc:  # the renamed map is fine; say why the game has the old one
            log.warning("job %s: reinstall failed: %s", job.info.id, exc)
            with job.lock:
                job.info.install_error = str(exc) or type(exc).__name__
                self._save(job)

    def regenerate(self, job: Job, request: RegenerateRequest) -> None:
        with job.lock:
            if job.info.status not in TERMINAL or job.result is None:
                raise JobBusyError("job has no finished map to regenerate")
            previous = job.result
            run_id = self._claim_run(job)
            self._emit(
                job,
                "status",
                status=JobStatus.QUEUED,
                message=f"Regenerate {request.characteristic} {request.difficulty} queued",
            )
        self._put(
            job,
            lambda: self.runner.regenerate(job.input, previous, request, self._progress_fn(job)),
            run_id,
            first=False,
        )

    def cancel(self, job: Job) -> None:
        """Drop a queued run, or stop a running one at its next progress report."""
        with job.lock:
            status = job.info.status
            if status == JobStatus.RUNNING:
                job.cancel_requested = True
                self._emit(job, "progress", status=JobStatus.RUNNING, message="Cancelling…")
            elif status == JobStatus.QUEUED:
                job.active_run = None
                self._emit_cancelled(job)
            else:
                raise JobNotActiveError(f"job is {status}, not queued or running")

    def remove(self, job: Job) -> None:
        """Forget a job that is not running (dropping it from the queue) and delete its files."""
        with job.lock:
            if job.info.status == JobStatus.RUNNING:
                raise JobBusyError("job is running; cancel it first")
            job.active_run = None
        with self._jobs_lock:
            self.jobs.pop(job.info.id, None)
        shutil.rmtree(job.input.work_dir, ignore_errors=True)

    def install(self, job: Job) -> InstallResult:
        """Copy the finished map into the game (``MapNotReady`` if there is none yet)."""
        with job.lock:
            result = job.result if job.info.status in TERMINAL else None
        if result is None:
            raise MapNotReady("map not ready")
        return self._install(job, result)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def all(self) -> list[Job]:
        """Every job, oldest first (the order they run in)."""
        with self._jobs_lock:
            return list(self.jobs.values())

    # --- internals --------------------------------------------------------------------

    def _queue_first_run(self, job: Job, message: str) -> None:
        with job.lock:
            run_id = self._claim_run(job)
            self._emit(job, "status", status=JobStatus.QUEUED, message=message)
        run = lambda: self.runner.run(job.input, self._progress_fn(job))  # noqa: E731
        self._put(job, run, run_id, first=True)

    def _save(self, job: Job) -> None:
        """Write the job's info to its folder (with ``job.lock`` held)."""
        try:
            store.save_info(job.input.work_dir, job.info, job.input.audio_path)
        except OSError as exc:  # e.g. removed meanwhile; the job still works in memory
            log.warning("job %s: could not save: %s", job.info.id, exc)

    def _claim_run(self, job: Job) -> int:
        """Make a new run the job's active one (with ``job.lock`` held once others see it)."""
        job.runs += 1
        job.active_run = job.runs
        job.cancel_requested = False
        return job.runs

    def _put(self, job: Job, fn: Callable[[], JobResult], run_id: int, first: bool) -> None:
        self.start()
        self._queue.put(_Run(job, fn, run_id, first))

    def _work(self) -> None:
        while (item := self._queue.get()) is not None and not self._stopping.is_set():
            job = item.job
            with job.lock:
                if job.active_run != item.run_id:  # cancelled or removed while queued
                    continue
                self._emit(job, "status", status=JobStatus.RUNNING, message="Started")
            self._run(item)

    def _run(self, item: _Run) -> None:
        job = item.job
        try:
            result = item.fn()
        except JobCancelled:
            with job.lock:
                job.active_run = None
            self._emit_cancelled(job)
            return
        except Exception as exc:  # report any runner failure on the job, keep serving
            log.exception("job %s failed", job.info.id)
            with job.lock:
                job.active_run = None
                job.info.error = str(exc) or type(exc).__name__
            self._emit(job, "status", status=JobStatus.FAILED, message=job.info.error or "")
            return
        try:
            store.save_result(job.input.work_dir, result)
        except OSError as exc:
            log.warning("job %s: could not save the map: %s", job.info.id, exc)
        with job.lock:
            job.result = result
            job.info.revision += 1
            job.info.error = None
            job.info.title = result.beatmap.song_name
            job.info.artist = result.beatmap.song_author
            job.info.installed = job.info.install_error = None  # the game has the old map
            job.active_run = None
            auto_install = item.first and job.info.install_when_done
        message = "Done"
        if auto_install:
            self._emit(job, "progress", status=JobStatus.RUNNING, message="Installing to game")
            try:
                self._install(job, result)
                message = "Done, installed to the game"
            except Exception as exc:  # the map is fine; say why it is not in the game
                log.warning("job %s: install failed: %s", job.info.id, exc)
                with job.lock:
                    job.info.install_error = str(exc) or type(exc).__name__
                message = f"Done, but installing failed: {job.info.install_error}"
        self._emit(job, "status", status=JobStatus.DONE, progress=1.0, message=message)

    def _install(self, job: Job, result: JobResult) -> InstallResult:
        installed = self.installer(job.input, result)
        with job.lock:
            job.info.installed = installed
            job.info.install_error = None
            self._save(job)
        return installed

    def _emit_cancelled(self, job: Job) -> None:
        # A cancelled regenerate keeps the earlier map, so the job stays done.
        if job.result is None:
            self._emit(job, "status", status=JobStatus.CANCELLED, message="Cancelled")
        else:
            self._emit(job, "status", status=JobStatus.DONE, progress=1.0, message="Cancelled")

    def _progress_fn(self, job: Job):
        def progress(stage: str, fraction: float, message: str = "") -> None:
            if job.cancel_requested:
                raise JobCancelled
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
            if type_ == "status" and status == JobStatus.RUNNING:
                info.started_at, info.finished_at = datetime.now(UTC), None
            elif type_ == "status" and status in TERMINAL:
                info.finished_at = datetime.now(UTC)
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
            if type_ == "status":
                self._save(job)
