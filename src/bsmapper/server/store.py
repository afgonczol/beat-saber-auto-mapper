"""Job folders on disk, so the queue and finished maps survive a server restart.

Each job's folder (``work_dir/<id>``) holds ``job.json`` (its ``JobInfo`` and the upload's
path) and, once a map exists, ``result.json`` (the map for the preview, its warnings and the
zip's path). Paths inside the job folder are stored relative to it, so the work folder can
be moved. Folders without a ``job.json`` (from older versions) are ignored.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import BaseModel, ValidationError

from bsmapper.schema import Beatmap
from bsmapper.server.models import DifficultyWarning, JobInfo
from bsmapper.server.runner import JobResult

log = logging.getLogger(__name__)

JOB_FILE = "job.json"
RESULT_FILE = "result.json"


class JobRecord(BaseModel):
    info: JobInfo
    audio: str


class ResultRecord(BaseModel):
    beatmap: Beatmap
    duration_s: float
    zip: str
    warnings: list[DifficultyWarning] = []
    notes: list[str] = []
    audio: str | None = None


def _rel(path: Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_info(job_dir: Path, info: JobInfo, audio: Path) -> None:
    _write(job_dir / JOB_FILE, JobRecord(info=info, audio=_rel(audio, job_dir)).model_dump_json())


def save_result(job_dir: Path, result: JobResult) -> None:
    record = ResultRecord(
        beatmap=result.beatmap,
        duration_s=result.duration_s,
        zip=_rel(result.zip_path, job_dir),
        warnings=result.warnings,
        notes=result.notes,
        audio=None if result.audio_path is None else _rel(result.audio_path, job_dir),
    )
    _write(job_dir / RESULT_FILE, record.model_dump_json())


def load(job_dir: Path) -> tuple[JobInfo, Path, JobResult | None] | None:
    """The saved job in ``job_dir``: its info, upload and map (None if it has none).

    None if the folder holds no readable job.
    """
    try:
        record = JobRecord.model_validate_json((job_dir / JOB_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as exc:
        log.warning("skipping job folder %s: %s", job_dir, exc)
        return None
    result = None
    path = job_dir / RESULT_FILE
    if path.is_file():
        try:
            saved = ResultRecord.model_validate_json(path.read_text(encoding="utf-8"))
            result = JobResult(
                beatmap=saved.beatmap,
                duration_s=saved.duration_s,
                zip_path=job_dir / saved.zip,
                warnings=saved.warnings,
                notes=saved.notes,
                audio_path=None if saved.audio is None else job_dir / saved.audio,
            )
        except (OSError, ValidationError) as exc:
            log.warning("job %s: unreadable map: %s", job_dir.name, exc)
    return record.info, job_dir / record.audio, result
