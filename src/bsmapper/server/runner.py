"""The boundary between the server and whatever produces maps.

The server only talks to a ``JobRunner``: ``PipelineJobRunner`` (the real WS7 pipeline,
``pipeline_runner``) or ``MockJobRunner`` (a canned map, for UI development).
Runners are called from the single worker thread, one job at a time, and may block.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from bsmapper.schema import Beatmap
from bsmapper.server.models import DifficultyWarning, JobOptions, RegenerateRequest

# progress(stage, fraction 0..1 of the whole run, human readable message)
ProgressFn = Callable[[str, float, str], None]


@dataclass(slots=True)
class JobInput:
    job_id: str
    audio_path: Path  # the uploaded file, untouched
    work_dir: Path  # private scratch folder for this job; runners write their output here
    options: JobOptions


@dataclass(slots=True)
class JobResult:
    beatmap: Beatmap
    duration_s: float
    zip_path: Path  # the downloadable, game-ready map zip
    warnings: list[DifficultyWarning] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # What the preview plays. None: the upload. Runners that pad the song (lead-in) set
    # this to the packaged song.egg so the audio lines up with the map's beats.
    audio_path: Path | None = None


class JobRunner(Protocol):
    def run(self, job: JobInput, progress: ProgressFn) -> JobResult:
        """Generate every requested mode x difficulty, with lighting, and package the zip."""
        ...

    def regenerate(
        self, job: JobInput, previous: JobResult, request: RegenerateRequest, progress: ProgressFn
    ) -> JobResult:
        """Redo one difficulty with a new seed/temperature; keep the others as they are."""
        ...
