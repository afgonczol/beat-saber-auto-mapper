"""Request and response models of the jobs API.

These pydantic models are the source of the OpenAPI spec (``web/openapi.json``). Map data
reuses the dataclasses from ``bsmapper.schema`` directly, so the preview JSON is exactly the
internal map model (times in beats).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from bsmapper.interfaces import Issue, LightingStyle
from bsmapper.schema import Beatmap, Characteristic, Difficulty


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobOptions(BaseModel):
    """What to generate. Sent as multipart form fields next to the audio file."""

    modes: list[Characteristic] = Field(default=[Characteristic.STANDARD], min_length=1)
    difficulties: list[Difficulty] = Field(
        default=[Difficulty.HARD, Difficulty.EXPERT, Difficulty.EXPERT_PLUS], min_length=1
    )
    lighting: LightingStyle = "v3"
    style: float | None = Field(
        default=None, ge=0.0, le=1.0, description="0 = flowy, 1 = technical, null = no preference."
    )
    seed: int = 0
    njs: float | None = Field(
        default=None, ge=1.0, le=40.0, description="NJS for every difficulty; null = defaults."
    )
    reaction_time: float | None = Field(
        default=None,
        ge=0.3,
        le=2.0,
        description="Seconds from spawn to strike for Expert and Expert+ (lower difficulties "
        "keep their defaults); null = defaults (0.8 s and 0.7 s).",
    )


class JobInfo(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    filename: str = Field(description="Name of the uploaded audio file.")
    options: JobOptions
    stage: str = Field(default="", description="Current pipeline stage, e.g. 'generate'.")
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""
    error: str | None = None
    revision: int = Field(default=0, description="Bumped every time the map changes.")
    last_seq: int = Field(
        default=0, description="seq of the latest event; pass it as ?after= to follow new ones."
    )


class JobEvent(BaseModel):
    """One server-sent event. ``type`` is also the SSE ``event:`` name and ``seq`` its ``id:``.

    A ``status`` event with status ``done`` or ``failed`` ends the current run; the server then
    closes the stream, so clients should close their EventSource on it.
    """

    seq: int
    type: Literal["status", "progress"]
    status: JobStatus
    stage: str = ""
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""


class DifficultyWarning(BaseModel):
    """A playability issue (``interfaces.Issue``) tagged with the difficulty it belongs to."""

    characteristic: Characteristic
    difficulty: Difficulty
    issue: Issue


class MapPreview(BaseModel):
    job_id: str
    revision: int
    duration_s: float = Field(description="Song length in seconds.")
    beatmap: Beatmap
    warnings: list[DifficultyWarning] = []
    notes: list[str] = Field(default=[], description="Job-level remarks, e.g. packaging fallbacks.")


class RegenerateRequest(BaseModel):
    characteristic: Characteristic
    difficulty: Difficulty
    seed: int
    temperature: float = Field(default=1.0, gt=0.0, le=2.0)


class InstallResult(BaseModel):
    path: str = Field(description="The installed map folder inside CustomLevels.")
    warnings: list[str] = Field(
        default=[], description="Things the player should know, e.g. Chroma is not installed."
    )


class ErrorResponse(BaseModel):
    detail: str
