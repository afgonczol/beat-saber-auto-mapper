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
    CANCELLED = "cancelled"


LightStyle = Literal["auto", "none", "pulse", "calm", "scenes", "flash", "accent"]


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
    intensity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Challenge within each difficulty: 0.5 = typical human map, 1 = about the "
        "hardest 10%; null = the server's configured default.",
    )
    repetition: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How much repeated music reuses earlier patterns (note_v1): 0 = never, "
        "0.5 = the learned amount, 1 = twice as strongly; null = the configured default.",
    )
    light_style: LightStyle | None = Field(
        default=None,
        description="Look of model lights (light_v2 checkpoints): auto = picked for the song, "
        "none = the model's choice, or a style; null = the configured default (else auto).",
    )
    arcs: bool | None = Field(
        default=None,
        description="Arcs on or off in every difficulty; null = the configured default "
        "(else from Hard up).",
    )
    chains: bool | None = Field(
        default=None,
        description="Chains on or off in every difficulty; null = the configured default "
        "(else from Expert up).",
    )
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
    environment: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9]{1,64}$",
        description="Info.dat environment (its look and colours), e.g. LizzoEnvironment, from "
        "GET /api/environments; null = picked for each song.",
    )
    title: str | None = Field(
        default=None,
        max_length=200,
        description="Song title in the map; null = from the file's tags, else its name.",
    )
    artist: str | None = Field(
        default=None,
        max_length=200,
        description="Song artist in the map; null = from the file's tags, else its name.",
    )


class EnvironmentInfo(BaseModel):
    name: str = Field(description="Info.dat environment name, e.g. TheWeekndEnvironment.")
    label: str = Field(description="Display name, e.g. The Weeknd.")
    maps: int = Field(
        description="Human maps its light effects come from; 0 = the built-in group-lighting "
        "template."
    )
    auto: bool = Field(description="One of the environments picked when none is chosen.")


class InstallResult(BaseModel):
    path: str = Field(description="The installed map folder inside CustomLevels.")
    warnings: list[str] = Field(
        default=[], description="Things the player should know, e.g. Chroma is not installed."
    )


class Feedback(BaseModel):
    """What the player thought of the map after playing it (docs/PLAYTEST.md)."""

    rating: Literal[-1, 0, 1] = Field(default=0, description="1 = good, -1 = needs work, 0 = none.")
    note: str = Field(default="", max_length=4000, description="What they noticed.")
    updated_at: datetime | None = Field(default=None, description="Set by the server.")


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
    install_when_done: bool = Field(
        default=False, description="Copy the map into the game as soon as the first run is done."
    )
    installed: InstallResult | None = Field(
        default=None,
        description="Where this revision of the map is installed in the game, if it is.",
    )
    install_error: str | None = Field(
        default=None, description="Why the automatic install failed (the map itself is fine)."
    )
    last_seq: int = Field(
        default=0, description="seq of the latest event; pass it as ?after= to follow new ones."
    )
    title: str = Field(default="", description="Song title of the generated map ('' before).")
    artist: str = Field(default="", description="Song artist of the generated map.")
    started_at: datetime | None = Field(default=None, description="When the latest run started.")
    finished_at: datetime | None = Field(
        default=None, description="When the latest run ended (done, failed or cancelled)."
    )
    source_id: str | None = Field(
        default=None, description="The job this one was remade from (POST .../rerun)."
    )
    feedback: Feedback | None = Field(default=None, description="Playtest rating and notes.")


class JobEvent(BaseModel):
    """One server-sent event. ``type`` is also the SSE ``event:`` name and ``seq`` its ``id:``.

    A ``status`` event with status ``done``, ``failed`` or ``cancelled`` ends the current run;
    the server then closes the stream, so clients should close their EventSource on it.
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


class RerunRequest(BaseModel):
    """Make the song again as a new job, from the upload the server already has."""

    options: JobOptions | None = Field(
        default=None, description="Options for the new job; null = the same as the source job."
    )
    install: bool = Field(default=False, description="Copy the new map into the game when done.")


class SongInfo(BaseModel):
    """The song title and artist the game shows for the map."""

    title: str = Field(min_length=1, max_length=200, pattern=r"\S")
    artist: str = Field(default="", max_length=200)


class InstallTarget(BaseModel):
    path: str = Field(description="Beat Saber's CustomLevels folder (from the config file).")
    found: bool = Field(description="The folder exists, so maps can be installed.")
    chroma: bool = Field(description="The Chroma mod is installed.")
    config: str = Field(description="The config file that sets the folder.")
    error: str | None = Field(default=None, description="Why the config could not be read.")


class JumpDefaults(BaseModel):
    njs: float
    reaction_time: float = Field(description="Seconds from spawn to strike.")


class ServerSettings(BaseModel):
    """What the server generates with when a job leaves an option out."""

    generator: str = Field(description="Where notes come from: model, stub or mock.")
    generator_detail: str = Field(default="", description="Checkpoint, or why it fell back.")
    intensity: float
    repetition: float | None = Field(description="null = the learned amount.")
    light_style: str
    arcs: bool | None = Field(description="null = from Hard up.")
    chains: bool | None = Field(description="null = from Expert up.")
    jump: dict[Difficulty, JumpDefaults] = Field(
        description="Default NJS and reaction time per difficulty, config overrides included."
    )
    install: InstallTarget


class ErrorResponse(BaseModel):
    detail: str
