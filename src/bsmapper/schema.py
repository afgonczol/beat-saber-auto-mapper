"""Internal, version-agnostic beatmap representation.

This module is the shared contract between every workstream. Readers convert
v2/v3/v4 files *into* these types, writers convert *out of* them, the tokenizer
and models only ever see these types. Changes here go through a dedicated
"contract change" PR (see CLAUDE.md).

All object times are in beats (float), relative to the song's initial BPM and
its BPM changes. Use ``Beatmap.beat_to_seconds`` / ``seconds_to_beat`` to convert.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal

GRID_COLUMNS = 4  # x: 0..3, left to right
GRID_ROWS = 3  # y: 0..2, bottom to top


class Characteristic(StrEnum):
    STANDARD = "Standard"
    ONE_SABER = "OneSaber"
    NO_ARROWS = "NoArrows"
    DEGREE_90 = "90Degree"
    DEGREE_360 = "360Degree"
    LIGHTSHOW = "Lightshow"


class Difficulty(StrEnum):
    EASY = "Easy"
    NORMAL = "Normal"
    HARD = "Hard"
    EXPERT = "Expert"
    EXPERT_PLUS = "ExpertPlus"

    @property
    def rank(self) -> int:
        """The ``_difficultyRank`` value Info.dat expects."""
        return {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}[self.value]


class Color(IntEnum):
    RED = 0  # left saber
    BLUE = 1  # right saber


class CutDirection(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    ANY = 8  # dot note


# --- Gameplay objects -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Note:
    beat: float
    x: int
    y: int
    color: Color
    direction: CutDirection
    angle_offset: int = 0


@dataclass(frozen=True, slots=True)
class Bomb:
    beat: float
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Obstacle:
    beat: float
    duration: float
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Arc:
    """v3 'slider'. Head and tail usually sit on notes of the same color."""

    beat: float
    color: Color
    x: int
    y: int
    direction: CutDirection
    tail_beat: float
    tail_x: int
    tail_y: int
    tail_direction: CutDirection
    head_multiplier: float = 1.0
    tail_multiplier: float = 1.0
    mid_anchor: int = 0


@dataclass(frozen=True, slots=True)
class Chain:
    """v3 'burst slider'."""

    beat: float
    color: Color
    x: int
    y: int
    direction: CutDirection
    tail_beat: float
    tail_x: int
    tail_y: int
    slice_count: int
    squish: float = 1.0


@dataclass(frozen=True, slots=True)
class RotationEvent:
    """90/360 rotation. ``degrees`` is relative and a multiple of 15."""

    beat: float
    degrees: int
    early: bool = True


# --- Lighting ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BasicEvent:
    """Classic (v2-style) light/environment event; also valid in v3/v4.

    ``type`` follows the game's event type ids (0-4 lights, 8 ring spin,
    9 ring zoom, 12/13 laser speed, ...). ``color`` is an optional Chroma RGBA.
    """

    beat: float
    type: int
    value: int
    float_value: float = 1.0
    color: tuple[float, float, float, float] | None = None
    custom_data: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ColorBoostEvent:
    beat: float
    on: bool


@dataclass(frozen=True, slots=True)
class EventBoxGroup:
    """v3 group lighting. ``boxes`` stays loosely typed until the lighting
    workstream refines it; renderers own the per-environment group ids."""

    beat: float
    group_id: int
    kind: Literal["color", "rotation", "translation", "fx"]
    boxes: tuple[dict[str, Any], ...] = ()


@dataclass(slots=True)
class Lightshow:
    basic_events: list[BasicEvent] = field(default_factory=list)
    boost_events: list[ColorBoostEvent] = field(default_factory=list)
    event_box_groups: list[EventBoxGroup] = field(default_factory=list)
    uses_chroma: bool = False


# --- Containers -------------------------------------------------------------


@dataclass(slots=True)
class DifficultyMap:
    characteristic: Characteristic
    difficulty: Difficulty
    njs: float = 16.0
    njs_offset: float = 0.0
    notes: list[Note] = field(default_factory=list)
    bombs: list[Bomb] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    chains: list[Chain] = field(default_factory=list)
    rotations: list[RotationEvent] = field(default_factory=list)
    # None means "use Beatmap.lightshow" (v4 style shared lightshow).
    lightshow: Lightshow | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class BpmChange:
    beat: float
    bpm: float


@dataclass(slots=True)
class Beatmap:
    song_name: str
    bpm: float
    audio_path: str = "song.egg"
    song_sub_name: str = ""
    song_author: str = ""
    mapper: str = "bsmapper"
    cover_path: str | None = None
    preview_start: float = 10.0
    preview_duration: float = 10.0
    environment: str = "DefaultEnvironment"
    song_time_offset: float = 0.0
    bpm_changes: list[BpmChange] = field(default_factory=list)
    difficulties: list[DifficultyMap] = field(default_factory=list)
    lightshow: Lightshow = field(default_factory=Lightshow)

    def _tempo_segments(self) -> list[tuple[float, float, float]]:
        """(start_beat, start_seconds, bpm) for each constant-tempo segment."""
        segments = [(0.0, 0.0, self.bpm)]
        for change in sorted(self.bpm_changes, key=lambda c: c.beat):
            if change.beat <= 0:
                segments[0] = (0.0, 0.0, change.bpm)
                continue
            start_beat, start_s, bpm = segments[-1]
            segments.append(
                (change.beat, start_s + (change.beat - start_beat) * 60.0 / bpm, change.bpm)
            )
        return segments

    def beat_to_seconds(self, beat: float) -> float:
        segments = self._tempo_segments()
        i = bisect.bisect_right([s[0] for s in segments], beat) - 1
        start_beat, start_s, bpm = segments[max(i, 0)]
        return start_s + (beat - start_beat) * 60.0 / bpm

    def seconds_to_beat(self, seconds: float) -> float:
        segments = self._tempo_segments()
        i = bisect.bisect_right([s[1] for s in segments], seconds) - 1
        start_beat, start_s, bpm = segments[max(i, 0)]
        return start_beat + (seconds - start_s) * bpm / 60.0
