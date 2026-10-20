"""Stage boundaries of the mapping pipeline.

Every workstream implements one of these protocols, so stages can be built,
tested and swapped independently (e.g. a stub generator while the ML model
trains). Changes here go through a "contract change" PR (see CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from bsmapper.schema import Beatmap, Characteristic, Difficulty, DifficultyMap, Lightshow

# --- Audio analysis -----------------------------------------------------------


@dataclass(slots=True)
class Section:
    start_s: float
    end_s: float
    label: str  # e.g. "intro", "verse", "chorus", "drop", "break", "outro"
    energy: float  # 0..1, relative to the song


@dataclass(slots=True)
class AudioAnalysis:
    duration_s: float
    bpm: float
    beat_times: np.ndarray  # seconds, one per detected beat
    downbeat_times: np.ndarray  # seconds
    sections: list[Section]
    # Frame-level features, all sampled at `frame_rate` Hz, keyed by name
    # ("mel", "onset_strength", "rms", ...). Shapes are (frames, dims) or (frames,).
    features: dict[str, np.ndarray] = field(default_factory=dict)
    frame_rate: float = 100.0
    # Beat-synchronous features at `ticks_per_beat` resolution, if computed.
    beat_features: dict[str, np.ndarray] = field(default_factory=dict)
    ticks_per_beat: int = 24


class AudioAnalyzer(Protocol):
    def analyze(self, audio_path: Path) -> AudioAnalysis: ...


# --- Note generation -----------------------------------------------------------


@dataclass(slots=True)
class GenerationRequest:
    characteristic: Characteristic
    difficulty: Difficulty
    target_nps: float | None = None  # None = model/heuristic picks
    style: float | None = None  # 0 = flowy .. 1 = technical; None = no preference
    njs: float | None = None
    seed: int = 0
    temperature: float = 1.0
    allow_bombs: bool = True
    allow_obstacles: bool = True
    allow_arcs: bool = True
    allow_chains: bool = True


class NoteGenerator(Protocol):
    def generate(self, analysis: AudioAnalysis, request: GenerationRequest) -> DifficultyMap: ...


# --- Lighting --------------------------------------------------------------------

LightingStyle = Literal["vanilla", "v3", "chroma"]


class LightingGenerator(Protocol):
    def generate(
        self, analysis: AudioAnalysis, beatmap: Beatmap, style: LightingStyle
    ) -> Lightshow: ...


# --- Map I/O -----------------------------------------------------------------------

FormatVersion = Literal["v2", "v3", "v4"]


class MapReader(Protocol):
    def read(self, map_dir: Path) -> Beatmap: ...


class MapWriter(Protocol):
    def write(self, beatmap: Beatmap, out_dir: Path, version: FormatVersion = "v3") -> Path: ...


# --- Playability -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Issue:
    rule: str  # e.g. "parity", "vision_block", "nps_ceiling"
    beat: float
    severity: Literal["error", "warning"]
    message: str


class PlayabilityChecker(Protocol):
    def check(self, dmap: DifficultyMap, beatmap: Beatmap) -> list[Issue]: ...
