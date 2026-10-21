"""The map's own beat grid: the human ground truth the dataset is aligned to.

A map's grid is its BPM, BPM changes and ``_songTimeOffset`` (beat 0 at that many
seconds, as in ``audio.benchmark``). Some mappers also place every note at a
constant fraction of a beat off that grid (they never set the offset); the
``phase`` corrects that by moving the whole map, its lights and the feature
sampling together, so notes stay where they were in the audio.

Features are resampled onto this grid (row ``j`` = map beat ``j / 24``), not onto
the grid WS2 detects. ``grid_drift`` measures how far the detected grid is from
the map's grid, as a quality stat for WS2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from bsmapper.audio.features import beat_synchronous
from bsmapper.audio.grid import BeatGrid
from bsmapper.schema import Beatmap, BpmChange, DifficultyMap, Lightshow
from bsmapper.tokenizer.grid import TICKS_PER_BEAT, WINDOW_BEATS

FEATURE_COLUMNS = ("mel", "onset_strength", "spectral_flux", "rms")
N_MELS = 80
FEATURE_DIM = N_MELS + len(FEATURE_COLUMNS) - 1
MEL_FLOOR_DB = -80.0  # value of "silence" rows past the end of the audio


@dataclass(frozen=True, slots=True)
class MapTiming:
    bpm: float
    bpm_changes: tuple[BpmChange, ...] = ()
    song_time_offset: float = 0.0
    phase: float = 0.0  # beats subtracted from every object; see the module docstring

    @classmethod
    def of(cls, beatmap: Beatmap, phase: float = 0.0) -> MapTiming:
        return cls(
            beatmap.bpm,
            tuple(sorted(beatmap.bpm_changes, key=lambda c: c.beat)),
            beatmap.song_time_offset,
            phase,
        )

    def _grid(self) -> BeatGrid:
        return BeatGrid(self.bpm, self.song_time_offset, list(self.bpm_changes))

    def seconds(self, beats: float | np.ndarray) -> np.ndarray:
        """Audio time of (phase-corrected) map beats."""
        return self._grid().beat_to_seconds(np.asarray(beats, dtype=np.float64) + self.phase)

    def beats(self, seconds: float | np.ndarray) -> np.ndarray:
        return self._grid().seconds_to_beat(np.asarray(seconds, dtype=np.float64)) - self.phase

    def beatmap(self) -> Beatmap:
        """A metadata-free Beatmap with this timing, for rules and NPS (no song offset:
        those only use time differences)."""
        changes = [BpmChange(c.beat - self.phase, c.bpm) for c in self.bpm_changes]
        return Beatmap(song_name="", bpm=self.bpm, bpm_changes=changes)

    def to_dict(self) -> dict:
        return {
            "bpm": self.bpm,
            "bpm_changes": [[c.beat, c.bpm] for c in self.bpm_changes],
            "song_time_offset": self.song_time_offset,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MapTiming:
        return cls(
            float(d["bpm"]),
            tuple(BpmChange(float(b), float(v)) for b, v in d.get("bpm_changes", [])),
            float(d.get("song_time_offset", 0.0)),
            float(d.get("phase", 0.0)),
        )


def off_grid_ratio(beats: np.ndarray, tolerance_ticks: float = 0.1) -> float:
    """Share of times more than ``tolerance_ticks`` away from the 1/24-beat grid."""
    if len(beats) == 0:
        return 0.0
    t = np.asarray(beats, dtype=np.float64) * TICKS_PER_BEAT
    return float(np.mean(np.abs(t - np.round(t)) > tolerance_ticks))


def grid_phase(
    beats: np.ndarray, tolerance_ticks: float = 0.1, min_gain: float = 0.1
) -> tuple[float, float, float]:
    """(phase, off-grid ratio before, after): the constant note offset to remove.

    The phase is the most common note position modulo half a beat
    (``audio.benchmark.note_phase``). It is kept only if it moves at least
    ``min_gain`` of the notes onto the grid; otherwise the phase is 0.
    """
    from bsmapper.audio.benchmark import note_phase

    before = off_grid_ratio(beats, tolerance_ticks)
    if before < min_gain or len(beats) < 16:
        return 0.0, before, before
    phase = note_phase(np.asarray(beats, dtype=np.float64))
    after = off_grid_ratio(np.asarray(beats) - phase, tolerance_ticks)
    if before - after < min_gain:
        return 0.0, before, before
    return phase, before, after


def shift_map(dmap: DifficultyMap, delta: float) -> DifficultyMap:
    """Every object moved by ``delta`` beats (spans keep their length)."""
    if delta == 0:
        return dmap

    def mv(objs):
        return [replace(o, beat=o.beat + delta) for o in objs]

    def mv_tail(objs):
        return [replace(o, beat=o.beat + delta, tail_beat=o.tail_beat + delta) for o in objs]

    return replace(
        dmap,
        notes=mv(dmap.notes),
        bombs=mv(dmap.bombs),
        obstacles=mv(dmap.obstacles),
        arcs=mv_tail(dmap.arcs),
        chains=mv_tail(dmap.chains),
        rotations=mv(dmap.rotations),
        lightshow=None if dmap.lightshow is None else shift_lightshow(dmap.lightshow, delta),
    )


def shift_lightshow(ls: Lightshow, delta: float) -> Lightshow:
    if delta == 0:
        return ls
    return Lightshow(
        basic_events=[replace(e, beat=e.beat + delta) for e in ls.basic_events],
        boost_events=[replace(e, beat=e.beat + delta) for e in ls.boost_events],
        event_box_groups=[replace(e, beat=e.beat + delta) for e in ls.event_box_groups],
        uses_chroma=ls.uses_chroma,
    )


def audio_windows(timing: MapTiming, duration_s: float) -> int:
    """16-beat windows needed to cover the audio on the map's grid."""
    end_beat = float(timing.beats(duration_s))
    return max(int(np.ceil(end_beat / WINDOW_BEATS - 1e-9)), 1)


def map_grid_features(
    frame_features: dict[str, np.ndarray],
    frame_rate: float,
    timing: MapTiming,
    duration_s: float,
    n_windows: int,
) -> np.ndarray:
    """(n_windows * 384, FEATURE_DIM) float16: mel (80), onset_strength, spectral_flux, rms.

    Rows whose time falls outside the audio (before 0 s or after the end) are
    silence: mel at -80 dB, the rest 0.
    """
    n_beats = n_windows * WINDOW_BEATS
    sync = beat_synchronous(
        {k: frame_features[k] for k in FEATURE_COLUMNS},
        frame_rate,
        timing.seconds,
        n_beats,
        TICKS_PER_BEAT,
    )
    cols = [sync["mel"]] + [sync[k][:, None] for k in FEATURE_COLUMNS[1:]]
    out = np.concatenate(cols, axis=1)
    t = timing.seconds(np.arange(n_beats * TICKS_PER_BEAT) / TICKS_PER_BEAT)
    outside = (t < 0) | (t > duration_s)
    out[outside, :N_MELS] = MEL_FLOOR_DB
    out[outside, N_MELS:] = 0.0
    return out.astype(np.float16)


@dataclass(frozen=True, slots=True)
class GridDrift:
    """Where the detected grid's beats land on the map's grid."""

    bpm_ratio: float  # detected bpm / map bpm (at beat 0)
    relation: str  # "same", "double", "half", "3/2", "2/3" or "other"
    median_ticks: float  # signed; > 0 means detected beats are late
    median_abs_ticks: float
    p95_abs_ticks: float
    within_1_tick: float
    within_2_ticks: float
    beats: int


_RELATIONS = (("same", 1.0, 1.0), ("double", 2.0, 0.5), ("half", 0.5, 1.0),
              ("3/2", 1.5, 0.5), ("2/3", 2 / 3, 1.0))  # fmt: skip


def grid_drift(detected: BeatGrid, timing: MapTiming, duration_s: float) -> GridDrift:
    """Phase error of the detected grid's beats on the map grid, in ticks.

    At double or 3/2 tempo the detected beats are scored modulo half a map beat,
    since every one of them still sits on a mappable position.
    """
    ratio = detected.bpm / timing.bpm
    relation, unit = "other", 1.0
    for name, r, u in _RELATIONS:
        if abs(ratio / r - 1) < 0.02:
            relation, unit = name, u
            break
    t = detected.beat_times(duration_s)
    t = t[(t >= 0) & (t <= duration_s)]
    if len(t) == 0:
        return GridDrift(ratio, relation, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    b = timing.beats(t)
    err = (b / unit - np.round(b / unit)) * unit * TICKS_PER_BEAT
    a = np.abs(err)
    return GridDrift(
        bpm_ratio=float(ratio),
        relation=relation,
        median_ticks=float(np.median(err)),
        median_abs_ticks=float(np.median(a)),
        p95_abs_ticks=float(np.percentile(a, 95)),
        within_1_tick=float(np.mean(a <= 1.0)),
        within_2_ticks=float(np.mean(a <= 2.0)),
        beats=len(t),
    )


ONSET_SHIFTS = range(-12, 13)  # ticks, half a beat each way


def onset_alignment(onset: np.ndarray, note_beats: list[float]) -> dict[str, float]:
    """How well the notes sit on onsets of the resampled ``onset_strength`` row.

    ``onset_ratio`` is the mean onset strength one tick after each note's tick (the
    feature's peaks lag the beat by about 12 ms, see ``audio.analyzer``) over the
    song mean; well-aligned maps score about 2-3, unrelated timing about 1.
    ``onset_best_shift`` is the shift in ticks (-12..12) where the ratio peaks; far
    from +1 means the map's grid, offset or audio do not line up.
    """
    ticks = np.unique(np.round(np.asarray(note_beats) * TICKS_PER_BEAT).astype(np.int64))
    mean = float(onset.mean()) if len(onset) else 0.0
    if len(ticks) < 8 or mean <= 0:
        return {
            "onset_ratio": float("nan"),
            "onset_best_shift": 0,
            "onset_best_ratio": float("nan"),
        }
    ratios = np.array([onset[np.clip(ticks + s, 0, len(onset) - 1)].mean() for s in ONSET_SHIFTS])
    ratios /= mean
    best = int(np.argmax(ratios))
    return {
        "onset_ratio": float(ratios[list(ONSET_SHIFTS).index(1)]),
        "onset_best_shift": int(ONSET_SHIFTS[best]),
        "onset_best_ratio": float(ratios[best]),
    }
