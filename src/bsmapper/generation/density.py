"""How dense a difficulty should be: the NPS the model is conditioned on.

Fitted on the dataset v0 train split (Standard, 5906 difficulties): NPS grows roughly
linearly with BPM within a difficulty (r 0.2 to 0.4). Other modes run at a fixed share of
Standard, from the ratio of their medians. Values stay inside Standard's interquartile range
(a little wider), so an odd BPM cannot ask for an extreme map.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from bsmapper.schema import Characteristic, Difficulty

# Standard: nps = slope * bpm + intercept, clamped to [lo, hi].
NPS_FIT: dict[Difficulty, tuple[float, float, float, float]] = {
    Difficulty.EASY: (0.0035, 1.09, 1.0, 2.1),
    Difficulty.NORMAL: (0.0068, 1.29, 1.5, 3.0),
    Difficulty.HARD: (0.0089, 1.87, 2.3, 4.0),
    Difficulty.EXPERT: (0.0116, 2.47, 3.2, 5.2),
    Difficulty.EXPERT_PLUS: (0.0168, 2.75, 4.0, 6.5),
}
MODE_SHARE: dict[Characteristic, float] = {
    Characteristic.STANDARD: 1.0,
    Characteristic.NO_ARROWS: 1.0,
    Characteristic.ONE_SABER: 0.73,
    Characteristic.DEGREE_90: 0.9,
    Characteristic.DEGREE_360: 0.9,
    Characteristic.LIGHTSHOW: 0.0,
}


def target_nps(difficulty: Difficulty, characteristic: Characteristic, bpm: float) -> float:
    """Typical notes per second for this difficulty and mode at ``bpm``."""
    slope, intercept, lo, hi = NPS_FIT[difficulty]
    nps = min(max(slope * bpm + intercept, lo), hi)
    return round(nps * MODE_SHARE.get(characteristic, 1.0), 2)


# Conditioning NPS = wanted NPS x gain. v0 reaches only part of the NPS it is conditioned on;
# these are 1 / the median achieved/requested ratio on Stronger, Toxic and a trap song (seed
# 0, rule masks on, after post-processing). Songs vary a lot (0.4 to 1.2), which is what the
# per-window feedback in ``model`` is for.
NPS_GAIN: dict[Difficulty, float] = {
    Difficulty.EASY: 1.12,
    Difficulty.NORMAL: 1.12,
    Difficulty.HARD: 1.27,
    Difficulty.EXPERT: 1.22,
    Difficulty.EXPERT_PLUS: 1.33,
}

# Longest gap between notes (beats) that may stay in a loud passage; longer ones only where
# the music is quiet. The per-window feedback resamples windows that break this.
GAP_LIMIT_BEATS: dict[Difficulty, float] = {
    Difficulty.EASY: 8.0,
    Difficulty.NORMAL: 6.0,
    Difficulty.HARD: 4.0,
    Difficulty.EXPERT: 2.0,
    Difficulty.EXPERT_PLUS: 2.0,
}
SILENT_RMS = 0.02  # beat-synchronous RMS (song-relative) below this is silence
QUIET_SHARE = 0.5  # a gap is in quiet music when its median RMS is below this x the song's


# note_v1 follows its NPS token much more closely (calibrated in ``NPS_GAIN_BY_VERSION``).
NPS_GAIN_V1: dict[Difficulty, float] = dict.fromkeys(Difficulty, 1.0)
NPS_GAIN_BY_VERSION: dict[int, dict[Difficulty, float]] = {1: NPS_GAIN, 2: NPS_GAIN_V1}


def gains_for(version: int) -> dict[Difficulty, float]:
    """Conditioning gains for a checkpoint's tokenizer version (1 = note_v0)."""
    return NPS_GAIN_BY_VERSION.get(version, NPS_GAIN)


def conditioning_nps(
    difficulty: Difficulty, wanted: float, gains: dict[Difficulty, float] | None = None
) -> float:
    return round(wanted * (gains or NPS_GAIN)[difficulty], 2)


class Loudness:
    """Song-relative loudness per tick (the RMS column of the model features)."""

    def __init__(self, rms_ticks: np.ndarray) -> None:
        self.rms = np.asarray(rms_ticks, dtype=np.float64)
        active = self.rms[self.rms > SILENT_RMS]
        self.typical = float(np.median(active)) if len(active) else 1.0
        self.quiet = QUIET_SHARE * self.typical

    def activity(self, start_beat: float, end_beat: float) -> float:
        """0 (silent) .. 1 (typical or louder) over a span of beats."""
        seg = self.rms[int(start_beat * 24) : int(end_beat * 24)]
        if len(seg) == 0:
            return 0.0
        return float(np.clip(seg.mean() / max(self.typical, 1e-6), 0.0, 1.0))

    def loud(self, start_beat: float, end_beat: float) -> bool:
        seg = self.rms[int(start_beat * 24) : int(end_beat * 24)]
        return len(seg) > 0 and float(np.median(seg)) >= self.quiet


def loud_gaps(beats: list[float], loudness: Loudness, limit: float) -> list[float]:
    """Gaps between consecutive note times longer than ``limit`` over loud music."""
    out = []
    for a, b in zip(beats, beats[1:], strict=False):
        if b - a > limit + 1e-6 and loudness.loud(a, b):
            out.append(b - a)
    return out


# Share of dataset v0 train difficulties that use each object type (bombs, arcs, chains;
# walls are in about 90% everywhere). Bombs are on by default when at least ~45% of human
# maps of that mode and difficulty use them: the conditioning flags mean "this map uses X".
_USES = {
    # characteristic: {difficulty: (bombs, arcs, chains)}
    Characteristic.STANDARD: {
        Difficulty.EASY: (0.27, 0.32, 0.11),
        Difficulty.NORMAL: (0.31, 0.33, 0.13),
        Difficulty.HARD: (0.39, 0.35, 0.16),
        Difficulty.EXPERT: (0.46, 0.34, 0.18),
        Difficulty.EXPERT_PLUS: (0.54, 0.36, 0.19),
    },
    Characteristic.ONE_SABER: {
        Difficulty.EASY: (0.44, 0.50, 0.12),
        Difficulty.NORMAL: (0.41, 0.66, 0.41),
        Difficulty.HARD: (0.47, 0.56, 0.42),
        Difficulty.EXPERT: (0.48, 0.51, 0.39),
        Difficulty.EXPERT_PLUS: (0.55, 0.64, 0.45),
    },
    Characteristic.DEGREE_360: {
        Difficulty.EASY: (0.08, 0.54, 0.31),
        Difficulty.NORMAL: (0.14, 0.57, 0.43),
        Difficulty.HARD: (0.30, 0.40, 0.30),
        Difficulty.EXPERT: (0.20, 0.16, 0.15),
        Difficulty.EXPERT_PLUS: (0.32, 0.16, 0.10),
    },
}
_USES[Characteristic.DEGREE_90] = _USES[Characteristic.DEGREE_360]
USE_THRESHOLD = 0.45

# Arcs and chains are on by default from these difficulties up, whatever share of human maps
# uses them (#12: a third of Standard Expert maps have arcs, so the share rule left them off).
# note_v1 learned them from 118k arcs in Standard Expert alone, and post-processing
# re-anchors or drops malformed ones (``postprocess.fix_spans``). With the flag on it places
# 12-40 arcs a minute on Hard to Expert+, the range of human maps that use arcs (median
# 13-18). Chains (fast bursts) start at Expert. Below these, the human share decides.
SPANS_FROM: dict[Characteristic, tuple[Difficulty | None, Difficulty | None]] = {
    # characteristic: (arcs from, chains from)
    Characteristic.STANDARD: (Difficulty.HARD, Difficulty.EXPERT),
    Characteristic.ONE_SABER: (Difficulty.HARD, Difficulty.EXPERT),
    Characteristic.DEGREE_90: (Difficulty.HARD, Difficulty.EXPERT),
    Characteristic.DEGREE_360: (Difficulty.HARD, Difficulty.EXPERT),
}


def _from(difficulty: Difficulty, lowest: Difficulty | None) -> bool:
    return lowest is not None and difficulty.rank >= lowest.rank


def object_flags(difficulty: Difficulty, characteristic: Characteristic) -> dict[str, bool]:
    """Default ``GenerationRequest`` allow_* flags: bombs where most human maps of this
    kind use them, arcs and chains from ``SPANS_FROM`` up (else where most maps use them)."""
    bombs, arcs, chains = _USES.get(characteristic, _USES[Characteristic.STANDARD])[difficulty]
    arcs_from, chains_from = SPANS_FROM.get(characteristic, (None, None))
    allow_arcs = arcs >= USE_THRESHOLD or _from(difficulty, arcs_from)
    allow_chains = chains >= USE_THRESHOLD or _from(difficulty, chains_from)
    if characteristic == Characteristic.NO_ARROWS:
        allow_arcs = allow_chains = False
    return {
        "allow_bombs": bombs >= USE_THRESHOLD,
        "allow_obstacles": True,
        "allow_arcs": allow_arcs,
        "allow_chains": allow_chains,
    }


def configured_objects(path: Path | None = None) -> dict[str, bool]:
    """``[generate] arcs`` / ``chains`` from ``~/.bsmapper/config.toml`` (only the keys set):
    true turns the type on for every difficulty, false off; unset keeps ``object_flags``."""
    from bsmapper.generation.install import config_path

    cfg = path or config_path()
    if not cfg.is_file():
        return {}
    with cfg.open("rb") as f:
        table = tomllib.load(f).get("generate", {})
    out = {}
    for key in ("bombs", "walls", "arcs", "chains"):
        value = table.get(key)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise ValueError(f"{cfg}: [generate] {key} must be true or false")
        out[key] = value
    return out
