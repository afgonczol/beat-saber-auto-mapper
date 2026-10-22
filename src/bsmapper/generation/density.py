"""How dense a difficulty should be: the NPS the model is conditioned on.

Fitted on the dataset v0 train split (Standard, 5906 difficulties): NPS grows roughly
linearly with BPM within a difficulty (r 0.2 to 0.4). Other modes run at a fixed share of
Standard, from the ratio of their medians. Values stay inside Standard's interquartile range
(a little wider), so an odd BPM cannot ask for an extreme map.
"""

from __future__ import annotations

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


# The v0 model reaches about 75-80% of the NPS it is conditioned on (Stronger, Expert+ and
# Normal at several targets), so heuristic targets are scaled up by this before conditioning.
NPS_CONDITIONING_GAIN = 1.25

# Share of dataset v0 train difficulties that use each object type (bombs, arcs, chains;
# walls are in about 90% everywhere). A type is on by default when at least ~45% of human
# maps of that mode and difficulty use it: the conditioning flags mean "this map uses X".
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


def object_flags(difficulty: Difficulty, characteristic: Characteristic) -> dict[str, bool]:
    """Default ``GenerationRequest`` allow_* flags: what most human maps of this kind use."""
    bombs, arcs, chains = _USES.get(characteristic, _USES[Characteristic.STANDARD])[difficulty]
    if characteristic == Characteristic.NO_ARROWS:
        arcs = chains = 0.0
    return {
        "allow_bombs": bombs >= USE_THRESHOLD,
        "allow_obstacles": True,
        "allow_arcs": arcs >= USE_THRESHOLD,
        "allow_chains": chains >= USE_THRESHOLD,
    }
