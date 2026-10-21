"""Note jump speed (NJS) and start-beat offset per difficulty.

Beat Saber spawns a note ``half jump duration`` (HJD) beats before its beat. The game
starts from 4 beats, halves that while the jump distance would exceed 18 m, adds the
map's offset and clamps the result at 0.25 beats. We pick an NJS per difficulty and the
offset that brings the reaction time (HJD in seconds) to a per-difficulty target, so a
slow song does not give Expert+ two seconds of warning and a fast one does not rush Easy.
"""

from __future__ import annotations

from dataclasses import dataclass

from bsmapper.schema import Difficulty

BASE_HJD_BEATS = 4.0
MAX_HALF_JUMP_DISTANCE_M = 17.999
MIN_HJD_BEATS = 0.25
OFFSET_STEP = 0.05


@dataclass(frozen=True, slots=True)
class JumpSetting:
    njs: float
    reaction_s: float  # target time from spawn to the note's beat


JUMP_TABLE: dict[Difficulty, JumpSetting] = {
    Difficulty.EASY: JumpSetting(10.0, 1.0),
    Difficulty.NORMAL: JumpSetting(10.0, 0.9),
    Difficulty.HARD: JumpSetting(12.0, 0.8),
    Difficulty.EXPERT: JumpSetting(16.0, 0.7),
    Difficulty.EXPERT_PLUS: JumpSetting(18.0, 0.6),
}


def default_hjd(njs: float, bpm: float) -> float:
    """The game's half jump duration in beats before the map's offset is added."""
    seconds_per_beat = 60.0 / bpm
    hjd = BASE_HJD_BEATS
    while njs * seconds_per_beat * hjd > MAX_HALF_JUMP_DISTANCE_M:
        hjd /= 2.0
    return hjd


def reaction_time(njs: float, offset: float, bpm: float) -> float:
    """Seconds between a note spawning and reaching the player, as the game computes it."""
    return max(default_hjd(njs, bpm) + offset, MIN_HJD_BEATS) * 60.0 / bpm


def jump_distance(njs: float, offset: float, bpm: float) -> float:
    return 2.0 * njs * reaction_time(njs, offset, bpm)


def njs_and_offset(
    difficulty: Difficulty, bpm: float, njs: float | None = None
) -> tuple[float, float]:
    """(NJS, start-beat offset) for ``difficulty`` at ``bpm``; ``njs`` overrides the table."""
    setting = JUMP_TABLE[difficulty]
    njs = njs if njs and njs > 0 else setting.njs
    target_hjd = setting.reaction_s * bpm / 60.0
    offset = target_hjd - default_hjd(njs, bpm)
    offset = round(round(offset / OFFSET_STEP) * OFFSET_STEP, 3)
    # Never let the clamp at 0.25 beats hide a too-negative offset.
    offset = max(offset, MIN_HJD_BEATS - default_hjd(njs, bpm))
    return njs, offset + 0.0  # normalise -0.0
