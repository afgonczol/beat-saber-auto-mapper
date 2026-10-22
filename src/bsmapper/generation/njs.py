"""Note jump speed (NJS) and start-beat offset per difficulty.

Beat Saber spawns a note ``half jump duration`` (HJD) beats before its beat. The game
starts from 4 beats, halves that while the jump distance would exceed 18 m, adds the
map's offset and clamps the result at 0.25 beats. We pick an NJS per difficulty and the
offset that brings the reaction time (HJD in seconds) to a per-difficulty target, so a
slow song does not give Expert+ two seconds of warning and a fast one does not rush Easy.

Defaults (``JUMP_TABLE``): Easy to Hard sit near the medians of the v0 human Standard
maps; Expert and Expert+ get more time than the median (about its 75th percentile), after
playtest feedback that 0.6 s on Expert+ felt rushed. Both can be overridden per
difficulty: ``--njs`` / ``--reaction-time`` on the CLI, ``njs`` / ``reaction_time`` in
the server's job options, and ``[jump.<Difficulty>]`` tables in ``~/.bsmapper/config.toml``::

    [jump.ExpertPlus]
    njs = 17
    reaction_time = 0.75
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from bsmapper.schema import Difficulty

BASE_HJD_BEATS = 4.0
MAX_HALF_JUMP_DISTANCE_M = 17.999
MIN_HJD_BEATS = 0.25
OFFSET_STEP = 0.05


@dataclass(frozen=True, slots=True)
class JumpSetting:
    njs: float
    reaction_s: float  # target time from spawn to the note's beat


# Human v0 Standard medians (NJS / reaction time): Easy 10 / 1.20 s, Normal 12 / 1.00 s,
# Hard 14 / 0.83 s (13 / 0.88 s at 90-120 BPM), Expert 16 / 0.69 s, Expert+ 18 / 0.59 s.
JUMP_TABLE: dict[Difficulty, JumpSetting] = {
    Difficulty.EASY: JumpSetting(10.0, 1.2),
    Difficulty.NORMAL: JumpSetting(12.0, 1.0),
    Difficulty.HARD: JumpSetting(13.0, 0.88),
    Difficulty.EXPERT: JumpSetting(16.0, 0.8),
    Difficulty.EXPERT_PLUS: JumpSetting(17.0, 0.7),  # about 23.8 m jump distance
}
MIN_REACTION_S, MAX_REACTION_S = 0.3, 2.0


@dataclass(frozen=True, slots=True)
class JumpOverride:
    """A per-difficulty override; None keeps the table's value."""

    njs: float | None = None
    reaction_s: float | None = None

    def over(self, other: JumpOverride | None) -> JumpOverride:
        """This override, with ``other`` filling the fields this one leaves out."""
        if other is None:
            return self
        return JumpOverride(
            self.njs if self.njs is not None else other.njs,
            self.reaction_s if self.reaction_s is not None else other.reaction_s,
        )


def configured_jumps(path: Path | None = None) -> dict[Difficulty, JumpOverride]:
    """``[jump.<Difficulty>]`` tables from the user config (``Expert+`` works as a name)."""
    from bsmapper.generation.install import config_path

    cfg = path or config_path()
    if not cfg.is_file():
        return {}
    with cfg.open("rb") as f:
        table = tomllib.load(f).get("jump", {})
    out = {}
    names = {d.value.lower(): d for d in Difficulty} | {"expert+": Difficulty.EXPERT_PLUS}
    for name, values in table.items():
        diff = names.get(str(name).lower())
        if diff is None:
            raise ValueError(f"{cfg}: unknown difficulty [jump.{name}]")
        out[diff] = JumpOverride(values.get("njs"), values.get("reaction_time"))
    return out


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
    difficulty: Difficulty,
    bpm: float,
    njs: float | None = None,
    reaction_s: float | None = None,
) -> tuple[float, float]:
    """(NJS, start-beat offset) for ``difficulty`` at ``bpm``.

    ``njs`` and ``reaction_s`` (seconds from spawn to the note's beat) override the table.
    """
    setting = JUMP_TABLE[difficulty]
    njs = njs if njs and njs > 0 else setting.njs
    reaction = reaction_s if reaction_s and reaction_s > 0 else setting.reaction_s
    reaction = min(max(reaction, MIN_REACTION_S), MAX_REACTION_S)
    target_hjd = reaction * bpm / 60.0
    offset = target_hjd - default_hjd(njs, bpm)
    offset = round(round(offset / OFFSET_STEP) * OFFSET_STEP, 3)
    # Never let the clamp at 0.25 beats hide a too-negative offset.
    offset = max(offset, MIN_HJD_BEATS - default_hjd(njs, bpm))
    return njs, offset + 0.0  # normalise -0.0
