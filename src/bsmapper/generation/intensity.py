"""Intensity: how hard a map is *within* its difficulty label.

``GenerationRequest.intensity`` runs from 0 to 1. 0.5 is a typical human map for the label
and BPM, 1.0 is about the hardest 10%. It maps linearly to a percentile of the human NPS
spread, from p10 at 0 to p90 at 1. Dataset v0 (Standard train split) gives, per
difficulty, NPS divided by the fitted median NPS at its BPM (``density.NPS_FIT``). The NPS
wanted in the map is ``target_nps`` x ratio(percentile) / ratio(p50), so 0.5 keeps the
earlier defaults.

**Above** ``SHIFT_ABOVE`` (0.75) the v0 model, which has no intensity token, is
conditioned on the next difficulty's DIFF token. An Expert map at intensity 0.8 then
decodes as DIFF_ExpertPlus, with Expert+ patterns at an NPS between the two. Expert+
cannot shift, so it gets the top of its own NPS range, plus the per-window density
feedback. NJS and reaction time always follow the **label** (``njs.JUMP_TABLE``).

The rules' hard ceilings follow the shift (``relaxed_rules``). The label's NPS ceiling is
raised to the human p95 peak 2-s NPS of the difficulty it is decoded as, and its minimum
same-hand swing interval is lowered to that difficulty's human p5. A shifted map may then
be as dense as most human maps of the harder difficulty, but never denser.

When a model gets its own intensity token (note_v1), pass ``model_intensity=True`` to
``plan``: the label stays, the percentile target is unchanged, and there is no shift.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bsmapper.generation.density import target_nps
from bsmapper.rules import RuleChecker, RulesConfig
from bsmapper.schema import Characteristic, Difficulty, DifficultyMap

DEFAULT_INTENSITY = 0.5
SHIFT_ABOVE = 0.75
P_LOW, P_HIGH = 0.1, 0.9  # percentiles at intensity 0 and 1

# Human v0 Standard train split, NPS / fitted median NPS at the map's BPM.
_Q = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
NPS_RATIO_QUANTILES: dict[Difficulty, tuple[float, ...]] = {
    Difficulty.EASY: (0.611, 0.758, 0.935, 1.177, 1.446, 1.617),
    Difficulty.NORMAL: (0.678, 0.806, 0.968, 1.168, 1.378, 1.522),
    Difficulty.HARD: (0.738, 0.846, 0.976, 1.130, 1.288, 1.380),
    Difficulty.EXPERT: (0.759, 0.867, 0.991, 1.117, 1.242, 1.359),
    Difficulty.EXPERT_PLUS: (0.736, 0.867, 0.999, 1.125, 1.251, 1.347),
}
# Same maps (300 sampled per difficulty, decoded from tokens): p95 of the densest 2 s, and
# p5 of the fastest same-hand swing (gaps over 0.09 s, the rules' stack/slider window).
HUMAN_PEAK_NPS_P95: dict[Difficulty, float] = {
    Difficulty.EASY: 5.5,
    Difficulty.NORMAL: 7.0,
    Difficulty.HARD: 9.5,
    Difficulty.EXPERT: 12.0,
    Difficulty.EXPERT_PLUS: 14.0,
}
HUMAN_SWING_INTERVAL_P5: dict[Difficulty, float] = {
    Difficulty.EASY: 0.207,
    Difficulty.NORMAL: 0.167,
    Difficulty.HARD: 0.135,
    Difficulty.EXPERT: 0.117,
    Difficulty.EXPERT_PLUS: 0.098,
}
_ORDER = sorted(Difficulty, key=lambda d: d.rank)


def clamp_intensity(intensity: float | None) -> float:
    return DEFAULT_INTENSITY if intensity is None else float(np.clip(intensity, 0.0, 1.0))


def percentile(intensity: float | None) -> float:
    return P_LOW + (P_HIGH - P_LOW) * clamp_intensity(intensity)


def nps_ratio(difficulty: Difficulty, q: float) -> float:
    """Human NPS at percentile ``q`` over the median, for ``difficulty``."""
    ratios = NPS_RATIO_QUANTILES[difficulty]
    at = float(np.interp(q, _Q, ratios))
    return at / float(np.interp(0.5, _Q, ratios))


def harder(difficulty: Difficulty) -> Difficulty:
    i = _ORDER.index(difficulty)
    return _ORDER[min(i + 1, len(_ORDER) - 1)]


@dataclass(frozen=True, slots=True)
class IntensityPlan:
    label: Difficulty  # what the map is called; NJS/reaction time follow it
    intensity: float
    percentile: float
    wanted_nps: float  # NPS wanted in the finished map
    condition_as: Difficulty  # DIFF token the model is conditioned on

    @property
    def shifted(self) -> bool:
        return self.condition_as != self.label


def plan(
    label: Difficulty,
    characteristic: Characteristic,
    bpm: float,
    intensity: float | None,
    *,
    model_intensity: bool = False,
) -> IntensityPlan:
    """Targets for a label at an intensity (see the module docstring)."""
    level = clamp_intensity(intensity)
    q = percentile(level)
    wanted = round(target_nps(label, characteristic, bpm) * nps_ratio(label, q), 2)
    shift = not model_intensity and level > SHIFT_ABOVE
    return IntensityPlan(label, level, q, wanted, harder(label) if shift else label)


def conditioning_cap(
    condition_as: Difficulty,
    characteristic: Characteristic,
    bpm: float,
    gains: dict[Difficulty, float] | None = None,
) -> float:
    """Highest NPS to condition the model on above intensity 0.75: the human p95 for
    ``condition_as`` at ``bpm``, times its conditioning gain. Above that v0 saw too few maps
    and its output gets erratic (at intensity 1.0 the boosted copy sometimes came out
    *sparser*)."""
    from bsmapper.generation.density import NPS_GAIN

    p95 = target_nps(condition_as, characteristic, bpm) * nps_ratio(condition_as, 0.95)
    return round(p95 * (gains or NPS_GAIN)[condition_as], 2)


def relaxed_rules(config: RulesConfig, label: Difficulty, intensity: float | None) -> RulesConfig:
    """``config`` (the label's thresholds) with the ceilings raised for a shifted label."""
    level = clamp_intensity(intensity)
    target = harder(label)
    if level <= SHIFT_ABOVE or target == label:
        return config
    density = dataclasses.replace(
        config.density, max_nps=max(config.density.max_nps, HUMAN_PEAK_NPS_P95[target])
    )
    swing = dataclasses.replace(
        config.swing,
        min_interval_s=min(config.swing.min_interval_s, HUMAN_SWING_INTERVAL_P5[target]),
    )
    return dataclasses.replace(config, density=density, swing=swing)


class IntensityChecker(RuleChecker):
    """A ``RuleChecker`` whose ceilings follow ``relaxed_rules`` for one intensity."""

    def __init__(self, base: RuleChecker | None, intensity: float | None) -> None:
        super().__init__(raw=getattr(base, "_raw", None))
        self.intensity = clamp_intensity(intensity)

    def config(self, dmap: DifficultyMap) -> RulesConfig:
        return relaxed_rules(super().config(dmap), dmap.difficulty, self.intensity)


def checker_for(base: RuleChecker, intensity: float | None) -> RuleChecker:
    """``base``, or a relaxed copy of it when ``intensity`` shifts labels."""
    if clamp_intensity(intensity) <= SHIFT_ABOVE:
        return base
    if isinstance(base, IntensityChecker) and base.intensity == clamp_intensity(intensity):
        return base
    return IntensityChecker(base, intensity)


def configured_generate(key: str, path: Path | None = None) -> float | None:
    """A 0..1 value from the ``[generate]`` table of ``~/.bsmapper/config.toml``, if set."""
    from bsmapper.generation.install import config_path

    cfg = path or config_path()
    if not cfg.is_file():
        return None
    with cfg.open("rb") as f:
        value = tomllib.load(f).get("generate", {}).get(key)
    if value is None:
        return None
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{cfg}: [generate] {key} must be between 0 and 1")
    return float(value)


def configured_intensity(path: Path | None = None) -> float | None:
    """``[generate] intensity`` from ``~/.bsmapper/config.toml``, if set."""
    return configured_generate("intensity", path)


def configured_repetition(path: Path | None = None) -> float | None:
    """``[generate] repetition`` from ``~/.bsmapper/config.toml``, if set."""
    return configured_generate("repetition", path)
