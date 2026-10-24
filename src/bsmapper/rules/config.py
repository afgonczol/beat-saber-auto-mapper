"""Rule thresholds, loaded from a per-difficulty YAML file.

``thresholds.yaml`` (next to this module) has three layers, merged in order:
``default`` -> ``difficulties.<Difficulty>`` -> ``characteristics.<Characteristic>``.
Every layer may set any key; unknown keys are an error so typos do not silently
fall back to defaults.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from bsmapper.schema import Characteristic, Difficulty

Severity = Literal["error", "warning", "off"]

DEFAULT_THRESHOLDS = Path(__file__).with_name("thresholds.yaml")

# Every rule name an Issue can carry. "error" rules are the hard decode masks.
RULES = (
    "parity",
    "swing_interval",
    "swing_speed",
    "same_cell",
    "overlap",
    "vision_block",
    "crossover",
    "note_in_wall",
    "nps_ceiling",
    "wall_duration",
    "dodge_gap",
    "wall_boxin",
    "rotation_whiplash",
    "rotation_range",
    "rotation_step",
    "arc_anchor",
    "arc_span",
    "arc_wall",
    "chain_anchor",
)


@dataclass(frozen=True, slots=True)
class ParityConfig:
    # A forehand cut must point within this many degrees of DOWN, a backhand of UP.
    max_wrist_deg: float = 100.0
    # Rotates the forehand centre from DOWN towards the hand's inward side (DOWN_LEFT
    # for the right hand). 0 treats both hands alike; 22.5 splits the 8 directions 4/4.
    inward_bias_deg: float = 0.0
    # Consecutive swings of one hand must turn at least this much (LEFT then LEFT is 0,
    # DOWN then DOWN_LEFT is 45). Below 90 so v3 angle offsets have some slack.
    min_direction_change_deg: float = 60.0
    # Swings turning at most this much may keep their parity (a curve, e.g. DOWN -> LEFT)
    # as well as flip it. 0 means strict forehand/backhand alternation.
    keep_max_turn_deg: float = 90.0
    # A swing turning at most slider_max_turn_deg that starts further along the previous
    # swing's path continues it (a slider with a gap), so it is not a parity break.
    allow_continuation: bool = True
    # Same-hand notes this close (and pointing the same way) are one swing (stack/slider).
    merge_window_s: float = 0.09
    # Max turn between consecutive notes of one slider/stack.
    slider_max_turn_deg: float = 45.0
    # A gap this long lets the player reset the hand, so parity starts over.
    reset_gap_s: float = 1.0
    # Bombs within this many columns of the hand, between two swings, allow a reset.
    bomb_reset_columns: int = 1
    # If false, dot notes are wildcards: they can be hit with either swing.
    dots_flip_parity: bool = False


@dataclass(frozen=True, slots=True)
class SwingConfig:
    # Minimum time from the end of one swing to the start of the next (same hand).
    min_interval_s: float = 0.06
    # Max saber travel between swings, in grid cells per second.
    max_speed: float = 60.0


@dataclass(frozen=True, slots=True)
class OverlapConfig:
    # Objects in the same cell closer than this along the track overlap (NJS * dt).
    min_distance_m: float = 0.6


@dataclass(frozen=True, slots=True)
class VisionConfig:
    # A note this long after a centre blocker is hidden by it.
    min_s: float = 0.05
    max_s: float = 0.25
    bombs_block: bool = True


@dataclass(frozen=True, slots=True)
class CrossoverConfig:
    # Red this many columns right of blue, within window_s, counts as crossed hands.
    # One-column crossovers are a common deliberate pattern in human maps.
    min_columns: int = 2
    window_s: float = 0.0


@dataclass(frozen=True, slots=True)
class DensityConfig:
    max_nps: float = 16.0
    window_s: float = 2.0


@dataclass(frozen=True, slots=True)
class WallConfig:
    min_duration_s: float = 0.0138
    # Gap between dodge walls that need the player in different places.
    min_dodge_gap_s: float = 0.15
    # Shorter walls are flashes, not dodges, and are left out of the gap check.
    min_dodge_duration_s: float = 0.1
    # Walls of any duration that close both centre columns at head height (left and
    # right lean) closer than this together leave nowhere to stand.
    boxin_window_s: float = 0.3


@dataclass(frozen=True, slots=True)
class RotationConfig:
    # Total rotation swing (max - min) allowed within window_s.
    window_s: float = 1.0
    max_swing_deg: float = 90.0
    # Max |cumulative rotation|; null for no limit (360). 90Degree maps keep within 45.
    max_abs_deg: float | None = None
    # Largest single rotation event. Human 90/360 maps never turn 75° or more at once.
    max_step_deg: float = 60.0


@dataclass(frozen=True, slots=True)
class SpansConfig:
    # Longest arc, head to tail (human p99 is 2.8 s).
    arc_max_s: float = 3.0
    # Longest chain, head to tail (human chains reach a quarter beat).
    chain_max_beats: float = 0.5


@dataclass(frozen=True, slots=True)
class RulesConfig:
    severity: dict[str, Severity] = field(default_factory=lambda: dict.fromkeys(RULES, "error"))
    parity: ParityConfig = ParityConfig()
    swing: SwingConfig = SwingConfig()
    overlap: OverlapConfig = OverlapConfig()
    vision: VisionConfig = VisionConfig()
    crossover: CrossoverConfig = CrossoverConfig()
    density: DensityConfig = DensityConfig()
    walls: WallConfig = WallConfig()
    rotation: RotationConfig = RotationConfig()
    spans: SpansConfig = SpansConfig()

    def enabled(self, rule: str) -> bool:
        return self.severity.get(rule, "error") != "off"


_SECTIONS = {
    "parity": ParityConfig,
    "swing": SwingConfig,
    "overlap": OverlapConfig,
    "vision": VisionConfig,
    "crossover": CrossoverConfig,
    "density": DensityConfig,
    "walls": WallConfig,
    "rotation": RotationConfig,
    "spans": SpansConfig,
}


def _merge(base: dict[str, Any], over: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _section(cls: type, values: dict[str, Any], where: str) -> Any:
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in {where}")
    return cls(**values)


def build_config(raw: dict[str, Any]) -> RulesConfig:
    """Build a config from one already merged layer (``default`` merged with overrides)."""
    unknown = set(raw) - set(_SECTIONS) - {"severity"}
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)} in thresholds")
    severity: dict[str, Severity] = dict.fromkeys(RULES, "error")
    for rule, sev in (raw.get("severity") or {}).items():
        if rule not in RULES:
            raise ValueError(f"unknown rule {rule!r} in severity")
        if sev not in ("error", "warning", "off"):
            raise ValueError(f"severity for {rule!r} must be error, warning or off, not {sev!r}")
        severity[rule] = sev
    sections = {name: _section(cls, raw.get(name) or {}, name) for name, cls in _SECTIONS.items()}
    return RulesConfig(severity=severity, **sections)


def load_raw(path: Path | str | None = None) -> dict[str, Any]:
    import yaml  # pyyaml ships with the "ml" extra

    with open(path or DEFAULT_THRESHOLDS, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"default", "difficulties", "characteristics"}
    if unknown:
        raise ValueError(f"unknown top-level key(s) {sorted(unknown)} in {path}")
    for name in raw.get("difficulties") or {}:
        Difficulty(name)
    for name in raw.get("characteristics") or {}:
        Characteristic(name)
    return raw


def config_for(
    difficulty: Difficulty | str,
    characteristic: Characteristic | str = Characteristic.STANDARD,
    path: Path | str | None = None,
    raw: dict[str, Any] | None = None,
) -> RulesConfig:
    """Thresholds for one (difficulty, characteristic), merged from the YAML layers."""
    raw = raw if raw is not None else load_raw(path)
    merged = _merge(raw.get("default") or {}, (raw.get("difficulties") or {}).get(str(difficulty)))
    merged = _merge(merged, (raw.get("characteristics") or {}).get(str(characteristic)))
    return build_config(merged)
