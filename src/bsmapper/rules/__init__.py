"""WS5: playability rules (parity, vision blocks, NPS, ...) and decode masks.

Whole-map check (eval, UI warnings)::

    issues = RuleChecker().check(dmap, beatmap)

Incremental use during decoding (WS7)::

    state = RuleChecker().state(dmap, beatmap)   # or MapState.for_map(dmap, beatmap)
    state.allowed(candidate)                     # no "error"-severity rule broken?
    mask = state.note_mask(beat, Color.RED)      # bool [x, y, direction]
    state.add(chosen)

Thresholds live in ``thresholds.yaml`` (per difficulty and characteristic).
"""

from bsmapper.rules.checker import RuleChecker, summarize
from bsmapper.rules.config import RULES, RulesConfig, config_for
from bsmapper.rules.state import MapState, ordered_objects

__all__ = [
    "RULES",
    "MapState",
    "RuleChecker",
    "RulesConfig",
    "config_for",
    "ordered_objects",
    "summarize",
]
