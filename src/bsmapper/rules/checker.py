"""Whole-map checker (``interfaces.PlayabilityChecker``) built on ``MapState``."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from bsmapper.interfaces import Issue
from bsmapper.rules.config import RulesConfig, config_for, load_raw
from bsmapper.rules.spans import check_spans
from bsmapper.rules.state import MapState, ordered_objects
from bsmapper.schema import Beatmap, DifficultyMap


class RuleChecker:
    """Runs every rule over a difficulty. Thresholds come from ``thresholds.yaml``."""

    def __init__(self, thresholds: Path | str | None = None, raw: dict[str, Any] | None = None):
        self._raw = raw if raw is not None else load_raw(thresholds)

    def config(self, dmap: DifficultyMap) -> RulesConfig:
        return config_for(dmap.difficulty, dmap.characteristic, raw=self._raw)

    def state(self, dmap: DifficultyMap, beatmap: Beatmap) -> MapState:
        """An empty incremental state with this difficulty's thresholds."""
        return MapState.for_map(dmap, beatmap, config=self.config(dmap))

    def check(self, dmap: DifficultyMap, beatmap: Beatmap) -> list[Issue]:
        state = self.state(dmap, beatmap)
        issues = _collapse_density(state.extend(ordered_objects(dmap)), state)
        spans = check_spans(dmap, beatmap, state.config)
        return sorted(issues + spans, key=lambda i: i.beat) if spans else issues


def _collapse_density(issues: list[Issue], state: MapState) -> list[Issue]:
    """Report one NPS issue per dense passage instead of one per note."""
    window = state.config.density.window_s
    out: list[Issue] = []
    last_s: float | None = None
    for issue in issues:
        if issue.rule == "nps_ceiling":
            t = state.seconds(issue.beat)
            dense_run = last_s is not None and t - last_s <= window
            last_s = t
            if dense_run:
                continue
        out.append(issue)
    return out


def summarize(issues: list[Issue], note_count: int) -> dict[str, dict[str, float]]:
    """Issue counts and rate per 100 notes, by rule (for eval and calibration)."""
    counts = Counter(i.rule for i in issues)
    per = max(note_count, 1) / 100.0
    return {rule: {"count": n, "per_100_notes": n / per} for rule, n in sorted(counts.items())}
