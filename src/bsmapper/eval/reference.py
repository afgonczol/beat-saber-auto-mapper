"""How a map compares with human maps of the same mode, difficulty and BPM.

The reference is the dataset manifest's training difficulties (``ProcessedDataset``
rows of ``train`` and ``val``; never the split being scored).

- ``nps_percentile``: where a map's NPS falls among human maps of its difficulty near its
  BPM (within ``BPM_BAND``, or the ``MIN_NEIGHBOURS`` nearest). 50 is a typical human map.
- ``equivalent_level``: the map's NPS placed on the ladder of human median NPS at that BPM
  (0 = Easy .. 4 = Expert+, interpolated). ``step_offset`` is that minus the map's own
  level: -1 means "as dense as a typical human map one difficulty lower".
- ``style_percentile``: the tech score's percentile among human maps of the difficulty
  (the dataset's ``style`` label is the same percentile).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from bsmapper.eval.metrics import percentile_of
from bsmapper.schema import Characteristic, Difficulty

LEVELS = tuple(Difficulty)  # Easy .. ExpertPlus, index = level
BPM_BAND = 0.15  # +-15 % BPM
MIN_NEIGHBOURS = 60
MIN_MODE_ROWS = 40  # fewer human maps of a mode and difficulty: use Standard's


class HumanReference:
    def __init__(self, rows: Sequence[dict[str, Any]], splits: Sequence[str] = ("train", "val")):
        self._by: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._tech: dict[str, np.ndarray] = {}
        groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
        tech: dict[str, list[float]] = {}
        for r in rows:
            if r["split"] not in splits:
                continue
            groups.setdefault((r["characteristic"], r["difficulty"]), []).append(
                (float(r["bpm"]), float(r["nps"]))
            )
            ts = r.get("tech_score")
            if ts is not None and math.isfinite(ts):
                tech.setdefault(r["difficulty"], []).append(float(ts))
        for k, v in groups.items():
            arr = np.array(v, dtype=np.float64)
            self._by[k] = (arr[:, 0], arr[:, 1])
        self._tech = {k: np.array(v) for k, v in tech.items()}

    def _group(self, mode: Characteristic, diff: Difficulty) -> tuple[np.ndarray, np.ndarray]:
        key = (str(mode), str(diff))
        got = self._by.get(key)
        if got is None or len(got[0]) < MIN_MODE_ROWS:
            got = self._by.get((str(Characteristic.STANDARD), str(diff)))
        return got if got is not None else (np.empty(0), np.empty(0))

    def neighbours(self, mode: Characteristic, diff: Difficulty, bpm: float) -> np.ndarray:
        """Human NPS values of this mode and difficulty near ``bpm``."""
        bpms, nps = self._group(mode, diff)
        if not len(bpms):
            return nps
        dist = np.abs(np.log(bpms / bpm))
        near = dist <= math.log1p(BPM_BAND)
        if near.sum() >= MIN_NEIGHBOURS:
            return nps[near]
        return nps[np.argsort(dist)[: min(MIN_NEIGHBOURS, len(dist))]]

    def nps_percentile(self, mode: Characteristic, diff: Difficulty, bpm: float, nps: float):
        return percentile_of(nps, self.neighbours(mode, diff, bpm))

    def median_ladder(self, mode: Characteristic, bpm: float) -> list[float]:
        """Median human NPS per level (Easy .. Expert+) at ``bpm``."""
        return [float(np.median(self.neighbours(mode, d, bpm))) for d in LEVELS]

    def equivalent_level(self, mode: Characteristic, bpm: float, nps: float) -> float:
        ladder = self.median_ladder(mode, bpm)
        if not math.isfinite(nps) or any(not math.isfinite(m) for m in ladder):
            return float("nan")
        ladder = list(np.maximum.accumulate(ladder))  # keep it monotone
        # Linear between neighbouring medians; past either end, the end segment's slope.
        last = len(ladder) - 2
        i = next((i for i in range(last) if nps <= ladder[i + 1]), last)
        span = max(ladder[i + 1] - ladder[i], 1e-6)
        return i + (nps - ladder[i]) / span

    def step_offset(self, mode: Characteristic, diff: Difficulty, bpm: float, nps: float):
        return self.equivalent_level(mode, bpm, nps) - LEVELS.index(diff)

    def style_percentile(self, diff: Difficulty, tech_score: float) -> float:
        return percentile_of(tech_score, self._tech.get(str(diff), np.empty(0)))
