"""Heuristic tech score: the label behind the flowy (0) .. technical (1) style slider.

``tech_features`` measures one difficulty from its swings. Notes of one colour that
follow each other within ``SWING_MERGE_BEATS`` (sliders, stacks, chains drawn as
notes) are one swing. The features are:

- ``angle_change``: mean of ``180 - angle_between(previous cut, this cut)`` over
  consecutive swings of a hand, divided by 180. A pure down-up-down flow scores 0,
  a same-direction reset scores 1, diagonals and sideways cuts land in between.
- ``reset_rate``: share of consecutive swings of a hand that cut within 90° of the
  previous cut (the hand has to reset instead of swinging back).
- ``crossover_rate``: share of swings that land on the far side of the other
  hand's most recent swing, within ``CROSSOVER_BEATS``.
- ``dot_ratio``: dot notes / notes.
- ``rhythm_irregularity``: mean ``|log2(gap / previous gap)|`` between a hand's
  swings (0 for a steady rhythm, 1 for a halving or doubling at every swing).
- ``odd_snap_ratio``: share of swings that start off the 1/4-beat grid
  (triplets, 1/6, 1/12 ...).
- ``travel``: mean grid distance between a hand's consecutive swings.

``calibrate_by_difficulty`` fits them against BeatLeader's ``techRating`` with one
ridge regression (``TechCalibration``) per difficulty, which turns them into one
``tech_score``: how many standard deviations of techRating a map would sit above its
difficulty's mean. ``style_by_difficulty`` then ranks the scores within each
difficulty into 0..1, so every difficulty uses the whole slider.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from bsmapper.rules.geometry import angle_between, cut_angle
from bsmapper.schema import Color, CutDirection, DifficultyMap, Note

FEATURES = (
    "angle_change",
    "reset_rate",
    "crossover_rate",
    "dot_ratio",
    "rhythm_irregularity",
    "odd_snap_ratio",
    "travel",
)
SWING_MERGE_BEATS = 0.125 + 1e-6
CROSSOVER_BEATS = 1.0
QUARTER_TICKS = 6  # 1/4 beat at 24 ticks per beat


@dataclass(frozen=True, slots=True)
class Swing:
    beat: float
    x: float
    y: float
    angle: float | None  # None: only dots


def swings(notes: Sequence[Note], color: Color) -> list[Swing]:
    """Group one hand's notes into swings, in time order."""
    out: list[Swing] = []
    group: list[Note] = []

    def flush() -> None:
        if not group:
            return
        angle = next((a for a in map(cut_angle, group) if a is not None), None)
        out.append(Swing(group[0].beat, group[0].x, group[0].y, angle))

    for n in sorted((n for n in notes if n.color == color), key=lambda n: n.beat):
        if group and n.beat - group[-1].beat > SWING_MERGE_BEATS:
            flush()
            group = []
        group.append(n)
    flush()
    return out


def tech_features(dmap: DifficultyMap) -> dict[str, float] | None:
    """The features above, or None when the difficulty has fewer than 16 notes."""
    notes = dmap.notes
    if len(notes) < 16:
        return None
    hands = {c: swings(notes, c) for c in Color}
    angle_dev: list[float] = []
    resets = pairs = 0
    log_ratios: list[float] = []
    travel: list[float] = []
    odd = total = 0
    for hand in hands.values():
        total += len(hand)
        odd += sum(1 for s in hand if round(s.beat * 24) % QUARTER_TICKS)
        for prev, cur in zip(hand, hand[1:], strict=False):
            travel.append(math.hypot(cur.x - prev.x, cur.y - prev.y))
            if prev.angle is not None and cur.angle is not None:
                diff = angle_between(prev.angle, cur.angle)
                angle_dev.append((180.0 - diff) / 180.0)
                pairs += 1
                resets += diff < 90.0
        gaps = np.diff([s.beat for s in hand])
        gaps = gaps[gaps > 0]
        if len(gaps) > 1:
            log_ratios.extend(np.abs(np.diff(np.log2(gaps))).tolist())

    crossed = checked = 0
    events = sorted((s.beat, c, s.x) for c, hand in hands.items() for s in hand)
    last: dict[Color, tuple[float, float]] = {}
    for beat, color, x in events:
        other = Color.BLUE if color == Color.RED else Color.RED
        if other in last and beat - last[other][0] <= CROSSOVER_BEATS:
            checked += 1
            ox = last[other][1]
            crossed += (x > ox) if color == Color.RED else (x < ox)
        last[color] = (beat, x)

    dots = sum(1 for n in notes if n.direction == CutDirection.ANY)
    return {
        "angle_change": float(np.mean(angle_dev)) if angle_dev else 0.0,
        "reset_rate": resets / pairs if pairs else 0.0,
        "crossover_rate": crossed / checked if checked else 0.0,
        "dot_ratio": dots / len(notes),
        "rhythm_irregularity": float(np.mean(log_ratios)) if log_ratios else 0.0,
        "odd_snap_ratio": odd / total if total else 0.0,
        "travel": float(np.mean(travel)) if travel else 0.0,
    }


def feature_matrix(rows: Sequence[dict[str, float] | None]) -> np.ndarray:
    """(n, len(FEATURES)); rows without features are NaN."""
    out = np.full((len(rows), len(FEATURES)), np.nan)
    for i, r in enumerate(rows):
        if r is not None:
            out[i] = [r[f] for f in FEATURES]
    return out


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_ranks(a), _ranks(b))


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), 0-based."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


@dataclass(slots=True)
class TechCalibration:
    """Ridge fit of BeatLeader techRating on the standardized features."""

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    intercept: float
    n: int = 0
    r_train: float = float("nan")
    r_cv: float = float("nan")  # 5-fold cross-validated Pearson r
    spearman_cv: float = float("nan")
    feature_r: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(
        cls, x: np.ndarray, y: np.ndarray, alpha: float = 1.0, folds: int = 5, seed: int = 0
    ) -> TechCalibration:
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(y) < 10:
            raise ValueError(f"need at least 10 rated difficulties to calibrate, got {len(y)}")
        cal = cls._fit(x, y, alpha)
        cal.n = len(y)
        cal.r_train = pearson(cal.predict(x), y)
        cal.feature_r = {f: pearson(x[:, i], y) for i, f in enumerate(FEATURES)}
        fold = np.random.default_rng(seed).permutation(len(y)) % folds
        pred = np.empty(len(y))
        for k in range(folds):
            test = fold == k
            pred[test] = cls._fit(x[~test], y[~test], alpha).predict(x[test])
        cal.r_cv = pearson(pred, y)
        cal.spearman_cv = spearman(pred, y)
        return cal

    @classmethod
    def _fit(cls, x: np.ndarray, y: np.ndarray, alpha: float) -> TechCalibration:
        mean, std = x.mean(axis=0), x.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        z = (x - mean) / std
        intercept = float(y.mean())
        a = z.T @ z + alpha * np.eye(z.shape[1])
        weights = np.linalg.solve(a, z.T @ (y - intercept))
        return cls(mean=mean, std=std, weights=weights, intercept=intercept)

    @classmethod
    def uncalibrated(cls, x: np.ndarray) -> TechCalibration:
        """Equal weights on the standardized features, for when nothing is rated."""
        ok = np.isfinite(x).all(axis=1)
        if not ok.any():
            mean, std = np.zeros(x.shape[1]), np.ones(x.shape[1])
        else:
            mean, std = x[ok].mean(axis=0), x[ok].std(axis=0)
        std = np.where(std > 0, std, 1.0)
        return cls(mean, std, np.full(x.shape[1], 1.0 / x.shape[1]), 0.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """tech_score on the techRating scale; NaN where a row has no features."""
        return ((x - self.mean) / self.std) @ self.weights + self.intercept

    def to_dict(self) -> dict:
        return {
            "features": list(FEATURES),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
            "intercept": self.intercept,
            "n": self.n,
            "r_train": self.r_train,
            "r_cv": self.r_cv,
            "spearman_cv": self.spearman_cv,
            "feature_r": self.feature_r,
        }


def calibrate_by_difficulty(
    x: np.ndarray, rating: np.ndarray, difficulties: Sequence[str], alpha: float = 1.0
) -> tuple[np.ndarray, dict[str, TechCalibration]]:
    """One fit per difficulty against the techRating standardized within that difficulty.

    BeatLeader's techRating grows with difficulty (and with density), so a single fit
    over all difficulties mostly learns which difficulty a map is. The style slider
    only compares maps of one difficulty, so each difficulty gets its own fit and the
    target is "standard deviations above this difficulty's mean techRating". A
    difficulty with fewer than 10 rated maps falls back to equal weights.
    Returns the scores (in those units) and the fit per difficulty.
    """
    diffs = np.asarray(difficulties)
    scores = np.full(len(diffs), np.nan)
    cals: dict[str, TechCalibration] = {}
    for d in np.unique(diffs):
        idx = np.flatnonzero(diffs == d)
        y = rating[idx]
        rated = np.isfinite(y)
        z = np.full(len(idx), np.nan)
        if rated.sum() > 1 and np.std(y[rated]) > 0:
            z[rated] = (y[rated] - y[rated].mean()) / y[rated].std()
        try:
            cal = TechCalibration.fit(x[idx], z, alpha=alpha)
        except ValueError:
            cal = TechCalibration.uncalibrated(x[idx])
        cals[str(d)] = cal
        scores[idx] = cal.predict(x[idx])
    return scores, cals


def style_by_difficulty(scores: np.ndarray, difficulties: Sequence[str]) -> np.ndarray:
    """Percentile rank of each score within its difficulty, in 0..1 (NaN stays NaN).

    Ranks are mid-ranks, ``(rank + 0.5) / n``, so the buckets ``STYLE_0..4`` each
    get about a fifth of every difficulty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    diffs = np.asarray(difficulties)
    out = np.full(len(scores), np.nan)
    for d in np.unique(diffs):
        idx = np.flatnonzero((diffs == d) & np.isfinite(scores))
        if len(idx):
            out[idx] = (_ranks(scores[idx]) + 0.5) / len(idx)
    return out
