"""Per-difficulty note metrics: onsets, density, rules, pattern distributions, style.

Everything here is a pure function of ``DifficultyMap``s on one tempo map. Times stay in
beats (ticks of 1/24 beat) except NPS, which uses the tempo map.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from bsmapper.data.tech import FEATURES, SWING_MERGE_BEATS, TechCalibration, tech_features
from bsmapper.rules import RULES, RuleChecker, summarize
from bsmapper.schema import Beatmap, Color, CutDirection, DifficultyMap, Note
from bsmapper.tokenizer import TICKS_PER_BEAT, notes_per_second

ONSET_TOLERANCES = (0, 1, 2)  # ticks
RHYTHM_MAX_TICKS = 96  # gaps longer than 4 beats share one bucket
DISTRIBUTIONS = ("direction", "position", "pattern", "pattern_2gram", "pattern_3gram", "rhythm")


# --- onsets -------------------------------------------------------------------------


def onset_ticks(dmap: DifficultyMap) -> np.ndarray:
    """Sorted unique ticks that hold at least one colour note."""
    return np.unique(np.round(np.array([n.beat for n in dmap.notes]) * TICKS_PER_BEAT)).astype(
        np.int64
    )


def match_onsets(ref: np.ndarray, est: np.ndarray, tolerance: int) -> int:
    """Greedy one-to-one matches of sorted ``est`` onsets to sorted ``ref`` onsets within
    ``tolerance`` ticks (both sorted, each onset used once; optimal for 1-D intervals)."""
    i = j = hits = 0
    while i < len(ref) and j < len(est):
        d = est[j] - ref[i]
        if abs(d) <= tolerance:
            hits += 1
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return hits


def onset_scores(ref: DifficultyMap, est: DifficultyMap) -> dict[str, float]:
    """Precision, recall and F1 of ``est``'s note onsets against ``ref``'s per tolerance."""
    r, e = onset_ticks(ref), onset_ticks(est)
    out: dict[str, float] = {}
    for tol in ONSET_TOLERANCES:
        hits = match_onsets(r, e, tol)
        p = hits / len(e) if len(e) else 0.0
        rc = hits / len(r) if len(r) else 0.0
        out[f"onset_p_{tol}"] = p
        out[f"onset_r_{tol}"] = rc
        out[f"onset_f1_{tol}"] = 2 * p * rc / (p + rc) if p + rc else 0.0
    return out


# --- pattern distributions ----------------------------------------------------------


def _swing_tokens(notes: Sequence[Note], color: Color) -> list[str]:
    """One token per swing of a hand: first arrow (or dot) direction and its cell."""
    return [_token(g) for g in swing_groups(notes, color)]


def swing_groups(notes: Sequence[Note], color: Color) -> list[list[Note]]:
    """One hand's notes grouped into swings (as ``data.tech.swings`` groups them)."""
    out: list[list[Note]] = []
    for n in sorted((n for n in notes if n.color == color), key=lambda n: n.beat):
        if out and n.beat - out[-1][-1].beat <= SWING_MERGE_BEATS:
            out[-1].append(n)
        else:
            out.append([n])
    return out


def _token(group: list[Note]) -> str:
    head = next((n for n in group if n.direction != CutDirection.ANY), group[0])
    return f"{int(head.direction)}@{group[0].x}{group[0].y}"


def distributions(dmap: DifficultyMap) -> dict[str, Counter]:
    """Counts for the distribution distances (pooled over maps before comparing).

    - ``direction``: cut direction of every note (8 = dot); ``position``: its cell ``xy``
    - ``pattern``: per swing (one hand's notes within 1/8 beat), direction and cell
    - ``pattern_2gram`` / ``pattern_3gram``: consecutive swings of one hand
    - ``rhythm``: gaps in ticks between consecutive note onsets (either hand)
    """
    notes = dmap.notes
    out: dict[str, Counter] = {name: Counter() for name in DISTRIBUTIONS}
    for n in notes:
        out["direction"][str(int(n.direction))] += 1
        out["position"][f"{n.x}{n.y}"] += 1
    for color in Color:
        toks = _swing_tokens(notes, color)
        hand = "L" if color == Color.RED else "R"
        out["pattern"].update(f"{hand}{t}" for t in toks)
        out["pattern_2gram"].update(f"{hand}{a}>{b}" for a, b in zip(toks, toks[1:], strict=False))
        out["pattern_3gram"].update(
            f"{hand}{a}>{b}>{c}" for a, b, c in zip(toks, toks[1:], toks[2:], strict=False)
        )
    ticks = onset_ticks(dmap)
    gaps = np.minimum(np.diff(ticks), RHYTHM_MAX_TICKS + 1)
    out["rhythm"].update(str(int(g)) for g in gaps)
    return out


def jensen_shannon(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence (base 2, 0 = identical, 1 = disjoint) of two count maps."""
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    b = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("nan")
    a, b = a / a.sum(), b / b.sum()
    m = 0.5 * (a + b)

    def kl(x: np.ndarray) -> float:
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / m[nz])))

    return max(0.0, 0.5 * kl(a) + 0.5 * kl(b))


def pool(counters: Iterable[Mapping[str, float]]) -> Counter:
    total: Counter = Counter()
    for c in counters:
        total.update(c)
    return total


# --- style ----------------------------------------------------------------------------


def calibrations(dataset_meta: Mapping) -> dict[str, TechCalibration]:
    """The dataset's per-difficulty tech-score fits (``dataset.json`` ``tech_calibration``)."""
    out = {}
    for diff, d in dataset_meta.get("tech_calibration", {}).items():
        if list(d.get("features", FEATURES)) != list(FEATURES):
            continue
        out[diff] = TechCalibration(
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            weights=np.asarray(d["weights"], dtype=np.float64),
            intercept=float(d["intercept"]),
            n=int(d.get("n", 0)),
        )
    return out


def tech_score(features: Mapping[str, float] | None, cal: TechCalibration | None) -> float:
    if features is None or cal is None:
        return float("nan")
    x = np.array([[features[f] for f in FEATURES]], dtype=np.float64)
    return float(cal.predict(x)[0])


# --- everything for one difficulty ------------------------------------------------------


def difficulty_metrics(
    dmap: DifficultyMap,
    beatmap: Beatmap,
    checker: RuleChecker,
    cal: TechCalibration | None = None,
) -> dict[str, float]:
    """Scalar metrics of one difficulty on its own (no reference map)."""
    notes = dmap.notes
    n = len(notes)
    out: dict[str, float] = {
        "notes": float(n),
        "bombs": float(len(dmap.bombs)),
        "obstacles": float(len(dmap.obstacles)),
        "arcs": float(len(dmap.arcs)),
        "chains": float(len(dmap.chains)),
        "nps": notes_per_second(dmap, beatmap),
        "red_share": sum(x.color == Color.RED for x in notes) / n if n else float("nan"),
        "dot_share": sum(x.direction == CutDirection.ANY for x in notes) / n if n else float("nan"),
    }
    out["color_imbalance"] = abs(out["red_share"] - 0.5) if n else float("nan")
    issues = checker.check(dmap, beatmap)
    per = max(n, 1) / 100.0
    out["errors_per_100"] = sum(i.severity == "error" for i in issues) / per
    out["warnings_per_100"] = sum(i.severity == "warning" for i in issues) / per
    for rule in RULES:
        out[f"rule_{rule}_per_100"] = 0.0
    for rule, s in summarize(issues, n).items():
        out[f"rule_{rule}_per_100"] = s["per_100_notes"]
    feats = tech_features(dmap)
    for f in FEATURES:
        out[f"tech_{f}"] = feats[f] if feats else float("nan")
    out["crossover_rate"] = out["tech_crossover_rate"]
    out["tech_score"] = tech_score(feats, cal)
    return out


def percentile_of(value: float, reference: Sequence[float]) -> float:
    """Share of ``reference`` below ``value`` (ties count half), as 0..100."""
    ref = np.asarray([r for r in reference if np.isfinite(r)], dtype=np.float64)
    if not len(ref) or not math.isfinite(value):
        return float("nan")
    below = np.sum(ref < value) + 0.5 * np.sum(ref == value)
    return float(100.0 * below / len(ref))
