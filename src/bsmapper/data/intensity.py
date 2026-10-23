"""The intensity label: how challenging a difficulty is for its label (0..1).

``GenerationRequest.intensity`` asks for "a typical human Expert" (0.5) up to
"about the hardest 10% of Expert maps" (1.0, bucket ``INTENSITY_4``), so the label
is a percentile rank within comparable maps:

* the group is the difficulty and the song's tempo (``bpm_bucket``): a fast song's
  Expert is denser than a slow song's, and both should count as typical Expert. The
  characteristic is part of the group where it has enough maps (rare modes fall
  back to all characteristics, see ``group_keys``);
* the score ranked is BeatLeader's ``passRating`` (how hard the map is to pass; its
  AI rates nearly every map), else its ``stars``, else the map's notes per second.

All three rank into the same uniform 0..1 scale within a group, so a map without a
BeatLeader rating gets its NPS percentile directly. ``calibration`` reports how well
the two agree on maps that have both (Spearman per difficulty), which is how the
fallback is checked; see docs/DATASET.md ("Intensity").
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

BPM_FOLD = (90.0, 180.0)  # tempo octave the BPM is folded into (mappers double BPMs)
BPM_EDGES = (110.0, 130.0, 150.0)  # four tempo buckets inside the octave
MIN_GROUP = 40  # smaller groups fall back to a coarser group


def fold_bpm(bpm: float) -> float:
    """``bpm`` moved by octaves into [90, 180)."""
    if not np.isfinite(bpm) or bpm <= 0:
        return float("nan")
    lo, hi = BPM_FOLD
    while bpm >= hi:
        bpm /= 2
    while bpm < lo:
        bpm *= 2
    return bpm


def bpm_bucket(bpm: float) -> int:
    folded = fold_bpm(bpm)
    return -1 if not np.isfinite(folded) else int(np.searchsorted(BPM_EDGES, folded, "right"))


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Mid-rank percentile in (0, 1); ties share their mean rank. NaN stays NaN."""
    out = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n == 0:
        return out
    v = x[ok]
    order = np.argsort(v, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n, dtype=np.float64)
    # average the ranks of ties
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = sums[inv] / counts[inv]
    out[ok] = (ranks + 0.5) / n
    return out


def group_keys(rows: Sequence[dict[str, Any]]) -> list[tuple]:
    """The ranking group of each row: (characteristic, difficulty, tempo), coarsened
    until it holds at least ``MIN_GROUP`` rows."""
    levels = [
        lambda r: (r["characteristic"], r["difficulty"], bpm_bucket(r["bpm"])),
        lambda r: (r["characteristic"], r["difficulty"]),
        lambda r: ("*", r["difficulty"], bpm_bucket(r["bpm"])),
        lambda r: ("*", r["difficulty"]),
    ]
    counts = [defaultdict(int) for _ in levels]
    for r in rows:
        for c, key in zip(counts, levels, strict=True):
            c[key(r)] += 1
    out = []
    for r in rows:
        for c, key in zip(counts, levels, strict=True):
            k = key(r)
            if c[k] >= MIN_GROUP:
                out.append(k)
                break
        else:
            out.append(levels[-1](r))
    return out


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) and f > 0 else float("nan")


def intensity_labels(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Intensity (0..1) per row and the calibration report.

    Rows need ``characteristic``, ``difficulty``, ``bpm``, ``nps``,
    ``bl_pass_rating`` and ``bl_stars`` (None/NaN when missing).
    """
    n = len(rows)
    groups = group_keys(rows)
    nps = np.array([_num(r["nps"]) for r in rows])
    rating = np.array([_num(r.get("bl_pass_rating")) for r in rows])
    stars = np.array([_num(r.get("bl_stars")) for r in rows])
    source = np.where(np.isfinite(rating), "pass_rating",
                      np.where(np.isfinite(stars), "stars", "nps"))  # fmt: skip
    p_bl = np.full(n, np.nan)
    p_nps = np.full(n, np.nan)
    members: dict[tuple, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    for idx in members.values():
        idx = np.asarray(idx)
        p_nps[idx] = percentile_rank(nps[idx])
        # pass rating where present, stars for the rest of the rated maps
        bl = np.where(np.isfinite(rating[idx]), rating[idx], np.nan)
        if np.isfinite(bl).sum() >= 2:
            p_bl[idx] = percentile_rank(bl)
        st_only = np.isnan(bl) & np.isfinite(stars[idx])
        if st_only.any() and np.isfinite(stars[idx]).sum() >= 2:
            p_bl[idx[st_only]] = percentile_rank(stars[idx])[st_only]
    intensity = np.where(np.isfinite(p_bl), p_bl, p_nps)
    intensity = np.where(np.isfinite(intensity), intensity, 0.5)
    return intensity, calibration(rows, p_bl, p_nps, source, groups)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = percentile_rank(a[ok]), percentile_rank(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def calibration(rows, p_bl, p_nps, source, groups) -> dict[str, Any]:
    """Per difficulty: counts by source, Spearman(BL percentile, NPS percentile) and how
    often the two land in the same INTENSITY bucket."""
    from bsmapper.tokenizer import intensity_bucket

    out: dict[str, Any] = {"groups": len(set(groups)), "by_difficulty": {}}
    diffs = sorted({r["difficulty"] for r in rows})
    for d in diffs:
        idx = np.array([i for i, r in enumerate(rows) if r["difficulty"] == d])
        both = idx[np.isfinite(p_bl[idx]) & np.isfinite(p_nps[idx])]
        same = [intensity_bucket(p_bl[i]) == intensity_bucket(p_nps[i]) for i in both]
        within = [abs(intensity_bucket(p_bl[i]) - intensity_bucket(p_nps[i])) <= 1 for i in both]
        out["by_difficulty"][d] = {
            "n": len(idx),
            "pass_rating": int((source[idx] == "pass_rating").sum()),
            "stars": int((source[idx] == "stars").sum()),
            "nps": int((source[idx] == "nps").sum()),
            "spearman_bl_vs_nps": _spearman(p_bl[both], p_nps[both]),
            "same_bucket": float(np.mean(same)) if same else float("nan"),
            "within_one_bucket": float(np.mean(within)) if within else float("nan"),
        }
    return out
