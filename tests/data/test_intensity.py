"""Intensity labels: percentile within (characteristic, difficulty, tempo)."""

import numpy as np
import pytest

from bsmapper.data.intensity import (
    bpm_bucket,
    fold_bpm,
    group_keys,
    intensity_labels,
    percentile_rank,
)


def test_bpm_folding_and_buckets():
    assert fold_bpm(240) == 120 and fold_bpm(60) == 120 and fold_bpm(175) == 175
    assert [bpm_bucket(b) for b in (95, 120, 140, 170, 340, 0)] == [0, 1, 2, 3, 3, -1]


def test_percentile_rank_ties_and_nan():
    r = percentile_rank(np.array([1.0, 2.0, 2.0, np.nan, 3.0]))
    assert r[:3].tolist() == [0.125, 0.5, 0.5] and np.isnan(r[3]) and r[4] == 0.875


def _rows(n, diff="Expert", char="Standard", bpm=128.0, rated=True, seed=0):
    rng = np.random.default_rng(seed)
    nps = rng.uniform(2, 8, n)
    return [
        {
            "characteristic": char,
            "difficulty": diff,
            "bpm": bpm,
            "nps": float(x),
            "bl_pass_rating": float(x + rng.normal(0, 0.3)) if rated else None,
            "bl_stars": None,
        }
        for x in nps
    ]


def test_labels_rank_within_group_and_fall_back_to_nps():
    rows = _rows(200) + _rows(100, rated=False, seed=1)
    x, cal = intensity_labels(rows)
    assert x.min() > 0 and x.max() < 1
    assert np.median(x[:200]) == pytest.approx(0.5, abs=0.02)
    # Unrated maps use their NPS percentile, on the same scale.
    unrated = x[200:]
    assert np.corrcoef(unrated, [r["nps"] for r in rows[200:]])[0, 1] > 0.9
    e = cal["by_difficulty"]["Expert"]
    assert e["pass_rating"] == 200 and e["nps"] == 100
    assert e["spearman_bl_vs_nps"] > 0.8


def test_small_groups_fall_back():
    rows = _rows(100) + _rows(5, char="90Degree", seed=2)
    keys = group_keys(rows)
    assert keys[0] == ("Standard", "Expert", 1)
    assert keys[-1] == ("*", "Expert", 1)
