# ruff: noqa: E402, E501, SIM115  (analysis script for docs/LIGHTING_V2.md)
"""Per-window readability stats and per-map style features of the gold human lightshows.

usage: python human_stats.py <processed_v1 dir> <out.pkl>
"""

import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bsmapper.lighting.events import LIGHT_TYPES
from bsmapper.lighting.metrics import constant_tempo, visible_changes, window_stats
from bsmapper.tokenizer import LightingTokenizer
from bsmapper.tokenizer.windows import TokenWindow

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else None
WT = 384


def windows_from(tokens, offsets):
    return [
        TokenWindow(k, list(tokens[offsets[k] : offsets[k + 1]]), [])
        for k in range(len(offsets) - 1)
    ]


def map_features(ls, feats, n_windows, bpm):
    """Style features of one map (beyond the window stats)."""
    b2s = constant_tempo(bpm)
    changes = visible_changes(ls)
    allc = [c for t in LIGHT_TYPES for c in changes[t]]
    n_beats = n_windows * 16
    secs = b2s(n_beats)
    onset = feats[:, 80].astype(np.float32)
    rms = feats[:, 82].astype(np.float32)
    f = {}
    if not allc:
        return f, rms
    ticks = np.array([round(c.beat * 24) for c in allc])
    ticks = ticks[ticks < len(onset)]
    frac = ticks % 24
    f["on_beat"] = float((frac == 0).mean())
    f["on_downbeat"] = float((ticks % 96 == 0).mean())
    f["on_half"] = float((frac == 12).mean())
    f["off_grid"] = float(((frac % 6) != 0).mean())  # finer than 1/4 beat
    # Onset lift: mean onset strength (max over +-1 tick) at change ticks vs. at all ticks
    pad = np.maximum(onset, np.maximum(np.roll(onset, 1), np.roll(onset, -1)))
    loud = rms > 0.05
    base = pad[loud].mean() if loud.any() else 1.0
    f["onset_lift"] = float(pad[ticks].mean() / max(base, 1e-6)) if len(ticks) else np.nan
    colors = [c.color for c in allc if c.lit]
    f["white_share"] = float(np.mean([c == 2 for c in colors])) if colors else 0.0
    f["boost_per_min"] = 60 * len(ls.boost_events) / secs
    # share of time boosted
    on, t0, total = False, 0.0, 0.0
    for b in sorted(ls.boost_events, key=lambda b: b.beat):
        if b.on and not on:
            on, t0 = True, b.beat
        elif not b.on and on:
            on, total = False, total + b.beat - t0
    if on:
        total += n_beats - t0
    f["boost_time"] = total / max(n_beats, 1)
    f["chroma"] = float(any(e.color is not None for e in ls.basic_events))
    return f, rms


def process(job):
    shard, items = job
    table = pq.read_table(ROOT / "lights" / f"{shard}.parquet").to_pylist()
    feat_cache = {}
    out = []
    tok = LightingTokenizer()
    for m in items:
        try:
            t = table[m["light_row"]]
            assert t["key"] == m["key"]
            ls = tok.decode(windows_from(t["tokens"], t["window_offsets"]))
            fs = m["feature_shard"]
            if fs not in feat_cache:
                feat_cache[fs] = np.load(ROOT / "features" / f"{fs}.npy", mmap_mode="r")
            nw = m["n_windows"]
            feats = np.asarray(feat_cache[fs][m["feature_row"] : m["feature_row"] + nw * WT])
            ws = window_stats(ls, nw * 16, constant_tempo(m["bpm"]))
            f, rms = map_features(ls, feats, nw, m["bpm"])
            energy = rms[: nw * WT].reshape(nw, WT).mean(axis=1)
            out.append(
                dict(
                    meta=m,
                    windows=[w.as_dict() for w in ws],
                    energy=energy,
                    features=f,
                )
            )
        except Exception as exc:  # noqa: BLE001
            print("fail", m["key"], exc)
    return out


def main():
    maps = pq.read_table(ROOT / "maps.parquet").to_pylist()
    keep = [
        m
        for m in maps
        if m["status"] == "kept"
        and m["gold"]
        and m["light_shard"] is not None
        and (m["light_dropped_box_groups"] or 0) == 0
        and not m["bpm_changes"]
    ]
    print(len(keep), "gold maps with basic lightshows")
    jobs = defaultdict(list)
    for m in keep:
        jobs[m["light_shard"]].append(
            {
                k: m[k]
                for k in (
                    "key",
                    "song_name",
                    "song_author",
                    "mapper",
                    "environment",
                    "bpm",
                    "n_windows",
                    "split",
                    "light_row",
                    "feature_shard",
                    "feature_row",
                    "light_events",
                    "light_chroma",
                    "duration_s",
                )
            }
        )
    results = []
    with Pool(12) as pool:
        for r in pool.imap_unordered(process, list(jobs.items())):
            results += r
            print(len(results), flush=True)
    with open(sys.argv[2], "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
