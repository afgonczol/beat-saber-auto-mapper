# ruff: noqa: E402, E501, SIM115  (analysis script for docs/LIGHTING_V2.md)
"""Assign generated lightshows to the human style clusters (nearest centroid)."""

import json
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr

sys.argv_saved = sys.argv
sys.path.insert(0, sys.argv[1])
from human_stats import map_features  # noqa: E402

from bsmapper.generation.model import song_features
from bsmapper.lighting.metrics import constant_tempo, window_stats

S = sys.argv[1]
model = json.load(open(f"{S}/styles_k6.json"))
mean, scale, centers = map(np.array, (model["mean"], model["scale"], model["centers"]))
NAMES = {
    0: "calm fades",
    1: "flash+zoom",
    2: "white+boost",
    3: "beat fades+laser sweeps",
    4: "sustained on/off",
    5: "strobe/sequenced",
}


def med(ws, key, idx):
    v = np.array([ws[k][key] for k in idx], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.nan


def features(ls, feats, bpm):
    nw = feats.shape[0] // 384
    ws = [w.as_dict() for w in window_stats(ls, nw * 16, constant_tempo(bpm))]
    f, rms = map_features(ls, feats, nw, bpm)
    e = rms[: nw * 384].reshape(nw, 384).mean(1)
    rel = e / e.max()
    live = [k for k in range(nw) if e[k] > 0.02]
    lit = [k for k in live if ws[k]["changes_per_s"] > 0]
    loud = [k for k in lit if rel[k] >= 0.7] or lit
    rate = np.array([ws[k]["changes_per_s"] for k in live])
    f["rate"] = med(ws, "changes_per_s", loud)
    tot = sum(med(ws, f"ch{t}_per_s", loud) for t in range(5)) or 1.0
    f["laser_share"] = (med(ws, "ch2_per_s", loud) + med(ws, "ch3_per_s", loud)) / tot
    f["ring_share"] = med(ws, "ch1_per_s", loud) / tot
    f["color_ratio"] = med(ws, "color_switches_per_s", loud) / max(f["rate"], 1e-6)
    for a in ("fade", "flash", "on", "transition", "off"):
        f[a] = med(ws, f"{a}_share", loud)
    f["strobe_ratio"] = med(ws, "strobe_per_s", loud) / max(f["rate"], 1e-6)
    f["hold_med"] = med(ws, "hold_median_s", loud)
    f["spin"] = med(ws, "spin_per_s", loud)
    f["zoom"] = med(ws, "zoom_per_s", loud)
    f["laser_speed"] = med(ws, "laser_speed_per_s", loud)
    f["energy_follow"] = float(spearmanr(e[live], rate).statistic) if rate.std() > 0 else 0.0
    lr = np.array([ws[k]["changes_per_s"] for k in loud])
    f["consistency_cv"] = float(lr.std() / max(lr.mean(), 1e-6))
    x = [
        np.log1p(f["rate"]),
        f["laser_share"],
        f["ring_share"],
        np.log1p(10 * f["color_ratio"]),
        f["fade"],
        f["flash"],
        f["on"],
        f["strobe_ratio"],
        np.log(max(f["hold_med"], 0.01)) if np.isfinite(f["hold_med"]) else 0.0,
        np.log1p(f["spin"]),
        np.log1p(f["zoom"]),
        np.log1p(f["laser_speed"]),
        f["on_beat"],
        f["off_grid"],
        min(f["onset_lift"], 5),
        f["boost_time"],
        f["white_share"],
        f["energy_follow"],
        min(f["consistency_cv"], 3),
    ]
    z = (np.nan_to_num(np.array(x, float)) - mean) / scale
    d = np.linalg.norm(centers - z, axis=1)
    return int(d.argmin()), f


if __name__ == "__main__":
    sweep = pickle.load(open(f"{S}/sweep.pkl", "rb"))
    cache = {}
    rows = []
    for (path, seed, name), ls in sorted(sweep.items(), key=str):
        if path not in cache:
            d = pickle.load(open(path, "rb"))
            cache[path] = (
                song_features(d["song"].analysis, d["beatmap"], d["song"].lead_in_s),
                d["beatmap"].bpm,
            )
        feats, bpm = cache[path]
        c, f = features(ls, feats, bpm)
        rows.append((path.rsplit("/", 1)[-1][:-4], seed, name, c, f))
        print(
            f"{rows[-1][0]:9s} seed {seed} {name:7s} -> {c} {NAMES[c]:24s} rate {f['rate']:5.1f} fade {f['fade']:.2f} on {f['on']:.2f} laser_share {f['laser_share']:.2f} laser_speed {f['laser_speed']:.2f}"
        )
    pickle.dump(rows, open(f"{S}/assign.pkl", "wb"))
