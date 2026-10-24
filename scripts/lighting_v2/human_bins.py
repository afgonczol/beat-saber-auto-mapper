# ruff: noqa: E402, E501, SIM115  (analysis script for docs/LIGHTING_V2.md)
"""Human window stats by song-relative energy: quantiles across windows (each map weighted
equally) of the readability metrics. Writes human_bins.json."""

import json
import pickle
import sys

import numpy as np

S = sys.argv[1]
data = pickle.load(open(f"{S}/human.pkl", "rb"))
BINS = [
    (0.0, 0.4, "calm <0.4"),
    (0.4, 0.7, "mid 0.4-0.7"),
    (0.7, 0.85, "loud 0.7-0.85"),
    (0.85, 1.01, "peak >=0.85"),
]
KEYS = [
    "changes_per_s",
    "active_ticks_per_s",
    "color_switches_per_s",
    "on_off_per_s",
    "strobe_per_s",
    "short_hold_share",
    "hold_p10_s",
    "hold_median_s",
    "ch0_per_s",
    "ch1_per_s",
    "ch2_per_s",
    "ch3_per_s",
    "ch4_per_s",
    "spin_per_s",
    "zoom_per_s",
    "laser_speed_per_s",
    "flash_share",
    "fade_share",
    "on_share",
    "transition_share",
    "off_share",
]
rows = {b[2]: {k: [] for k in KEYS} | {"w": []} for b in BINS}
for m in data:
    e = np.asarray(m["energy"], float)
    if e.max() <= 0:
        continue
    rel = e / e.max()
    ws = m["windows"]
    # skip the silent tail and windows with no lights at all (not part of the show)
    idx = [k for k in range(len(ws)) if e[k] > 0.02 and ws[k]["changes_per_s"] > 0]
    per_bin = {}
    for k in idx:
        for lo, hi, name in BINS:
            if lo <= rel[k] < hi:
                per_bin.setdefault(name, []).append(k)
    for name, ks in per_bin.items():
        for k in ks:
            for key in KEYS:
                rows[name][key].append(ws[k][key])
            rows[name]["w"].append(1.0 / len(ks))


def wq(v, w, qs):
    v, w = np.asarray(v, float), np.asarray(w, float)
    ok = np.isfinite(v)
    v, w = v[ok], w[ok]
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    return [float(v[min(np.searchsorted(c, q), len(v) - 1)]) for q in qs]


QS = [0.1, 0.5, 0.9, 0.95, 0.99]
out = {}
for name in rows:
    out[name] = {"n_windows": len(rows[name]["w"])}
    for key in KEYS:
        out[name][key] = wq(rows[name][key], rows[name]["w"], QS)
json.dump(out, open(f"{S}/human_bins.json", "w"), indent=1)
for name, d in out.items():
    print(f"== {name}  windows={d['n_windows']}")
    for key in KEYS:
        print(f"  {key:22s} " + " ".join(f"{x:7.3f}" for x in d[key]))
