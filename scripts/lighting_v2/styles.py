# ruff: noqa: E402, E501, SIM115  (analysis script for docs/LIGHTING_V2.md)
"""Cluster per-map style features of the gold human lightshows."""

import json
import pickle
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

S = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 0
data = pickle.load(open(f"{S}/human.pkl", "rb"))


def med(ws, key, idx):
    v = np.array([ws[k][key] for k in idx], float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else np.nan


maps = []
for m in data:
    e = np.asarray(m["energy"], float)
    ws = m["windows"]
    if e.max() <= 0 or not m["features"]:
        continue
    rel = e / e.max()
    live = [k for k in range(len(ws)) if e[k] > 0.02]
    lit = [k for k in live if ws[k]["changes_per_s"] > 0]
    if len(lit) < 4:
        continue
    loud = [k for k in lit if rel[k] >= 0.7] or lit
    rate = np.array([ws[k]["changes_per_s"] for k in live])
    f = dict(m["features"])
    f["rate"] = med(ws, "changes_per_s", loud)
    if f["rate"] < 0.3:
        continue
    tot = sum(med(ws, f"ch{t}_per_s", loud) for t in range(5)) or 1.0
    f["laser_share"] = (med(ws, "ch2_per_s", loud) + med(ws, "ch3_per_s", loud)) / tot
    f["back_share"] = med(ws, "ch0_per_s", loud) / tot
    f["ring_share"] = med(ws, "ch1_per_s", loud) / tot
    f["center_share"] = med(ws, "ch4_per_s", loud) / tot
    f["color_ratio"] = med(ws, "color_switches_per_s", loud) / max(f["rate"], 1e-6)
    for a in ("fade", "flash", "on", "transition", "off"):
        f[a] = med(ws, f"{a}_share", loud)
    f["strobe_ratio"] = med(ws, "strobe_per_s", loud) / max(f["rate"], 1e-6)
    f["hold_med"] = med(ws, "hold_median_s", loud)
    f["spin"] = med(ws, "spin_per_s", loud)
    f["zoom"] = med(ws, "zoom_per_s", loud)
    f["laser_speed"] = med(ws, "laser_speed_per_s", loud)
    f["dark_share"] = 1 - len(lit) / max(len(live), 1)
    if len(live) > 4 and rate.std() > 0:
        f["energy_follow"] = float(spearmanr(e[live], rate).statistic)
    else:
        f["energy_follow"] = 0.0
    lr = np.array([ws[k]["changes_per_s"] for k in loud])
    f["consistency_cv"] = float(lr.std() / max(lr.mean(), 1e-6))
    f["meta"] = {
        k: m["meta"][k] for k in ("key", "song_name", "song_author", "mapper", "environment", "bpm")
    }
    maps.append(f)

print(len(maps), "maps")
FEATS = [
    ("log_rate", lambda f: np.log1p(f["rate"])),
    ("laser_share", lambda f: f["laser_share"]),
    ("ring_share", lambda f: f["ring_share"]),
    ("color_ratio", lambda f: np.log1p(10 * f["color_ratio"])),
    ("fade", lambda f: f["fade"]),
    ("flash", lambda f: f["flash"]),
    ("on", lambda f: f["on"]),
    ("strobe_ratio", lambda f: f["strobe_ratio"]),
    ("log_hold", lambda f: np.log(max(f["hold_med"], 0.01)) if np.isfinite(f["hold_med"]) else 0.0),
    ("log_spin", lambda f: np.log1p(f["spin"])),
    ("log_zoom", lambda f: np.log1p(f["zoom"])),
    ("log_laser_speed", lambda f: np.log1p(f["laser_speed"])),
    ("on_beat", lambda f: f["on_beat"]),
    ("off_grid", lambda f: f["off_grid"]),
    ("onset_lift", lambda f: min(f["onset_lift"], 5) if np.isfinite(f["onset_lift"]) else 1.0),
    ("boost_time", lambda f: f["boost_time"]),
    ("white_share", lambda f: f["white_share"]),
    ("energy_follow", lambda f: f["energy_follow"]),
    ("consistency_cv", lambda f: min(f["consistency_cv"], 3)),
]
X = np.array([[fn(f) for _, fn in FEATS] for f in maps], float)
X = np.nan_to_num(X)
scaler = StandardScaler().fit(X)
Z = scaler.transform(X)
if not K:
    for k in range(3, 10):
        km = KMeans(k, n_init=10, random_state=0).fit(Z)
        print(
            k,
            round(silhouette_score(Z, km.labels_, sample_size=3000, random_state=0), 3),
            np.bincount(km.labels_),
        )
    sys.exit()
km = KMeans(K, n_init=20, random_state=0).fit(Z)
lab = km.labels_
names = [n for n, _ in FEATS]
raw = [
    "rate",
    "laser_share",
    "ring_share",
    "color_ratio",
    "fade",
    "flash",
    "on",
    "strobe_ratio",
    "hold_med",
    "spin",
    "zoom",
    "laser_speed",
    "on_beat",
    "on_half",
    "off_grid",
    "onset_lift",
    "boost_time",
    "white_share",
    "energy_follow",
    "consistency_cv",
    "chroma",
    "dark_share",
]
summary = []
for c in range(K):
    idx = np.where(lab == c)[0]
    print(f"\n=== cluster {c}: {len(idx)} maps")
    zc = Z[idx].mean(0)
    top = np.argsort(-np.abs(zc))[:6]
    print("  distinctive:", ", ".join(f"{names[i]} {zc[i]:+.2f}" for i in top))
    row = {"cluster": c, "n": int(len(idx))}
    for r in raw:
        v = np.array([maps[i][r] for i in idx], float)
        row[r] = float(np.nanmedian(v))
    print("  medians:", " ".join(f"{r}={row[r]:.2f}" for r in raw))
    d = np.linalg.norm(Z[idx] - km.cluster_centers_[c], axis=1)
    ex = [maps[idx[i]]["meta"] for i in np.argsort(d)[:8]]
    for e in ex:
        print(
            f"   - {e['key']} {e['song_name']} - {e['song_author']} (mapper {e['mapper']}, {e['environment']}, {e['bpm']:.0f} bpm)"
        )
    row["examples"] = ex
    envs = {}
    for i in idx:
        envs[maps[i]["meta"]["environment"]] = envs.get(maps[i]["meta"]["environment"], 0) + 1
    row["top_envs"] = sorted(envs.items(), key=lambda x: -x[1])[:4]
    print("  envs:", row["top_envs"])
    summary.append(row)
json.dump(
    {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "centers": km.cluster_centers_.tolist(),
        "features": names,
        "clusters": summary,
        "labels": {maps[i]["meta"]["key"]: int(lab[i]) for i in range(len(maps))},
    },
    open(f"{S}/styles_k{K}.json", "w"),
    indent=1,
    default=str,
)
