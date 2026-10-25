"""Build ``light_styles.json`` for a dataset: style labels, the auto-pick and scene tables.

    bsmapper lights styles data/processed_v1          # writes data/processed_v1/light_styles.json

Reads the gold maps with a basic-event lightshow on a constant tempo (``maps.parquet``,
the light shards and the feature shards), then:

1. per map: ``styles.map_style_features``, ``styles.song_vector``, and per window its
   scene (``tokenizer.lighting.window_scene``) and loudness bin;
2. leaves out lightshows too sparse to describe a style and Chroma / strobe sequences
   (``styles.is_sequenced``);
3. clusters the rest into ``k`` styles (k-means on standardised features), names them
   by ``styles.STYLE_PROTOTYPES``;
4. fits the auto-pick (song features -> style) on the train split, and reports its
   accuracy on the val split;
5. counts human scenes per style and loudness bin (train split).

The file holds ``{"model": StyleModel, "labels": {key: style}, "excluded": {key:
reason}}``. Light training (``training.data.LightWindows``, version 2) reads the labels;
the trainer copies the model into the checkpoint.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from bsmapper.lighting.styles import (
    STYLE_DESCRIPTIONS,
    StyleModel,
    fit_softmax,
    is_sequenced,
    kmeans,
    loudness_bin,
    map_style_features,
    name_clusters,
    song_vector,
    style_vector,
    window_loudness,
)
from bsmapper.tokenizer import WINDOW_TICKS, LightingTokenizer, TokenWindow, window_scene

log = logging.getLogger(__name__)

STYLE_FILE = "light_styles.json"
DEFAULT_K = 5
MIN_SCENE_P = 0.003  # rarer scenes are left out of the tables


def _pq():
    import pyarrow.parquet as pq

    return pq


def candidate_maps(root: Path) -> list[dict[str, Any]]:
    """Gold maps with a basic-event lightshow, no dropped box groups and one tempo."""
    maps = _pq().read_table(root / "maps.parquet").to_pylist()
    return [
        m
        for m in maps
        if m.get("status") == "kept"
        and m.get("gold")
        and m.get("light_shard") is not None
        and not (m.get("light_dropped_box_groups") or 0)
        and not m.get("bpm_changes")
    ]


def _process(job: tuple[str, str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    root, shard, items = job
    root_p = Path(root)
    table = _pq().read_table(root_p / "lights" / f"{shard}.parquet").to_pylist()
    features: dict[str, np.ndarray] = {}
    tok = LightingTokenizer()
    out = []
    for m in items:
        try:
            row = table[m["light_row"]]
            if row["key"] != m["key"]:
                raise ValueError(f"light row {m['light_row']} of {shard} is not {m['key']}")
            tokens, offsets = row["tokens"], row["window_offsets"]
            bodies = [list(tokens[offsets[k] : offsets[k + 1]]) for k in range(len(offsets) - 1)]
            lightshow = tok.decode([TokenWindow(k, b) for k, b in enumerate(bodies)])
            fs = m["feature_shard"]
            if fs not in features:
                features[fs] = np.load(root_p / "features" / f"{fs}.npy", mmap_mode="r")
            n = m["n_windows"]
            feats = np.asarray(
                features[fs][m["feature_row"] : m["feature_row"] + n * WINDOW_TICKS],
                dtype=np.float32,
            )
            loud = window_loudness(feats, n)
            out.append(
                {
                    "key": m["key"],
                    "split": m["split"],
                    "features": map_style_features(lightshow, feats, m["bpm"]),
                    "song": song_vector(feats, m["bpm"]).tolist(),
                    "scenes": [window_scene(b) for b in bodies[:n]],
                    "bins": [loudness_bin(float(r)) for r in loud],
                }
            )
        except Exception as exc:  # noqa: BLE001 - one bad map must not stop the build
            log.warning("light styles: skipping %s (%s)", m["key"], exc)
    return out


def collect(root: Path, workers: int = 8, limit: int | None = None) -> list[dict[str, Any]]:
    maps = candidate_maps(root)[:limit]
    jobs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in maps:
        jobs[m["light_shard"]].append(m)
    args = [(str(root), shard, items) for shard, items in sorted(jobs.items())]
    if workers <= 1:
        return [r for job in args for r in _process(job)]
    with Pool(workers) as pool:
        return [r for rows in pool.imap_unordered(_process, args) for r in rows]


def scene_tables(
    rows: list[dict[str, Any]], labels: dict[str, int], names: list[str]
) -> dict[str, dict[str, list[list[float]]]]:
    counts: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        if r["key"] not in labels or r["split"] != "train":
            continue
        style = names[labels[r["key"]]]
        for scene, b in zip(r["scenes"], r["bins"], strict=False):
            counts[style][b][tuple(scene)] += 1
            counts["*"][b][tuple(scene)] += 1
    tables: dict[str, dict[str, list[list[float]]]] = {}
    for style, bins in counts.items():
        tables[style] = {}
        for b, counter in sorted(bins.items()):
            total = sum(counter.values())
            kept = [(s, c / total) for s, c in counter.most_common() if c / total >= MIN_SCENE_P]
            norm = sum(p for _, p in kept) or 1.0
            tables[style][str(b)] = [[*s, round(p / norm, 5)] for s, p in kept]
    return tables


def build(
    root: Path, k: int = DEFAULT_K, workers: int = 8, limit: int | None = None, seed: int = 0
) -> dict[str, Any]:
    """Build and write ``root/light_styles.json``; returns its contents."""
    rows = collect(root, workers, limit)
    excluded: dict[str, str] = {}
    usable = []
    for r in rows:
        f = r["features"]
        if f is None:
            excluded[r["key"]] = "sparse"
        elif is_sequenced(f):
            excluded[r["key"]] = "sequenced"
        else:
            usable.append(r)
    if len(usable) < k:
        raise ValueError(f"only {len(usable)} usable lightshows for {k} styles")
    x = np.array([style_vector(r["features"]) for r in usable])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    centers, labels_arr = kmeans(z, k, seed=seed)
    names = name_clusters(centers)
    labels = {r["key"]: int(c) for r, c in zip(usable, labels_arr, strict=True)}

    songs = np.array([r["song"] for r in usable])
    train = np.array([r["split"] == "train" for r in usable])
    s_mean, s_scale = songs[train].mean(axis=0), songs[train].std(axis=0)
    s_scale[s_scale == 0] = 1.0
    sz = (songs - s_mean) / s_scale
    coef, intercept = fit_softmax(sz[train], labels_arr[train], k)
    model = StyleModel(
        names=names,
        mean=mean.tolist(),
        scale=scale.tolist(),
        centers=centers.tolist(),
        song_mean=s_mean.tolist(),
        song_scale=s_scale.tolist(),
        coef=coef.tolist(),
        intercept=intercept.tolist(),
    )
    model.scenes = scene_tables(rows, labels, names)
    val = ~train
    picked = np.array([model.song_probs(s).argmax() for s in songs])
    counts = np.bincount(labels_arr, minlength=k)
    model.stats = {
        "maps": len(rows),
        "labelled": len(usable),
        "excluded": dict(Counter(excluded.values())),
        "sizes": {names[i]: int(counts[i]) for i in range(k)},
        "descriptions": {n: STYLE_DESCRIPTIONS.get(n, "") for n in names},
        "auto_pick_val_accuracy": float((picked[val] == labels_arr[val]).mean())
        if val.any()
        else None,
        "majority_val_accuracy": float((labels_arr[val] == counts.argmax()).mean())
        if val.any()
        else None,
        "val_maps": int(val.sum()),
    }
    out = {"model": model.to_dict(), "labels": labels, "excluded": excluded}
    (root / STYLE_FILE).write_text(json.dumps(out), encoding="utf-8")
    return out


def load_labels(path: Path | str) -> tuple[StyleModel, dict[str, int]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return StyleModel.from_dict(d["model"]), {k: int(v) for k, v in d["labels"].items()}
