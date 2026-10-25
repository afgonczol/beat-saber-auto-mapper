# ruff: noqa: E402, E501  (analysis script for docs/LIGHTING_V2.md §5)
"""Style and scene adherence of a light model on val maps (WS8b phase 2).

usage: python phase2_eval.py <checkpoint> <dataset> <out.json> [n_maps] [seed]

For each labelled gold val map it decodes the lights on the map's human audio features
and hardest note track, once per condition:

- ``none``: no style and no scene (what note_v1 does; v1 checkpoints only run this);
- ``style=<name>``: the style token only;
- ``style+scene``: the map's own human style and human window scenes (oracle scenes);

then labels each generated show with the dataset's style model (``StyleModel.assign``)
and measures visible changes / colour flips per second on loud windows and how often each
window's scene matches the requested one. Human values of the same maps are included.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bsmapper.lighting.metrics import constant_tempo, window_stats
from bsmapper.lighting.style_build import load_labels
from bsmapper.lighting.styles import LOUD, map_style_features, window_loudness
from bsmapper.models import LightSampler, SamplingOptions, WindowRefs, load_model
from bsmapper.tokenizer import (
    LIGHT_GRAMMAR,
    LightingTokenizer,
    LightPrompt,
    TokenWindow,
    version_for_vocab,
    window_scene,
)
from bsmapper.training import DatasetStore

CKPT, ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
RANK = {"Easy": 0, "Normal": 1, "Hard": 2, "Expert": 3, "ExpertPlus": 4}

device = "cuda" if torch.cuda.device_count() else "cpu"
model = load_model(CKPT, device)
sampler = LightSampler(model)
style_model, labels = load_labels(ROOT / "light_styles.json")
store = DatasetStore(ROOT, ["val"])
version = version_for_vocab(model.cfg.note_vocab)
maps = {}
for i, r in enumerate(store.rows):
    if r["key"] in labels and r["key"] in store.lights and not r.get("bpm_changes"):
        best = maps.get(r["key"])
        if best is None or RANK[r["difficulty"]] > RANK[store.rows[best]["difficulty"]]:
            maps[r["key"]] = i
keys = sorted(maps)[:: max(1, len(maps) // N)][:N]
tok = LightingTokenizer()


def show_stats(bodies, feats, bpm):
    ls = tok.decode([TokenWindow(k, b) for k, b in enumerate(bodies)])
    n = len(bodies)
    loud = window_loudness(feats, n)
    ws = window_stats(ls, n * 16, constant_tempo(bpm))
    idx = [k for k in range(n) if loud[k] >= LOUD] or list(range(n))
    f = map_style_features(ls, feats, bpm)
    return {
        "style": None if f is None else style_model.names[style_model.assign(f)],
        "changes_per_s": float(np.median([ws[k].changes_per_s for k in idx])),
        "flips_per_s": float(np.median([ws[k].color_switches_per_s for k in idx])),
        "dark_loud_windows": int(sum(ws[k].changes_per_s < 0.5 for k in idx)),
    }


def scene_match(bodies, scenes):
    hits = [0, 0, 0]
    for b, want in zip(bodies, scenes, strict=True):
        got = window_scene(b)
        hits[0] += abs(got[0] - want[0]) <= 1
        hits[1] += got[1] == want[1]
        hits[2] += abs(got[2] - want[2]) <= 1
    return [h / max(len(bodies), 1) for h in hits]


conditions = ["none"]
if sampler.supports_prompts:
    conditions += [f"style={n}" for n in style_model.names] + ["style+scene"]
records = []
t0 = time.time()
for key in keys:
    i = maps[key]
    r = store.rows[i]
    track, lights = store.notes[i], store.lights[key]
    n = min(track.n_windows, lights.n_windows)
    feats = np.concatenate([store.features(i, k) for k in range(n)]).astype(np.float32)
    prefix = store.conditioning(i).prefix_ids(version=version)
    notes = [TokenWindow(k, track.body(k), [], prefix) for k in range(n)]
    starts = [store.ref_start(i, k, 0.85) for k in range(n)]
    human = [lights.body(k) for k in range(n)]
    human_scenes = [window_scene(b) for b in human]
    rec = {"key": key, "label": style_model.names[labels[key]], "bpm": r["bpm"], "windows": n,
           "human": show_stats(human, feats, r["bpm"])}  # fmt: skip
    for cond in conditions:
        style = None
        if cond.startswith("style="):
            style = style_model.index(cond.split("=")[1])
        elif cond == "style+scene":
            style = labels[key]
        prompts = [LightPrompt(style)] * n
        if cond == "style+scene":
            prompts = [p.with_scene(s) for p, s in zip(prompts, human_scenes, strict=True)]
        refs = WindowRefs([-1 if s is None else s for s in starts], 384, LIGHT_GRAMMAR)
        out = sampler.generate(
            feats, notes, SamplingOptions(temperature=0.9, top_p=0.95, seed=SEED),
            prompts=prompts, ref_fn=refs if sampler.supports_prompts else None,
        )  # fmt: skip
        bodies = [w.body for w in out]
        stats = show_stats(bodies, feats, r["bpm"])
        stats["scene_match"] = scene_match(bodies, human_scenes)
        rec[cond] = stats
    records.append(rec)
    print(f"{len(records)}/{len(keys)} {key} ({rec['label']}) {time.time() - t0:.0f}s", flush=True)

summary = {}
for cond in ["human", *conditions]:
    rows = [rec[cond] for rec in records]
    s = {
        "changes_per_s": float(np.median([x["changes_per_s"] for x in rows])),
        "flips_per_s": float(np.median([x["flips_per_s"] for x in rows])),
        "dark_loud_windows": int(sum(x["dark_loud_windows"] for x in rows)),
        "styles": {n: sum(x["style"] == n for x in rows) for n in style_model.names},
    }
    if cond.startswith("style="):
        s["hit_rate"] = s["styles"][cond.split("=")[1]] / len(rows)
    if cond == "style+scene":
        s["hit_rate"] = float(np.mean([rec[cond]["style"] == rec["label"] for rec in records]))
    if cond != "human":
        s["scene_match"] = np.mean([x["scene_match"] for x in rows], axis=0).round(3).tolist()
    summary[cond] = s
    print(cond, json.dumps(s))
OUT.write_text(json.dumps({"checkpoint": str(CKPT), "summary": summary, "maps": records}, indent=1))
