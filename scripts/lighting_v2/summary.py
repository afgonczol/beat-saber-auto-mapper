# ruff: noqa: E402, E501, SIM115  (analysis script for docs/LIGHTING_V2.md)
"""Before/after table from the sweep: per song, mean over seeds (and seed 0) of the jitter metrics."""

import importlib.util
import json
import pickle
import sys
from collections import defaultdict

import numpy as np

from bsmapper.lighting import RuleLightingGenerator, render
from bsmapper.lighting.metrics import motion_stats, pattern_repeat, span_stats, visible_changes
from bsmapper.lighting.profiles import get_profile
from bsmapper.lighting.readability import limit_lightshow, section_intensity
from bsmapper.lighting.rules import RuleLightshow

sys.path.insert(0, sys.argv[1])
from assign import features  # noqa: E402

from bsmapper.generation.model import song_features

S = sys.argv[1]
spec = importlib.util.spec_from_file_location("render_old", f"{S}/render_old.py")
render_old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_old)
sweep = pickle.load(open(f"{S}/sweep.pkl", "rb"))
JITTER = {
    "stronger": (152.0, 236.0)
}  # beats; Toxic: the section with the most colour flips (before)
modes = {
    "before (main)": ("none", False, True),
    "renderer fix only": ("none", False, False),
    "renderer + post-filter": ("none", True, False),
    "quick win default (types mask + filter)": ("types", True, False),
    "opt-in: + motion-gap mask": ("motion", True, False),
    "opt-in: + hold mask": ("holds", True, False),
}
songs = {}
for path, seed, _mode in sweep:
    songs.setdefault(path, set()).add(seed)
out = {}
for path, seeds in songs.items():
    name = path.rsplit("/", 1)[-1][:-4]
    d = pickle.load(open(path, "rb"))
    song, bm = d["song"], d["beatmap"]
    b2s = bm.beat_to_seconds
    plan = RuleLightingGenerator(time_offset_s=song.lead_in_s).plan(song.analysis, bm)
    secs = plan.sections
    inten = section_intensity([s.start for s in secs], [s.intensity for s in secs])
    feats = song_features(song.analysis, bm, song.lead_in_s)
    lo, hi = secs[0].start, secs[-1].end
    if name not in JITTER:
        base = sweep[(path, 0, "none")]
        ch = visible_changes(base)
        best = max(
            secs, key=lambda s: span_stats(base, s.start, s.end, b2s, ch).color_switches_per_s
        )
        JITTER[name] = (best.start, best.end)
    js, je = JITTER[name]
    res = defaultdict(list)
    for mode, (mname, filt, old) in modes.items():
        for seed in sorted(seeds):
            if (path, seed, mname) not in sweep:
                continue
            ls = sweep[(path, seed, mname)]
            rep = None
            if filt:
                ls, rep = limit_lightshow(ls, b2s, inten, fallback=plan.lightshow)
            r = (
                render_old.render_v3(ls, get_profile("WeaveEnvironment"))
                if old
                else render(RuleLightshow(ls, secs, plan.end_beat), "v3", "WeaveEnvironment")
            )
            ch = visible_changes(ls)
            w = span_stats(ls, lo, hi, b2s, ch)
            j = span_stats(ls, js, je, b2s, ch)
            mw = motion_stats(r, lo, hi, b2s)
            mj = motion_stats(r, js, je, b2s)
            cluster, _ = features(ls, feats, bm.bpm)
            res[mode].append(
                dict(
                    seed=seed,
                    changes=w.changes_per_s,
                    flips=w.color_switches_per_s,
                    moments=w.active_ticks_per_s,
                    laser=w.laser_speed_per_s,
                    spins=w.spin_per_s,
                    strobe=w.strobe_per_s,
                    j_changes=j.changes_per_s,
                    j_flips=j.color_switches_per_s,
                    j_laser=j.laser_speed_per_s,
                    whips=mw.whips_per_s * 60,
                    maxdeg=mw.max_deg_per_s,
                    j_maxdeg=mj.max_deg_per_s,
                    repeat=pattern_repeat(ch, lo, hi),
                    cluster=cluster,
                    filt=rep.summary() if rep else "",
                )
            )
    out[name] = dict(jitter=(b2s(js), b2s(je)), res=res)
    print(f"\n### {name}: jitter section {b2s(js):.0f}-{b2s(je):.0f} s; seeds {sorted(seeds)}")
    cols = [
        "changes",
        "flips",
        "moments",
        "laser",
        "spins",
        "j_flips",
        "j_laser",
        "whips",
        "maxdeg",
        "repeat",
    ]
    print(f"{'mode':42s} " + " ".join(f"{c:>8s}" for c in cols) + "  styles(seed0..)")
    for mode, rows in res.items():
        m = {c: np.nanmean([r[c] for r in rows]) for c in cols}
        mx = max(r["maxdeg"] for r in rows)
        print(
            f"{mode:42s} "
            + " ".join(f"{m[c]:8.2f}" if c != "maxdeg" else f"{mx:8.0f}" for c in cols)
            + "  "
            + ",".join(str(r["cluster"]) for r in rows)
        )
        r0 = rows[0]
        print(
            f"{'   seed 0':42s} "
            + " ".join(f"{r0[c]:8.2f}" if c != "maxdeg" else f"{r0[c]:8.0f}" for c in cols)
            + ("  " + r0["filt"] if r0["filt"] else "")
        )
json.dump(out, open(f"{S}/summary.json", "w"), indent=1, default=float)
