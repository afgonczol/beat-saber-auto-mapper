# ruff: noqa: E402, E501, SIM115  (analysis script for docs/LIGHTING_V2.md)
"""Compact per-section diagnosis (basic + rendered) for several lightshow variants.
usage: diagnose.py <song.pkl> <human_bins.json> name=path|base|base+filter ...
"""

import json
import pickle
import sys

from bsmapper.generation.model import song_features
from bsmapper.lighting import RuleLightingGenerator, render
from bsmapper.lighting.metrics import motion_stats, span_stats, visible_changes
from bsmapper.lighting.readability import limit_lightshow, section_intensity
from bsmapper.lighting.rules import RuleLightshow

d = pickle.load(open(sys.argv[1], "rb"))
human = json.load(open(sys.argv[2]))
song, bm = d["song"], d["beatmap"]
b2s = bm.beat_to_seconds
plan = RuleLightingGenerator(time_offset_s=song.lead_in_s).plan(song.analysis, bm)
secs = plan.sections
feats = song_features(song.analysis, bm, song.lead_in_s)
rms = feats[:, 82]
emax = max(rms[int(s.start * 24) : int(s.end * 24)].mean() for s in secs if s.end > s.start)
inten = section_intensity([s.start for s in secs], [s.intensity for s in secs])
BINS = [
    (0.0, 0.4, "calm <0.4"),
    (0.4, 0.7, "mid 0.4-0.7"),
    (0.7, 0.85, "loud 0.7-0.85"),
    (0.85, 1.01, "peak >=0.85"),
]
variants = []
for spec in sys.argv[3:]:
    name, src = spec.split("=")
    old = src.endswith("@old")
    src = src.replace("@old", "")
    filt = src.endswith("+filter")
    src = src.replace("+filter", "")
    ls = d["basic"] if src == "base" else pickle.load(open(src, "rb"))
    if filt:
        ls, rep = limit_lightshow(ls, b2s, inten, fallback=plan.lightshow)
        print(name, "filter:", dict(rep.dropped), "recolored", rep.recolored, "moved", rep.moved)
    variants.append((name, ls, old))
COLS = [
    "changes_per_s",
    "active_ticks_per_s",
    "color_switches_per_s",
    "strobe_per_s",
    "hold_p10_s",
    "spin_per_s",
    "zoom_per_s",
    "laser_speed_per_s",
]
summary = {}
import importlib.util

spec_ = importlib.util.spec_from_file_location(
    "render_old", sys.argv[1].rsplit("/", 1)[0] + "/render_old.py"
)
render_old = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(render_old)
from bsmapper.lighting.profiles import get_profile

for name, ls, old in variants:
    ch = visible_changes(ls)
    r = (
        render_old.render_v3(ls, get_profile("WeaveEnvironment"))
        if old
        else render(RuleLightshow(ls, secs, plan.end_beat), "v3", "WeaveEnvironment")
    )
    print(
        f"\n== {name}: {'section':16} {'e':>4} "
        + " ".join(f"{c[:9]:>9}" for c in COLS)
        + "  rot_mv/s  whip/s  maxdeg/s | human p50/p90 chg, col, ticks"
    )
    rows = []
    for s in secs:
        st = span_stats(ls, s.start, s.end, b2s, ch).as_dict()
        mo = motion_stats(r, s.start, s.end, b2s)
        e = rms[int(s.start * 24) : int(s.end * 24)].mean() / emax
        h = human[next(n for lo, hi, n in BINS if lo <= e < hi)]
        print(
            f"   {b2s(s.start):4.0f}-{b2s(s.end):4.0f}s {s.label:8} {e:4.2f} "
            + " ".join(f"{st[c]:9.2f}" for c in COLS)
            + f"  {mo.moves_per_s:8.2f} {mo.whips_per_s:7.2f} {mo.max_deg_per_s:9.0f} | {h['changes_per_s'][1]:.1f}/{h['changes_per_s'][2]:.1f}, {h['color_switches_per_s'][1]:.1f}/{h['color_switches_per_s'][2]:.1f}, {h['active_ticks_per_s'][1]:.1f}/{h['active_ticks_per_s'][2]:.1f}"
        )
        rows.append(
            dict(
                section=s.label,
                start_s=b2s(s.start),
                end_s=b2s(s.end),
                energy=float(e),
                **{c: st[c] for c in COLS},
                rot_moves_per_s=mo.moves_per_s,
                whips_per_s=mo.whips_per_s,
                max_deg_per_s=mo.max_deg_per_s,
            )
        )
    whole = span_stats(ls, secs[0].start, secs[-1].end, b2s, ch).as_dict()
    mo = motion_stats(r, secs[0].start, secs[-1].end, b2s)
    print(
        "   whole song: "
        + " ".join(f"{c}={whole[c]:.2f}" for c in COLS)
        + f" rot_moves={mo.moves_per_s:.2f} whips={mo.whips_per_s:.2f} maxdeg={mo.max_deg_per_s:.0f}"
    )
    summary[name] = dict(
        sections=rows,
        whole={
            **{c: whole[c] for c in COLS},
            "rot_moves_per_s": mo.moves_per_s,
            "whips_per_s": mo.whips_per_s,
            "max_deg_per_s": mo.max_deg_per_s,
        },
    )
out = sys.argv[1].replace(".pkl", "_diag.json")
json.dump(summary, open(out, "w"), indent=1)
