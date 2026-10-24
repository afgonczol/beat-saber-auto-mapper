# ruff: noqa: E402, E501, SIM115  (analysis script for docs/LIGHTING_V2.md)
"""Seeds x decode-mask settings for cached songs. Saves basic lightshows."""

import os
import pickle
import sys
import time
from dataclasses import replace

from bsmapper.generation.lighting import ModelLightingGenerator
from bsmapper.generation.runtime import load_cached
from bsmapper.lighting.readability import ReadabilityConfig

CHECKPOINT = os.environ.get("BSMAPPER_CHECKPOINT", "checkpoints/note_v1/best.pt")

model = load_cached(CHECKPOINT, None)
base = ReadabilityConfig()
settings = {
    "none": None,
    "types": replace(base, decode_mask=True),
    "motion": replace(base, decode_mask=True, decode_motion_gaps=True),
    "holds": replace(base, decode_mask=True, decode_light_holds=True, decode_motion_gaps=True),
}
out = {}
for path in sys.argv[1:]:
    d = pickle.load(open(path, "rb"))
    song, bm = d["song"], d["beatmap"]
    for seed in range(4):
        for name, cfg in settings.items():
            gen = ModelLightingGenerator(
                model, time_offset_s=song.lead_in_s, seed=seed, readability=cfg
            )
            t0 = time.time()
            out[(path, seed, name)] = gen.basic_lightshow(song.analysis, bm)
            print(path[-12:], seed, name, round(time.time() - t0, 1), flush=True)
            pickle.dump(out, open(path.rsplit("/", 1)[0] + "/sweep.pkl", "wb"))
