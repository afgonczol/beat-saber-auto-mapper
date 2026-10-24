# ruff: noqa: E402, E501, SIM115  (analysis script for docs/LIGHTING_V2.md)
"""Generate notes for a song once (current defaults) and pickle song + beatmap + basic lights.

usage: python gen_cache.py <audio> <out.pkl>
"""

import logging
import os
import pickle
import sys
import time
from pathlib import Path

from bsmapper.generation import make_pipeline
from bsmapper.generation.intensity import configured_intensity
from bsmapper.generation.lighting import ModelLightingGenerator
from bsmapper.generation.pipeline import GenerateOptions

CHECKPOINT = os.environ.get("BSMAPPER_CHECKPOINT", "checkpoints/note_v1/best.pt")

logging.basicConfig(level=logging.INFO)
audio, out = Path(sys.argv[1]), Path(sys.argv[2])
t0 = time.time()
pipe = make_pipeline(checkpoint=CHECKPOINT)
assert pipe.light_model is not None
song = pipe.analyze(audio)
opts = GenerateOptions(intensity=configured_intensity(), arcs=True)
beatmap, results = pipe.build(song, opts)
print("built", time.time() - t0, pipe.last_lighting)
gen = ModelLightingGenerator(pipe.light_model, time_offset_s=song.lead_in_s, seed=opts.seed)
basic = gen.basic_lightshow(song.analysis, beatmap)
rendered = beatmap.lightshow
beatmap.lightshow = None
with out.open("wb") as f:
    pickle.dump({"song": song, "beatmap": beatmap, "basic": basic, "rendered": rendered}, f)
print("done", time.time() - t0)
