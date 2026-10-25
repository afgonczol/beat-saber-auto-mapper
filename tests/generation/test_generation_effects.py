"""Human effect libraries in the pipeline (WS8c): discovery, environment and lights."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from gen_synthetic import FakeAnalyzer

from bsmapper.generation import GenerateOptions, Pipeline
from bsmapper.generation.runtime import find_light_effects
from bsmapper.lighting.effects import EFFECTS_VERSION, INDEX_FILE
from bsmapper.lighting.effects_build import describe

ALL = {"c": 0, "f": 1, "p": 1, "t": 0, "r": 0, "n": 0, "s": 0, "l": 0.0, "d": 0}


def make_library(root: Path, env: str = "PyroEnvironment") -> Path:
    groups = []
    for beat in range(64):
        event = {"b": 0.0, "i": 0, "c": beat % 3, "s": 1.0, "f": 0, "sb": 0.0, "sf": 0}
        box = {"f": dict(ALL), "w": 0.25, "d": 1, "r": 0.0, "t": 1, "b": 0, "i": 0, "e": [event]}
        groups.append([float(beat), "color", beat % 4, [box]])
    source = {"bpm": 120.0, "bars": 16, "groups": groups, "basic": [], "boost": []}
    folder = root / env
    folder.mkdir(parents=True)
    with gzip.open(folder / "abc.json.gz", "wt", encoding="utf-8") as f:
        json.dump(source, f)
    entry = {"key": "abc", "mapper": "someone", "song": "x", "bars": 16}
    entry |= describe(groups, [], 16)
    doc = {"version": EFFECTS_VERSION, "environment": env, "maps": [entry]}
    (folder / INDEX_FILE).write_text(json.dumps(doc), encoding="utf-8")
    return root


def song_file(tmp_path: Path) -> Path:
    path = tmp_path / "Some Artist - Some Song.mp3"
    path.write_bytes(b"not decoded; the fake analyzer ignores it")
    return path


def test_find_light_effects(tmp_path: Path) -> None:
    assert find_light_effects() is None  # tests never see a checkout's library
    root = make_library(tmp_path / "fx")
    assert find_light_effects(root) == root
    with pytest.raises(FileNotFoundError):
        find_light_effects(tmp_path / "missing")


def test_pipeline_lights_v3_from_the_effect_library(tmp_path: Path) -> None:
    root = make_library(tmp_path / "fx")
    pipeline = Pipeline(analyzer=FakeAnalyzer(), effects_root=root)
    song = pipeline.analyze(song_file(tmp_path))
    beatmap, _ = pipeline.build(song, GenerateOptions())
    assert beatmap.environment == "PyroEnvironment"
    assert pipeline.last_effects == [("abc", "someone")]
    assert {g.group_id for g in beatmap.lightshow.event_box_groups} <= {0, 1, 2, 3}

    beatmap, _ = pipeline.build(song, GenerateOptions(light_effects=False))
    assert beatmap.environment == "WeaveEnvironment"  # the profile renderer's default
    assert pipeline.last_effects == []

    beatmap, _ = pipeline.build(song, GenerateOptions(lighting="chroma"))
    assert pipeline.last_effects == [] and not beatmap.lightshow.event_box_groups


def test_model_lighting_renders_with_effects(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from gen_synthetic import make_analysis

    from bsmapper.generation import lighting as lighting_mod
    from bsmapper.generation import tempo_beatmap
    from bsmapper.lighting import EffectRenderer
    from bsmapper.models import MapperModel, ModelConfig
    from bsmapper.tokenizer import LIGHT_VOCAB_SIZES

    tiny = dict(
        d_model=32, n_heads=4, ff_mult=2, dropout=0.0, encoder_layers=1, decoder_layers=1,
        light_layers=1, note_memory_layers=1, max_len=192, light_max_len=192,
        note_memory_len=128, light_vocab=LIGHT_VOCAB_SIZES[1],
    )  # fmt: skip
    torch.manual_seed(0)
    model = MapperModel(ModelConfig(**tiny)).eval()
    monkeypatch.setattr(lighting_mod, "check_density", lambda *a: None)  # a random tiny model
    analysis = make_analysis(duration=20.0)
    frames = int(20.0 * analysis.frame_rate)
    analysis.features["chroma"] = np.random.default_rng(0).random((frames, 12)).astype(np.float32)
    beatmap = tempo_beatmap(analysis)
    beatmap.environment = "PyroEnvironment"
    effects = EffectRenderer(make_library(tmp_path / "fx"))
    gen = lighting_mod.ModelLightingGenerator(
        model, max_body_tokens=24, readability=None, effects=effects, light_style="calm"
    )
    lightshow = gen.generate(analysis, beatmap, "v3")
    assert lightshow.event_box_groups and effects.last_sources == [("abc", "someone")]
    assert gen._refs is not None  # structure references feed the section repeats
    assert effects.style == "calm"  # a named style biases the effects, even for v1 models
