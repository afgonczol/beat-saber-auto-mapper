"""ModelGenerator, model lighting and generator selection, on a tiny random checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gen_synthetic import FakeAnalyzer, make_analysis  # noqa: E402

from bsmapper.generation import (  # noqa: E402
    GenerateOptions,
    ModelGenerator,
    ModelSettings,
    Pipeline,
    RuleMasker,
    make_pipeline,
    postprocess,
    resolve,
    song_features,
    tempo_beatmap,
)
from bsmapper.generation import runtime as runtime_mod  # noqa: E402
from bsmapper.generation.density import NPS_CONDITIONING_GAIN, target_nps  # noqa: E402
from bsmapper.generation.lighting import (  # noqa: E402
    ModelLightingGenerator,
    SparseLightshowError,
    check_density,
)
from bsmapper.generation.model import request_mask  # noqa: E402
from bsmapper.interfaces import GenerationRequest  # noqa: E402
from bsmapper.models import MapperModel, ModelConfig, StepState, model_state  # noqa: E402
from bsmapper.rules import RuleChecker  # noqa: E402
from bsmapper.schema import (  # noqa: E402
    BasicEvent,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Lightshow,
)
from bsmapper.tokenizer import NOTE_GRAMMAR, NOTE_VOCAB  # noqa: E402

V = NOTE_VOCAB
FAST = ModelSettings(max_body_tokens=48, style_guidance=1.5)


@pytest.fixture(scope="module")
def tiny_model() -> MapperModel:
    torch.manual_seed(0)
    cfg = ModelConfig(
        d_model=32,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        encoder_layers=1,
        decoder_layers=1,
        light_layers=1,
        note_memory_layers=1,
        max_len=192,
        light_max_len=192,
        note_memory_len=128,
    )
    return MapperModel(cfg).eval()


@pytest.fixture(scope="module")
def checkpoint(tiny_model, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save(model_state(tiny_model), path)
    return path


@pytest.fixture(scope="module")
def analysis():
    return make_analysis(duration=20.0)  # 40 beats: 3 windows


def req(diff=Difficulty.EXPERT, mode=Characteristic.STANDARD, **kw) -> GenerationRequest:
    return GenerationRequest(mode, diff, **kw)


def test_features_match_the_training_layout(analysis) -> None:
    feats = song_features(analysis, tempo_beatmap(analysis))
    assert feats.shape == (3 * 384, 83) and feats.dtype == np.float32
    assert feats[-1, :80].max() == -80.0 and feats[-1, 80:].max() == 0.0  # past the end
    assert feats[: 4 * 24, 80].max() == pytest.approx(1.0)  # an onset row on beat 0


def test_generate_many_decodes_and_post_processes(tiny_model, analysis) -> None:
    gen = ModelGenerator(tiny_model, FAST)
    requests = [req(Difficulty.EXPERT, style=0.8, seed=3), req(Difficulty.HARD, seed=3)]
    dmaps = gen.generate_many(analysis, requests)
    assert [(d.characteristic, d.difficulty) for d in dmaps] == [
        (Characteristic.STANDARD, Difficulty.EXPERT),
        (Characteristic.STANDARD, Difficulty.HARD),
    ]
    beatmap = tempo_beatmap(analysis)
    for dmap in dmaps:
        assert dmap.njs > 0
        fixed, report = postprocess(dmap, beatmap)
        assert report.errors == []
    # One RNG per batch: the same batch and seed give the same maps.
    again = ModelGenerator(tiny_model, FAST).generate_many(analysis, requests)
    assert [d.notes for d in again] == [d.notes for d in dmaps]
    assert ModelGenerator(tiny_model, FAST).generate(analysis, requests[1]).difficulty == "Hard"


def test_heuristic_nps_is_scaled_and_explicit_kept() -> None:
    filled = ModelGenerator._with_nps(req(Difficulty.EXPERT_PLUS), 120.0)
    expected = target_nps(Difficulty.EXPERT_PLUS, Characteristic.STANDARD, 120.0)
    assert filled.target_nps == pytest.approx(expected * NPS_CONDITIONING_GAIN, abs=0.01)
    assert ModelGenerator._with_nps(req(target_nps=2.0), 120.0).target_nps == 2.0


def test_request_masks() -> None:
    m = request_mask(req(mode=Characteristic.ONE_SABER, allow_bombs=False))
    assert not m[V.id("NOTE", Color.RED)] and m[V.id("NOTE", Color.BLUE)] and not m[V.id("BOMB")]
    assert not m[V.ids("ROT")].any()
    assert request_mask(req(mode=Characteristic.DEGREE_360))[V.ids("ROT")].all()
    dots = request_mask(req(mode=Characteristic.NO_ARROWS))
    assert not dots[V.id("ARC")] and not dots[V.id("CHAIN")]


def test_rule_masker_blocks_a_parity_break(analysis) -> None:
    beatmap = tempo_beatmap(analysis)
    r = req(Difficulty.EXPERT)
    masker = RuleMasker(
        [r], [DifficultyMap(r.characteristic, r.difficulty)], beatmap, RuleChecker()
    )
    parser = NOTE_GRAMMAR.parser()
    tokens = [V.id("TIME", 96), V.id("NOTE", Color.RED), V.id("POS", (1, 0))]
    tokens += [V.id("DIR", CutDirection.DOWN), V.id("TIME", 108), V.id("NOTE", Color.RED)]
    tokens += [V.id("POS", (1, 0))]
    for t in tokens:
        parser.feed(t)
    mask = masker.mask(StepState(0, 0, tokens, parser))
    assert not mask[V.id("DIR", CutDirection.DOWN)]  # down again, half a beat later
    assert mask[V.id("DIR", CutDirection.UP)]
    assert len(masker.states[0].notes) == 1  # the first note reached the rules state


def test_model_lighting_renders_through_ws8(tiny_model, analysis, monkeypatch) -> None:
    beatmap = tempo_beatmap(analysis)
    beatmap.environment = "WeaveEnvironment"
    dense = Lightshow([BasicEvent(b / 2, b % 5, 1 + b % 7) for b in range(80)])
    gen = ModelLightingGenerator(tiny_model)
    monkeypatch.setattr(gen, "basic_lightshow", lambda a, bm: dense)
    lights = gen.generate(analysis, beatmap, "v3")
    assert lights.event_box_groups  # v3 group lighting from WS8's renderer
    with pytest.raises(SparseLightshowError):
        check_density(Lightshow([BasicEvent(1.0, 0, 1)]), beatmap, analysis.duration_s)


def test_pipeline_model_lighting_falls_back_to_rules(tiny_model, tmp_path) -> None:
    pipeline = Pipeline(
        ModelGenerator(tiny_model, FAST), light_model=tiny_model, analyzer=FakeAnalyzer(20.0)
    )
    song = pipeline.analyze(tmp_path / "A - B.mp3")
    opts = GenerateOptions(difficulties=[Difficulty.HARD, Difficulty.EXPERT])
    beatmap, results = pipeline.build(song, opts)
    assert len(results) == 2 and all(r.seconds > 0 for r in results)
    assert pipeline.last_lighting in ("model", "rules")
    if pipeline.last_lighting == "rules":  # a random model's lights are too sparse
        assert "rule-based" in pipeline.last_notes[0]
    assert beatmap.lightshow.event_box_groups
    _, rules_only = pipeline.light(song, beatmap, GenerateOptions(lighting_source="rules"))
    assert rules_only is None


def test_resolve_and_make_pipeline(checkpoint, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BSMAPPER_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setattr(runtime_mod, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    assert resolve().generator == "stub" and "no model checkpoint" in resolve().reason
    with pytest.raises(FileNotFoundError):
        resolve("model")
    setup = resolve("auto", "auto", checkpoint)
    assert (setup.generator, setup.lighting, setup.checkpoint) == ("model", "model", checkpoint)
    assert resolve("model", "rules", checkpoint).lighting == "rules"
    assert resolve("stub", "auto", checkpoint).lighting == "rules"

    cfg = tmp_path / "config.toml"
    cfg.write_text(f"[model]\ncheckpoint = '{checkpoint.as_posix()}'\n", encoding="utf-8")
    monkeypatch.setenv("BSMAPPER_CONFIG", str(cfg))
    assert resolve().checkpoint == checkpoint

    a = make_pipeline(device="cpu")
    b = make_pipeline(device="cpu")
    assert isinstance(a.generator, ModelGenerator) and a.light_model is not None
    assert a.generator.model is b.generator.model  # loaded once per process
    assert a.mapper == "bsmapper"
