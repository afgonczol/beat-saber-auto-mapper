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
from bsmapper.generation.density import NPS_GAIN, Loudness  # noqa: E402
from bsmapper.generation.lighting import (  # noqa: E402
    ModelLightingGenerator,
    SparseLightshowError,
    check_density,
)
from bsmapper.generation.model import DensityFeedback, request_mask  # noqa: E402
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


def test_conditioning_gains_are_sane() -> None:
    # The wanted NPS (and its intensity percentile) is tested in test_generation_intensity.
    assert all(1.0 <= g <= 1.5 for g in NPS_GAIN.values())


def test_boosted_candidates_decode_in_the_same_batch(tiny_model, analysis) -> None:
    gen = ModelGenerator(tiny_model, FAST)
    (dmap,) = gen.generate_many(analysis, [req(Difficulty.EXPERT, seed=1)])
    boosted, windows = gen.last_boosted[0]
    assert windows == 3 and 0 <= boosted <= windows
    off = ModelGenerator(tiny_model, ModelSettings(max_body_tokens=48, density_boosts=()))
    off.generate_many(analysis, [req(Difficulty.EXPERT, seed=1)])
    assert off.last_boosted == [(0, 3)]


def note_body(ticks: list[int]) -> list[int]:
    body, color = [], Color.RED
    for t in ticks:
        body += [V.id("TIME", t), V.id("NOTE", color), V.id("POS", (1, 0)),
                 V.id("DIR", CutDirection.ANY)]  # fmt: skip
        color = Color.BLUE if color == Color.RED else Color.RED
    return body


def test_density_feedback_keeps_the_fuller_window_and_syncs_rows(analysis) -> None:
    beatmap = tempo_beatmap(analysis)
    r = req(Difficulty.EXPERT, target_nps=4.0)
    metas = [DifficultyMap(r.characteristic, r.difficulty) for _ in range(2)]
    masker = RuleMasker([r, r], metas, beatmap, RuleChecker())
    loud = Loudness(np.full(48 * 24, 0.6))
    fb = DensityFeedback(masker, [[0, 1]], [4.0], [2.0], loud, beatmap)
    sparse = note_body([0, 24 * 12])  # two notes, a 12-beat hole
    full = note_body(list(range(0, 384, 12)))  # every half beat
    bodies = [sparse, full]
    fb.on_window(0, bodies)
    assert bodies[0] == full and bodies[1] == full  # the base row continues from the choice
    assert fb.boosted == [1]
    assert len(masker.states[0].notes) == len(masker.states[1].notes) == 32
    # a window that is fine as it is stays with the base candidate
    bodies = [note_body(list(range(0, 384, 12))), note_body(list(range(0, 384, 6)))]
    fb.on_window(1, bodies)
    assert fb.boosted == [1]
    quiet = Loudness(np.zeros(48 * 24))
    calm = DensityFeedback(masker, [[0, 1]], [4.0], [2.0], quiet, beatmap)
    assert calm.score(0, 2, []) < calm.score(0, 2, [32.0 + i / 2 for i in range(32)])


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


def test_high_intensity_conditions_on_the_next_difficulty(tiny_model, analysis) -> None:
    gen = ModelGenerator(tiny_model, FAST)
    (dmap,) = gen.generate_many(analysis, [req(Difficulty.EXPERT, intensity=0.9, seed=2)])
    assert dmap.difficulty == Difficulty.EXPERT  # the label stays
    (p,) = gen.last_plans
    assert p.condition_as == Difficulty.EXPERT_PLUS and p.wanted_nps > 0
    fixed, report = postprocess(dmap, tempo_beatmap(analysis))
    assert report.errors == []


def repeating_analysis(duration: float = 48.0):
    """A synthetic song whose second half repeats the first (chroma, mel, onsets)."""
    a = make_analysis(duration=duration)
    n = len(a.features["rms"])
    rng = np.random.default_rng(0)
    half = n // 2
    pattern = rng.random((half, 12)).astype(np.float32)
    chroma = np.concatenate([pattern, pattern, pattern[: n - 2 * half]])
    a.features["chroma"] = chroma
    mel = a.features["mel"].copy()
    mel[half : 2 * half] = mel[:half]
    a.features["mel"] = mel
    return a


def test_reference_windows_from_self_similarity(tiny_model) -> None:
    from bsmapper.generation.model import reference_starts

    a = repeating_analysis()
    beatmap = tempo_beatmap(a)
    refs = reference_starts(a, beatmap, song_features(a, beatmap))
    assert refs is not None and len(refs) == 6
    assert (refs >= 0).any()  # the repeated half finds the first one
    assert all(r < k * 384 for k, r in enumerate(refs) if r >= 0)  # only earlier spans
    assert reference_starts(make_analysis(), beatmap, song_features(a, beatmap)) is None


def test_v1_decodes_with_references_and_repetition(tiny_model) -> None:
    gen = ModelGenerator(tiny_model, FAST)
    assert gen.version == 2 and gen.needs_chroma and gen.shifts_intensity
    a = repeating_analysis()
    (dmap,) = gen.generate_many(a, [req(Difficulty.EXPERT, intensity=0.8, repetition=1.0)])
    assert dmap.difficulty == Difficulty.EXPERT
    assert gen.last_refs is not None
    fixed, report = postprocess(dmap, tempo_beatmap(a))
    assert report.errors == []
    off = ModelGenerator(tiny_model, ModelSettings(max_body_tokens=48, references=False))
    assert not off.needs_chroma
    off.generate_many(a, [req(Difficulty.EXPERT)])
    assert off.last_refs is None


def test_lighting_prefix_follows_the_checkpoint_version(analysis) -> None:
    from bsmapper.tokenizer.vocab import NOTE_VOCAB_SIZES

    torch.manual_seed(0)
    cfg = ModelConfig(
        d_model=32, n_heads=4, ff_mult=2, dropout=0.0, encoder_layers=1, decoder_layers=1,
        light_layers=1, note_memory_layers=1, max_len=192, light_max_len=192,
        note_memory_len=128, note_vocab=NOTE_VOCAB_SIZES[1],
    )  # fmt: skip
    v0_sized = MapperModel(cfg).eval()
    assert ModelGenerator(v0_sized).version == 1
    beatmap = tempo_beatmap(analysis)
    beatmap.difficulties.append(DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT))
    gen = ModelLightingGenerator(v0_sized, max_body_tokens=24)
    lights = gen.basic_lightshow(analysis, beatmap)  # would index past v0's vocabulary before
    assert isinstance(lights, Lightshow)


def test_note_v1_is_the_default_checkpoint(tmp_path, monkeypatch) -> None:
    from bsmapper.generation.runtime import DEFAULT_CHECKPOINTS, find_checkpoint

    monkeypatch.setattr(runtime_mod, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.chdir(tmp_path)
    for d in DEFAULT_CHECKPOINTS:
        (tmp_path / d).parent.mkdir(parents=True)
        (tmp_path / d).write_bytes(b"x")
    assert find_checkpoint() == tmp_path / DEFAULT_CHECKPOINTS[0]
    assert "note_v1" in str(DEFAULT_CHECKPOINTS[0])
    (tmp_path / DEFAULT_CHECKPOINTS[0]).unlink()
    assert find_checkpoint() == tmp_path / DEFAULT_CHECKPOINTS[1]
