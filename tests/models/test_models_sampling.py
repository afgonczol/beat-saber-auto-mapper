import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bsmapper.models import (  # noqa: E402
    LightSampler,
    MapperModel,
    ModelConfig,
    NoteSampler,
    SamplingOptions,
    load_model,
    model_state,
    pad_features,
    sample_token,
)
from bsmapper.schema import Characteristic, Color, Difficulty  # noqa: E402
from bsmapper.tokenizer import (  # noqa: E402
    LIGHT_GRAMMAR,
    NOTE_GRAMMAR,
    NOTE_VOCAB,
    WINDOW_TICKS,
    Conditioning,
    MapTokenizer,
    tail_context,
)

COND = Conditioning(Characteristic.STANDARD, Difficulty.EXPERT, nps=5.0, style=0.7)


@pytest.fixture(scope="module")
def model() -> MapperModel:
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
def features() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(2 * WINDOW_TICKS + 100, 83)).astype(np.float32)


def test_sample_token_respects_mask_greedy_and_top_p():
    rng = np.random.default_rng(0)
    logits = np.array([5.0, 4.0, 0.0, -1.0])
    allowed = np.array([False, True, True, True])
    assert sample_token(logits, allowed, 0.0, 1.0, rng) == 1
    # top_p this small keeps only the most likely allowed token.
    assert {sample_token(logits, allowed, 1.0, 0.01, rng) for _ in range(50)} == {1}
    draws = {sample_token(logits, allowed, 5.0, 1.0, rng) for _ in range(300)}
    assert draws == {1, 2, 3}
    with pytest.raises(ValueError):
        sample_token(logits, np.zeros(4, dtype=bool), 1.0, 1.0, rng)


def test_pad_features_fills_whole_windows_with_silence():
    x = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
    out = pad_features(x)
    assert out.shape == (WINDOW_TICKS, 3)
    assert np.array_equal(out[10], x.min(axis=0))
    assert pad_features(x, 2).shape == (2 * WINDOW_TICKS, 3)


def test_generate_is_grammatical_seeded_and_batched(model, features):
    opts = SamplingOptions(temperature=1.0, top_p=0.9, seed=7, max_body_tokens=60)
    conds = [COND, Conditioning(Characteristic.ONE_SABER, Difficulty.HARD)]
    a = NoteSampler(model).generate(features, conds, opts)
    b = NoteSampler(model).generate(features, conds, opts)
    assert [[w.body for w in ws] for ws in a] == [[w.body for w in ws] for ws in b]
    assert len(a) == 2 and all(len(ws) == 3 for ws in a)  # 868 frames -> 3 windows
    for ws, cond in zip(a, conds, strict=True):
        for k, w in enumerate(ws):
            assert w.index == k
            assert w.prefix == cond.prefix_ids()
            NOTE_GRAMMAR.parse(w.body)  # raises if not grammatical
            assert len(w.body) <= 60 + 32
        assert ws[1].context == tail_context(ws[0].body, NOTE_GRAMMAR)
        dmap = MapTokenizer().decode(ws)
        assert dmap.characteristic == cond.characteristic
    other = NoteSampler(model).generate(features, conds, SamplingOptions(seed=8))
    assert [w.body for w in other[0]] != [w.body for w in a[0]]


def test_mask_fn_is_applied_and_sees_the_parser(model, features):
    seen = []
    blue = NOTE_VOCAB.id("NOTE", Color.BLUE)

    def no_blue(state):
        seen.append((state.window, state.sequence, len(state.body)))
        m = np.ones(len(NOTE_VOCAB), dtype=bool)
        m[blue] = False
        return m

    (ws,) = NoteSampler(model).generate(
        features, COND, SamplingOptions(seed=1, max_body_tokens=80), mask_fn=no_blue
    )
    assert all(blue not in w.body for w in ws)
    assert {w for w, _, _ in seen} == {0, 1, 2}

    def impossible(state):  # all-false masks fall back to the grammar mask
        return np.zeros(len(NOTE_VOCAB), dtype=bool)

    (ws,) = NoteSampler(model).generate(
        features, COND, SamplingOptions(seed=1, max_body_tokens=20), mask_fn=impossible
    )
    for w in ws:
        NOTE_GRAMMAR.parse(w.body)


def test_style_guidance_runs_and_changes_logits(model, features):
    plain = NoteSampler(model).generate(
        features, COND, SamplingOptions(temperature=0, max_body_tokens=40)
    )
    guided = NoteSampler(model).generate(
        features, COND, SamplingOptions(temperature=0, max_body_tokens=40, style_guidance=3.0)
    )
    for w in guided[0]:
        NOTE_GRAMMAR.parse(w.body)
    assert len(guided[0]) == len(plain[0])


def test_light_sampler_follows_note_windows(model, features, tmp_path):
    (notes,) = NoteSampler(model).generate(features, COND, SamplingOptions(max_body_tokens=40))
    lights = LightSampler(model).generate(
        features, notes, SamplingOptions(seed=2, max_body_tokens=40)
    )
    assert len(lights) == len(notes)
    for w in lights:
        LIGHT_GRAMMAR.parse(w.body)
        assert w.prefix == []


def test_checkpoint_roundtrip(model, tmp_path):
    path = tmp_path / "m.pt"
    torch.save(model_state(model), path)
    loaded = load_model(path)
    assert loaded.cfg == model.cfg
    for (k, a), b in zip(model.state_dict().items(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b), k
