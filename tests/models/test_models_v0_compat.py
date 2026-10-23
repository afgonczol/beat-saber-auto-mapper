"""note_v0 (tokenizer v1) must keep loading and decoding exactly as before tokenizer v2."""

import json

import pytest

torch = pytest.importorskip("torch")

from v0_golden import GOLDEN, checkpoint, click_track, generate  # noqa: E402

from bsmapper.models import MapperModel, ModelConfig, NoteSampler, SamplingOptions  # noqa: E402
from bsmapper.schema import Characteristic, Difficulty  # noqa: E402
from bsmapper.tokenizer import (  # noqa: E402
    NOTE_VOCAB_SIZES,
    REF,
    Conditioning,
    MapTokenizer,
)


def _tiny(vocab: int) -> MapperModel:
    torch.manual_seed(0)
    cfg = ModelConfig(
        d_model=32, n_heads=2, ff_mult=2, encoder_layers=1, decoder_layers=1,
        light_layers=0, note_vocab=vocab, max_len=512,
    )  # fmt: skip
    return MapperModel(cfg).eval()


def test_v1_model_gets_v1_prompts():
    model = _tiny(NOTE_VOCAB_SIZES[1])
    sampler = NoteSampler(model)
    assert sampler.version == 1 and not sampler.supports_references
    cond = Conditioning(Characteristic.STANDARD, Difficulty.EXPERT, intensity=0.9)
    calls = []

    def ref_fn(k, b, bodies):
        calls.append(k)
        return []

    opts = SamplingOptions(seed=1, max_body_tokens=24, repetition_guidance=1.5)
    (windows,) = sampler.generate(click_track(2), [cond], opts, ref_fn=ref_fn)
    assert not calls  # a v1 model never sees references
    for w in windows:
        assert len(w.prefix) == 8 and w.ref is None
        assert REF not in w.input_ids()
        assert max(w.input_ids()) < NOTE_VOCAB_SIZES[1]
    MapTokenizer().decode(windows)


def test_v2_model_gets_intensity_and_refs():
    model = _tiny(NOTE_VOCAB_SIZES[2])
    sampler = NoteSampler(model)
    assert sampler.version == 2
    cond = Conditioning(Characteristic.STANDARD, Difficulty.EXPERT, intensity=0.9)
    seen = []

    def ref_fn(k, b, bodies):
        seen.append((k, len(bodies)))
        return [] if k else None

    opts = SamplingOptions(seed=1, max_body_tokens=24, repetition_guidance=1.5)
    (windows,) = sampler.generate(click_track(3), [cond], opts, ref_fn=ref_fn)
    assert seen == [(0, 0), (1, 1), (2, 2)]
    assert len(windows[0].prefix) == 9 and windows[0].ref is None
    assert windows[1].ref == [] and windows[1].input_ids()[10] == REF
    MapTokenizer().decode(windows)


@pytest.mark.slow
def test_note_v0_decodes_exactly_as_before():
    from bsmapper.models import load_model

    path = checkpoint()
    if path is None:
        pytest.skip("checkpoints/note_v0/best.pt is not available")
    golden = json.loads(GOLDEN.read_text())["windows"]
    torch.manual_seed(0)
    assert generate(load_model(path, "cpu")) == golden
