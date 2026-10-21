import pytest

torch = pytest.importorskip("torch")

from bsmapper.models import MapperModel, ModelConfig  # noqa: E402
from bsmapper.tokenizer import PAD  # noqa: E402


def tiny(**kw) -> ModelConfig:
    base = dict(
        d_model=32,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        encoder_layers=1,
        decoder_layers=2,
        light_layers=1,
        note_memory_layers=1,
        max_len=128,
        light_max_len=128,
        note_memory_len=64,
    )
    return ModelConfig(**{**base, **kw})


def test_default_size_is_in_the_planned_range():
    with torch.device("meta"):
        full = MapperModel(ModelConfig())
        notes_only = MapperModel(ModelConfig(light_layers=0))
    assert 60e6 <= full.num_parameters() <= 100e6
    assert 40e6 <= notes_only.num_parameters() < full.num_parameters()
    assert not hasattr(notes_only, "lights")


def test_shapes_and_padding_do_not_leak():
    torch.manual_seed(0)
    m = MapperModel(tiny()).eval()
    feats = torch.randn(2, 384, 83)
    tokens = torch.randint(4, 600, (2, 12))
    padded = torch.cat([tokens, torch.zeros(2, 5, dtype=torch.long)], dim=1)
    mem = m.encode_audio(feats)
    a = m.note_logits(mem, tokens)
    b = m.note_logits(mem, padded)
    assert a.shape == (2, 12, m.cfg.note_vocab)
    assert torch.allclose(a, b[:, :12], atol=1e-5)
    assert not torch.isnan(b).any()
    lights = m.light_logits(mem, padded, torch.randint(4, 400, (2, 7)))
    assert lights.shape == (2, 7, m.cfg.light_vocab)


def test_wrong_frame_count_is_an_error():
    m = MapperModel(tiny())
    with pytest.raises(ValueError, match="frames"):
        m.encode_audio(torch.randn(1, 100, 83))


def test_kv_cache_matches_teacher_forcing_with_left_padding():
    torch.manual_seed(0)
    m = MapperModel(tiny()).eval()
    mem = m.encode_audio(torch.randn(2, 384, 83))
    seqs = [torch.randint(4, 600, (9,)), torch.randint(4, 600, (6,))]
    full = [m.note_logits(mem[b : b + 1], s[None])[0] for b, s in enumerate(seqs)]
    # Prompts of 5 and 2 tokens, left-padded to 5, then the rest one token at a time.
    prompt = torch.full((2, 5), PAD)
    valid = torch.zeros(2, 5, dtype=torch.bool)
    prompt[0], valid[0] = seqs[0][:5], True
    prompt[1, 3:], valid[1, 3:] = seqs[1][:2], True
    cache = m.notes.start(mem)
    out = m.notes.step(prompt, cache, valid)
    assert torch.allclose(out[0, -1], full[0][4], atol=1e-4)
    assert torch.allclose(out[1, -1], full[1][1], atol=1e-4)
    for i in range(4):
        step = torch.stack([seqs[0][5 + i], seqs[1][2 + i]])[:, None]
        out = m.notes.step(step, cache)
        assert torch.allclose(out[0, -1], full[0][5 + i], atol=1e-4)
        assert torch.allclose(out[1, -1], full[1][2 + i], atol=1e-4)


def test_gradient_checkpointing_gives_the_same_gradients():
    torch.manual_seed(0)
    plain = MapperModel(tiny())
    ckpt = MapperModel(tiny(grad_checkpointing=True))
    ckpt.load_state_dict(plain.state_dict())
    feats = torch.randn(2, 384, 83)
    tokens = torch.randint(4, 600, (2, 10))
    grads = []
    for m in (plain, ckpt):
        m.train()
        loss = m.note_logits(m.encode_audio(feats), tokens).logsumexp(-1).mean()
        loss.backward()
        grads.append(m.encoder.stem[0].weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_feature_stats_normalize_the_input():
    m = MapperModel(tiny()).eval()
    m.set_feature_stats(torch.full((83,), -40.0), torch.full((83,), 20.0))
    assert torch.equal(m.encoder.feat_mean, torch.full((83,), -40.0))
    # Shifting the input by the mean shift and scaling by std is a no-op after normalizing.
    x = torch.randn(1, 384, 83)
    a = m.encode_audio(x * 20 - 40)
    m.set_feature_stats(torch.zeros(83), torch.ones(83))
    assert torch.allclose(a, m.encode_audio(x), atol=1e-4)
