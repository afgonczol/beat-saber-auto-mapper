import numpy as np
import pytest
from training_helpers import REPO

from bsmapper.tokenizer import BOS, EOS, LIGHT_GRAMMAR, NOTE_GRAMMAR, NOTE_VOCAB, PREFIX_LEN, SEP
from bsmapper.training import DataConfig, DatasetStore, TrainConfig
from bsmapper.training.data import (
    IGNORE,
    LightWindows,
    NoteWindows,
    collate,
    note_sequence,
    window_weights,
)


def test_store_reads_the_layout(synthetic_root):
    store = DatasetStore(synthetic_root, ["train"])
    val = DatasetStore(synthetic_root, ["val"])
    assert len(store) == 8 and len(val) == 4  # 4 + 2 maps x 2 difficulties
    assert store.features(0, 2).shape == (384, 83)
    assert store.notes[0].n_windows == 3
    for k in range(3):
        NOTE_GRAMMAR.parse(store.notes[0].body(k))
        LIGHT_GRAMMAR.parse(store.lights[store.rows[0]["key"]].body(k))
    assert store.notes[0].body(-1) == [] and store.notes[0].body(3) == []
    mean, std = store.feature_stats()
    assert mean.shape == (83,) and (std > 0).all()
    cond = store.conditioning(0, drop_style=True)
    assert cond.style is None and cond.nps is not None


def test_note_sequence_layout_and_crop():
    prefix, context, body = [10] * PREFIX_LEN, [20, 21], [30, 31, 32]
    ids, start = note_sequence(prefix, context, body, 100)
    assert ids == [BOS, *prefix, *context, SEP, *body, EOS]
    assert ids[start] == SEP and ids[start + 1] == 30
    ids, start = note_sequence(prefix, context, body, 14)
    assert len(ids) == 14 and ids[start] == SEP
    # When even the prompt does not fit, the context goes first.
    ids, start = note_sequence(prefix, [20] * 40, body, 14)
    assert ids[: PREFIX_LEN + 2] == [BOS, *prefix, SEP]


def test_examples_and_collate(synthetic_root):
    store = DatasetStore(synthetic_root, ["train"])
    cfg = DataConfig(style_dropout=1.0, nps_dropout=0.0, context_dropout=0.0)
    ds = NoteWindows(store, cfg)
    item = ds[1]  # window 1: has a context
    assert item["tokens"][0] == BOS
    assert item["tokens"][1 + 3] == NOTE_VOCAB.id("STYLE", None)  # dropped for CFG
    assert item["tokens"][item["loss_start"]] == SEP
    items = [ds[0], ds[1], ds[2]]  # drawn once: training examples are random
    batch = collate(items)
    assert batch["features"].shape == (3, 384, 83)
    assert batch["inputs"].shape == batch["targets"].shape
    t = batch["targets"]
    for b in range(3):
        start = items[b]["loss_start"]
        assert (t[b, :start] == IGNORE).all()
        assert t[b, start] != IGNORE
    lights = LightWindows(store, DataConfig(lights_gold_only=True))
    li = lights[0]
    assert li["note_tokens"][0] == BOS and li["tokens"][li["loss_start"]] == SEP
    assert "note_tokens" in collate([lights[0], lights[1]])


def test_sampling_weights_oversample_rare_modes_but_not_easy():
    from bsmapper.training.store import TokenTrack

    class FakeStore:
        rows = [
            {"characteristic": "Standard", "difficulty": "Expert", "gold": False},
            {"characteristic": "360Degree", "difficulty": "Expert", "gold": False},
            {"characteristic": "360Degree", "difficulty": "Easy", "gold": False},
            {"characteristic": "Standard", "difficulty": "Expert", "gold": True},
        ]
        notes = [TokenTrack(np.array([5, 6], dtype=np.int16), np.array([0, 2, 2]))] * 4

    cfg = DataConfig(gold_weight=2.0, empty_window_weight=0.5)
    index, w = window_weights(FakeStore(), cfg)
    assert index[:2] == [(0, 0), (0, 1)]
    w = w.reshape(4, 2)
    assert w[0].tolist() == [1.0, 0.5]  # second window is empty
    assert w[1, 0] == cfg.characteristic_weights["360Degree"]
    assert w[2, 0] == 1.0  # Easy keeps weight 1 in a rare mode
    assert w[3, 0] == 2.0


@pytest.mark.parametrize("name", ["debug.yaml", "note_v0.yaml"])
def test_shipped_configs_load(name):
    cfg = TrainConfig.load(REPO / "configs" / name, overrides=["optim.max_steps=5"])
    assert cfg.optim.max_steps == 5
    assert isinstance(cfg.optim.betas, tuple)


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("optim:\n  learning_rate: 1\n")
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig.load(p)
    with pytest.raises(ValueError):
        TrainConfig.load(REPO / "configs" / "debug.yaml", overrides=["nonsense"])


def test_reference_and_intensity_examples(synthetic_root):
    from bsmapper.tokenizer import REF, REF_END, span_tokens

    store = DatasetStore(synthetic_root, ["train"])
    assert store.version == 2
    assert store.ref_start(0, 0, 0.8) is None and store.ref_start(0, 2, 0.8) == 0
    assert store.ref_start(0, 2, 1.5) is None  # below the threshold
    keep = DataConfig(ref_dropout=0.0, intensity_dropout=0.0, context_dropout=0.0)
    ds = NoteWindows(store, keep)
    ids = ds[2]["tokens"]  # window 2 repeats window 0
    assert NOTE_VOCAB.group(ids[1 + 8]) == "INTENSITY" and ids[1 + 8] != NOTE_VOCAB.id(
        "INTENSITY", None
    )
    ref = ids[ids.index(REF) + 1 : ids.index(REF_END)]
    assert ref == span_tokens(store.notes[0], 0, NOTE_GRAMMAR) == store.notes[0].body(0)
    assert ids[ds[2]["loss_start"]] == SEP
    assert REF not in ds[1]["tokens"]  # window 1 has no reference
    drop = NoteWindows(store, DataConfig(ref_dropout=1.0, intensity_dropout=1.0))
    ids = drop[2]["tokens"]
    assert REF not in ids and ids[1 + 8] == NOTE_VOCAB.id("INTENSITY", None)
    # A tokenizer v1 model (note_v0) gets the v1 layout: 8 prefix tokens, no references.
    v1 = NoteWindows(store, keep, version=1)
    ids = v1[2]["tokens"]
    assert ids[1 + 8] != REF and REF not in ids and max(ids) < 655


def test_note_sequence_drops_reference_before_context():
    ids, start = note_sequence([10] * 9, [20] * 4, [30] * 3, 20, ref=[40] * 10)
    assert 40 not in ids and ids.count(20) == 4 and ids[start] == SEP
