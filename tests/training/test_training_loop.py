import math

import numpy as np
import pytest
import torch
from training_helpers import tiny_model

from bsmapper.models import MapperModel, ModelConfig, NoteSampler, SamplingOptions, load_model
from bsmapper.tokenizer import NOTE_GRAMMAR, Conditioning, MapTokenizer, tail_context
from bsmapper.training import DataConfig, DatasetStore, TrainConfig
from bsmapper.training.data import NoteWindows, collate
from bsmapper.training.trainer import Trainer, lr_lambda, token_loss


def _cfg(root, out, **optim) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "model": tiny_model(),
            "data": {"root": str(root), "lights_gold_only": False},
            "optim": {"warmup_steps": 2, "max_steps": 6, "batch_size": 4,
                      "light_batch_size": 2, **optim},
            "run": {"out_dir": str(out), "device": "cpu", "precision": "fp32",
                    "log_every": 2, "eval_every": 3, "eval_batches": 1, "save_every": 3},
        }
    )  # fmt: skip


def test_lr_schedule_warms_up_then_decays():
    lrs = [lr_lambda(s, 10, 100, 0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[10] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:], strict=False))
    assert lr_lambda(100, 10, 100, 0.1) == pytest.approx(0.1)


def test_train_save_resume(synthetic_root, tmp_path):
    cfg = _cfg(synthetic_root, tmp_path / "run")
    trainer = Trainer(cfg)
    result = trainer.train()
    assert trainer.step == 6
    assert math.isfinite(result["note_loss"]) and math.isfinite(result["light_loss"])
    assert "val_note_loss" in result
    for name in ("last.pt", "best.pt", "step_0000003.pt", "step_0000006.pt", "config.yaml"):
        assert (tmp_path / "run" / name).exists(), name
    assert any((tmp_path / "run" / "tb").iterdir())
    # Feature normalization is fitted to the data and saved with the weights.
    loaded = load_model(tmp_path / "run" / "last.pt")
    assert not torch.equal(loaded.encoder.feat_mean, torch.zeros(83))

    cfg.optim.max_steps = 8
    resumed = Trainer(cfg)
    assert resumed.step == 6
    for a, b in zip(resumed.model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    resumed.train()
    assert resumed.step == 8
    fresh = Trainer(cfg, resume=False)
    assert fresh.step == 0


def test_notes_only_model_trains(synthetic_root, tmp_path):
    cfg = _cfg(synthetic_root, tmp_path / "run", max_steps=2)
    cfg.model.light_layers = 0
    trainer = Trainer(cfg)
    assert trainer.lights is None
    assert "light_loss" not in trainer.train()


def test_overfits_a_tiny_batch_on_cpu(synthetic_root):
    """The acceptance check for #8: the model memorizes a few windows exactly."""
    torch.manual_seed(0)
    store = DatasetStore(synthetic_root, ["train"])
    data = DataConfig(style_dropout=0.0, nps_dropout=0.0, context_dropout=0.0)
    ds = NoteWindows(store, data, train=False)
    picks = [0, 1, 2, 4]  # windows 0..2 of one difficulty, window 1 of the next
    batch = collate([ds[j] for j in picks])
    cfg = ModelConfig(**{**tiny_model(), "d_model": 64, "decoder_layers": 2, "light_layers": 0})
    model = MapperModel(cfg)
    mean, std = store.feature_stats()
    model.set_feature_stats(mean, std)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss, correct, count = token_loss(
            model.note_logits(model.encode_audio(batch["features"]), batch["inputs"]),
            batch["targets"],
            0.0,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert correct.item() == count.item()

    # Greedy decoding reproduces the memorized windows token for token.
    i = ds.index[0][0]
    feats = np.concatenate([store.features(i, k) for k in range(3)])
    (windows,) = NoteSampler(model.eval()).generate(
        feats, store.conditioning(i), SamplingOptions(temperature=0.0)
    )
    truth = [store.notes[i].body(k) for k in range(3)]
    assert [w.body for w in windows] == truth
    assert windows[1].context == tail_context(truth[0], NOTE_GRAMMAR)
    dmap = MapTokenizer().decode(windows)
    assert len(dmap.notes) == sum(len(NOTE_GRAMMAR.parse(b)) for b in truth)
    assert isinstance(store.conditioning(i), Conditioning)
