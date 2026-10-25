"""WS8b phase 2: v2 light examples, the lights-only fine-tune and v2 light sampling."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from training_helpers import tiny_model

from bsmapper.lighting.styles import STYLE_FEATURE_NAMES, StyleModel
from bsmapper.models import (
    LightSampler,
    MapperModel,
    ModelConfig,
    NoteSampler,
    SamplingOptions,
    WindowRefs,
    load_model,
    model_state,
)
from bsmapper.models.checkpoint import load_weights
from bsmapper.schema import Characteristic, Difficulty
from bsmapper.tokenizer import (
    LIGHT_GRAMMAR,
    LIGHT_REF,
    LIGHT_REF_END,
    LIGHT_VOCAB,
    LIGHT_VOCAB_SIZES,
    WINDOW_TICKS,
    Conditioning,
    LightPrompt,
    window_scene,
)
from bsmapper.training import DataConfig, DatasetStore, TrainConfig
from bsmapper.training.data import LightWindows
from bsmapper.training.trainer import Trainer

V = LIGHT_VOCAB


def _style_model() -> dict:
    n = len(STYLE_FEATURE_NAMES)
    return StyleModel(
        names=["pulse", "calm"], mean=[0.0] * n, scale=[1.0] * n, centers=[[0.0] * n, [1.0] * n]
    ).to_dict()


def _styles(store: DatasetStore, skip: int = 1) -> dict:
    keys = sorted({r["key"] for r in store.rows})
    labels = {k: i % 2 for i, k in enumerate(keys[skip:])}
    return {"model": _style_model(), "labels": labels, "excluded": {}}


def _no_dropout(**kw) -> DataConfig:
    return DataConfig(
        lights_gold_only=False,
        context_dropout=0.0,
        light_style_dropout=0.0,
        scene_dropout=0.0,
        light_ref_dropout=0.0,
        ref_threshold=0.5,
        **kw,
    )


def test_v2_light_examples_carry_style_scene_and_reference(synthetic_root) -> None:
    store = DatasetStore(synthetic_root)
    styles = _styles(store)
    ds = LightWindows(store, _no_dropout(), light_version=2, styles=styles)
    assert {key for key, _ in ds.index} == set(styles["labels"])  # labelled maps only
    seen_ref = False
    for j, (key, k) in enumerate(ds.index):
        item = ds[j]
        ids = item["tokens"]
        prompt = LightPrompt.from_prefix(ids[1:5])
        assert prompt.style == styles["labels"][key]
        assert prompt.scene == window_scene(store.lights[key].body(k))
        if LIGHT_REF in ids:
            seen_ref = True
            assert ids[5] == LIGHT_REF and LIGHT_REF_END in ids
            assert k >= 2  # the synthetic references start at window 2
    assert seen_ref
    # Version 1: no prompt and every map, as note_v1 trained.
    v1 = LightWindows(store, _no_dropout(), light_version=1, styles=styles)
    assert len(v1) > len(ds)
    assert V.group(v1[0]["tokens"][1]) in ("SEP", "TIME")
    # Dropout to the _NONE tokens.
    drop = _no_dropout()
    drop.light_style_dropout = drop.scene_dropout = 1.0
    none = LightWindows(store, drop, light_version=2, styles=styles)[0]["tokens"][1:5]
    assert none == LightPrompt().prefix_ids()


def _light_v1_checkpoint(path) -> None:
    torch.manual_seed(0)
    model = MapperModel(ModelConfig(**tiny_model(), light_vocab=LIGHT_VOCAB_SIZES[1]))
    model.set_feature_stats(np.full(83, 0.5), np.full(83, 2.0))
    torch.save(model_state(model), path)


def test_load_weights_grows_the_light_vocabulary(tmp_path) -> None:
    _light_v1_checkpoint(tmp_path / "v1.pt")
    old = torch.load(tmp_path / "v1.pt", weights_only=False)["model"]
    model = MapperModel(ModelConfig(**tiny_model()))
    report = load_weights(model, tmp_path / "v1.pt")
    assert "lights.embed.weight" in report["grown"] and not report["missing"]
    emb = model.lights.embed.weight.detach()
    assert emb.shape[0] == len(V)
    assert torch.equal(emb[:500], old["lights.embed.weight"])
    assert torch.allclose(emb[500], old["lights.embed.weight"].mean(0))
    assert model.lights.head.weight is model.lights.embed.weight  # still tied
    small = MapperModel(ModelConfig(**{**tiny_model(), "d_model": 16}))
    with pytest.raises(ValueError, match="cannot grow"):
        load_weights(small, tmp_path / "v1.pt")


def test_lights_only_fine_tune_freezes_notes(synthetic_root, tmp_path) -> None:
    _light_v1_checkpoint(tmp_path / "v1.pt")
    store = DatasetStore(synthetic_root)
    (tmp_path / "styles.json").write_text(json.dumps(_styles(store)))
    cfg = TrainConfig.from_dict(
        {
            "model": tiny_model(),
            "data": {"root": str(synthetic_root), "lights_gold_only": False,
                     "light_styles": str(tmp_path / "styles.json"), "ref_threshold": 0.5},
            "optim": {"warmup_steps": 1, "max_steps": 4, "batch_size": 4, "light_batch_size": 2,
                      "lr": 1e-2},
            "run": {"out_dir": str(tmp_path / "run"), "device": "cpu", "precision": "fp32",
                    "log_every": 2, "eval_every": 2, "eval_batches": 1, "save_every": 2,
                    "train_notes": False, "init_from": str(tmp_path / "v1.pt"),
                    "freeze": ["encoder", "notes"]},
        }
    )  # fmt: skip
    trainer = Trainer(cfg)
    assert trainer.light_version == 2 and trainer.best_metric == "light_loss"
    src = torch.load(tmp_path / "v1.pt", weights_only=False)["model"]
    result = trainer.train()
    assert "light_loss" in result and "note_loss" not in result
    state = trainer.model.state_dict()
    for name in ("encoder.stem.0.weight", "encoder.feat_mean", "notes.embed.weight"):
        assert torch.equal(state[name], src[name]), name  # frozen, normalisation kept
    assert not torch.equal(state["lights.blocks.0.ff.0.weight"], src["lights.blocks.0.ff.0.weight"])
    best = load_model(tmp_path / "run" / "last.pt")  # the synthetic data has no val split
    assert best.cfg.light_vocab == len(V)
    assert best.light_styles == _style_model()
    with pytest.raises(ValueError, match="unknown modules"):
        Trainer(TrainConfig.from_dict({**cfg.to_dict(), "run": {**cfg.to_dict()["run"],
                "out_dir": str(tmp_path / "x"), "freeze": ["decoder"]}}))  # fmt: skip


def test_v2_light_sampler_uses_prompts_references_and_guidance() -> None:
    torch.manual_seed(0)
    cfg = {**tiny_model(), "max_len": 192, "light_max_len": 256, "note_memory_len": 128}
    model = MapperModel(ModelConfig(**cfg)).eval()
    feats = np.random.default_rng(0).normal(size=(3 * WINDOW_TICKS, 83)).astype(np.float32)
    cond = Conditioning(Characteristic.STANDARD, Difficulty.EXPERT)
    (notes,) = NoteSampler(model).generate(feats, cond, SamplingOptions(max_body_tokens=24))
    sampler = LightSampler(model)
    assert sampler.supports_prompts
    prompts = [LightPrompt(1, 3, 2, 1)] * 3
    seen: list[int] = []

    def refs(k, b, bodies):
        seen.append(len(bodies))
        return WindowRefs([-1, -1, 0], 64, LIGHT_GRAMMAR)(k, b, bodies)

    opts = SamplingOptions(seed=1, max_body_tokens=24)
    lights = sampler.generate(feats, notes, opts, prompts=prompts, ref_fn=refs, style_guidance=2.0)
    assert [w.prefix for w in lights] == [p.prefix_ids() for p in prompts]
    assert seen == [0, 1, 2]  # references see the windows generated so far
    for w in lights:
        LIGHT_GRAMMAR.parse(w.body)
    # A v1 lighting decoder ignores prompts and references.
    v1 = MapperModel(ModelConfig(**cfg, light_vocab=LIGHT_VOCAB_SIZES[1])).eval()
    old = LightSampler(v1).generate(feats, notes, opts, prompts=prompts, ref_fn=refs)
    assert all(w.prefix == [] for w in old) and len(seen) == 3
