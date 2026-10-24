"""Checkpoint files: ``torch.save`` of a dict with the model config and weights.

    {"format": 1, "model_config": {...}, "model": state_dict, "step": int,
     "train_config": {...}, "optimizer": ..., "scheduler": ..., "rng": ...}

Only ``model_config`` and ``model`` are needed for inference (``load_model``). A
checkpoint whose lighting decoder uses the v2 light prompt (WS8b) also carries
``light_styles``, the ``lighting.styles.StyleModel`` it was trained with; ``load_model``
puts it on ``model.light_styles`` (a dict, or None).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from bsmapper.models.config import ModelConfig
from bsmapper.models.mapper import MapperModel

CHECKPOINT_FORMAT = 1


def model_state(model: MapperModel) -> dict[str, Any]:
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    return {
        "format": CHECKPOINT_FORMAT,
        "model_config": model.cfg.to_dict(),
        "model": model.state_dict(),
    }


def load_model(path: Path | str, device: torch.device | str = "cpu") -> MapperModel:
    """An eval-mode model from a checkpoint file (training state is ignored)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unsupported checkpoint format {ckpt.get('format')}")
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    cfg.grad_checkpointing = False
    model = MapperModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.light_styles = ckpt.get("light_styles")
    return model.to(device).eval()


def load_weights(model: MapperModel, path: Path | str) -> dict[str, list[str]]:
    """Initialise ``model`` from a checkpoint's weights, for fine-tuning.

    Tensors whose first dimension grew (embeddings and tied heads of a larger, appended
    vocabulary) keep the old rows; the new rows start at the mean of the old ones. Any
    other shape mismatch is an error. Returns the names of grown and missing tensors.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unsupported checkpoint format {ckpt.get('format')}")
    own = model.state_dict()
    state, grown = {}, []
    for name, tensor in ckpt["model"].items():
        if name not in own:
            raise ValueError(f"{path}: {name} is not in this model")
        target = own[name]
        if tensor.shape != target.shape:
            if tensor.shape[1:] != target.shape[1:] or tensor.shape[0] > target.shape[0]:
                raise ValueError(
                    f"{name}: cannot grow {tuple(tensor.shape)} to {tuple(target.shape)}"
                )
            new = tensor.new_empty(target.shape)
            new[: tensor.shape[0]] = tensor
            new[tensor.shape[0] :] = tensor.float().mean(dim=0).to(tensor.dtype)
            tensor = new
            grown.append(name)
        state[name] = tensor
    missing = [n for n in own if n not in state]
    model.load_state_dict(state, strict=not missing)
    return {"grown": grown, "missing": missing}
