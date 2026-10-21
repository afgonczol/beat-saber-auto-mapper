"""Checkpoint files: ``torch.save`` of a dict with the model config and weights.

    {"format": 1, "model_config": {...}, "model": state_dict, "step": int,
     "train_config": {...}, "optimizer": ..., "scheduler": ..., "rng": ...}

Only ``model_config`` and ``model`` are needed for inference (``load_model``).
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
    return model.to(device).eval()
