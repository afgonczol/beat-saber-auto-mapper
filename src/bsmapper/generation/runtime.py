"""Choosing and loading the generator: model checkpoint lookup, device, process cache.

The checkpoint is, in order: ``--model PATH``, ``[model] checkpoint`` in
``~/.bsmapper/config.toml``, then ``checkpoints/note_v0/best.pt`` under the current
folder or the repository. ``generator="auto"`` uses the model when a checkpoint is found
and torch is installed, else the stub. Models load once per process and device (CUDA when
available, else CPU), so the server and repeated CLI calls in one process share them.
"""

from __future__ import annotations

import logging
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bsmapper.generation.install import config_path

log = logging.getLogger(__name__)

GeneratorChoice = Literal["auto", "model", "stub"]
LightingSource = Literal["auto", "model", "rules"]
# Newest first: note_v1 (intensity token, structure memory), then note_v0.
DEFAULT_CHECKPOINTS = (
    Path("checkpoints") / "note_v1" / "best.pt",
    Path("checkpoints") / "note_v0" / "best.pt",
)
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINTS[0]
REPO_ROOT = Path(__file__).resolve().parents[3]

_models: dict[tuple[str, str], object] = {}
_lock = threading.Lock()


def configured_checkpoint(path: Path | None = None) -> Path | None:
    """``[model] checkpoint`` from the user config, if set."""
    cfg = path or config_path()
    if not cfg.is_file():
        return None
    with cfg.open("rb") as f:
        value = tomllib.load(f).get("model", {}).get("checkpoint")
    return Path(value).expanduser() if value else None


def find_checkpoint(explicit: Path | str | None = None) -> Path | None:
    """The checkpoint to use, or None when there is none (an explicit path must exist)."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"model checkpoint not found: {path}")
        return path
    candidates = [configured_checkpoint()]
    for default in DEFAULT_CHECKPOINTS:
        candidates += [Path.cwd() / default, REPO_ROOT / default]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_cached(path: Path, device: str | None = None):
    """``bsmapper.models.load_model``, once per (file, device) in this process."""
    from bsmapper.models import load_model

    device = pick_device(device)
    key = (str(Path(path).resolve()), device)
    with _lock:
        model = _models.get(key)
        if model is None:
            try:
                model = load_model(path, device=device)
            except RuntimeError as exc:  # CUDA out of memory or broken driver
                if device == "cpu":
                    raise
                log.warning("loading %s on %s failed (%s); using the CPU", path, device, exc)
                return load_cached(path, "cpu")
            _models[key] = model
            log.info("loaded %s on %s", path, device)
    return model


@dataclass(slots=True)
class GeneratorSetup:
    generator: str  # "model" or "stub"
    lighting: str  # "model" or "rules"
    checkpoint: Path | None
    reason: str = ""  # why the stub was chosen, for the user


def resolve(
    generator: GeneratorChoice = "auto",
    lighting: LightingSource = "auto",
    checkpoint: Path | str | None = None,
) -> GeneratorSetup:
    """Decide which note generator and lighting source to use."""
    wants_model = generator != "stub" or lighting == "model"
    path = find_checkpoint(checkpoint) if wants_model else None
    reason = ""
    if path is None:
        reason = "no model checkpoint found"
    elif not torch_available():
        reason = "torch is not installed (pip install -e .[ml])"
    for choice, option in ((generator, "--generator"), (lighting, "--lighting-source")):
        if choice == "model" and reason:
            raise FileNotFoundError(
                f"{option} model: {reason}; pass --model or set [model] checkpoint in "
                f"{config_path()}"
            )
    gen = "model" if generator == "model" or (generator == "auto" and not reason) else "stub"
    light = "model" if lighting == "model" or (lighting == "auto" and gen == "model") else "rules"
    return GeneratorSetup(gen, light, path if not reason else None, reason)


def make_pipeline(
    generator: GeneratorChoice = "auto",
    lighting: LightingSource = "auto",
    checkpoint: Path | str | None = None,
    device: str | None = None,
    **pipeline_kwargs,
):
    """A ``Pipeline`` with the chosen generator and lighting source (model loaded once)."""
    from bsmapper.generation.pipeline import Pipeline

    setup = resolve(generator, lighting, checkpoint)
    model = None
    if setup.generator == "model" or setup.lighting == "model":
        model = load_cached(setup.checkpoint, device)  # type: ignore[arg-type]
    note_gen = None
    if setup.generator == "model":
        from bsmapper.generation.model import ModelGenerator

        note_gen = ModelGenerator(model)
    light_model = model if setup.lighting == "model" else None
    pipeline = Pipeline(note_gen, light_model=light_model, **pipeline_kwargs)
    pipeline.setup = setup
    return pipeline
