"""Training configuration, read from ``configs/*.yaml``.

The YAML has four sections that map onto the dataclasses below::

    model: {...}     # bsmapper.models.ModelConfig
    data: {...}      # DataConfig
    optim: {...}     # OptimConfig
    run: {...}       # RunConfig

Unknown keys are errors, so a typo does not silently fall back to a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from bsmapper.models.config import ModelConfig


@dataclass(slots=True)
class DataConfig:
    root: str = "data/processed"  # a built dataset, or "synthetic" for a generated toy one
    train_split: str = "train"
    val_split: str = "val"
    context_events: int = 16  # previous-window tail objects in the decoder prefix
    max_note_tokens: int = 1024  # BOS prefix context SEP body EOS, cropped beyond this
    max_light_tokens: int = 1024
    max_light_body: int = 1024  # skip light windows with longer bodies (lightID spam)
    note_memory_tokens: int = 512  # note track tokens the lighting decoder sees
    # Conditioning dropout: STYLE -> STYLE_NONE for classifier-free guidance on style.
    style_dropout: float = 0.2
    nps_dropout: float = 0.1
    context_dropout: float = 0.05  # train some windows without the previous tail
    # note_v1 (tokenizer v2): INTENSITY -> INTENSITY_NONE, and the reference window.
    intensity_dropout: float = 0.15
    ref_dropout: float = 0.3  # train windows that have a reference without it too
    ref_threshold: float = 0.85  # keep a reference when its audio similarity is at least this
    max_ref_tokens: int = 256  # cap of the reference block's events
    # Sampling weights. A difficulty window's weight is the product of these.
    characteristic_weights: dict[str, float] = field(
        default_factory=lambda: {
            "Standard": 1.0,
            "OneSaber": 2.0,
            "90Degree": 4.0,
            "360Degree": 3.0,
        }
    )
    difficulty_weights: dict[str, float] = field(
        default_factory=lambda: {
            "Easy": 1.0,
            "Normal": 1.0,
            "Hard": 1.0,
            "Expert": 1.0,
            "ExpertPlus": 1.0,
        }
    )
    oversample_easy: bool = False  # Easy is low priority: rare-mode weights skip it
    gold_weight: float = 1.5
    empty_window_weight: float = 0.25  # windows with no objects (intros, breaks)
    lights_gold_only: bool = True
    hardest_notes_for_lights: float = 0.5  # else a random difficulty of the map
    # Synthetic data (root: synthetic).
    synthetic_maps: int = 16
    synthetic_windows: int = 4
    num_workers: int = 0


@dataclass(slots=True)
class OptimConfig:
    lr: float = 3e-4
    min_lr_ratio: float = 0.1  # cosine decays to lr * min_lr_ratio
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    max_steps: int = 100_000
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    batch_size: int = 32
    grad_accum: int = 1
    light_batch_size: int = 16
    light_loss_weight: float = 0.5
    # z-loss: weight * mean(logsumexp(logits)^2) on both decoders' targets. It keeps the
    # output logits from drifting (PaLM); 0 = off (note_v0).
    z_loss: float = 0.0


@dataclass(slots=True)
class GuardConfig:
    """Divergence guard (``training.guard``): roll back to the last good checkpoint when
    the pre-clip gradient norm or the train loss blows up."""

    enabled: bool = True
    grad_ratio: float = 10.0  # a step is a spike when grad_norm > ratio x running median
    grad_steps: int = 100  # this many spike steps within ``grad_window`` -> roll back
    grad_window: int = 150
    loss_jump: float = 1.0  # fast loss EMA above the slow EMA by this much -> roll back
    nonfinite_steps: int = 20  # non-finite loss or gradient steps in a row -> roll back
    min_history: int = 300  # steps of history before the guard may fire
    history: int = 1000  # steps in the running grad-norm median
    max_rollbacks: int = 3  # then give up (the run stops with an error)
    lr_factor: float = 0.5  # learning rate multiplier per rollback


@dataclass(slots=True)
class RunConfig:
    out_dir: str = "checkpoints/run"
    device: str = "auto"  # auto | cpu | cuda
    precision: str = "bf16"  # bf16 | fp32 (bf16 only on CUDA)
    # Cap PyTorch's share of VRAM: past it you get an OOM error instead of Windows
    # silently spilling into shared system memory (which is ~50x slower).
    cuda_memory_fraction: float = 0.92
    seed: int = 0
    train_notes: bool = True
    train_lights: bool = True
    log_every: int = 50
    eval_every: int = 2000
    eval_batches: int = 50
    save_every: int = 2000
    keep_checkpoints: int = 3
    compile: bool = False
    # Denser snapshots early, when a run is most likely to blow up.
    early_save_every: int = 1000
    early_save_until: int = 10000
    guard: GuardConfig = field(default_factory=lambda: GuardConfig())


@dataclass(slots=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: asdict(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainConfig:
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(
            model=ModelConfig.from_dict(d.get("model") or {}),
            data=_build(DataConfig, d.get("data") or {}),
            optim=_build(OptimConfig, d.get("optim") or {}),
            run=_build(RunConfig, d.get("run") or {}),
        )

    @classmethod
    def load(cls, path: Path | str, overrides: list[str] | None = None) -> TrainConfig:
        import yaml

        d = yaml.safe_load(Path(path).read_text()) or {}
        for item in overrides or []:
            apply_override(d, item)
        return cls.from_dict(d)


def apply_override(d: dict[str, Any], item: str) -> None:
    """``section.key=value`` (value parsed as YAML), e.g. ``optim.max_steps=500``."""
    import yaml

    path, sep, raw = item.partition("=")
    if not sep or "." not in path:
        raise ValueError(f"override must look like section.key=value, got {item!r}")
    *parents, leaf = path.split(".")
    node = d
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def _build(cls, d: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    if cls is RunConfig and isinstance(d.get("guard"), dict):
        d = {**d, "guard": _build(GuardConfig, d["guard"])}
    return cls(**{k: tuple(v) if k == "betas" else v for k, v in d.items()})
