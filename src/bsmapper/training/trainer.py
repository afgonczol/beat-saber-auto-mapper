"""The training loop: AdamW, warmup + cosine schedule, label smoothing, bf16 autocast,
TensorBoard logging and resumable checkpoints.

Each optimizer step takes a note batch and, when lighting is trained, a light
batch; the loss is ``note_loss + light_loss_weight * light_loss``. The audio
encoder is shared by both decoders.

    out_dir/last.pt            latest state (resumed automatically)
    out_dir/best.pt            lowest validation note loss (light loss when notes are
                               not trained)
    out_dir/good.pt            newest snapshot taken while the divergence guard saw a
                               healthy run: the rollback target
    out_dir/step_NNNNNNN.pt    periodic snapshots (the newest ``keep_checkpoints``),
                               every ``early_save_every`` steps up to ``early_save_until``
    out_dir/guard.log          divergence guard events (rollbacks, giving up)
    out_dir/config.yaml        the resolved config
    out_dir/tb/                TensorBoard events
"""

from __future__ import annotations

import contextlib
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from bsmapper.models.checkpoint import CHECKPOINT_FORMAT, load_weights, model_state
from bsmapper.models.mapper import MapperModel
from bsmapper.tokenizer import light_version_for_vocab, version_for_vocab
from bsmapper.training.config import TrainConfig
from bsmapper.training.data import (
    IGNORE,
    LightWindows,
    NoteWindows,
    collate,
    infinite_loader,
    load_light_styles,
)
from bsmapper.training.guard import DivergenceGuard
from bsmapper.training.store import DatasetStore

log = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def lr_lambda(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def token_loss(logits: torch.Tensor, targets: torch.Tensor, smoothing: float):
    """Mean cross-entropy over non-ignored targets, and (correct, count) for accuracy."""
    flat = logits.float().reshape(-1, logits.shape[-1])
    loss = F.cross_entropy(
        flat, targets.reshape(-1), ignore_index=IGNORE, label_smoothing=smoothing
    )
    mask = targets != IGNORE
    correct = ((logits.argmax(-1) == targets) & mask).sum()
    return loss, correct, mask.sum()


def z_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """mean(logsumexp(logits)^2) over the non-ignored targets (PaLM's z-loss)."""
    mask = targets != IGNORE
    z = torch.logsumexp(logits.float(), dim=-1)
    return (z[mask] ** 2).mean() if mask.any() else z.sum() * 0.0


class TrainingDiverged(RuntimeError):
    """The divergence guard rolled back ``max_rollbacks`` times and gave up."""


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim < 2 or name.endswith(".pos") or "memory_type" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class _NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def add_text(self, *args, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _writer(path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover - tensorboard is in the ml extra
        log.warning("tensorboard is not installed; not logging to %s", path)
        return _NullWriter()
    return SummaryWriter(str(path))


class Trainer:
    def __init__(self, cfg: TrainConfig, resume: bool = True) -> None:
        self.cfg = cfg
        run, data = cfg.run, cfg.data
        self.out_dir = Path(run.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _seed_everything(run.seed)
        self.device = resolve_device(run.device)
        self.amp = run.precision == "bf16" and self.device.type == "cuda"
        if self.device.type == "cuda" and run.cuda_memory_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(
                run.cuda_memory_fraction, self.device.index or 0
            )

        root = Path(data.root)
        if data.root == "synthetic":
            from bsmapper.training.synthetic import write_synthetic_dataset

            root = self.out_dir / "synthetic"
            if not (root / "dataset.json").exists():
                write_synthetic_dataset(root, data.synthetic_maps, data.synthetic_windows, run.seed)
        self.train_store = DatasetStore(root, [data.train_split])
        val_store = DatasetStore(root, [data.val_split])
        self.val_store = val_store if len(val_store) else None

        self.model = MapperModel(cfg.model)
        self.train_lights = run.train_lights and cfg.model.has_lighting
        self.train_notes = run.train_notes
        if not (self.train_notes or self.train_lights):
            raise ValueError("nothing to train: train_notes and train_lights are both off")
        self.version = version_for_vocab(cfg.model.note_vocab)  # prompt layout of this model
        self.light_version = (
            light_version_for_vocab(cfg.model.light_vocab) if cfg.model.has_lighting else 1
        )
        styles = load_light_styles(self.train_store, data) if self.light_version >= 2 else None
        self.light_styles = styles  # the style file; its "model" goes into checkpoints
        self.notes = NoteWindows(self.train_store, data, version=self.version)
        self.lights = (
            LightWindows(
                self.train_store,
                data,
                version=self.version,
                light_version=self.light_version,
                styles=styles,
            )
            if self.train_lights
            else None
        )
        if self.lights is not None and not len(self.lights):
            log.warning("no light windows to train on (gold-only and none gold?); lights off")
            self.lights, self.train_lights = None, False

        init = run.init_from and not (resume and (self.out_dir / "last.pt").exists())
        if init:
            report = load_weights(self.model, run.init_from)
            log.info("initialised from %s (grown: %s)", run.init_from, report["grown"] or "none")
        self.frozen = set(run.freeze)
        unknown = self.frozen - {name for name, _ in self.model.named_children()}
        if unknown:
            raise ValueError(f"run.freeze names unknown modules: {sorted(unknown)}")
        for name, module in self.model.named_children():
            if name in self.frozen:
                module.requires_grad_(False)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            _param_groups(self.model, cfg.optim.weight_decay),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            fused=self.device.type == "cuda",
        )
        o = cfg.optim
        self.lr_scale = 1.0  # halved by each rollback of the divergence guard
        self.rollbacks = 0
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: self.lr_scale * lr_lambda(s, o.warmup_steps, o.max_steps, o.min_lr_ratio),
        )
        self.guard = DivergenceGuard(run.guard)
        self.step = 0
        self.best_val = float("inf")
        if resume and (self.out_dir / "last.pt").exists():
            self.load(self.out_dir / "last.pt")
        elif not init:  # a fine-tune keeps the normalisation its encoder was trained with
            mean, std = self.train_store.feature_stats(seed=run.seed)
            self.model.set_feature_stats(mean, std)
        self.forward_model = torch.compile(self.model) if run.compile else self.model
        self._write_config()

    # --- losses -------------------------------------------------------------------

    def _autocast(self):
        if self.amp:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _to(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def _train_mode(self) -> None:
        """Train mode, except for frozen modules (no dropout where nothing learns)."""
        self.model.train()
        for name, module in self.model.named_children():
            if name in self.frozen:
                module.eval()

    def _audio(self, features: torch.Tensor) -> torch.Tensor:
        if "encoder" in self.frozen:
            with torch.no_grad():
                return self.forward_model.encode_audio(features)
        return self.forward_model.encode_audio(features)

    @property
    def best_metric(self) -> str:
        return "note_loss" if self.train_notes else "light_loss"

    def note_loss(self, batch: dict[str, torch.Tensor], with_z: bool = False):
        m = self.forward_model
        with self._autocast():
            memory = self._audio(batch["features"])
            logits = m.note_logits(memory, batch["inputs"])
        out = token_loss(logits, batch["targets"], self.cfg.optim.label_smoothing)
        return (*out, z_loss(logits, batch["targets"])) if with_z else out

    def light_loss(self, batch: dict[str, torch.Tensor], with_z: bool = False):
        m = self.forward_model
        with self._autocast():
            memory = self._audio(batch["features"])
            logits = m.light_logits(memory, batch["note_tokens"], batch["inputs"])
        out = token_loss(logits, batch["targets"], self.cfg.optim.label_smoothing)
        return (*out, z_loss(logits, batch["targets"])) if with_z else out

    def _loaders(self, seed: int):
        o, workers = self.cfg.optim, self.cfg.data.num_workers
        note_iter = (
            infinite_loader(self.notes, o.batch_size, workers, seed) if self.train_notes else None
        )
        light_iter = (
            infinite_loader(self.lights, o.light_batch_size, workers, seed + 1)
            if self.lights is not None
            else None
        )  # fmt: skip
        return note_iter, light_iter

    def _save_every(self) -> int:
        run = self.cfg.run
        early = min(run.early_save_every, run.save_every)
        return early if self.step <= run.early_save_until else run.save_every

    def _guard_event(self, msg: str, writer) -> None:
        log.error("DIVERGENCE GUARD: %s", msg)
        writer.add_text("guard", msg, self.step)
        with open(self.out_dir / "guard.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} step {self.step}: {msg}\n")

    def rollback(self, reason: str, writer) -> None:
        """Back to ``good.pt`` with the learning rate cut, or raise ``TrainingDiverged``."""
        g = self.cfg.run.guard
        at = f"{reason} at step {self.step} ({self.guard.describe()})"
        good = self.out_dir / "good.pt"
        if self.rollbacks >= g.max_rollbacks or not good.exists():
            why = "no good checkpoint yet" if not good.exists() else f"{self.rollbacks} rollbacks"
            self._guard_event(f"GAVE UP: {at}; {why}", writer)
            raise TrainingDiverged(f"training diverged ({at}); {why}")
        rollbacks, scale = self.rollbacks + 1, self.lr_scale * g.lr_factor
        self.load(good)
        self.rollbacks, self.lr_scale = rollbacks, scale
        for group, base in zip(self.optimizer.param_groups, self.scheduler.base_lrs, strict=True):
            group["lr"] = base * self.scheduler.lr_lambdas[0](self.scheduler.last_epoch)
        self.guard.after_rollback()
        self._guard_event(
            f"ROLLBACK {rollbacks}/{g.max_rollbacks}: {at}; back to step {self.step}, "
            f"lr x{scale:g}, new data order",
            writer,
        )

    # --- loop ---------------------------------------------------------------------

    def train(self, max_steps: int | None = None) -> dict[str, float]:
        o, run = self.cfg.optim, self.cfg.run
        total = max_steps if max_steps is not None else o.max_steps
        # A resumed run draws new batches, and so does each rollback.
        note_iter, light_iter = self._loaders(run.seed + self.step + 7919 * self.rollbacks)
        writer = _writer(self.out_dir / "tb")
        if self.step == 0:
            writer.add_text("config", f"```\n{self._config_yaml()}\n```")
            writer.add_scalar("model/params_M", self.model.num_parameters() / 1e6, 0)
        self._train_mode()
        last: dict[str, float] = {}
        meter = _Meter()
        t0 = time.perf_counter()
        while self.step < total:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(o.grad_accum):
                loss = torch.zeros((), device=self.device)
                if note_iter is not None:
                    nb = self._to(next(note_iter))
                    nl, nc, nn_, nz = self.note_loss(nb, with_z=True)
                    loss = loss + nl + o.z_loss * nz
                    meter.add("note", nl.item(), nc.item(), nn_.item())
                    meter.add_value("note_z", nz.item())
                    step_loss += nl.item() / o.grad_accum
                if light_iter is not None:
                    lb = self._to(next(light_iter))
                    ll, lc, ln, lz = self.light_loss(lb, with_z=True)
                    loss = loss + o.light_loss_weight * (ll + o.z_loss * lz)
                    meter.add("light", ll.item(), lc.item(), ln.item())
                    if note_iter is None:
                        step_loss += ll.item() / o.grad_accum
                (loss / o.grad_accum).backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), o.grad_clip))
            if math.isfinite(grad_norm):
                self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            meter.grad_norm += grad_norm if math.isfinite(grad_norm) else 0.0
            meter.grad_norm_max = max(meter.grad_norm_max, grad_norm)
            meter.steps += 1
            reason = self.guard.update(step_loss, grad_norm) if run.guard.enabled else None
            if reason is not None:
                self.rollback(reason, writer)
                note_iter, light_iter = self._loaders(run.seed + self.step + 7919 * self.rollbacks)
                meter, t0 = _Meter(), time.perf_counter()
                continue

            if self.step % run.log_every == 0 or self.step == total:
                elapsed = time.perf_counter() - t0
                last = meter.summary(elapsed)
                last["lr"] = self.scheduler.get_last_lr()[0]
                last["grad_norm_median"] = self.guard.median
                if self.device.type == "cuda":
                    last["gpu_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 1024**3
                for k, v in last.items():
                    writer.add_scalar(f"train/{k}", v, self.step)
                log.info("step %d %s", self.step, _fmt(last))
                meter, t0 = _Meter(), time.perf_counter()
            if self.val_store is not None and (
                self.step % run.eval_every == 0 or self.step == total
            ):
                val = self.evaluate()
                for k, v in val.items():
                    writer.add_scalar(f"val/{k}", v, self.step)
                log.info("step %d val %s", self.step, _fmt(val))
                last.update({f"val_{k}": v for k, v in val.items()})
                if val.get(self.best_metric, float("inf")) < self.best_val:
                    self.best_val = val[self.best_metric]
                    self.save(self.out_dir / "best.pt")
                self._train_mode()
            every = self._save_every()
            if self.step % every == 0 or self.step == total:
                self.save(self.out_dir / "last.pt")
                if self.step % every == 0:
                    self.save(self.out_dir / f"step_{self.step:07d}.pt")
                    self._prune()
                if self.guard.healthy:
                    self.save(self.out_dir / "good.pt")
        writer.flush()
        writer.close()
        return last

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Validation loss / token accuracy on a fixed, unweighted set of batches."""
        assert self.val_store is not None
        self.model.eval()
        cfg = self.cfg
        out: dict[str, float] = {}
        tasks = []
        if self.train_notes:
            notes = NoteWindows(self.val_store, cfg.data, train=False, version=self.version)
            tasks.append(("note", notes, self.note_loss))
        if self.train_lights:
            lights = LightWindows(
                self.val_store,
                cfg.data,
                train=False,
                version=self.version,
                light_version=self.light_version,
                styles=self.light_styles,
            )
            if len(lights):
                tasks.append(("light", lights, self.light_loss))
        for name, ds, fn in tasks:
            gen = np.random.default_rng(1234)
            bs = cfg.optim.batch_size if name == "note" else cfg.optim.light_batch_size
            n = min(len(ds), bs * cfg.run.eval_batches)
            picks = gen.choice(len(ds), n, replace=False)
            loss_sum = correct = count = 0.0
            for s in range(0, n, bs):
                batch = self._to(collate([ds[int(j)] for j in picks[s : s + bs]]))
                loss, c, k = fn(batch)
                loss_sum += loss.item() * k.item()
                correct += c.item()
                count += k.item()
            if count:
                out[f"{name}_loss"] = loss_sum / count
                out[f"{name}_acc"] = correct / count
        return out

    # --- checkpoints ----------------------------------------------------------------

    def save(self, path: Path) -> None:
        state = model_state(self.model)
        if self.light_styles is not None:
            state["light_styles"] = self.light_styles["model"]
        state.update(
            step=self.step,
            best_val=self.best_val,
            rollbacks=self.rollbacks,
            lr_scale=self.lr_scale,
            train_config=self.cfg.to_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            rng={
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        )
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: unsupported checkpoint format")
        if ckpt["model_config"] != self.cfg.model.to_dict():
            log.warning("resuming %s with a different model config than it was saved with", path)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt["step"]
        self.best_val = ckpt.get("best_val", float("inf"))
        self.rollbacks = ckpt.get("rollbacks", 0)
        self.lr_scale = ckpt.get("lr_scale", 1.0)
        rng = ckpt.get("rng") or {}
        if "python" in rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        log.info("resumed from %s at step %d", path, self.step)

    def _prune(self) -> None:
        snaps = sorted(self.out_dir.glob("step_*.pt"))
        for p in snaps[: max(0, len(snaps) - self.cfg.run.keep_checkpoints)]:
            p.unlink()

    def _config_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.cfg.to_dict(), sort_keys=False)

    def _write_config(self) -> None:
        (self.out_dir / "config.yaml").write_text(self._config_yaml())


class _Meter:
    def __init__(self) -> None:
        self.sums: dict[str, list[float]] = {}
        self.values: dict[str, list[float]] = {}
        self.grad_norm = 0.0
        self.grad_norm_max = 0.0
        self.steps = 0

    def add_value(self, name: str, value: float) -> None:
        self.values.setdefault(name, []).append(value)

    def add(self, name: str, loss: float, correct: float, count: float) -> None:
        s = self.sums.setdefault(name, [0.0, 0.0, 0.0, 0])
        s[0] += loss
        s[1] += correct
        s[2] += count
        s[3] += 1

    def summary(self, elapsed: float) -> dict[str, float]:
        out: dict[str, float] = {}
        tokens = 0.0
        for name, (loss, correct, count, n) in self.sums.items():
            out[f"{name}_loss"] = loss / max(n, 1)
            out[f"{name}_acc"] = correct / max(count, 1)
            tokens += count
        for name, vals in self.values.items():
            out[name] = sum(vals) / len(vals)
        out["grad_norm"] = self.grad_norm / max(self.steps, 1)
        out["grad_norm_max"] = self.grad_norm_max
        out["target_tokens_per_s"] = tokens / max(elapsed, 1e-9)
        out["steps_per_s"] = self.steps / max(elapsed, 1e-9)
        return out


def _fmt(d: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4g}" for k, v in d.items())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
