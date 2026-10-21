"""The training loop: AdamW, warmup + cosine schedule, label smoothing, bf16 autocast,
TensorBoard logging and resumable checkpoints.

Each optimizer step takes a note batch and, when lighting is trained, a light
batch; the loss is ``note_loss + light_loss_weight * light_loss``. The audio
encoder is shared by both decoders.

    out_dir/last.pt            latest state (resumed automatically)
    out_dir/best.pt            lowest validation note loss
    out_dir/step_NNNNNNN.pt    periodic snapshots (the newest ``keep_checkpoints``)
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

from bsmapper.models.checkpoint import CHECKPOINT_FORMAT, model_state
from bsmapper.models.mapper import MapperModel
from bsmapper.training.config import TrainConfig
from bsmapper.training.data import IGNORE, LightWindows, NoteWindows, collate, infinite_loader
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
        self.notes = NoteWindows(self.train_store, data)
        self.lights = LightWindows(self.train_store, data) if self.train_lights else None
        if self.lights is not None and not len(self.lights):
            log.warning("no light windows to train on (gold-only and none gold?); lights off")
            self.lights, self.train_lights = None, False

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            _param_groups(self.model, cfg.optim.weight_decay),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            fused=self.device.type == "cuda",
        )
        o = cfg.optim
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: lr_lambda(s, o.warmup_steps, o.max_steps, o.min_lr_ratio),
        )
        self.step = 0
        self.best_val = float("inf")
        if resume and (self.out_dir / "last.pt").exists():
            self.load(self.out_dir / "last.pt")
        else:
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

    def note_loss(self, batch: dict[str, torch.Tensor]):
        m = self.forward_model
        with self._autocast():
            memory = m.encode_audio(batch["features"])
            logits = m.note_logits(memory, batch["inputs"])
        return token_loss(logits, batch["targets"], self.cfg.optim.label_smoothing)

    def light_loss(self, batch: dict[str, torch.Tensor]):
        m = self.forward_model
        with self._autocast():
            memory = m.encode_audio(batch["features"])
            logits = m.light_logits(memory, batch["note_tokens"], batch["inputs"])
        return token_loss(logits, batch["targets"], self.cfg.optim.label_smoothing)

    # --- loop ---------------------------------------------------------------------

    def train(self, max_steps: int | None = None) -> dict[str, float]:
        cfg, o, run = self.cfg, self.cfg.optim, self.cfg.run
        total = max_steps if max_steps is not None else o.max_steps
        seed = run.seed + self.step  # a resumed run draws new batches
        workers = cfg.data.num_workers
        note_iter = (
            infinite_loader(self.notes, o.batch_size, workers, seed) if self.train_notes else None
        )
        light_iter = (
            infinite_loader(self.lights, o.light_batch_size, workers, seed + 1)
            if self.lights is not None
            else None
        )  # fmt: skip
        writer = _writer(self.out_dir / "tb")
        if self.step == 0:
            writer.add_text("config", f"```\n{self._config_yaml()}\n```")
            writer.add_scalar("model/params_M", self.model.num_parameters() / 1e6, 0)
        self.model.train()
        last: dict[str, float] = {}
        meter = _Meter()
        t0 = time.perf_counter()
        while self.step < total:
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(o.grad_accum):
                loss = torch.zeros((), device=self.device)
                if note_iter is not None:
                    nb = self._to(next(note_iter))
                    nl, nc, nn_ = self.note_loss(nb)
                    loss = loss + nl
                    meter.add("note", nl.item(), nc.item(), nn_.item())
                if light_iter is not None:
                    lb = self._to(next(light_iter))
                    ll, lc, ln = self.light_loss(lb)
                    loss = loss + o.light_loss_weight * ll
                    meter.add("light", ll.item(), lc.item(), ln.item())
                (loss / o.grad_accum).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), o.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            meter.grad_norm += float(grad_norm)
            meter.steps += 1

            if self.step % run.log_every == 0 or self.step == total:
                elapsed = time.perf_counter() - t0
                last = meter.summary(elapsed)
                last["lr"] = self.scheduler.get_last_lr()[0]
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
                if val.get("note_loss", float("inf")) < self.best_val:
                    self.best_val = val["note_loss"]
                    self.save(self.out_dir / "best.pt")
                self.model.train()
            if self.step % run.save_every == 0 or self.step == total:
                self.save(self.out_dir / "last.pt")
                if self.step % run.save_every == 0:
                    self.save(self.out_dir / f"step_{self.step:07d}.pt")
                    self._prune()
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
            tasks.append(
                ("note", NoteWindows(self.val_store, cfg.data, train=False), self.note_loss)
            )
        if self.train_lights:
            lights = LightWindows(self.val_store, cfg.data, train=False)
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
        state.update(
            step=self.step,
            best_val=self.best_val,
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
        self.grad_norm = 0.0
        self.steps = 0

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
        out["grad_norm"] = self.grad_norm / max(self.steps, 1)
        out["target_tokens_per_s"] = tokens / max(elapsed, 1e-9)
        out["steps_per_s"] = self.steps / max(elapsed, 1e-9)
        return out


def _fmt(d: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4g}" for k, v in d.items())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
