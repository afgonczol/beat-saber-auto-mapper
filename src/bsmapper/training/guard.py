"""Divergence detection for the training loop.

The first note_v1 run blew up right after warmup. The pre-clip gradient norm went from
about 1 to 1e10 within 900 steps, and the loss went back to chance and never recovered.
Clipping cannot undo that, so the trainer watches for it and rolls back instead (see
``Trainer``). Per optimizer step, ``DivergenceGuard.update`` gets the train loss and the
pre-clip gradient norm, and returns why the run should roll back (or None):

* ``grad``: in the last ``grad_window`` steps, at least ``grad_steps`` had a gradient
  norm above ``grad_ratio`` x the running median of the previous ``history`` steps;
* ``loss``: a fast EMA of the loss (about 20 steps) is ``loss_jump`` above a slow one
  (about 500 steps);
* ``nonfinite``: ``nonfinite_steps`` steps in a row had a NaN/inf loss or gradient.

Spike steps do not enter the median or the slow EMA, so a blow-up cannot raise the bar
it is measured against. ``healthy`` says whether a checkpoint taken now is a safe
rollback target: there is enough history and no spike or loss rise is building.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from bsmapper.training.config import GuardConfig


class DivergenceGuard:
    def __init__(self, cfg: GuardConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        """Forget everything (a new run)."""
        self.grads: deque[float] = deque(maxlen=self.cfg.history)
        self.spikes: deque[bool] = deque(maxlen=self.cfg.grad_window)
        self.fast: float | None = None
        self.slow: float | None = None
        self.nonfinite = 0
        self.seen = 0

    def after_rollback(self) -> None:
        """Clear the spike state but keep the baselines from before the blow-up (spike
        steps never entered them), so a run that diverges again right away is caught."""
        self.spikes.clear()
        self.nonfinite = 0
        self.fast = self.slow

    @property
    def median(self) -> float:
        return float(np.median(self.grads)) if self.grads else float("nan")

    @property
    def ready(self) -> bool:
        return self.seen >= self.cfg.min_history

    @property
    def healthy(self) -> bool:
        if not self.ready or self.nonfinite:
            return False
        if sum(self.spikes) > max(1, self.cfg.grad_steps // 10):
            return False
        return self.fast is None or self.slow is None or self.fast <= self.slow + 0.3

    def update(self, loss: float, grad_norm: float) -> str | None:
        cfg = self.cfg
        if not (math.isfinite(loss) and math.isfinite(grad_norm)):
            self.nonfinite += 1
            return "nonfinite" if self.nonfinite >= cfg.nonfinite_steps else None
        self.nonfinite = 0
        self.seen += 1
        spike = self.ready and grad_norm > cfg.grad_ratio * self.median
        self.spikes.append(spike)
        if not spike:
            self.grads.append(grad_norm)
        self.fast = loss if self.fast is None else 0.95 * self.fast + 0.05 * loss
        if self.slow is None:
            self.slow = loss
        elif not (spike or (self.ready and self.fast > self.slow + 0.3)):
            self.slow = 0.998 * self.slow + 0.002 * loss
        if not self.ready:
            return None
        if sum(self.spikes) >= cfg.grad_steps:
            return "grad"
        if self.fast > self.slow + cfg.loss_jump:
            return "loss"
        return None

    def describe(self) -> str:
        return (
            f"grad median {self.median:.3g}, spikes {sum(self.spikes)}/{len(self.spikes)}, "
            f"loss ema fast {self.fast:.3f} slow {self.slow:.3f}, nonfinite {self.nonfinite}"
        )
