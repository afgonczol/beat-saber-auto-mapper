"""The divergence guard: detection on synthetic sequences, and rollback in the trainer."""

import math

import numpy as np
import pytest
from training_helpers import tiny_model

from bsmapper.training import TrainConfig
from bsmapper.training.config import GuardConfig
from bsmapper.training.guard import DivergenceGuard
from bsmapper.training.trainer import Trainer, TrainingDiverged


def _feed(guard, losses, grads):
    events = []
    for i, (loss, grad) in enumerate(zip(losses, grads, strict=True)):
        reason = guard.update(loss, grad)
        if reason:
            events.append((i, reason))
    return events


def _steady(n, seed=0):
    rng = np.random.default_rng(seed)
    return list(2.0 + 0.05 * rng.normal(size=n)), list(1.0 + 0.2 * rng.random(size=n))


def test_quiet_run_and_isolated_spikes_do_not_fire():
    guard = DivergenceGuard(GuardConfig())
    losses, grads = _steady(2000)
    for i in range(400, 2000, 25):
        grads[i] = 500.0  # a lone bad batch now and then
    assert _feed(guard, losses, grads) == []
    assert guard.healthy and 1.0 < guard.median < 1.2


def test_sustained_gradient_spike_fires_after_grad_steps():
    guard = DivergenceGuard(GuardConfig())
    losses, grads = _steady(1000)
    # Like note_v1's first run: the norm climbs past 10x the median and stays there.
    for i in range(500, 1000):
        grads[i] = 1.0 * 1.05 ** (i - 500)
    events = _feed(guard, losses, grads)
    first = events[0]
    assert first[1] == "grad"
    crossed = next(i for i in range(500, 1000) if grads[i] > 10 * guard.median)
    assert abs(first[0] - (crossed + 99)) <= 3  # about the 100th spike step
    assert not guard.healthy


def test_loss_jump_fires():
    guard = DivergenceGuard(GuardConfig())
    losses, grads = _steady(800)
    losses[500:] = [3.5] * 300
    events = _feed(guard, losses, grads)
    assert events and events[0][1] == "loss" and 500 < events[0][0] < 560


def test_nonfinite_steps_fire_and_reset():
    guard = DivergenceGuard(GuardConfig(nonfinite_steps=5))
    losses, grads = _steady(400)
    assert _feed(guard, losses, grads) == []
    assert guard.update(float("nan"), 1.0) is None
    assert not guard.healthy
    assert guard.update(2.0, 1.0) is None  # one bad step is forgiven
    events = [guard.update(2.0, math.inf) for _ in range(5)]
    assert events == [None] * 4 + ["nonfinite"]


def test_guard_waits_for_history():
    guard = DivergenceGuard(GuardConfig(min_history=300))
    assert _feed(guard, [2.0] * 299, [1.0] * 298 + [1e9]) == []
    assert not guard.healthy and not guard.ready


def _cfg(root, out, max_steps, max_rollbacks=3) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "model": tiny_model(),
            "data": {"root": str(root), "lights_gold_only": False},
            "optim": {"warmup_steps": 2, "max_steps": max_steps, "batch_size": 4,
                      "light_batch_size": 2, "z_loss": 1e-4},
            "run": {"out_dir": str(out), "device": "cpu", "precision": "fp32",
                    "log_every": 2, "eval_every": 100, "save_every": 100,
                    "early_save_every": 2, "early_save_until": 100,
                    "guard": {"min_history": 4, "grad_steps": 3, "grad_window": 5,
                              "history": 50, "max_rollbacks": max_rollbacks}},
        }
    )  # fmt: skip


def _inject_spike(trainer, start, stop_after_rollbacks=1):
    """Scale the note loss by 1e6 from step ``start`` until enough rollbacks happened."""
    real = trainer.note_loss

    def spiky(batch, with_z=False):
        out = real(batch, with_z)
        if trainer.step >= start and trainer.rollbacks < stop_after_rollbacks:
            return (out[0] * 1e6, *out[1:])
        return out

    trainer.note_loss = spiky


def test_trainer_rolls_back_cuts_lr_and_finishes(synthetic_root, tmp_path):
    out = tmp_path / "run"
    trainer = Trainer(_cfg(synthetic_root, out, max_steps=14))
    _inject_spike(trainer, start=8)
    trainer.train()
    assert trainer.step == 14 and trainer.rollbacks == 1 and trainer.lr_scale == 0.5
    log = (out / "guard.log").read_text()
    # The 1e6x loss trips the loss EMA one step before the gradient count.
    assert "ROLLBACK 1/3: loss at step 9" in log and "back to step 8" in log
    assert (out / "good.pt").exists()
    # The counters survive a restart.
    resumed = Trainer(_cfg(synthetic_root, out, max_steps=14))
    assert resumed.rollbacks == 1 and resumed.lr_scale == 0.5
    assert resumed.scheduler.lr_lambdas[0](14) == pytest.approx(0.5 * 0.1)  # end of cosine


def test_trainer_gives_up_after_max_rollbacks(synthetic_root, tmp_path):
    out = tmp_path / "run"
    trainer = Trainer(_cfg(synthetic_root, out, max_steps=40, max_rollbacks=1))
    _inject_spike(trainer, start=8, stop_after_rollbacks=99)
    with pytest.raises(TrainingDiverged):
        trainer.train()
    log = (out / "guard.log").read_text()
    assert "ROLLBACK 1/1" in log and "GAVE UP" in log
