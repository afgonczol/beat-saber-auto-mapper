"""WS6: data loading, training loop and checkpoints (needs the ``ml`` extra).

    cfg = TrainConfig.load("configs/note_v0.yaml", overrides=["optim.max_steps=1000"])
    Trainer(cfg).train()

or ``bsmapper train configs/note_v0.yaml``. See docs/MODEL.md.
"""

from bsmapper.training.config import DataConfig, OptimConfig, RunConfig, TrainConfig
from bsmapper.training.store import DatasetStore, TokenTrack

__all__ = [
    "DataConfig",
    "DatasetStore",
    "OptimConfig",
    "RunConfig",
    "TokenTrack",
    "TrainConfig",
]
