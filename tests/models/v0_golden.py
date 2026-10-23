"""The note_v0 compatibility check: fixed inputs whose output was recorded before
tokenizer v2, so ``test_note_v0_decodes_exactly_as_before`` can compare token for token.

    python tests/models/v0_golden.py record   # (only with the pre-v2 code) rewrites the JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "note_v0" / "best.pt"
MAIN_CHECKPOINT = Path("G:/ClaudeProjects/beat-saber-auto-mapper/checkpoints/note_v0/best.pt")
GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "note_v0_golden.json"


def checkpoint() -> Path | None:
    for p in (CHECKPOINT, MAIN_CHECKPOINT):
        if p.exists():
            return p
    return None


def click_track(n_windows: int = 3) -> np.ndarray:
    """(n_windows * 384, 83) synthetic features: a kick on every beat, hats on off-beats."""
    rng = np.random.default_rng(7)
    ticks = n_windows * 384
    x = np.full((ticks, 83), -70.0, dtype=np.float32)
    x[:, 80:] = 0.0
    x[:, :80] += rng.normal(0, 2, (ticks, 80)).astype(np.float32)
    for t in range(0, ticks, 24):
        loud = 1.0 if (t // 24) % 4 == 0 else 0.7
        x[t : t + 6, :20] = -10.0
        x[t : t + 2, 80] = loud
        x[t : t + 2, 81] = loud * 0.8
        x[t + 12 : t + 14, 60:80] = -25.0
        x[t + 12, 80] = 0.4
    x[:, 82] = 0.5
    return x


def generate(model) -> list[list[list[int]]]:
    from bsmapper.models import NoteSampler, SamplingOptions
    from bsmapper.schema import Characteristic, Difficulty
    from bsmapper.tokenizer import Conditioning

    conds = [
        Conditioning(Characteristic.STANDARD, Difficulty.EXPERT_PLUS, nps=5.5, style=0.7),
        Conditioning(Characteristic.STANDARD, Difficulty.HARD, nps=3.0),
    ]
    opts = SamplingOptions(temperature=0.9, top_p=0.95, seed=3, style_guidance=1.5)
    windows = NoteSampler(model, "cpu").generate(click_track(), conds, opts)
    return [[w.input_ids() for w in seq] for seq in windows]


if __name__ == "__main__" and sys.argv[1:] == ["record"]:
    import torch

    from bsmapper.models import load_model

    torch.manual_seed(0)
    out = generate(load_model(checkpoint(), "cpu"))
    GOLDEN.write_text(json.dumps({"checkpoint": "note_v0/best.pt", "windows": out}) + "\n")
    print("wrote", GOLDEN, [len(w) for seq in out for w in seq])
