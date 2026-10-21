"""A toy dataset in the WS3b on-disk layout, for tests and the CPU debug config.

Each map has "onsets" on a random 1/2-beat pattern. The onset feature column
spikes there and one mel band per grid column lights up; notes sit on exactly
those ticks, alternate colour and up/down swings, and take their column from
the lit band. Lights flash on every note. So a model can learn the mapping, and
a tiny one can overfit it quickly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bsmapper.models.config import FEATURE_DIM
from bsmapper.schema import Characteristic, Color, CutDirection, Difficulty
from bsmapper.tokenizer import (
    LIGHT_VOCAB,
    NOTE_VOCAB,
    TICKS_PER_BEAT,
    WINDOW_TICKS,
    Conditioning,
)
from bsmapper.tokenizer.grammar import serialize

N_MELS = 80
ONSET_COL = N_MELS
_DENSITY = {
    Difficulty.EASY: 0.15,
    Difficulty.NORMAL: 0.25,
    Difficulty.HARD: 0.4,
    Difficulty.EXPERT: 0.55,
    Difficulty.EXPERT_PLUS: 0.7,
}


def _note_events(onsets: np.ndarray, columns: np.ndarray, start: int) -> list:
    v = NOTE_VOCAB
    events = []
    for n, (tick, x) in enumerate(zip(onsets, columns, strict=True)):
        color = Color.RED if n % 2 == 0 else Color.BLUE
        direction = CutDirection.DOWN if (n // 2) % 2 == 0 else CutDirection.UP
        y = 0 if direction == CutDirection.DOWN else 2
        ids = (v.id("NOTE", color), v.id("POS", (int(x), y)), v.id("DIR", direction))
        events.append((int(tick) - start, ids))
    return events


def _light_events(onsets: np.ndarray, start: int) -> list:
    v = LIGHT_VOCAB
    return [
        (int(t) - start, (v.id("EVT", n % 5), v.id("VAL", 1 + 4 * (n % 2))))
        for n, t in enumerate(onsets)
    ]


def _in_window(ticks: np.ndarray, k: int) -> np.ndarray:
    return ticks[(ticks >= k * WINDOW_TICKS) & (ticks < (k + 1) * WINDOW_TICKS)]


def _track(per_window: list[list], vocab) -> tuple[np.ndarray, np.ndarray]:
    bodies = [serialize(events, 0, vocab) for events in per_window]
    offsets = np.cumsum([0, *(len(b) for b in bodies)]).astype(np.int32)
    return np.asarray([t for b in bodies for t in b], dtype=np.int16), offsets


def write_synthetic_dataset(
    root: Path | str,
    n_maps: int = 16,
    n_windows: int = 4,
    seed: int = 0,
    difficulties: tuple[Difficulty, ...] = (Difficulty.EXPERT, Difficulty.EXPERT_PLUS),
    val_maps: int = 2,
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    for sub in ("features", "tokens", "lights"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest: list[dict] = []
    splits = {"train": [], "val": []}
    for m in range(n_maps):
        splits["val" if m >= n_maps - val_maps else "train"].append(m)

    for split, maps in splits.items():
        shard = f"{split}-000"
        feats = []
        tok = {
            "key": [],
            "characteristic": [],
            "difficulty": [],
            "tokens": [],
            "window_offsets": [],
        }
        lights = {"key": [], "tokens": [], "window_offsets": []}
        row0 = 0
        for m in maps:
            key = f"syn{m:03d}"
            ticks = n_windows * WINDOW_TICKS
            f = rng.normal(-60.0, 3.0, size=(ticks, FEATURE_DIM)).astype(np.float32)
            f[:, ONSET_COL:] = np.abs(rng.normal(0.0, 0.01, size=(ticks, FEATURE_DIM - N_MELS)))
            half = TICKS_PER_BEAT // 2
            slots = np.arange(0, ticks, half)
            density = max(_DENSITY[d] for d in difficulties)
            hits = slots[rng.random(len(slots)) < density]
            cols = rng.integers(0, 4, size=len(hits))
            f[hits, ONSET_COL] = 1.0
            for t, x in zip(hits, cols, strict=True):
                f[t, 10 + 15 * x : 20 + 15 * x] = -10.0
            feats.append(f.astype(np.float16))
            # Lighting follows the densest difficulty's notes.
            lights["key"].append(key)
            lt, lo = _track(
                [_light_events(_in_window(hits, k), k * WINDOW_TICKS) for k in range(n_windows)],
                LIGHT_VOCAB,
            )
            lights["tokens"].append(lt)
            lights["window_offsets"].append(lo)
            for d in difficulties:
                keep = rng.random(len(hits)) < _DENSITY[d] / density
                on, cs = hits[keep], cols[keep]
                per_window = []
                for k in range(n_windows):
                    sel = (on >= k * WINDOW_TICKS) & (on < (k + 1) * WINDOW_TICKS)
                    per_window.append(_note_events(on[sel], cs[sel], k * WINDOW_TICKS))
                tokens, offsets = _track(per_window, NOTE_VOCAB)
                nps = len(on) / (ticks / TICKS_PER_BEAT * 0.5)  # 120 BPM
                style = float(rng.random())
                cond = Conditioning(Characteristic.STANDARD, d, nps=nps, style=style)
                manifest.append(
                    {
                        "key": key,
                        "split": split,
                        "characteristic": cond.characteristic.value,
                        "difficulty": d.value,
                        "nps": nps,
                        "style": style,
                        "flag_bombs": False,
                        "flag_walls": False,
                        "flag_arcs": False,
                        "flag_chains": False,
                        "gold": True,
                        "n_windows": n_windows,
                        "feature_shard": shard,
                        "feature_row": row0,
                        "token_shard": shard,
                        "token_row": len(tok["key"]),
                        "light_shard": shard,
                        "light_row": len(lights["key"]) - 1,
                    }
                )
                tok["key"].append(key)
                tok["characteristic"].append(cond.characteristic.value)
                tok["difficulty"].append(d.value)
                tok["tokens"].append(tokens)
                tok["window_offsets"].append(offsets)
            row0 += ticks
        np.save(root / "features" / f"{shard}.npy", np.concatenate(feats))
        pq.write_table(_table(pa, tok), root / "tokens" / f"{shard}.parquet")
        pq.write_table(_table(pa, lights), root / "lights" / f"{shard}.parquet")
    pq.write_table(pa.Table.from_pylist(manifest), root / "manifest.parquet")
    meta = {
        "dataset_version": "synthetic",
        "ticks_per_beat": TICKS_PER_BEAT,
        "window_ticks": WINDOW_TICKS,
        "feature_dim": FEATURE_DIM,
        "note_vocab_size": len(NOTE_VOCAB),
        "light_vocab_size": len(LIGHT_VOCAB),
    }
    (root / "dataset.json").write_text(json.dumps(meta, indent=1))
    return root


def _table(pa, rows: dict[str, list]):
    cols = {}
    for k, v in rows.items():
        if k == "tokens":
            cols[k] = pa.array(v, pa.list_(pa.int16()))
        elif k == "window_offsets":
            cols[k] = pa.array(v, pa.list_(pa.int32()))
        else:
            cols[k] = pa.array(v, pa.string())
    return pa.table(cols)
