"""Read-only access to a built dataset (the ``bsmapper data build`` layout, WS3b).

    root/dataset.json                     layout and vocab sizes
    root/manifest.parquet                 one row per kept difficulty (labels + pointers)
    root/features/<shard>.npy             float16 (rows, 83); map beat j/24 is row j
    root/tokens/<shard>.parquet           key, characteristic, difficulty, tokens, window_offsets
    root/lights/<shard>.parquet           key, tokens, window_offsets (one row per map)

A difficulty's window ``k`` body is ``tokens[offsets[k]:offsets[k+1]]`` and its
features are rows ``feature_row + k*384 ..`` of its feature shard. Only window
bodies are stored; prefixes and tail contexts are rebuilt per sample.

This module only depends on that file layout (not on ``bsmapper.data``), so the
trainer can read datasets built by any builder that writes it, including the
synthetic one in ``training.synthetic``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bsmapper.schema import Characteristic, Difficulty
from bsmapper.tokenizer import LIGHT_VOCAB, NOTE_VOCAB, WINDOW_TICKS, Conditioning


def _pq():
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("reading datasets needs pyarrow: pip install -e .[ml]") from e
    return pq


@dataclass(slots=True)
class TokenTrack:
    """All window bodies of one difficulty (or one lightshow)."""

    tokens: np.ndarray  # int16
    offsets: np.ndarray  # int32, n_windows + 1 entries

    @property
    def n_windows(self) -> int:
        return len(self.offsets) - 1

    def body(self, k: int) -> list[int]:
        if k < 0 or k >= self.n_windows:
            return []
        return self.tokens[self.offsets[k] : self.offsets[k + 1]].astype(np.int64).tolist()

    def body_len(self, k: int) -> int:
        return int(self.offsets[k + 1] - self.offsets[k])


def _arrays(column, dtype) -> list[np.ndarray]:
    return [np.asarray(x, dtype=dtype) for x in column.to_numpy(zero_copy_only=False)]


def _finite(x: Any) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


class DatasetStore:
    """Manifest rows of one or more splits, their token tracks and lazily mapped features."""

    def __init__(self, root: Path | str, splits: Sequence[str] = ("train",)) -> None:
        pq = _pq()
        self.root = Path(root)
        self.meta: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        for key, vocab in (("note_vocab_size", NOTE_VOCAB), ("light_vocab_size", LIGHT_VOCAB)):
            if key in self.meta and self.meta[key] != len(vocab):
                raise ValueError(
                    f"dataset {key} {self.meta[key]} != tokenizer {len(vocab)}; rebuild it"
                )
        manifest = pq.read_table(self.root / "manifest.parquet").to_pylist()
        self.rows = [r for r in manifest if r["split"] in splits]
        self.notes: list[TokenTrack] = []
        self.lights: dict[str, TokenTrack] = {}
        tables: dict[tuple[str, str], dict[str, list]] = {}

        def table(kind: str, shard: str) -> dict[str, list]:
            if (kind, shard) not in tables:
                t = pq.read_table(self.root / kind / f"{shard}.parquet")
                tables[(kind, shard)] = {
                    "key": t.column("key").to_pylist(),
                    "tokens": _arrays(t.column("tokens"), np.int16),
                    "window_offsets": _arrays(t.column("window_offsets"), np.int32),
                }
            return tables[(kind, shard)]

        for r in self.rows:
            t = table("tokens", r["token_shard"])
            i = r["token_row"]
            if t["key"][i] != r["key"]:
                raise ValueError(f"token row {i} of {r['token_shard']} is not map {r['key']}")
            track = TokenTrack(t["tokens"][i], t["window_offsets"][i])
            if track.n_windows != r["n_windows"]:
                raise ValueError(f"{r['key']}: {track.n_windows} windows != {r['n_windows']}")
            self.notes.append(track)
            shard = r.get("light_shard")
            if shard is not None and r["key"] not in self.lights:
                lt = table("lights", shard)
                j = r["light_row"]
                self.lights[r["key"]] = TokenTrack(lt["tokens"][j], lt["window_offsets"][j])
        self._features: dict[str, np.ndarray] = {}

    # Memory maps are opened lazily, and not pickled to DataLoader workers.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_features"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        return len(self.rows)

    def features(self, i: int, k: int) -> np.ndarray:
        """(384, feature_dim) float16 features of row ``i``'s window ``k``."""
        r = self.rows[i]
        shard = r["feature_shard"]
        if shard not in self._features:
            self._features[shard] = np.load(self.root / "features" / f"{shard}.npy", mmap_mode="r")
        start = r["feature_row"] + k * WINDOW_TICKS
        return np.asarray(self._features[shard][start : start + WINDOW_TICKS])

    def conditioning(self, i: int, drop_nps: bool = False, drop_style: bool = False):
        r = self.rows[i]
        return Conditioning(
            Characteristic(r["characteristic"]),
            Difficulty(r["difficulty"]),
            nps=None if drop_nps else _finite(r["nps"]),
            style=None if drop_style else _finite(r.get("style")),
            bombs=bool(r["flag_bombs"]),
            walls=bool(r["flag_walls"]),
            arcs=bool(r["flag_arcs"]),
            chains=bool(r["flag_chains"]),
        )

    def feature_stats(
        self, max_windows: int = 2000, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-feature mean and std over a random sample of windows."""
        rng = np.random.default_rng(seed)
        picks = [(i, k) for i, r in enumerate(self.rows) for k in range(r["n_windows"])]
        if len(picks) > max_windows:
            picks = [picks[j] for j in rng.choice(len(picks), max_windows, replace=False)]
        x = np.concatenate([self.features(i, k).astype(np.float32) for i, k in picks])
        return x.mean(axis=0), x.std(axis=0)
