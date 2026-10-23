"""Read a built dataset (``bsmapper data build`` output) back for training.

    ds = ProcessedDataset("data/processed")
    rows = ds.rows("train")                     # manifest rows (dicts), one per difficulty
    feats = ds.features(rows[0])                # (n_windows * 384, 83) float16 memmap view
    windows = ds.note_windows(rows[0])          # TokenWindows with prefix and context
    lights = ds.light_windows(rows[0])          # or None when the map has no lightshow

Tokens are stored as window bodies only. ``note_windows`` rebuilds each window's
context from the previous body (``tokenizer.tail_context``) and the conditioning
prefix from the manifest, so training can drop or change NPS/style per sample.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from bsmapper.schema import Characteristic, Difficulty
from bsmapper.tokenizer import (
    LIGHT_GRAMMAR,
    NOTE_GRAMMAR,
    Conditioning,
    TokenWindow,
    tail_context,
)
from bsmapper.tokenizer.grid import WINDOW_TICKS

_KEEP = object()


class ProcessedDataset:
    def __init__(self, root: Path | str) -> None:
        import pyarrow.parquet as pq

        self.root = Path(root)
        self.meta: dict[str, Any] = json.loads((self.root / "dataset.json").read_text())
        self._rows: list[dict[str, Any]] = pq.read_table(self.root / "manifest.parquet").to_pylist()
        self._tables = cache(self._load_table)
        self._feature_files = cache(self._load_features)

    def rows(self, split: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self._rows if split is None or r["split"] == split]

    def _load_table(self, kind: str, shard: str) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq

        return pq.read_table(self.root / kind / f"{shard}.parquet").to_pylist()

    def _load_features(self, shard: str) -> np.ndarray:
        return np.load(self.root / "features" / f"{shard}.npy", mmap_mode="r")

    def features(self, row: dict[str, Any]) -> np.ndarray:
        start = row["feature_row"]
        return self._feature_files(row["feature_shard"])[
            start : start + row["n_windows"] * WINDOW_TICKS
        ]

    def conditioning(
        self, row: dict[str, Any], nps: Any = _KEEP, style: Any = _KEEP, intensity: Any = _KEEP
    ):
        """Ground-truth conditioning; pass ``nps=None`` / ``style=None`` /
        ``intensity=None`` to drop them (dataset v0 has no intensity)."""
        level = row.get("intensity")
        if level is not None and not np.isfinite(level):
            level = None
        return Conditioning(
            Characteristic(row["characteristic"]),
            Difficulty(row["difficulty"]),
            nps=row["nps"] if nps is _KEEP else nps,
            style=(row["style"] if np.isfinite(row["style"]) else None)
            if style is _KEEP
            else style,
            bombs=row["flag_bombs"],
            walls=row["flag_walls"],
            arcs=row["flag_arcs"],
            chains=row["flag_chains"],
            intensity=level if intensity is _KEEP else intensity,
        )

    def note_windows(
        self,
        row: dict[str, Any],
        conditioning: Conditioning | None = None,
        context_events: int = 16,
    ) -> list[TokenWindow]:
        t = self._tables("tokens", row["token_shard"])[row["token_row"]]
        assert t["key"] == row["key"] and t["difficulty"] == row["difficulty"]
        prefix = (conditioning or self.conditioning(row)).prefix_ids()
        return _windows(t["tokens"], t["window_offsets"], prefix, NOTE_GRAMMAR, context_events)

    def light_windows(
        self, row: dict[str, Any], context_events: int = 16
    ) -> list[TokenWindow] | None:
        if row.get("light_shard") is None:
            return None
        t = self._tables("lights", row["light_shard"])[row["light_row"]]
        assert t["key"] == row["key"]
        return _windows(t["tokens"], t["window_offsets"], [], LIGHT_GRAMMAR, context_events)


def _style(row: dict[str, Any]) -> float | None:
    style = row["style"]
    return style if style is not None and np.isfinite(style) else None


def _windows(tokens, offsets, prefix, grammar, context_events) -> list[TokenWindow]:
    out: list[TokenWindow] = []
    prev: list[int] = []
    for k in range(len(offsets) - 1):
        body = [int(t) for t in tokens[offsets[k] : offsets[k + 1]]]
        context = tail_context(prev, grammar, context_events) if k else []
        out.append(TokenWindow(k, body, context, list(prefix)))
        prev = body
    return out
