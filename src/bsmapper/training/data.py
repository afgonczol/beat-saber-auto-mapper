"""Training examples: one 16-beat window of one difficulty (notes) or one map (lights).

A note example is the decoder sequence ``BOS prefix [REF ref REF_END] context SEP
body EOS`` with the loss on ``body EOS`` only, plus the window's audio features. The
prefix is rebuilt from the manifest each time, with STYLE / NPS / INTENSITY randomly
replaced by their ``_NONE`` tokens (classifier-free guidance), and the context is
the previous window's tail. For a tokenizer v2 model (note_v1), a window whose
reference span (``data.structure``) clears ``ref_threshold`` also gets the human
notes of that span as a reference block, except ``ref_dropout`` of the time.

A light example is ``BOS context SEP body EOS`` in the lighting vocabulary, the same
audio, and the note track of one of the map's difficulties (``BOS prefix SEP body EOS``)
for the lighting decoder's note memory.

For a lighting decoder with the v2 light vocabulary (WS8b), a light example is ``BOS
LSTYLE LEVEL PAL FLIPS [REF ref REF_END] context SEP body EOS``: the map's style label
(``data.light_styles``, from ``bsmapper lights styles``), the window's own scene
(``tokenizer.window_scene``) and the human light events of the window's reference
span. Each part has its own dropout to ``_NONE`` / no block, so generation can leave
any of them out. With ``light_labelled_only`` only labelled maps are trained on, which
leaves out Chroma lightID / strobe sequences and too sparse lightshows.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from bsmapper.schema import Difficulty
from bsmapper.tokenizer import (
    BOS,
    EOS,
    LIGHT_GRAMMAR,
    LIGHT_REF,
    LIGHT_REF_END,
    LIGHT_TOKENIZER_VERSION,
    NOTE_GRAMMAR,
    PAD,
    REF,
    REF_END,
    SEP,
    TOKENIZER_VERSION,
    LightPrompt,
    span_tokens,
    tail_context,
    window_scene,
)
from bsmapper.training.config import DataConfig
from bsmapper.training.store import DatasetStore

log = logging.getLogger(__name__)

IGNORE = -100
_DIFF_RANK = {d.value: i for i, d in enumerate(Difficulty)}


def note_sequence(
    prefix: list[int],
    context: list[int],
    body: list[int],
    max_len: int,
    ref: list[int] | None = None,
    ref_ids: tuple[int, int] = (REF, REF_END),
):
    """(ids, loss_start): the first target position predicts the first body token.

    ``ref_ids`` are the vocabulary's ``REF`` / ``REF_END`` (the light vocabulary's for
    light examples)."""
    block = [] if ref is None else [ref_ids[0], *ref, ref_ids[1]]
    ids = [BOS, *prefix, *block, *context, SEP, *body, EOS]
    body_offset = len(prefix) + len(block) + len(context) + 2
    if len(ids) > max_len:
        if body_offset + 1 > max_len:  # too long: drop the reference, then the context
            if ref is not None:
                return note_sequence(prefix, context, body, max_len)
            return note_sequence(prefix, [], body, max_len)
        ids = ids[:max_len]
    return ids, body_offset - 1


def window_weights(
    store: DatasetStore, cfg: DataConfig
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """(row, window) index and its sampling weight, for note training."""
    index: list[tuple[int, int]] = []
    weights: list[float] = []
    for i, r in enumerate(store.rows):
        w = cfg.difficulty_weights.get(r["difficulty"], 1.0)
        if cfg.oversample_easy or r["difficulty"] != Difficulty.EASY.value:
            w *= cfg.characteristic_weights.get(r["characteristic"], 1.0)
        if r.get("gold"):
            w *= cfg.gold_weight
        track = store.notes[i]
        for k in range(track.n_windows):
            index.append((i, k))
            weights.append(w * (cfg.empty_window_weight if track.body_len(k) == 0 else 1.0))
    return index, np.asarray(weights, dtype=np.float64)


class NoteWindows(Dataset):
    def __init__(
        self,
        store: DatasetStore,
        cfg: DataConfig,
        train: bool = True,
        version: int = TOKENIZER_VERSION,
    ) -> None:
        self.store, self.cfg, self.train = store, cfg, train
        self.version = version  # the model's tokenizer version: prefix layout, references
        self.index, self.weights = window_weights(store, cfg)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, j: int) -> dict:
        i, k = self.index[j]
        cfg, store = self.cfg, self.store
        rng = random if self.train else random.Random(j)
        cond = store.conditioning(
            i,
            drop_nps=rng.random() < cfg.nps_dropout,
            drop_style=rng.random() < cfg.style_dropout,
            drop_intensity=rng.random() < cfg.intensity_dropout,
        )
        track = store.notes[i]
        context = []
        if k and not rng.random() < cfg.context_dropout:
            context = tail_context(track.body(k - 1), NOTE_GRAMMAR, cfg.context_events)
        ref = None
        if self.version >= 2:
            s = store.ref_start(i, k, cfg.ref_threshold)
            if s is not None and not rng.random() < cfg.ref_dropout:
                ref = span_tokens(track, s, NOTE_GRAMMAR, cfg.max_ref_tokens)
        ids, start = note_sequence(
            cond.prefix_ids(version=self.version),
            context,
            track.body(k),
            cfg.max_note_tokens,
            ref,
        )
        return {"features": store.features(i, k), "tokens": ids, "loss_start": start}


def load_light_styles(store: DatasetStore, cfg: DataConfig) -> dict | None:
    """The ``bsmapper lights styles`` file of the dataset (``cfg.light_styles``), if any."""
    if not cfg.light_styles:
        return None
    path = Path(cfg.light_styles)
    if not path.is_absolute():
        path = store.root / path
    if not path.is_file():
        log.warning("no light style file %s: lights train with LSTYLE_NONE", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class LightWindows(Dataset):
    def __init__(
        self,
        store: DatasetStore,
        cfg: DataConfig,
        train: bool = True,
        version: int = TOKENIZER_VERSION,
        light_version: int = 1,
        styles: dict | None = None,
    ) -> None:
        """``version`` is the note prompt layout (note memory), ``light_version`` the light
        one; ``styles`` the light style file's contents (version 2 only)."""
        self.store, self.cfg, self.train = store, cfg, train
        self.version = version
        if not 1 <= light_version <= LIGHT_TOKENIZER_VERSION:
            raise ValueError(f"unknown light tokenizer version {light_version}")
        self.light_version = light_version
        self.labels: dict[str, int] | None = None
        if light_version >= 2 and styles is not None:
            self.labels = {k: int(v) for k, v in styles["labels"].items()}
        labelled_only = self.labels is not None and cfg.light_labelled_only
        rows_by_key: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(store.rows):
            if labelled_only and r["key"] not in self.labels:  # type: ignore[operator]
                continue
            if r["key"] in store.lights and (r.get("gold") or not cfg.lights_gold_only):
                rows_by_key[r["key"]].append(i)
        self.rows_by_key = dict(rows_by_key)
        self.index: list[tuple[str, int]] = []
        for key, rows in self.rows_by_key.items():
            track = store.lights[key]
            n = min(track.n_windows, store.notes[rows[0]].n_windows)
            for k in range(n):
                if track.body_len(k) <= cfg.max_light_body:
                    self.index.append((key, k))
        lengths = [store.lights[key].body_len(k) for key, k in self.index]
        self.weights = np.where(np.asarray(lengths) == 0, cfg.empty_window_weight, 1.0)

    def __len__(self) -> int:
        return len(self.index)

    def _note_row(self, key: str, rng) -> int:
        rows = self.rows_by_key[key]
        if rng.random() < self.cfg.hardest_notes_for_lights:
            return max(rows, key=lambda i: _DIFF_RANK[self.store.rows[i]["difficulty"]])
        return rng.choice(rows)

    def __getitem__(self, j: int) -> dict:
        key, k = self.index[j]
        cfg, store = self.cfg, self.store
        rng = random if self.train else random.Random(j)
        i = self._note_row(key, rng)
        track = store.lights[key]
        body = track.body(k)
        context = []
        if k and not rng.random() < cfg.context_dropout:
            context = tail_context(track.body(k - 1), LIGHT_GRAMMAR, cfg.context_events)
        prefix: list[int] = []
        ref = None
        if self.light_version >= 2:
            style = None
            if self.labels is not None and not rng.random() < cfg.light_style_dropout:
                style = self.labels.get(key)
            prompt = LightPrompt(style)
            if not rng.random() < cfg.scene_dropout:
                prompt = prompt.with_scene(window_scene(body))
            prefix = prompt.prefix_ids()
            s = store.ref_start(i, k, cfg.ref_threshold)
            if s is not None and not rng.random() < cfg.light_ref_dropout:
                ref = span_tokens(track, s, LIGHT_GRAMMAR, cfg.max_light_ref_tokens)
        ids, start = note_sequence(
            prefix, context, body, cfg.max_light_tokens, ref, (LIGHT_REF, LIGHT_REF_END)
        )
        notes, _ = note_sequence(
            store.conditioning(i).prefix_ids(version=self.version),
            [],
            store.notes[i].body(k),
            cfg.note_memory_tokens,
        )
        return {
            "features": store.features(i, k),
            "tokens": ids,
            "loss_start": start,
            "note_tokens": notes,
        }


def _pad(seqs: list[list[int]], value: int) -> torch.Tensor:
    out = torch.full((len(seqs), max(len(s) for s in seqs)), value, dtype=torch.long)
    for b, s in enumerate(seqs):
        out[b, : len(s)] = torch.as_tensor(s, dtype=torch.long)
    return out


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    """Inputs are ``tokens[:-1]``, targets ``tokens[1:]`` with IGNORE before the body."""
    tokens = _pad([it["tokens"] for it in items], PAD)
    targets = tokens[:, 1:].clone()
    targets[targets == PAD] = IGNORE
    for b, it in enumerate(items):
        targets[b, : it["loss_start"]] = IGNORE
    batch = {
        "features": torch.from_numpy(np.stack([it["features"] for it in items])),
        "inputs": tokens[:, :-1],
        "targets": targets,
    }
    if "note_tokens" in items[0]:
        batch["note_tokens"] = _pad([it["note_tokens"] for it in items], PAD)
    return batch


def infinite_loader(
    dataset: NoteWindows | LightWindows,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    weighted: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    """Batches forever, drawn with replacement by the dataset's sampling weights."""
    generator = torch.Generator().manual_seed(seed)
    weights = dataset.weights if weighted else np.ones(len(dataset))
    epoch = 0
    while True:
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=max(len(dataset), batch_size),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate,
            num_workers=num_workers,
            drop_last=True,
            persistent_workers=False,
            worker_init_fn=_seed_worker,
        )
        yield from loader
        epoch += 1


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
