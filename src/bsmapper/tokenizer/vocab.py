"""Token vocabularies: the note/gameplay vocabulary and the separate lighting one.

A vocabulary is an ordered list of *groups*. Each group holds tokens of one kind
(``TIME``, ``POS``, ``DIR`` ...) and every token carries a Python value
(``TIME`` -> tick, ``POS`` -> (x, y), ``NOTE`` -> Color ...). Ids are stable as
long as the group tables below do not change, so treat edits here like a
checkpoint-format change.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from bsmapper.schema import Characteristic, Color, CutDirection, Difficulty
from bsmapper.tokenizer.grid import (
    ANGLE_MAX,
    ANGLE_STEP,
    FRAC_MAX,
    GRID_COLUMNS,
    GRID_ROWS,
    MAX_SLICES,
    ROTATION_MAX,
    ROTATION_STEP,
    TICKS_PER_BEAT,
    WALL_MAX_HEIGHT,
    WINDOW_TICKS,
)

PAD, BOS, EOS, SEP = 0, 1, 2, 3  # the same ids in every vocabulary

# NPS bucket edges (notes per second). Bucket k covers [EDGES[k-1], EDGES[k]).
NPS_EDGES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0)
STYLE_BUCKETS = 5
DUR_MAX_BEATS = 16  # whole-beat duration tokens DUR_1..DUR_16, repeated for longer spans

# Event types the lighting vocabulary covers: lights, rings, lasers, utility, special.
LIGHT_EVENT_TYPES = (*range(0, 20), 40, 41, 42, 43)
LIGHT_VALUE_MAX = 31
LIGHT_FLOAT_MAX = 40  # float_value k / 20 (grid.FRAC_STEPS) for k in 0..40, i.e. 0.0..2.0
# Chroma colours are quantized to this palette (RGB, alpha dropped): 12 hues and white.
PALETTE: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 1.0, 0.0),
    (0.5, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.5),
    (0.0, 1.0, 1.0),
    (0.0, 0.5, 1.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 0.0, 0.5),
    (1.0, 1.0, 1.0),
)

Namer = Callable[[str, Any], str]


def _default_name(group: str, value: Any) -> str:
    if value is None:
        return group
    if isinstance(value, tuple):
        return "_".join([group, *(str(v) for v in value)])
    return f"{group}_{value}"


def _enum_name(group: str, value: Any) -> str:
    return f"{group}_{value.name}"


def _flag_name(group: str, value: Any) -> str:
    return f"{group}_{'ON' if value else 'OFF'}"


def _signed_name(group: str, value: Any) -> str:
    return f"{group}_{value:+d}"


class Vocab:
    """Ordered token groups. ``id(group, value)`` <-> ``group(id)``/``value(id)``."""

    def __init__(self, groups: Sequence[tuple[str, Sequence[Any], Namer]]) -> None:
        self._names: list[str] = []
        self._groups: list[str] = []
        self._values: list[Any] = []
        self._ids: dict[tuple[str, Any], int] = {}
        self._group_ids: dict[str, list[int]] = {}
        for group, values, namer in groups:
            if group in self._group_ids:
                raise ValueError(f"duplicate group {group}")
            ids = self._group_ids[group] = []
            for value in values:
                token_id = len(self._names)
                self._names.append(namer(group, value))
                self._groups.append(group)
                self._values.append(value)
                self._ids[(group, value)] = token_id
                ids.append(token_id)
        if len(set(self._names)) != len(self._names):
            raise ValueError("token names are not unique")
        self._by_name = {name: i for i, name in enumerate(self._names)}

    def __len__(self) -> int:
        return len(self._names)

    def id(self, group: str, value: Any = None) -> int:
        try:
            return self._ids[(group, value)]
        except KeyError:
            raise KeyError(f"no token {group} {value!r} in the vocabulary") from None

    def has(self, group: str, value: Any = None) -> bool:
        return (group, value) in self._ids

    def group(self, token_id: int) -> str:
        return self._groups[token_id]

    def value(self, token_id: int) -> Any:
        return self._values[token_id]

    def name(self, token_id: int) -> str:
        return self._names[token_id]

    def ids(self, *groups: str) -> list[int]:
        return [i for g in groups for i in self._group_ids[g]]

    def groups(self) -> list[str]:
        return list(self._group_ids)

    def names(self, ids: Iterable[int]) -> list[str]:
        return [self._names[i] for i in ids]

    def from_names(self, names: Iterable[str]) -> list[int]:
        return [self._by_name[n] for n in names]

    def mask(self, ids: Iterable[int]) -> np.ndarray:
        m = np.zeros(len(self), dtype=bool)
        m[list(ids)] = True
        return m


def _specials() -> list[tuple[str, Sequence[Any], Namer]]:
    return [(name, [None], _default_name) for name in ("PAD", "BOS", "EOS", "SEP")]


def _time_group() -> tuple[str, Sequence[Any], Namer]:
    return ("TIME", range(WINDOW_TICKS), _default_name)


def _duration_groups() -> list[tuple[str, Sequence[Any], Namer]]:
    return [
        ("DUR", range(1, DUR_MAX_BEATS + 1), _default_name),  # whole beats
        ("DURT", range(TICKS_PER_BEAT), _default_name),  # remaining ticks
    ]


def build_note_vocab() -> Vocab:
    cells = [(x, y) for y in range(GRID_ROWS) for x in range(GRID_COLUMNS)]
    angles = [a for a in range(-ANGLE_MAX, ANGLE_MAX + 1, ANGLE_STEP) if a != 0]
    rotations = [
        r for r in range(-ROTATION_MAX + ROTATION_STEP, ROTATION_MAX + 1, ROTATION_STEP) if r
    ]
    return Vocab(
        [
            *_specials(),
            # Conditioning prefix.
            ("CHAR", list(Characteristic), _enum_name),
            ("DIFF", list(Difficulty), _enum_name),
            ("NPS", [*range(len(NPS_EDGES) + 1), None], _nullable_name),
            ("STYLE", [*range(STYLE_BUCKETS), None], _nullable_name),
            ("BOMBS", [True, False], _flag_name),
            ("WALLS", [True, False], _flag_name),
            ("ARCS", [True, False], _flag_name),
            ("CHAINS", [True, False], _flag_name),
            # Events. Head groups are in canonical order: objects on the same tick
            # are sorted by their token ids, so rotations come first, then walls,
            # bombs, notes (red before blue), arcs and chains.
            _time_group(),
            ("ROT", rotations, _signed_name),
            ("WALL", [None], _default_name),
            ("BOMB", [None], _default_name),
            ("NOTE", list(Color), lambda g, v: v.name),  # RED / BLUE
            ("ARC", [None], _default_name),
            ("CHAIN", [None], _default_name),
            ("POS", cells, _default_name),
            ("DIR", list(CutDirection), _enum_name),
            ("ANGLE", angles, _signed_name),
            ("LATE", [True], lambda g, v: g),  # late rotation (value True: present)
            ("WX", range(GRID_COLUMNS), _default_name),
            ("WW", range(1, GRID_COLUMNS + 1), _default_name),
            (
                "WH",
                [(y, h) for y in range(GRID_ROWS) for h in range(1, WALL_MAX_HEIGHT + 1)],
                _default_name,
            ),
            *_duration_groups(),
            ("FRAC", range(FRAC_MAX + 1), _default_name),  # value k / 20
            ("ANCHOR", range(3), _default_name),
            ("SLICE", range(1, MAX_SLICES + 1), _default_name),
        ]
    )


def _nullable_name(group: str, value: Any) -> str:
    return f"{group}_NONE" if value is None else f"{group}_{value}"


def build_light_vocab() -> Vocab:
    return Vocab(
        [
            *_specials(),
            _time_group(),
            ("EVT", LIGHT_EVENT_TYPES, _default_name),
            ("BOOST", [True, False], _flag_name),
            ("VAL", range(LIGHT_VALUE_MAX + 1), _default_name),
            ("FLOAT", range(LIGHT_FLOAT_MAX + 1), _default_name),  # value k / 20
            ("COLOR", range(len(PALETTE)), _default_name),
        ]
    )


NOTE_VOCAB = build_note_vocab()
LIGHT_VOCAB = build_light_vocab()
assert NOTE_VOCAB.id("PAD") == PAD and NOTE_VOCAB.id("SEP") == SEP
assert LIGHT_VOCAB.id("BOS") == BOS and LIGHT_VOCAB.id("EOS") == EOS
