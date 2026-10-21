"""``DifficultyMap`` <-> windowed token sequences for the note model.

Objects (after ``TIME_t``, which is relative to the window start)::

    note   RED|BLUE POS_x_y DIR_d [ANGLE_a]
    bomb   BOMB POS_x_y
    wall   WALL WX_x WW_w WH_y_h <span>
    arc    ARC RED|BLUE POS_x_y DIR_d FRAC_head <span> POS_x_y DIR_d FRAC_tail ANCHOR_m
    chain  CHAIN RED|BLUE POS_x_y DIR_d <span> POS_x_y SLICE_n FRAC_squish
    rot    ROT_+15 .. ROT_+180 [LATE]

``<span>`` is ``DUR_16* [DUR_b] DURT_r`` (wall duration, or head-to-tail time).
Arcs and chains carry their own head colour/position/direction, so they do not
depend on a note being there; in real maps a note usually precedes them on the
same tick, which the canonical order (notes before arcs and chains) makes easy
to copy.

Encoding is strict: every time and value must already be on the grids in
``grid`` (run ``snap_map`` first), otherwise ``OffGridError`` is raised. Then
``decode(encode(m))`` equals ``canonical_order(m)`` exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import Any

from bsmapper.schema import (
    Arc,
    Bomb,
    Chain,
    Characteristic,
    Difficulty,
    DifficultyMap,
    Note,
    Obstacle,
    RotationEvent,
)
from bsmapper.tokenizer.conditioning import PREFIX_LEN, Conditioning
from bsmapper.tokenizer.grammar import (
    DUR,
    EncodedEvent,
    Grammar,
    ObjectSpec,
    ParsedEvent,
    span_ids,
)
from bsmapper.tokenizer.grid import (
    FRAC_STEPS,
    OffGridError,
    frac_index,
    grid_tick,
    tick_to_beat,
)
from bsmapper.tokenizer.vocab import NOTE_VOCAB, Vocab
from bsmapper.tokenizer.windows import (
    DEFAULT_CONTEXT_EVENTS,
    TokenWindow,
    build_windows,
    parse_windows,
)

NOTE_SPECS = (
    ObjectSpec("ROT", (), ("LATE",)),
    ObjectSpec("WALL", ("WX", "WW", "WH", DUR)),
    ObjectSpec("BOMB", ("POS",)),
    ObjectSpec("NOTE", ("POS", "DIR"), ("ANGLE",)),
    ObjectSpec("ARC", ("NOTE", "POS", "DIR", "FRAC", DUR, "POS", "DIR", "FRAC", "ANCHOR")),
    ObjectSpec("CHAIN", ("NOTE", "POS", "DIR", DUR, "POS", "SLICE", "FRAC")),
)
NOTE_GRAMMAR = Grammar(NOTE_VOCAB, NOTE_SPECS)

GameplayObject = Note | Bomb | Obstacle | Arc | Chain | RotationEvent


def _token(vocab: Vocab, group: str, value: Any = None) -> int:
    if not vocab.has(group, value):
        raise OffGridError(f"{group} {value!r} is outside the vocabulary; run snap_map first")
    return vocab.id(group, value)


def object_event(obj: GameplayObject, vocab: Vocab = NOTE_VOCAB) -> EncodedEvent:
    """(tick, token ids) of one gameplay object."""
    t = _token
    tick = grid_tick(obj.beat)
    match obj:
        case Note():
            ids = [t(vocab, "NOTE", obj.color), t(vocab, "POS", (obj.x, obj.y))]
            ids.append(t(vocab, "DIR", obj.direction))
            if obj.angle_offset:
                ids.append(t(vocab, "ANGLE", obj.angle_offset))
        case Bomb():
            ids = [t(vocab, "BOMB"), t(vocab, "POS", (obj.x, obj.y))]
        case Obstacle():
            ids = [t(vocab, "WALL"), t(vocab, "WX", obj.x), t(vocab, "WW", obj.width)]
            ids.append(t(vocab, "WH", (obj.y, obj.height)))
            ids += _span(vocab, grid_tick(obj.duration))
        case Arc():
            ids = [t(vocab, "ARC"), t(vocab, "NOTE", obj.color), t(vocab, "POS", (obj.x, obj.y))]
            ids.append(t(vocab, "DIR", obj.direction))
            ids.append(t(vocab, "FRAC", frac_index(obj.head_multiplier)))
            ids += _span(vocab, grid_tick(obj.tail_beat) - tick)
            ids += [t(vocab, "POS", (obj.tail_x, obj.tail_y)), t(vocab, "DIR", obj.tail_direction)]
            ids += [t(vocab, "FRAC", frac_index(obj.tail_multiplier))]
            ids.append(t(vocab, "ANCHOR", obj.mid_anchor))
        case Chain():
            ids = [t(vocab, "CHAIN"), t(vocab, "NOTE", obj.color)]
            ids += [t(vocab, "POS", (obj.x, obj.y)), t(vocab, "DIR", obj.direction)]
            ids += _span(vocab, grid_tick(obj.tail_beat) - tick)
            ids += [t(vocab, "POS", (obj.tail_x, obj.tail_y)), t(vocab, "SLICE", obj.slice_count)]
            ids.append(t(vocab, "FRAC", frac_index(obj.squish)))
        case RotationEvent():
            ids = [t(vocab, "ROT", obj.degrees)]
            if not obj.early:
                ids.append(t(vocab, "LATE", True))
        case _:
            raise TypeError(f"not a gameplay object: {obj!r}")
    if tick < 0:
        raise OffGridError(f"{type(obj).__name__} at beat {obj.beat} is before beat 0")
    return tick, tuple(ids)


def _span(vocab: Vocab, ticks: int) -> list[int]:
    if ticks < 0:
        raise OffGridError("negative duration; run snap_map first")
    return span_ids(vocab, ticks)


def _objects(dmap: DifficultyMap) -> Iterable[GameplayObject]:
    yield from dmap.rotations
    yield from dmap.obstacles
    yield from dmap.bombs
    yield from dmap.notes
    yield from dmap.arcs
    yield from dmap.chains


def map_events(dmap: DifficultyMap, vocab: Vocab = NOTE_VOCAB) -> list[EncodedEvent]:
    """All gameplay objects as events, in canonical order: by tick, then by token ids."""
    return sorted(object_event(obj, vocab) for obj in _objects(dmap))


def canonical_order(dmap: DifficultyMap) -> DifficultyMap:
    """``dmap`` with each object list in the order ``decode`` returns it."""

    def key(obj: GameplayObject) -> EncodedEvent:
        return object_event(obj)

    return replace(
        dmap,
        notes=sorted(dmap.notes, key=key),
        bombs=sorted(dmap.bombs, key=key),
        obstacles=sorted(dmap.obstacles, key=key),
        arcs=sorted(dmap.arcs, key=key),
        chains=sorted(dmap.chains, key=key),
        rotations=sorted(dmap.rotations, key=key),
    )


def _note(beat: float, e: ParsedEvent, tick: int) -> Note:
    (x, y), direction, angle = e.values
    return Note(beat, x, y, e.head, direction, angle or 0)


def _bomb(beat: float, e: ParsedEvent, tick: int) -> Bomb:
    ((x, y),) = e.values
    return Bomb(beat, x, y)


def _wall(beat: float, e: ParsedEvent, tick: int) -> Obstacle:
    x, width, (y, height), span = e.values
    return Obstacle(beat, tick_to_beat(span), x, y, width, height)


def _arc(beat: float, e: ParsedEvent, tick: int) -> Arc:
    color, (x, y), direction, head_k, span, (tx, ty), tail_dir, tail_k, anchor = e.values
    head_mult, tail_mult = head_k / FRAC_STEPS, tail_k / FRAC_STEPS
    tail = tick_to_beat(tick + span)
    return Arc(beat, color, x, y, direction, tail, tx, ty, tail_dir, head_mult, tail_mult, anchor)


def _chain(beat: float, e: ParsedEvent, tick: int) -> Chain:
    color, (x, y), direction, span, (tx, ty), slices, squish_k = e.values
    tail = tick_to_beat(tick + span)
    return Chain(beat, color, x, y, direction, tail, tx, ty, slices, squish_k / FRAC_STEPS)


def _rotation(beat: float, e: ParsedEvent, tick: int) -> RotationEvent:
    (late,) = e.values
    return RotationEvent(beat, e.head, early=late is None)


_BUILDERS: dict[str, tuple[str, Callable[[float, ParsedEvent, int], GameplayObject]]] = {
    "NOTE": ("notes", _note),
    "BOMB": ("bombs", _bomb),
    "WALL": ("obstacles", _wall),
    "ARC": ("arcs", _arc),
    "CHAIN": ("chains", _chain),
    "ROT": ("rotations", _rotation),
}


def events_to_map(
    events: Iterable[tuple[int, ParsedEvent]],
    characteristic: Characteristic,
    difficulty: Difficulty,
) -> DifficultyMap:
    """Build a map from (absolute tick, event) pairs."""
    dmap = DifficultyMap(characteristic, difficulty)
    for tick, event in events:
        field_name, build = _BUILDERS[event.kind]
        getattr(dmap, field_name).append(build(tick_to_beat(tick), event, tick))
    return dmap


class MapTokenizer:
    """Encode a difficulty into 16-beat token windows and decode them back."""

    vocab = NOTE_VOCAB
    grammar = NOTE_GRAMMAR
    prefix_len = PREFIX_LEN

    def __init__(self, context_events: int = DEFAULT_CONTEXT_EVENTS) -> None:
        self.context_events = context_events

    def encode(
        self,
        dmap: DifficultyMap,
        conditioning: Conditioning | None = None,
        n_windows: int | None = None,
    ) -> list[TokenWindow]:
        """Windows for ``dmap``. ``n_windows`` defaults to the window of the last object;
        pass the song length in windows to include trailing empty windows."""
        cond = conditioning or Conditioning.from_map(dmap)
        return build_windows(
            map_events(dmap, self.vocab),
            self.grammar,
            prefix=cond.prefix_ids(self.vocab),
            n_windows=n_windows,
            context_events=self.context_events,
        )

    def decode(
        self,
        windows: Sequence[TokenWindow],
        meta: DifficultyMap | None = None,
    ) -> DifficultyMap:
        """Rebuild the map. Characteristic and difficulty come from the first window's
        prefix, or from ``meta``; NJS, offset, label and lightshow are copied from
        ``meta`` (they are not tokenized)."""
        if windows and windows[0].prefix:
            cond = Conditioning.from_prefix(windows[0].prefix, self.vocab)
            characteristic, difficulty = cond.characteristic, cond.difficulty
        elif meta is not None:
            characteristic, difficulty = meta.characteristic, meta.difficulty
        else:
            raise ValueError("windows have no prefix; pass meta for characteristic/difficulty")
        dmap = events_to_map(parse_windows(windows, self.grammar), characteristic, difficulty)
        if meta is not None:
            dmap.njs, dmap.njs_offset = meta.njs, meta.njs_offset
            dmap.label, dmap.lightshow = meta.label, meta.lightshow
        return dmap

    def decode_ids(
        self, sequences: Sequence[Sequence[int]], meta: DifficultyMap | None = None
    ) -> DifficultyMap:
        """``decode`` for full ``input_ids()`` sequences, one per consecutive window."""
        windows = [TokenWindow.from_ids(ids, k, self.prefix_len) for k, ids in enumerate(sequences)]
        return self.decode(windows, meta)
