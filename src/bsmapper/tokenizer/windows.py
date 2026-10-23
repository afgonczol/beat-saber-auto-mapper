"""16-beat windows with the previous window's tail as context.

A model input for window k is::

    BOS  <prefix>  [REF <ref> REF_END]  <context>  SEP  <body>  EOS

``prefix`` is the conditioning (empty for lighting), ``context`` is the last
``context_events`` objects of window k-1 with TIME tokens relative to window
k-1, and ``body`` is window k's events with TIME tokens relative to window k.
Objects belong to the window their start time falls in; spans (walls, arc and
chain tails) may reach into later windows.

The optional reference block (tokenizer v2, note_v1) holds the events of an
earlier 16-beat span whose music matches window k (``data.structure``), with TIME
tokens relative to the span's start (``span_tokens``). The model learns from human
maps how much of it to copy, mirror or vary. A window without a reference has no
block at all; ``REF REF_END`` means the matching span was empty.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from bsmapper.tokenizer.grammar import EncodedEvent, Grammar, ParsedEvent, serialize
from bsmapper.tokenizer.grid import TICKS_PER_BEAT, WINDOW_TICKS
from bsmapper.tokenizer.vocab import BOS, EOS, NOTE_VOCAB, SEP

REF = NOTE_VOCAB.id("REF")
REF_END = NOTE_VOCAB.id("REF_END")
DEFAULT_MAX_REF_TOKENS = 256

DEFAULT_CONTEXT_EVENTS = 16


@dataclass(slots=True)
class TokenWindow:
    index: int
    body: list[int]
    context: list[int] = field(default_factory=list)
    prefix: list[int] = field(default_factory=list)
    ref: list[int] | None = None  # reference events (without REF / REF_END), None = none

    @property
    def start_tick(self) -> int:
        return self.index * WINDOW_TICKS

    @property
    def start_beat(self) -> float:
        return self.start_tick / TICKS_PER_BEAT

    @property
    def body_offset(self) -> int:
        """Position of the first body token in ``input_ids()``."""
        return 1 + len(self.prefix) + len(self.ref_block()) + len(self.context) + 1

    def ref_block(self) -> list[int]:
        return [] if self.ref is None else [REF, *self.ref, REF_END]

    def input_ids(self) -> list[int]:
        return [BOS, *self.prefix, *self.ref_block(), *self.context, SEP, *self.body, EOS]

    @classmethod
    def from_ids(cls, ids: Sequence[int], index: int, prefix_len: int) -> TokenWindow:
        """Split a full ``input_ids()`` sequence back into its parts."""
        ids = list(ids)
        if not ids or ids[0] != BOS:
            raise ValueError("window sequence must start with BOS")
        prefix = ids[1 : 1 + prefix_len]
        start, ref = 1 + prefix_len, None
        if start < len(ids) and ids[start] == REF:
            end = ids.index(REF_END, start)
            ref, start = ids[start + 1 : end], end + 1
        sep = ids.index(SEP, start)
        body = ids[sep + 1 :]
        if body and body[-1] == EOS:
            body = body[:-1]
        return cls(index, body, ids[start:sep], prefix, ref)


def window_count(events: Sequence[EncodedEvent]) -> int:
    return events[-1][0] // WINDOW_TICKS + 1 if events else 0


def build_windows(
    events: Sequence[EncodedEvent],
    grammar: Grammar,
    prefix: Sequence[int] = (),
    n_windows: int | None = None,
    context_events: int = DEFAULT_CONTEXT_EVENTS,
) -> list[TokenWindow]:
    """Cut time-sorted events into windows. ``n_windows`` defaults to the last event's."""
    count = window_count(events) if n_windows is None else n_windows
    buckets: list[list[EncodedEvent]] = [[] for _ in range(count)]
    for event in events:
        k = event[0] // WINDOW_TICKS
        if k >= count:
            raise ValueError(f"event at tick {event[0]} is past window {count - 1}")
        buckets[k].append(event)
    vocab = grammar.vocab
    windows = []
    for k, bucket in enumerate(buckets):
        tail = buckets[k - 1][-context_events:] if k and context_events > 0 else []
        windows.append(
            TokenWindow(
                index=k,
                body=serialize(bucket, k * WINDOW_TICKS, vocab),
                context=serialize(tail, (k - 1) * WINDOW_TICKS, vocab),
                prefix=list(prefix),
            )
        )
    return windows


def tail_context(
    body: Sequence[int], grammar: Grammar, context_events: int = DEFAULT_CONTEXT_EVENTS
) -> list[int]:
    """Context tokens for the next window, from this window's body (generated or not)."""
    if context_events <= 0:
        return []
    events = grammar.parse(body)[-context_events:]
    return serialize(((e.tick, e.ids) for e in events), 0, grammar.vocab)


def span_tokens(
    bodies: Sequence[Sequence[int]],
    start_tick: int,
    grammar: Grammar,
    max_tokens: int = DEFAULT_MAX_REF_TOKENS,
) -> list[int]:
    """Events starting in [start_tick, start_tick + 16 beats) of consecutive window
    ``bodies`` (window 0 first), with TIME relative to ``start_tick``.

    Used for reference blocks, from human windows in training and from generated
    windows at sampling time. Cut at an event boundary to at most ``max_tokens``.
    """
    lo, hi = start_tick, start_tick + WINDOW_TICKS
    events: list[EncodedEvent] = []
    for k in range(max(lo // WINDOW_TICKS, 0), min(hi // WINDOW_TICKS + 1, len(bodies))):
        origin = k * WINDOW_TICKS
        for e in grammar.parse(bodies[k]):
            if lo <= origin + e.tick < hi:
                events.append((origin + e.tick, e.ids))
    ids = serialize(events, lo, grammar.vocab)
    if len(ids) <= max_tokens:
        return ids
    out: list[int] = []
    last = None
    for tick, tokens in events:
        piece = tokens if tick == last else (grammar.vocab.id("TIME", tick - lo), *tokens)
        if len(out) + len(piece) > max_tokens:
            break
        out.extend(piece)
        last = tick
    return out


def parse_windows(
    windows: Iterable[TokenWindow], grammar: Grammar
) -> list[tuple[int, ParsedEvent]]:
    """(absolute tick, event) for every body event, in window order."""
    out = []
    for window in windows:
        for event in grammar.parse(window.body):
            out.append((window.start_tick + event.tick, event))
    return out
