"""Event grammar shared by the note and lighting tokenizers.

A window body is ``(TIME_t | object)* EOS``. ``TIME_t`` sets the current tick
(strictly increasing inside a window) and every object that follows sits on it
until the next ``TIME``. An object is a head token followed by fixed slots and
then optional trailing modifiers, as described by an ``ObjectSpec``. The slot
``"DUR"`` is a span: ``DUR_16* [DUR_b] DURT_r`` = 16*n + b beats + r ticks.

``Parser`` reads tokens one at a time. It is what decoding uses, and its
``allowed_mask`` gives the grammatically valid next tokens, so generation (WS7)
can mask the model's logits and the playability rules (WS5) can inspect the
partial state (``time``, ``events``, ``partial``) before choosing their own masks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bsmapper.tokenizer.grid import TICKS_PER_BEAT
from bsmapper.tokenizer.vocab import DUR_MAX_BEATS, EOS, Vocab

DUR = "DUR"  # pseudo slot: a span in ticks


class TokenGrammarError(ValueError):
    """A token sequence does not follow the event grammar."""


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    head: str  # group of the head token
    slots: tuple[str, ...]  # required slot groups, in order ("DUR" = span)
    optional: tuple[str, ...] = ()  # trailing optional groups, in order


@dataclass(frozen=True, slots=True)
class ParsedEvent:
    tick: int  # relative to the window start
    kind: str  # head group
    head: Any  # value of the head token
    values: tuple[Any, ...]  # slot values, then optional values (None when absent)
    ids: tuple[int, ...]  # the object's tokens, without TIME


# A serialized object: (tick, its token ids starting with the head, without TIME).
EncodedEvent = tuple[int, tuple[int, ...]]


def span_ids(vocab: Vocab, ticks: int) -> list[int]:
    """Canonical tokens for a non-negative span: DUR_16 * n, DUR_b, DURT_r."""
    if ticks < 0:
        raise ValueError(f"negative span {ticks}")
    beats, rest = divmod(ticks, TICKS_PER_BEAT)
    ids = [vocab.id("DUR", DUR_MAX_BEATS)] * (beats // DUR_MAX_BEATS)
    if beats % DUR_MAX_BEATS:
        ids.append(vocab.id("DUR", beats % DUR_MAX_BEATS))
    ids.append(vocab.id("DURT", rest))
    return ids


def serialize(events: Iterable[EncodedEvent], origin: int, vocab: Vocab) -> list[int]:
    """Token ids for time-sorted events, with a TIME token whenever the tick changes."""
    ids: list[int] = []
    last = None
    for tick, tokens in events:
        rel = tick - origin
        if rel != last:
            if last is not None and rel < last:
                raise ValueError("events are not sorted by time")
            ids.append(vocab.id("TIME", rel))
            last = rel
        ids.extend(tokens)
    return ids


class Grammar:
    def __init__(self, vocab: Vocab, specs: Sequence[ObjectSpec]) -> None:
        self.vocab = vocab
        self.specs = {spec.head: spec for spec in specs}
        self.time_ids = np.array(vocab.ids("TIME"))
        self.time_values = np.array([vocab.value(i) for i in self.time_ids])
        self._group_masks = {g: vocab.mask(vocab.ids(g)) for g in vocab.groups()}
        self.head_mask = np.zeros(len(vocab), dtype=bool)
        for head in self.specs:
            self.head_mask |= self._group_masks[head]
        self._dur_max_id = vocab.id("DUR", DUR_MAX_BEATS) if vocab.has("DUR", 1) else -1

    def group_mask(self, group: str) -> np.ndarray:
        return self._group_masks[group]

    def parser(self, terminator: int = EOS) -> Parser:
        return Parser(self, terminator)

    def parse(self, ids: Iterable[int], terminator: int = EOS) -> list[ParsedEvent]:
        """Parse a whole body. The terminator is optional at the end."""
        parser = self.parser(terminator)
        for token_id in ids:
            parser.feed(token_id)
        return parser.finish()


class Parser:
    """Incremental parser for one window body (or one tail context, ended by SEP)."""

    def __init__(self, grammar: Grammar, terminator: int = EOS) -> None:
        self.g = grammar
        self.vocab = grammar.vocab
        self.terminator = terminator
        self.time: int | None = None  # current tick, None before the first TIME
        self.events: list[ParsedEvent] = []
        self.done = False
        self._spec: ObjectSpec | None = None
        self._head: Any = None
        self._values: list[Any] = []
        self._ids: list[int] = []
        self._slot = 0  # next required slot; == len(slots) once the object is complete
        self._opt = 0  # next optional slot that may still follow
        self._span = 0
        self._span_more = True  # DUR may still follow inside the current span

    # --- state for callers -----------------------------------------------------

    @property
    def in_object(self) -> bool:
        """True while a started object still needs required slots."""
        return self._spec is not None and self._slot < len(self._spec.slots)

    @property
    def partial(self) -> tuple[str, Any, tuple[Any, ...]] | None:
        """(kind, head value, values so far) of the object being read, if any.

        A complete object that may still take an optional modifier is reported
        here too; it moves to ``events`` once the next object, TIME or end arrives.
        """
        if self._spec is None:
            return None
        return self._spec.head, self._head, tuple(self._values)

    # --- feeding ---------------------------------------------------------------

    def feed(self, token_id: int) -> list[ParsedEvent]:
        """Consume one token. Returns the events it completed (usually none or one)."""
        if self.done:
            raise TokenGrammarError("token after the end of the window")
        vocab = self.vocab
        group = vocab.group(token_id)
        value = vocab.value(token_id)
        spec = self._spec

        if spec is not None and self._slot < len(spec.slots):
            self._fill_slot(spec, token_id, group, value)
            if self._slot == len(spec.slots) and not spec.optional:
                return [self._emit()]
            return []

        if spec is not None and group in spec.optional[self._opt :]:
            k = spec.optional.index(group, self._opt)
            self._values[len(spec.slots) + k] = value
            self._ids.append(token_id)
            self._opt = k + 1
            return [self._emit()] if self._opt == len(spec.optional) else []

        completed = [self._emit()] if spec is not None else []
        if token_id == self.terminator:
            self.done = True
        elif group == "TIME":
            if self.time is not None and value <= self.time:
                raise TokenGrammarError(f"TIME {value} does not advance past {self.time}")
            self.time = value
        elif group in self.g.specs:
            if self.time is None:
                raise TokenGrammarError(f"{vocab.name(token_id)} before any TIME token")
            new = self.g.specs[group]
            self._start(new, token_id, value)
            if not new.slots and not new.optional:
                completed.append(self._emit())
        else:
            raise TokenGrammarError(f"unexpected {vocab.name(token_id)}")
        return completed

    def finish(self) -> list[ParsedEvent]:
        """End of input: flush a pending object and return every event."""
        if self.in_object:
            raise TokenGrammarError("token sequence ends inside an object")
        if self._spec is not None:
            self._emit()
        self.done = True
        return self.events

    def allowed_mask(self) -> np.ndarray:
        """Boolean mask over the vocabulary of tokens that may come next."""
        g = self.g
        mask = np.zeros(len(self.vocab), dtype=bool)
        if self.done:
            return mask
        spec = self._spec
        if spec is not None and self._slot < len(spec.slots):
            slot = spec.slots[self._slot]
            if slot == DUR:
                mask |= g.group_mask("DURT")
                if self._span_more:
                    mask |= g.group_mask("DUR")
            else:
                mask |= g.group_mask(slot)
            return mask
        if spec is not None:
            for group in spec.optional[self._opt :]:
                mask |= g.group_mask(group)
        later = g.time_ids if self.time is None else g.time_ids[g.time_values > self.time]
        mask[later] = True
        if self.time is not None:
            mask |= g.head_mask
        mask[self.terminator] = True
        return mask

    # --- internals -------------------------------------------------------------

    def _start(self, spec: ObjectSpec, token_id: int, value: Any) -> None:
        self._spec, self._head = spec, value
        self._values = []
        self._ids = [token_id]
        self._slot = self._opt = 0
        self._span, self._span_more = 0, True
        if not spec.slots:
            self._values.extend([None] * len(spec.optional))

    def _fill_slot(self, spec: ObjectSpec, token_id: int, group: str, value: Any) -> None:
        slot = spec.slots[self._slot]
        if slot == DUR and group == "DUR" and self._span_more:
            self._span += value * TICKS_PER_BEAT
            self._span_more = token_id == self.g._dur_max_id
        elif slot == DUR and group == "DURT":
            self._values.append(self._span + value)
            self._span, self._span_more = 0, True
            self._slot += 1
        elif slot == group:
            self._values.append(value)
            self._slot += 1
        else:
            expected = "a duration" if slot == DUR else slot
            raise TokenGrammarError(
                f"{spec.head} expects {expected}, got {self.vocab.name(token_id)}"
            )
        self._ids.append(token_id)
        if self._slot == len(spec.slots):
            self._values.extend([None] * len(spec.optional))

    def _emit(self) -> ParsedEvent:
        spec = self._spec
        assert spec is not None and self.time is not None
        event = ParsedEvent(self.time, spec.head, self._head, tuple(self._values), tuple(self._ids))
        self.events.append(event)
        self._spec = None
        return event
