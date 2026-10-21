"""``Lightshow`` <-> windowed tokens in the separate lighting vocabulary.

Events (after ``TIME_t``)::

    basic event   EVT_type VAL_value [FLOAT_k] [COLOR_i]
    colour boost  BOOST_ON | BOOST_OFF

``FLOAT_k`` (float_value = k/20) is left out when the float value is 1.0, and
``COLOR_i`` indexes ``vocab.PALETTE``; it is present only for Chroma-coloured
events. Lighting windows have no conditioning prefix; the lighting model gets
its conditioning from the audio and the note track instead.

Unlike gameplay objects, the order of light events on the same tick matters
(the last one wins), so events keep their original order within a tick (boosts
first). ``prepare_lightshow`` makes a lightshow representable; see its docstring
for what is lossy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from bsmapper.schema import BasicEvent, ColorBoostEvent, Lightshow
from bsmapper.tokenizer.grammar import EncodedEvent, Grammar, ObjectSpec
from bsmapper.tokenizer.grid import (
    DEFAULT_TOLERANCE_TICKS,
    FRAC_STEPS,
    OffGridError,
    Snapper,
    SnapReport,
    grid_tick,
    tick_to_beat,
)
from bsmapper.tokenizer.vocab import (
    LIGHT_EVENT_TYPES,
    LIGHT_FLOAT_MAX,
    LIGHT_VALUE_MAX,
    LIGHT_VOCAB,
    PALETTE,
    Vocab,
)
from bsmapper.tokenizer.windows import (
    DEFAULT_CONTEXT_EVENTS,
    TokenWindow,
    build_windows,
    parse_windows,
)

LIGHT_SPECS = (
    ObjectSpec("BOOST", ()),
    ObjectSpec("EVT", ("VAL",), ("FLOAT", "COLOR")),
)
LIGHT_GRAMMAR = Grammar(LIGHT_VOCAB, LIGHT_SPECS)
_EVENT_TYPES = frozenset(LIGHT_EVENT_TYPES)


def nearest_palette_index(color: tuple[float, ...]) -> int:
    rgb = [min(max(c, 0.0), 1.0) for c in color[:3]]

    def distance(i: int) -> float:
        return sum((p - c) ** 2 for p, c in zip(PALETTE[i], rgb, strict=True))

    return min(range(len(PALETTE)), key=distance)


def palette_color(index: int) -> tuple[float, float, float, float]:
    return (*PALETTE[index], 1.0)


def prepare_lightshow(
    lightshow: Lightshow, tolerance_ticks: float = DEFAULT_TOLERANCE_TICKS
) -> tuple[Lightshow, SnapReport]:
    """Move a lightshow onto the lighting vocabulary.

    Lossy, by design: times snap to 1/24 beat; values clamp to 0..31; float values
    round to 0.05 and clamp to 0..2; Chroma colours snap to the nearest ``PALETTE``
    entry with alpha 1; ``custom_data`` (lightID filters, gradients ...) is dropped;
    event types outside ``LIGHT_EVENT_TYPES`` and events before beat 0 are dropped.
    v3 event box groups are not tokenized (WS8 renders them from basic events), so
    they are dropped and counted. ``uses_chroma`` becomes "any event has a colour".
    """
    s = Snapper(tolerance_ticks)
    basic: list[BasicEvent] = []
    for e in lightshow.basic_events:
        s.begin()
        beat = s.time(e.beat)
        s.end()
        if beat < 0:
            s.drop("before_start")
            continue
        if e.type not in _EVENT_TYPES:
            s.drop("event_type")
            continue
        value = max(0, min(LIGHT_VALUE_MAX, e.value))
        k = max(0, min(LIGHT_FLOAT_MAX, round(e.float_value * FRAC_STEPS)))
        color = None if e.color is None else palette_color(nearest_palette_index(e.color))
        s.quantized("value", e.value, value)
        s.quantized("float_value", e.float_value, k / FRAC_STEPS)
        if color != e.color:
            s.quantized("color", 0, 1)
        if e.custom_data:
            s.quantized("custom_data", 0, 1)
        basic.append(BasicEvent(beat, e.type, value, k / FRAC_STEPS, color, None))

    boosts: list[ColorBoostEvent] = []
    for b in lightshow.boost_events:
        s.begin()
        beat = s.time(b.beat)
        s.end()
        if beat < 0:
            s.drop("before_start")
        else:
            boosts.append(replace(b, beat=beat))

    if lightshow.event_box_groups:
        s.report.dropped["event_box_group"] += len(lightshow.event_box_groups)
    uses_chroma = any(e.color is not None for e in basic)
    return Lightshow(basic, boosts, [], uses_chroma), s.report


def light_events(lightshow: Lightshow, vocab: Vocab = LIGHT_VOCAB) -> list[EncodedEvent]:
    """Events sorted by tick; within a tick boosts first, then the original order."""
    events: list[EncodedEvent] = []
    for b in lightshow.boost_events:
        events.append((grid_tick(b.beat), (vocab.id("BOOST", b.on),)))
    for e in lightshow.basic_events:
        if not (vocab.has("EVT", e.type) and vocab.has("VAL", e.value)):
            raise OffGridError(f"{e} is outside the vocabulary; run prepare_lightshow first")
        ids = [vocab.id("EVT", e.type), vocab.id("VAL", e.value)]
        if e.float_value != 1.0:
            k = round(e.float_value * FRAC_STEPS)
            if k / FRAC_STEPS != e.float_value or not vocab.has("FLOAT", k):
                raise OffGridError(f"float value {e.float_value} is off the 0.05 grid")
            ids.append(vocab.id("FLOAT", k))
        if e.color is not None:
            index = nearest_palette_index(e.color)
            if palette_color(index) != tuple(e.color) or e.custom_data:
                raise OffGridError(f"{e} has a colour or custom data outside the vocabulary")
            ids.append(vocab.id("COLOR", index))
        elif e.custom_data:
            raise OffGridError(f"{e} has custom data; run prepare_lightshow first")
        events.append((grid_tick(e.beat), tuple(ids)))
    if any(tick < 0 for tick, _ in events):
        raise OffGridError("light event before beat 0")
    return sorted(events, key=lambda event: event[0])


class LightingTokenizer:
    """Encode a lightshow into 16-beat token windows and decode it back."""

    vocab = LIGHT_VOCAB
    grammar = LIGHT_GRAMMAR

    def __init__(self, context_events: int = DEFAULT_CONTEXT_EVENTS) -> None:
        self.context_events = context_events

    def encode(self, lightshow: Lightshow, n_windows: int | None = None) -> list[TokenWindow]:
        return build_windows(
            light_events(lightshow, self.vocab),
            self.grammar,
            n_windows=n_windows,
            context_events=self.context_events,
        )

    def decode(self, windows: Sequence[TokenWindow]) -> Lightshow:
        lightshow = Lightshow()
        for tick, e in parse_windows(windows, self.grammar):
            beat = tick_to_beat(tick)
            if e.kind == "BOOST":
                lightshow.boost_events.append(ColorBoostEvent(beat, e.head))
                continue
            value, k, color = e.values
            lightshow.basic_events.append(
                BasicEvent(
                    beat,
                    e.head,
                    value,
                    1.0 if k is None else k / FRAC_STEPS,
                    None if color is None else palette_color(color),
                )
            )
        lightshow.uses_chroma = any(e.color is not None for e in lightshow.basic_events)
        return lightshow

    def decode_ids(self, sequences: Sequence[Sequence[int]]) -> Lightshow:
        return self.decode([TokenWindow.from_ids(ids, k, 0) for k, ids in enumerate(sequences)])
