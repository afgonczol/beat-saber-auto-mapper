"""``Lightshow`` <-> windowed tokens in the separate lighting vocabulary.

Events (after ``TIME_t``)::

    basic event   EVT_type VAL_value [FLOAT_k] [COLOR_i]
    colour boost  BOOST_ON | BOOST_OFF

``FLOAT_k`` (float_value = k/20) is left out when the float value is 1.0, and
``COLOR_i`` indexes ``vocab.PALETTE``; it is present only for Chroma-coloured
events. Version 1 lighting windows (note_v0, note_v1) have no conditioning prefix;
the lighting model gets its conditioning from the audio and the note track.

Version 2 (WS8b) adds a four-token prompt, ``LightPrompt``::

    BOS  LSTYLE_s LEVEL_l PAL_p FLIPS_f  [REF <ref> REF_END]  <context>  SEP  <body>  EOS

``LSTYLE`` is the map's lighting style (``lighting.styles``). The three scene tokens
describe the window: light events per beat (``LEVEL``), the colour mix of its lit
events (``PAL``) and the share of lit events that change their light's colour
(``FLIPS``, how calm the palette is). ``window_scene`` reads them
from a human window in training; at generation a section-level plan
(``lighting.scenes``) sets them. Each has a ``_NONE`` token for "no preference". The
optional reference block holds the light events of an earlier span whose music
matches (``data.structure``), with this vocabulary's ``REF`` / ``REF_END``.

Unlike gameplay objects, the order of light events on the same tick matters
(the last one wins), so events keep their original order within a tick (boosts
first). ``prepare_lightshow`` makes a lightshow representable; see its docstring
for what is lossy.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, replace

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
    BOS,
    EOS,
    LIGHT_EVENT_TYPES,
    LIGHT_FLIP_EDGES,
    LIGHT_FLOAT_MAX,
    LIGHT_LEVEL_EDGES,
    LIGHT_PALETTES,
    LIGHT_RED_EDGES,
    LIGHT_STYLE_SLOTS,
    LIGHT_VALUE_MAX,
    LIGHT_VOCAB,
    PALETTE,
    SEP,
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
LIGHT_REF = LIGHT_VOCAB.id("REF")
LIGHT_REF_END = LIGHT_VOCAB.id("REF_END")
LIGHT_PREFIX_LEN = 4
SCENE_LIGHT_TYPES = range(5)  # back, ring, left, right, centre lights
_PAL = {name: i for i, name in enumerate(LIGHT_PALETTES)}
WHITE_SHARE = 0.25  # this share of white among lit events makes a white palette


@dataclass(frozen=True, slots=True)
class LightPrompt:
    """The version 2 light prompt. None = no preference (the ``_NONE`` token)."""

    style: int | None = None  # index into the checkpoint's style names
    level: int | None = None  # LEVEL bucket of light events per beat
    palette: int | None = None  # index into ``vocab.LIGHT_PALETTES``
    flips: int | None = None  # FLIPS bucket of colour flips per lit event

    @property
    def scene(self) -> tuple[int | None, int | None, int | None]:
        return self.level, self.palette, self.flips

    def with_scene(self, scene: tuple[int | None, int | None, int | None]) -> LightPrompt:
        return replace(self, level=scene[0], palette=scene[1], flips=scene[2])

    def drop_style(self) -> LightPrompt:
        return replace(self, style=None)

    def drop_scene(self) -> LightPrompt:
        return replace(self, level=None, palette=None, flips=None)

    def prefix_ids(self, vocab: Vocab = LIGHT_VOCAB) -> list[int]:
        if self.style is not None and not 0 <= self.style < LIGHT_STYLE_SLOTS:
            raise ValueError(f"light style {self.style} is outside 0..{LIGHT_STYLE_SLOTS - 1}")
        return [
            vocab.id("LSTYLE", self.style),
            vocab.id("LEVEL", self.level),
            vocab.id("PAL", self.palette),
            vocab.id("FLIPS", self.flips),
        ]

    @classmethod
    def from_prefix(cls, ids: Sequence[int], vocab: Vocab = LIGHT_VOCAB) -> LightPrompt:
        groups = tuple(vocab.group(i) for i in ids)
        if groups != ("LSTYLE", "LEVEL", "PAL", "FLIPS"):
            raise ValueError(f"not a light prompt: {vocab.names(ids)}")
        return cls(*(vocab.value(i) for i in ids))


def level_bucket(events_per_beat: float) -> int:
    return bisect.bisect_right(LIGHT_LEVEL_EDGES, events_per_beat)


def flips_bucket(flip_share: float) -> int:
    return bisect.bisect_right(LIGHT_FLIP_EDGES, flip_share)


def palette_index(blue: int, red: int, white: int) -> int:
    """The palette of a window from its counts of lit blue, red and white events."""
    lit = blue + red + white
    if not lit:
        return _PAL["DARK"]
    if white >= WHITE_SHARE * lit:
        return _PAL["WHITE"]
    r = red / max(red + blue, 1)
    names = ("BLUE", "BLUE_LEAD", "EVEN", "RED_LEAD", "RED")
    return _PAL[names[bisect.bisect_right(LIGHT_RED_EDGES, r)]]


def window_scene(body: Sequence[int], beats: float = 16.0) -> tuple[int, int, int]:
    """(level, palette, flips) of a window body in the lighting vocabulary.

    Counts events of the five light types: all of them for the level; for the palette
    and flips only lit ones (values 1..12: blue, red, white; 0 is off). A flip is a lit
    event whose colour differs from the previous lit event of the same type; ``FLIPS``
    buckets flips per lit event.
    """
    events = LIGHT_GRAMMAR.parse(body)
    count, colours, flips = 0, [0, 0, 0], 0
    last: dict[int, int] = {}
    for e in events:
        if e.kind != "EVT" or e.head not in SCENE_LIGHT_TYPES:
            continue
        count += 1
        value = e.values[0]
        if not 1 <= value <= 12:
            continue
        colour = (value - 1) // 4  # 0 blue, 1 red, 2 white
        colours[colour] += 1
        if e.head in last and last[e.head] != colour:
            flips += 1
        last[e.head] = colour
    blue, red, white = colours
    return (
        level_bucket(count / beats),
        palette_index(blue, red, white),
        flips_bucket(flips / max(blue + red + white, 1)),
    )


def light_prompt_ids(
    prefix: Sequence[int], ref: Sequence[int] | None, context: Sequence[int]
) -> list[int]:
    """``BOS prefix [REF ref REF_END] context SEP`` in the lighting vocabulary."""
    block = [] if ref is None else [LIGHT_REF, *ref, LIGHT_REF_END]
    return [BOS, *prefix, *block, *context, SEP]


def light_input_ids(
    prefix: Sequence[int], ref: Sequence[int] | None, context: Sequence[int], body: Sequence[int]
) -> list[int]:
    return [*light_prompt_ids(prefix, ref, context), *body, EOS]


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
