"""Readability metrics for a vanilla lightshow: how often and how fast the lights change.

A light channel (basic event types 0-4) has a visible state: dark, or lit in a colour.
An event is a *visible change* when it alters that state or pulses the light (flash or
fade). Repeating the current state (an "on red" while already on red) is invisible and
not counted, and of several events on one channel at one tick only the last one counts,
as in the game.

``span_stats`` measures one span of a lightshow (a section, a 16-beat window or the whole
song), in seconds, so songs of different tempo compare:

- ``changes_per_s``: visible changes per second, summed over the five channels;
- ``color_switches_per_s``: a light coming on (or pulsing) in a different colour from
  its previous lit event, fades included: a red fade one beat after a blue fade is still
  read as a colour flip;
- ``on_off_per_s``: dark-to-lit and lit-to-dark toggles (a fade counts as both);
- ``active_ticks_per_s``: distinct moments at which any light changes;
- ``short_hold_share``: share of a channel's holds (time to its next change) shorter
  than ``SHORT_HOLD_S``; ``strobe_per_s``: changes after a hold under ``STROBE_S``;
- ``hold_p10_s`` / ``hold_median_s``: hold time quantiles over all channels;
- ``flash_share`` / ``fade_share`` / ``on_share`` / ``transition_share`` / ``off_share``;
- ``spin_per_s`` / ``zoom_per_s`` / ``laser_speed_per_s``: ring and laser motion events.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np

from bsmapper.lighting.events import (
    LASER_SPEED_TYPES,
    LIGHT_TYPES,
    RING_SPIN,
    RING_ZOOM,
    Action,
    decode_value,
)
from bsmapper.schema import BasicEvent, Lightshow

SHORT_HOLD_S = 0.15  # holds shorter than this read as flicker
STROBE_S = 0.1  # 10 Hz and faster reads as strobe
DARK = 0.05  # brightness (float value) at or below which a light counts as off
BRIGHTNESS_STEP = 0.3  # a brightness change this big on a lit light is visible

BeatToSeconds = Callable[[float], float]


def constant_tempo(bpm: float) -> BeatToSeconds:
    spb = 60.0 / bpm
    return lambda beat: beat * spb


@dataclass(frozen=True, slots=True)
class Change:
    """A visible change on one light channel."""

    beat: float
    type: int
    lit: bool  # state right after the change (a fade counts as lit, then dark)
    color: int | None
    action: Action | None  # None for off


@dataclass(slots=True)
class SpanStats:
    seconds: float
    changes_per_s: float = 0.0
    color_switches_per_s: float = 0.0
    on_off_per_s: float = 0.0
    active_ticks_per_s: float = 0.0
    short_hold_share: float = 0.0
    strobe_per_s: float = 0.0
    hold_p10_s: float = float("nan")
    hold_median_s: float = float("nan")
    flash_share: float = 0.0
    fade_share: float = 0.0
    on_share: float = 0.0
    transition_share: float = 0.0
    off_share: float = 0.0
    spin_per_s: float = 0.0
    zoom_per_s: float = 0.0
    laser_speed_per_s: float = 0.0
    channel_changes_per_s: tuple[float, ...] = (0.0,) * len(LIGHT_TYPES)

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        per_channel = d.pop("channel_changes_per_s")
        for t, v in zip(LIGHT_TYPES, per_channel, strict=True):
            d[f"ch{t}_per_s"] = v
        return d


def _last_per_tick(events: Iterable[BasicEvent]) -> list[BasicEvent]:
    """One event per (type, beat): the last one, as the game shows it."""
    last: dict[tuple[int, float], BasicEvent] = {}
    for e in events:
        last[(e.type, round(e.beat, 4))] = e
    return sorted(last.values(), key=lambda e: (e.beat, e.type))


def visible_changes(lightshow: Lightshow) -> dict[int, list[Change]]:
    """Visible changes per light channel, in time order.

    An event at (or gliding to) brightness ``DARK`` or less turns the light off. A
    brightness step of ``BRIGHTNESS_STEP`` or more on a lit light is a change.
    """
    out: dict[int, list[Change]] = {t: [] for t in LIGHT_TYPES}
    lights = _last_per_tick(e for e in lightshow.basic_events if e.type in LIGHT_TYPES)
    # channel -> (lit, colour, brightness)
    state: dict[int, tuple[bool, int | None, float]] = {t: (False, None, 0.0) for t in LIGHT_TYPES}
    for e in lights:
        decoded = decode_value(e.value)
        lit_now, color_now, bright_now = state[e.type]
        if decoded is None or (e.float_value <= DARK and decoded[1] != Action.FADE):
            if lit_now:
                out[e.type].append(Change(e.beat, e.type, False, None, None))
            state[e.type] = (False, None, 0.0)
            continue
        color, action = int(decoded[0]), decoded[1]
        pulse = action in (Action.FLASH, Action.FADE)
        step = abs(e.float_value - bright_now) >= BRIGHTNESS_STEP
        if pulse or not lit_now or color != color_now or step:
            out[e.type].append(Change(e.beat, e.type, True, color, action))
        state[e.type] = (action != Action.FADE, color, e.float_value)
    return out


def span_stats(
    lightshow: Lightshow,
    start: float,
    end: float,
    beat_to_s: BeatToSeconds,
    changes: dict[int, list[Change]] | None = None,
) -> SpanStats:
    """Readability statistics of the lightshow between beats ``start`` and ``end``."""
    seconds = max(beat_to_s(end) - beat_to_s(start), 1e-6)
    stats = SpanStats(seconds=seconds)
    changes = changes if changes is not None else visible_changes(lightshow)
    holds: list[float] = []
    actions: list[Action | None] = []
    ticks: set[float] = set()
    per_channel: list[float] = []
    n_color = n_onoff = n_strobe = 0
    for t in LIGHT_TYPES:
        seq = changes[t]
        inside = 0
        last_color: int | None = None  # colour of the channel's previous lit event
        for i, c in enumerate(seq):
            if c.beat >= end:
                break
            if c.beat < start:
                last_color = c.color if c.lit else last_color
                continue
            inside += 1
            actions.append(c.action)
            ticks.add(round(c.beat, 4))
            prev = seq[i - 1] if i else None
            if prev is not None:
                hold = beat_to_s(c.beat) - beat_to_s(prev.beat)
                holds.append(hold)
                n_strobe += hold < STROBE_S
            if c.lit:
                n_color += last_color is not None and c.color != last_color
                last_color = c.color
            prev_dark = prev is None or not prev.lit or prev.action == Action.FADE
            if c.lit == prev_dark:  # toggled between dark and lit
                n_onoff += 1
            if c.action == Action.FADE:
                n_onoff += 1  # and back to dark
        per_channel.append(inside / seconds)
    n = len(actions)
    stats.changes_per_s = n / seconds
    stats.channel_changes_per_s = tuple(per_channel)
    stats.color_switches_per_s = n_color / seconds
    stats.on_off_per_s = n_onoff / seconds
    stats.active_ticks_per_s = len(ticks) / seconds
    stats.strobe_per_s = n_strobe / seconds
    if holds:
        h = np.asarray(holds)
        stats.short_hold_share = float((h < SHORT_HOLD_S).mean())
        stats.hold_p10_s = float(np.quantile(h, 0.1))
        stats.hold_median_s = float(np.median(h))
    if n:
        stats.flash_share = actions.count(Action.FLASH) / n
        stats.fade_share = actions.count(Action.FADE) / n
        stats.on_share = actions.count(Action.ON) / n
        stats.transition_share = actions.count(Action.TRANSITION) / n
        stats.off_share = actions.count(None) / n

    motion = [e for e in lightshow.basic_events if start <= e.beat < end]
    stats.spin_per_s = sum(e.type == RING_SPIN for e in motion) / seconds
    stats.zoom_per_s = sum(e.type == RING_ZOOM for e in motion) / seconds
    speeds = _last_per_tick(e for e in motion if e.type in LASER_SPEED_TYPES)
    stats.laser_speed_per_s = len(speeds) / seconds
    return stats


def window_stats(
    lightshow: Lightshow, n_beats: float, beat_to_s: BeatToSeconds, window: float = 16.0
) -> list[SpanStats]:
    """``span_stats`` for consecutive windows of ``window`` beats."""
    changes = visible_changes(lightshow)
    n = int(np.ceil(n_beats / window))
    return [
        span_stats(lightshow, k * window, (k + 1) * window, beat_to_s, changes) for k in range(n)
    ]


def pattern_repeat(
    changes: dict[int, list[Change]], start: float, end: float, bar: float = 4.0
) -> float:
    """How much each bar repeats one of the two bars before it, 0..1.

    A bar is the set of (position in the bar, channel, colour) of its lit changes; the
    score of a bar is its best Jaccard overlap with the previous two bars, averaged over
    bars with lights. Human lightshows repeat patterns bar to bar; random ones do not.
    """
    bars: dict[int, set[tuple[int, int, int | None]]] = {}
    for t, seq in changes.items():
        for c in seq:
            if c.lit and start <= c.beat < end:
                k = int((c.beat - start) // bar)
                pos = round((c.beat - start - k * bar) * 24)
                bars.setdefault(k, set()).add((pos, t, c.color))
    scores = []
    for k, cur in bars.items():
        best = 0.0
        for back in (1, 2):
            prev = bars.get(k - back)
            if prev:
                best = max(best, len(cur & prev) / len(cur | prev))
        if k - 1 in bars or k - 2 in bars:
            scores.append(best)
    return float(np.mean(scores)) if scores else float("nan")


def together_share(changes: dict[int, list[Change]], start: float, end: float) -> float:
    """Share of light changes made at a moment when three or more channels change."""
    counts: dict[float, int] = {}
    for seq in changes.values():
        for c in seq:
            if start <= c.beat < end:
                key = round(c.beat, 4)
                counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    return sum(n for n in counts.values() if n >= 3) / total if total else float("nan")


# --- v3 motion ---------------------------------------------------------------------

WHIP_DEG_PER_S = 720.0  # two turns a second or faster reads as a whip


@dataclass(slots=True)
class MotionStats:
    """Rotation of v3 light groups between two beats.

    ``moves_per_s``: rotation targets per second, averaged over the groups that move;
    ``whips_per_s``: moves faster than ``WHIP_DEG_PER_S`` (per moving group, per second);
    ``max_deg_per_s``: the fastest move.
    """

    seconds: float
    moves_per_s: float = 0.0
    whips_per_s: float = 0.0
    max_deg_per_s: float = 0.0


def _angle_moves(lightshow: Lightshow) -> dict[int, list[tuple[float, float, float]]]:
    """Per rotation group: (start beat, end beat, degrees turned) of each move."""
    events: dict[int, list[tuple[float, dict]]] = {}
    for g in lightshow.event_box_groups:
        if g.kind != "rotation":
            continue
        for box in g.boxes:
            for e in box.get("l", ()):
                events.setdefault(g.group_id, []).append((g.beat + e.get("b", 0.0), e))
    moves: dict[int, list[tuple[float, float, float]]] = {}
    for gid, evs in events.items():
        evs.sort(key=lambda x: x[0])
        angle, last = 0.0, None
        out = moves.setdefault(gid, [])
        for beat, e in evs:
            if e.get("p", 0):  # extends the previous state: no movement
                last = beat
                continue
            target = float(e.get("r", 0.0))
            diff = abs((target - angle + 180.0) % 360.0 - 180.0) + 360.0 * e.get("l", 0)
            if last is not None and diff > 1.0:
                out.append((last, beat, diff))
            angle, last = target, beat
    return moves


def motion_stats(
    lightshow: Lightshow, start: float, end: float, beat_to_s: BeatToSeconds
) -> MotionStats:
    seconds = max(beat_to_s(end) - beat_to_s(start), 1e-6)
    stats = MotionStats(seconds=seconds)
    per_group = []
    whips = []
    for moves in _angle_moves(lightshow).values():
        inside = [(a, b, d) for a, b, d in moves if start <= b < end]
        if not inside:
            continue
        per_group.append(len(inside))
        speeds = [d / max(beat_to_s(b) - beat_to_s(a), 1e-3) for a, b, d in inside]
        whips.append(sum(v > WHIP_DEG_PER_S for v in speeds))
        stats.max_deg_per_s = max(stats.max_deg_per_s, max(speeds))
    if per_group:
        stats.moves_per_s = float(np.mean(per_group)) / seconds
        stats.whips_per_s = float(np.mean(whips)) / seconds
    return stats
