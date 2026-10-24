"""Readability limits for a vanilla lightshow: minimum holds, change-rate caps, anti-strobe.

The ML lightshow can change the scene faster than the eye can follow: lights flip
between red and blue every beat, laser speeds change every half beat, and ring spins
come in pairs a beat apart. Section energy sets how much change is allowed. Every limit
interpolates between a *calm* value (section intensity ``calm_intensity`` or below) and
a *peak* value (``peak_intensity`` or above).

Two ways to apply the limits, and the generator uses both:

- ``limit_lightshow`` is a post-filter on basic events. It keeps the moments that
  matter most (downbeats, then beats, then half beats, then the rest) and drops or
  recolours what breaks a limit. It works on any lightshow, rules or model.
- ``ReadabilityMask`` is a decode-time mask for ``LightSampler`` (a ``MaskFn``). It
  forbids an event on a channel that is still holding, and a colour flip inside the
  colour hold, so the model writes something else there instead of having it deleted
  afterwards. The post-filter still runs after it, for the moment cap (a token mask
  cannot see future moments) and for anything the sampler's fallback let through.

Limits, per section intensity (calm -> peak):

| limit | calm | peak |
|---|---|---|
| ``min_hold_s``: between two visible changes of one light channel | 0.25 s | 0.12 s |
| ``color_hold_beats``: between colour flips of one channel (1) | 4 beats | 2 beats |
| ``max_moments_per_s``: moments with any light change, in any 1 s | 4 | 8 |
| ``laser_speed_gap_beats``: between laser speed changes on one side | 2 beats | 1 beat |

(1) An earlier flip keeps the channel's colour: the hit stays, the flip goes.

Ring spins need ``spin_gap_beats`` (4) between them, zooms ``zoom_gap_beats`` (2), and
colour boost ``boost_gap_beats`` (4) between toggles. Left and right laser speed
changes closer than ``sync_lasers_beats`` move to the earlier one's beat, so both sides
change together.
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from bsmapper.lighting.events import (
    LASER_SPEED_TYPES,
    LEFT_LASER_SPEED,
    LIGHT_TYPES,
    RIGHT_LASER_SPEED,
    RING_SPIN,
    RING_ZOOM,
    VANILLA_TYPES,
    LightColor,
    decode_value,
    light_value,
)
from bsmapper.lighting.metrics import BeatToSeconds
from bsmapper.schema import BasicEvent, ColorBoostEvent, Lightshow

IntensityAt = Callable[[float], float]  # map beat -> section intensity 0..1
TICKS_PER_BEAT = 24
EPS = 1e-6


@dataclass(frozen=True, slots=True)
class ReadabilityConfig:
    # (calm, peak) pairs, interpolated by section intensity.
    min_hold_s: tuple[float, float] = (0.25, 0.12)
    color_hold_beats: tuple[float, float] = (4.0, 2.0)
    max_moments_per_s: tuple[float, float] = (4.0, 8.0)
    laser_speed_gap_beats: tuple[float, float] = (2.0, 1.0)
    # On v3 a spin is a full turn of the inner lights, far more motion than a vanilla
    # ring spin: at most one per bar.
    spin_gap_beats: float = 4.0
    zoom_gap_beats: float = 2.0
    boost_gap_beats: float = 4.0
    sync_lasers_beats: float = 0.5
    # Decode with ``ReadabilityMask``. Off by default: in a sweep over 4 seeds on Stronger
    # and Toxic no mask setting made the lights calmer than the post-filter alone, and
    # every one changed the sample (one blocked token flips the style for the rest of the
    # song, because each window continues from the tail of the one before); Toxic often
    # lost its lasers. With the mask on, it removes event types no renderer shows, plus:
    decode_mask: bool = False
    decode_light_holds: bool = False  # ``min_hold_s`` per light channel
    decode_motion_gaps: bool = False  # spin, zoom, laser speed and boost gaps
    calm_intensity: float = 0.3
    peak_intensity: float = 0.9
    # Dark-window fill: a window of ``fill_window_beats`` at section intensity
    # ``fill_min_intensity`` or more, with fewer than ``dark_changes_per_s`` visible light
    # changes, takes the fallback lightshow's light events (``limit_lightshow(fallback=)``).
    fill_window_beats: float = 16.0
    fill_min_intensity: float = 0.5
    dark_changes_per_s: float = 0.5

    def level(self, intensity: float) -> float:
        span = max(self.peak_intensity - self.calm_intensity, EPS)
        return float(np.clip((intensity - self.calm_intensity) / span, 0.0, 1.0))

    def at(self, pair: tuple[float, float], intensity: float) -> float:
        calm, peak = pair
        return calm + (peak - calm) * self.level(intensity)


@dataclass(slots=True)
class ReadabilityReport:
    dropped: Counter[str] = field(default_factory=Counter)
    recolored: int = 0
    moved: int = 0
    filled_windows: int = 0

    def total(self) -> int:
        return sum(self.dropped.values()) + self.recolored + self.moved + self.filled_windows

    def summary(self) -> str:
        dropped = ", ".join(f"{k} {v}" for k, v in sorted(self.dropped.items())) or "none"
        return (
            f"dropped {dropped}; recoloured {self.recolored}; moved {self.moved}; "
            f"filled {self.filled_windows} dark windows"
        )


def section_intensity(starts: Sequence[float], intensities: Sequence[float]) -> IntensityAt:
    """An ``IntensityAt`` from section start beats and their intensities."""
    starts, values = list(starts), list(intensities)
    if not starts:
        return lambda beat: 1.0

    def at(beat: float) -> float:
        return values[max(bisect.bisect_right(starts, beat + EPS) - 1, 0)]

    return at


def moment_priority(beat: float, beats_per_bar: int = 4) -> int:
    """0 downbeat, 1 beat, 2 half beat, 3 anything else."""
    tick = round(beat * TICKS_PER_BEAT)
    if tick % (TICKS_PER_BEAT * beats_per_bar) == 0:
        return 0
    if tick % TICKS_PER_BEAT == 0:
        return 1
    if tick % (TICKS_PER_BEAT // 2) == 0:
        return 2
    return 3


def _last_per_tick(events: Sequence[BasicEvent]) -> list[BasicEvent]:
    last: dict[tuple[int, int], BasicEvent] = {}
    for e in events:
        last[(e.type, round(e.beat * TICKS_PER_BEAT))] = e
    return sorted(last.values(), key=lambda e: (e.beat, e.type))


def _gap_ok(accepted: list[float], t: float, gap: float) -> bool:
    """True if ``t`` is at least ``gap`` from every time in sorted ``accepted``."""
    i = bisect.bisect_left(accepted, t)
    if i < len(accepted) and accepted[i] - t < gap - EPS:
        return False
    return not (i > 0 and t - accepted[i - 1] < gap - EPS)


def _window_ok(accepted: list[float], t: float, width: float, cap: float) -> bool:
    """True if adding ``t`` keeps every window of ``width`` at ``cap`` moments or fewer."""
    lo = bisect.bisect_left(accepted, t - width + EPS)
    hi = bisect.bisect_right(accepted, t + width - EPS)
    near = sorted([*accepted[lo:hi], t])
    # Only windows that contain t can change; each starts at a moment in (t - width, t].
    for start in (x for x in near if x <= t):
        if sum(start <= x < start + width - EPS for x in near) > cap:
            return False
    return True


def _limit_lights(
    events: list[BasicEvent],
    b2s: BeatToSeconds,
    level: IntensityAt,
    cfg: ReadabilityConfig,
    report: ReadabilityReport,
) -> list[BasicEvent]:
    """Holds and moment caps, keeping the most important moments first."""
    moments: dict[float, list[BasicEvent]] = {}
    for e in events:
        moments.setdefault(e.beat, []).append(e)
    order = sorted(moments, key=lambda b: (moment_priority(b), b))
    accepted_s: list[float] = []  # accepted moments, seconds
    channel_s: dict[int, list[float]] = {t: [] for t in LIGHT_TYPES}
    kept: list[BasicEvent] = []
    for beat in order:
        t = b2s(beat)
        intensity = level(beat)
        hold = cfg.at(cfg.min_hold_s, intensity)
        survivors = []
        for e in moments[beat]:
            if _gap_ok(channel_s[e.type], t, hold):
                survivors.append(e)
            else:
                report.dropped["hold"] += 1
        if not survivors:
            continue
        cap = cfg.at(cfg.max_moments_per_s, intensity)
        if not _window_ok(accepted_s, t, 1.0, cap):
            report.dropped["moment_cap"] += len(survivors)
            continue
        bisect.insort(accepted_s, t)
        for e in survivors:
            bisect.insort(channel_s[e.type], t)
        kept += survivors
    return kept


def _limit_colors(
    events: list[BasicEvent], level: IntensityAt, cfg: ReadabilityConfig, report: ReadabilityReport
) -> list[BasicEvent]:
    """A channel keeps its colour for ``color_hold_beats``; earlier flips are recoloured."""
    color: dict[int, LightColor] = {}
    since: dict[int, float] = {}
    out = []
    for e in events:
        decoded = decode_value(e.value)
        if decoded is None:
            out.append(e)
            continue
        c, action = decoded
        cur = color.get(e.type)
        hold = cfg.at(cfg.color_hold_beats, level(e.beat))
        if cur is not None and c != cur and e.beat - since[e.type] < hold - EPS:
            out.append(replace(e, value=light_value(cur, action), color=None))
            report.recolored += 1
            continue
        if cur is None or c != cur:
            color[e.type], since[e.type] = c, e.beat
        out.append(e)
    return out


def _limit_gap(
    events: list[BasicEvent], gap_at: Callable[[float], float], reason: str, report
) -> list[BasicEvent]:
    out, last = [], -1e9
    for e in events:
        if e.beat - last < gap_at(e.beat) - EPS:
            report.dropped[reason] += 1
            continue
        out.append(e)
        last = e.beat
    return out


def _sync_lasers(events: list[BasicEvent], within: float, report) -> list[BasicEvent]:
    """Move a laser speed change onto the other side's change when they are close."""
    by_side = {t: [e.beat for e in events if e.type == t] for t in LASER_SPEED_TYPES}
    out = []
    for e in events:
        other = RIGHT_LASER_SPEED if e.type == LEFT_LASER_SPEED else LEFT_LASER_SPEED
        near = [b for b in by_side[other] if 0 < e.beat - b < within - EPS]
        if near:
            out.append(replace(e, beat=max(near)))
            report.moved += 1
        else:
            out.append(e)
    return out


def _limit_boosts(
    boosts: list[ColorBoostEvent], gap: float, report: ReadabilityReport
) -> list[ColorBoostEvent]:
    """Boost toggles at least ``gap`` apart. A toggle back within the gap cancels the
    one before it, so a short blip of boost disappears instead of staying on."""
    out: list[ColorBoostEvent] = []
    state = False
    for b in sorted(boosts, key=lambda b: b.beat):
        if b.on == state:
            continue  # no change
        if out and b.beat - out[-1].beat < gap - EPS:
            out.pop()
            report.dropped["boost"] += 2
        else:
            out.append(b)
        state = out[-1].on if out else False
    return out


def fill_dark_windows(
    events: list[BasicEvent],
    fallback: Lightshow,
    beat_to_s: BeatToSeconds,
    intensity_at: IntensityAt,
    cfg: ReadabilityConfig,
    report: ReadabilityReport,
) -> list[BasicEvent]:
    """Give loud windows that are (nearly) dark the fallback's light events.

    The model sometimes starts a song dark and stays dark: an empty window gives the next
    one an empty context. Toxic's first 89 s had under 1.5 changes/s at full loudness.
    """
    from bsmapper.lighting.metrics import span_stats, visible_changes

    end = max((e.beat for e in events), default=0.0)
    end = max(end, max((e.beat for e in fallback.basic_events), default=0.0))
    width = cfg.fill_window_beats
    shown = Lightshow(events)
    changes = visible_changes(shown)
    spare = [e for e in fallback.basic_events if e.type in LIGHT_TYPES]
    dark: list[tuple[float, float]] = []
    start = 0.0
    while start <= end:
        stop = start + width
        if intensity_at(start + width / 2) >= cfg.fill_min_intensity:
            rate = span_stats(shown, start, stop, beat_to_s, changes).changes_per_s
            if rate < cfg.dark_changes_per_s and any(start <= e.beat < stop for e in spare):
                dark.append((start, stop))
        start = stop
    if not dark:
        return events
    report.filled_windows = len(dark)

    def in_dark(beat: float) -> bool:
        return any(a <= beat < b for a, b in dark)

    kept = [e for e in events if not (e.type in LIGHT_TYPES and in_dark(e.beat))]
    added = [e for e in spare if in_dark(e.beat)]
    return sorted(kept + added, key=lambda e: (e.beat, e.type))


def limit_lightshow(
    lightshow: Lightshow,
    beat_to_s: BeatToSeconds,
    intensity_at: IntensityAt,
    config: ReadabilityConfig | None = None,
    fallback: Lightshow | None = None,
) -> tuple[Lightshow, ReadabilityReport]:
    """The lightshow with the readability limits applied, and what was changed.

    With ``fallback`` (e.g. the L0 rule lightshow), loud windows the lightshow leaves
    dark first get the fallback's light events. Event box groups and other event types
    pass through unchanged.
    """
    cfg = config or ReadabilityConfig()
    report = ReadabilityReport()
    events = _last_per_tick(lightshow.basic_events)
    if fallback is not None:
        events = fill_dark_windows(events, fallback, beat_to_s, intensity_at, cfg, report)
    lights = [e for e in events if e.type in LIGHT_TYPES]
    lights = _limit_lights(lights, beat_to_s, intensity_at, cfg, report)
    lights = _limit_colors(
        sorted(lights, key=lambda e: (e.beat, e.type)), intensity_at, cfg, report
    )

    spins = _limit_gap(
        [e for e in events if e.type == RING_SPIN], lambda b: cfg.spin_gap_beats, "spin", report
    )
    zooms = _limit_gap(
        [e for e in events if e.type == RING_ZOOM], lambda b: cfg.zoom_gap_beats, "zoom", report
    )
    speeds = _sync_lasers(
        [e for e in events if e.type in LASER_SPEED_TYPES], cfg.sync_lasers_beats, report
    )
    speeds = _last_per_tick(speeds)
    limited_speeds: list[BasicEvent] = []
    for side in LASER_SPEED_TYPES:
        limited_speeds += _limit_gap(
            [e for e in speeds if e.type == side],
            lambda b: cfg.at(cfg.laser_speed_gap_beats, intensity_at(b)),
            "laser_speed",
            report,
        )
    handled = {*LIGHT_TYPES, RING_SPIN, RING_ZOOM, *LASER_SPEED_TYPES}
    others = [e for e in events if e.type not in handled]
    basic = sorted(lights + spins + zooms + limited_speeds + others, key=lambda e: (e.beat, e.type))
    boosts = _limit_boosts(list(lightshow.boost_events), cfg.boost_gap_beats, report)
    return (
        Lightshow(
            basic_events=basic,
            boost_events=boosts,
            event_box_groups=list(lightshow.event_box_groups),
            uses_chroma=lightshow.uses_chroma,
        ),
        report,
    )


# --- decode-time mask ------------------------------------------------------------


class ReadabilityMask:
    """A ``LightSampler`` mask (``models.MaskFn``) that applies the limits while decoding.

    Before an event head it forbids event types that no renderer shows (anything outside
    the vanilla types, e.g. the extra light channels 6/7/10/11 of a few legacy
    environments). With ``decode_light_holds`` it also forbids ``EVT_t`` for a light
    channel inside its minimum hold; with ``decode_motion_gaps``, a spin, zoom or laser
    speed change inside its gap and ``BOOST`` inside the boost gap. Opt-in: see
    ``ReadabilityConfig.decode_mask``.

    It deliberately leaves light *values* alone. Forcing a colour pushed note_v1 into a
    different style for the rest of the song (on/transition pulses instead of fades),
    because each window continues from the previous window's tail. Colour holds and the
    moment cap are ``limit_lightshow``'s job. The mask follows one sequence
    (LightSampler decodes one).
    """

    def __init__(
        self,
        beat_to_s: BeatToSeconds,
        intensity_at: IntensityAt,
        config: ReadabilityConfig | None = None,
    ) -> None:
        from bsmapper.tokenizer import LIGHT_GRAMMAR, LIGHT_VOCAB
        from bsmapper.tokenizer.vocab import LIGHT_EVENT_TYPES

        self.b2s, self.intensity_at = beat_to_s, intensity_at
        self.cfg = config or ReadabilityConfig()
        self.grammar = LIGHT_GRAMMAR
        self.size = len(LIGHT_VOCAB)
        self._evt = {t: LIGHT_VOCAB.id("EVT", t) for t in VANILLA_TYPES}
        self._unused = [
            LIGHT_VOCAB.id("EVT", t) for t in LIGHT_EVENT_TYPES if t not in VANILLA_TYPES
        ]
        self._boost = LIGHT_VOCAB.ids("BOOST")
        self.last: dict[int, float] = {}  # event type -> beat of its last event
        self.last_boost = -1e9
        self._window = -1
        self._body: list[int] = []
        self._seen = 0  # events of the current window already applied
        self.masked = 0  # steps at which the mask removed something

    def _apply(self, window: int, events) -> None:
        for e in events:
            beat = window * 16 + e.tick / TICKS_PER_BEAT
            if e.kind == "BOOST":
                self.last_boost = beat
            else:
                self.last[e.head] = beat

    def _sync(self, state) -> None:
        if state.window != self._window:
            if self._window >= 0:  # the previous window's last events, now complete
                done = self.grammar.parser()
                for token in self._body:
                    done.feed(token)
                self._apply(self._window, done.finish()[self._seen :])
            self._window, self._seen = state.window, 0
        self._body = state.body
        new = state.parser.events[self._seen :]
        self._apply(self._window, new)
        self._seen += len(new)

    def __call__(self, state) -> np.ndarray | None:
        self._sync(state)
        parser = state.parser
        if parser.time is None or parser.in_object:
            return None
        beat = state.window * 16 + parser.time / TICKS_PER_BEAT
        intensity = self.intensity_at(beat)
        cfg = self.cfg
        mask = np.ones(self.size, dtype=bool)
        mask[self._unused] = False
        self.masked += 1
        if cfg.decode_light_holds:
            t = self.b2s(beat)
            hold = cfg.at(cfg.min_hold_s, intensity)
            for typ in LIGHT_TYPES:
                if typ in self.last and t - self.b2s(self.last[typ]) < hold - EPS:
                    mask[self._evt[typ]] = False
        if not cfg.decode_motion_gaps:
            return mask
        laser_gap = cfg.at(cfg.laser_speed_gap_beats, intensity)
        gaps = {
            RING_SPIN: cfg.spin_gap_beats,
            RING_ZOOM: cfg.zoom_gap_beats,
            LEFT_LASER_SPEED: laser_gap,
            RIGHT_LASER_SPEED: laser_gap,
        }
        for typ, gap in gaps.items():
            if typ in self.last and beat - self.last[typ] < gap - EPS:
                mask[self._evt[typ]] = False
        if beat - self.last_boost < cfg.boost_gap_beats - EPS:
            mask[self._boost] = False
        return mask
