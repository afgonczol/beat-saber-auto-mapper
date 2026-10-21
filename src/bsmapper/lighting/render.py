"""Renderers: a vanilla lightshow (from the rules or, later, the ML model) in each style.

- ``render_vanilla``: basic events as they are; valid in every legacy environment.
- ``render_v3``: event box groups for one environment profile, plus the basic events the
  environment still reacts to and colour boost.
- ``render_chroma``: basic events with RGB from the section palettes and gradients on
  the centre lights. Writers then list Chroma as a suggestion, never a requirement.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from bsmapper.lighting.events import (
    CENTER_LIGHTS,
    LEFT_LASER_SPEED,
    LIGHT_TYPES,
    RIGHT_LASER_SPEED,
    RING_SPIN,
    RING_ZOOM,
    VANILLA_TYPES,
    Action,
    LightColor,
    decode_value,
    light_value,
)
from bsmapper.lighting.palettes import RGB
from bsmapper.lighting.profiles import EnvironmentProfile, Motion
from bsmapper.lighting.rules import SectionPlan
from bsmapper.schema import BasicEvent, EventBoxGroup, Lightshow

# --- vanilla -----------------------------------------------------------------------


def render_vanilla(lightshow: Lightshow) -> Lightshow:
    """Plain basic events: drops any colour, custom data and non-vanilla event type."""
    events = [
        replace(e, color=None, custom_data=None)
        for e in lightshow.basic_events
        if e.type in VANILLA_TYPES
    ]
    return Lightshow(basic_events=events, boost_events=list(lightshow.boost_events))


# --- v3 group lighting ----------------------------------------------------------------

FLASH_PEAK = 1.2  # brightness multiplier at the start of a flash or fade
FLASH_DECAY = 0.25  # beats for a flash to settle
FADE_OUT = 1.0  # beats for a fade to reach black
SPIN_BEATS = 2.0
ZOOM_BEATS = 1.0
SPIN_ANGLES = (45.0, 0.0, 90.0, 0.0)
ZOOM_SPREAD_FACTOR = 3.0
EASE_IN_OUT_QUAD = 3
EASE_NONE = -1

# Index filter that selects every light of a group, in v3.3 shape.
ALL_LIGHTS = {"c": 0, "f": 1, "p": 1, "t": 0, "r": 0, "n": 0, "s": 0, "l": 0.0, "d": 0}


def _color_event(beat: float, transition: int, color: int, brightness: float) -> dict[str, Any]:
    return {
        "b": beat,
        "i": transition,
        "c": color,
        "s": round(brightness, 3),
        "f": 0,
        "sb": 0.0,
        "sf": 0,
    }


def color_events(value: int, brightness: float) -> list[dict[str, Any]]:
    """v3 colour events (relative beats) that reproduce one basic light value."""
    decoded = decode_value(value)
    if decoded is None:
        return [_color_event(0.0, 0, 0, 0.0)]
    color, action = decoded
    c = int(color)
    if action == Action.ON:
        return [_color_event(0.0, 0, c, brightness)]
    if action == Action.FLASH:
        return [
            _color_event(0.0, 0, c, brightness * FLASH_PEAK),
            _color_event(FLASH_DECAY, 1, c, brightness),
        ]
    if action == Action.FADE:
        return [
            _color_event(0.0, 0, c, brightness * FLASH_PEAK),
            _color_event(FADE_OUT, 1, c, 0.0),
        ]
    return [_color_event(0.0, 1, c, brightness)]  # transition


def color_box(events: list[dict[str, Any]], wave: float = 0.0) -> dict[str, Any]:
    return {
        "f": dict(ALL_LIGHTS),
        "w": wave,
        "d": 1,
        "r": 0.0,
        "t": 1,
        "b": 0,
        "i": 0,
        "e": events,
    }


def rotation_box(
    motion: Motion, events: list[dict[str, Any]], spread: float, mirror: bool = False
) -> dict[str, Any]:
    return {
        "f": dict(ALL_LIGHTS),
        "w": 0.0,
        "d": 1,
        "s": spread,
        "t": 1,
        "b": 0,
        "i": 0,
        "a": motion.axis,
        "r": int(mirror),
        "l": events,
    }


def _rotation_event(
    beat: float, extend: bool, angle: float, ease: int = EASE_NONE, loops: int = 0, direction=0
) -> dict[str, Any]:
    return {"b": beat, "p": int(extend), "e": ease, "l": loops, "r": angle, "o": direction}


def _turn(duration: float, angle: float, loops: int = 0, direction: int = 0) -> list[dict]:
    """Hold the previous angle, then ease to ``angle`` over ``duration`` beats."""
    return [
        _rotation_event(0.0, True, 0.0),
        _rotation_event(duration, False, angle, EASE_IN_OUT_QUAD, loops, direction),
    ]


class _Groups:
    """Collects boxes into one EventBoxGroup per (beat, kind, group id)."""

    def __init__(self) -> None:
        self.boxes: dict[tuple[float, str, int], list[dict[str, Any]]] = defaultdict(list)

    def add(self, beat: float, kind: str, group_id: int, box: dict[str, Any]) -> None:
        self.boxes[(beat, kind, group_id)].append(box)

    def build(self) -> list[EventBoxGroup]:
        return [
            EventBoxGroup(beat, gid, kind, tuple(boxes))  # type: ignore[arg-type]
            for (beat, kind, gid), boxes in sorted(self.boxes.items())
        ]


def _spins(events: list[BasicEvent], motion: Motion, zoom_native: bool, out: _Groups) -> None:
    """Ring spins turn the spin groups; ring zooms fan their lights out and back."""
    beats = sorted(
        {e.beat for e in events if e.type == RING_SPIN or (e.type == RING_ZOOM and not zoom_native)}
    )
    spin_beats = {e.beat for e in events if e.type == RING_SPIN}
    angle, spread, n_spins = 0.0, motion.spread, 0
    for beat in beats:
        zoom = any(e.beat == beat and e.type == RING_ZOOM for e in events) and not zoom_native
        if zoom:
            wide = motion.spread * ZOOM_SPREAD_FACTOR or 45.0
            spread = motion.spread if spread != motion.spread else wide
        if beat in spin_beats:
            angle = SPIN_ANGLES[n_spins % len(SPIN_ANGLES)]
            direction = 1 + n_spins % 2  # alternate clockwise / counter-clockwise
            rot = _turn(SPIN_BEATS, angle, loops=1, direction=direction)
            n_spins += 1
        else:
            rot = _turn(ZOOM_BEATS, angle)
        for gid in motion.groups:
            out.add(beat, "rotation", gid, rotation_box(motion, rot, spread))


def sweep_period(speed: int) -> float:
    return 4.0 if speed <= 2 else 2.0 if speed <= 5 else 1.0


def sweep_amplitude(speed: int) -> float:
    return min(15.0 + 5.0 * speed, 60.0)


def _sweeps(
    events: list[BasicEvent], type_: int, motion: Motion, mirror: bool, end: float, out: _Groups
) -> None:
    """Laser speed events become back-and-forth sweeps on the sweep groups."""
    speeds = sorted((e for e in events if e.type == type_), key=lambda e: e.beat)
    for i, event in enumerate(speeds):
        stop = speeds[i + 1].beat if i + 1 < len(speeds) else end
        if event.value <= 0:
            steps = [(event.beat, _turn(1.0, 0.0))]
        else:
            period, amp = sweep_period(event.value), sweep_amplitude(event.value)
            steps = []
            beat, k = event.beat, 0
            while beat < stop - 1e-9:
                angle = amp if k % 2 == 0 else 360.0 - amp  # automatic direction swings via 0
                steps.append((beat, _turn(min(period, stop - beat), angle)))
                beat, k = beat + period, k + 1
        for beat, rot in steps:
            for gid in motion.groups:
                out.add(beat, "rotation", gid, rotation_box(motion, rot, motion.spread, mirror))


def render_v3(lightshow: Lightshow, profile: EnvironmentProfile) -> Lightshow:
    """Event box groups for ``profile``'s environment.

    Light events become colour boxes on the channel's groups, ring spins/zooms and laser
    speeds become rotation boxes, and the basic events the environment still uses are
    kept as they are. Colour boost always passes through.
    """
    events = sorted(lightshow.basic_events, key=lambda e: (e.beat, e.type))
    out = _Groups()
    for e in events:
        channel = profile.channels.get(e.type) if e.type in LIGHT_TYPES else None
        if channel is None:
            continue
        for gid in channel.groups:
            out.add(
                e.beat, "color", gid, color_box(color_events(e.value, e.float_value), channel.wave)
            )

    end = max((e.beat for e in events), default=0.0)
    if profile.spin is not None:
        _spins(events, profile.spin, RING_ZOOM in profile.basic_types, out)
    if profile.left_sweep is not None:
        _sweeps(events, LEFT_LASER_SPEED, profile.left_sweep, False, end, out)
    if profile.right_sweep is not None:
        _sweeps(events, RIGHT_LASER_SPEED, profile.right_sweep, True, end, out)

    basic = [
        replace(e, color=None, custom_data=None) for e in events if e.type in profile.basic_types
    ]
    return Lightshow(
        basic_events=basic,
        boost_events=list(lightshow.boost_events),
        event_box_groups=out.build(),
    )


# --- Chroma ------------------------------------------------------------------------

GRADIENT_MIN_BEATS = 4.0  # centre-light spans at least this long get a gradient
GRADIENT_LEAD = 0.25  # the gradient ends this many beats before the next event


def _rgba(rgb: RGB) -> tuple[float, float, float, float]:
    return (round(rgb[0], 3), round(rgb[1], 3), round(rgb[2], 3), 1.0)


def _section_at(sections: Sequence[SectionPlan], starts: list[float], beat: float) -> SectionPlan:
    return sections[max(bisect.bisect_right(starts, beat) - 1, 0)]


def _palette_color(section: SectionPlan, color: LightColor) -> RGB:
    palette = section.palette
    return {LightColor.RED: palette.left, LightColor.BLUE: palette.right}.get(color, palette.accent)


def render_chroma(lightshow: Lightshow, sections: Sequence[SectionPlan]) -> Lightshow:
    """Vanilla events coloured from the section palettes, with centre-light gradients.

    A gradient is a Chroma-coloured transition event: the light glides from the previous
    event's colour to the transition's colour over the gap between them. Every centre
    light span of at least ``GRADIENT_MIN_BEATS`` glides to the section's other colour.
    """
    base = render_vanilla(lightshow)
    if not sections:
        return base
    sections = sorted(sections, key=lambda s: s.start)
    starts = [s.start for s in sections]

    coloured: list[BasicEvent] = []
    for e in base.basic_events:
        decoded = decode_value(e.value) if e.type in LIGHT_TYPES else None
        if decoded is None:
            coloured.append(e)
            continue
        section = _section_at(sections, starts, e.beat)
        coloured.append(replace(e, color=_rgba(_palette_color(section, decoded[0]))))

    center = [e for e in coloured if e.type == CENTER_LIGHTS]
    for cur, nxt in zip(center, center[1:], strict=False):
        decoded = decode_value(cur.value)
        if decoded is None or decoded[1] == Action.FADE:
            continue
        if nxt.beat - cur.beat < GRADIENT_MIN_BEATS:
            continue
        section = _section_at(sections, starts, cur.beat)
        other = LightColor.BLUE if decoded[0] == LightColor.RED else LightColor.RED
        coloured.append(
            BasicEvent(
                beat=nxt.beat - GRADIENT_LEAD,
                type=CENTER_LIGHTS,
                value=light_value(other, Action.TRANSITION),
                float_value=cur.float_value,
                color=_rgba(_palette_color(section, other)),
            )
        )

    coloured.sort(key=lambda e: (e.beat, e.type))
    return Lightshow(basic_events=coloured, boost_events=base.boost_events, uses_chroma=True)
