"""Basic (v2-style) light event ids and values, the lighting package's semantic channels.

The rule baseline (and later the ML model) writes vanilla basic events. Renderers read
these channels and turn them into v3 event box groups or Chroma colours.
"""

from __future__ import annotations

from enum import IntEnum

# Event types (``BasicEvent.type``).
BACK_LASERS = 0
RING_LIGHTS = 1
LEFT_LASERS = 2
RIGHT_LASERS = 3
CENTER_LIGHTS = 4
RING_SPIN = 8
RING_ZOOM = 9
LEFT_LASER_SPEED = 12
RIGHT_LASER_SPEED = 13

LIGHT_TYPES = (BACK_LASERS, RING_LIGHTS, LEFT_LASERS, RIGHT_LASERS, CENTER_LIGHTS)
LASER_SPEED_TYPES = (LEFT_LASER_SPEED, RIGHT_LASER_SPEED)
VANILLA_TYPES = frozenset((*LIGHT_TYPES, RING_SPIN, RING_ZOOM, *LASER_SPEED_TYPES))

OFF = 0


class LightColor(IntEnum):
    """Same numbering as v3 colour events: the environment's primary, secondary and white."""

    RED = 0  # primary / left colour
    BLUE = 1  # secondary / right colour
    WHITE = 2


class Action(IntEnum):
    ON = 0
    FLASH = 1  # brighter for a moment, then stays on
    FADE = 2  # brighter for a moment, then fades out
    TRANSITION = 3  # glide from the previous state to this one


# value = base + action; 0 is off.
_BASE = {LightColor.BLUE: 1, LightColor.RED: 5, LightColor.WHITE: 9}


def light_value(color: LightColor, action: Action) -> int:
    return _BASE[color] + int(action)


def decode_value(value: int) -> tuple[LightColor, Action] | None:
    """(colour, action) of a light event value, or None for off and unknown values."""
    for color, base in _BASE.items():
        if base <= value < base + 4:
            return color, Action(value - base)
    return None
