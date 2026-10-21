"""Cut-direction geometry shared by the rules.

Angles follow the game: degrees counter-clockwise (x right, y up) from a DOWN cut,
so DOWN = 0, RIGHT = 90, UP = 180, LEFT = -90. A v3 ``angle_offset`` adds on top.
"""

from __future__ import annotations

import math

from bsmapper.schema import CutDirection, Note

BASE_ANGLE: dict[CutDirection, float] = {
    CutDirection.DOWN: 0.0,
    CutDirection.DOWN_RIGHT: 45.0,
    CutDirection.RIGHT: 90.0,
    CutDirection.UP_RIGHT: 135.0,
    CutDirection.UP: 180.0,
    CutDirection.UP_LEFT: -135.0,
    CutDirection.LEFT: -90.0,
    CutDirection.DOWN_LEFT: -45.0,
}


def cut_angle(note: Note) -> float | None:
    """The note's cut angle, or None for a dot."""
    base = BASE_ANGLE.get(note.direction)
    return None if base is None else base + note.angle_offset


def angle_between(a: float, b: float) -> float:
    """Unsigned difference of two angles, in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def cut_vector(angle: float) -> tuple[float, float]:
    """Unit vector of the saber's motion for a cut angle (grid units, y up)."""
    r = math.radians(angle)
    return (math.sin(r), -math.cos(r))
