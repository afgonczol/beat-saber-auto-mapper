"""The 24-ticks-per-beat grid, value quantizers and ``snap_map``.

Everything the tokenizer emits sits on a grid: times on 1/24 beat, angle offsets
on 5 degrees, arc multipliers and chain squish on 0.05, rotations on 15 degrees.
``snap_map`` moves a parsed map onto those grids and reports what it had to
change, so the dataset builder (WS3b) can reject maps that are too far off.
After ``snap_map`` a map round-trips through the tokenizer exactly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from bsmapper.schema import (
    GRID_COLUMNS,
    GRID_ROWS,
    Arc,
    Bomb,
    Chain,
    DifficultyMap,
    Note,
    Obstacle,
    RotationEvent,
)

TICKS_PER_BEAT = 24  # LCM of 1/8 and 1/6 beat: covers 1/32 streams and triplets
WINDOW_BEATS = 16
WINDOW_TICKS = WINDOW_BEATS * TICKS_PER_BEAT  # 384

# Beats are stored with 6 decimals everywhere (see formats._common.BEAT_DECIMALS).
BEAT_DECIMALS = 6
# How far (in ticks) a time may sit from the grid and still count as "on grid".
# 6-decimal rounding of a grid time is off by at most 1.2e-5 ticks.
ON_GRID_EPS = 1e-3
# snap_map counts an object as off-grid when snapping moves it by more than this.
DEFAULT_TOLERANCE_TICKS = 0.1

# Value grids. Values outside a range are clamped, values between steps are rounded.
ANGLE_STEP = 5
ANGLE_MAX = 45  # angle offsets in -45..45, in steps of 5
FRAC_STEPS = 20  # arc multipliers and chain squish: k / 20
FRAC_MAX = 60  # ... for k in 0..60, i.e. 0.0..3.0
MAX_SLICES = 32  # chain slice counts in 1..32
ROTATION_STEP = 15
ROTATION_MAX = 180  # rotations in -180..180 (no 0), in steps of 15

# Wall shapes the vocabulary can express.
WALL_MAX_HEIGHT = 5


class OffGridError(ValueError):
    """A time or value is not on the tokenizer grid. Run ``snap_map`` first."""


def beat_to_tick(beat: float) -> int:
    return round(beat * TICKS_PER_BEAT)


def tick_to_beat(tick: int) -> float:
    return round(tick / TICKS_PER_BEAT, BEAT_DECIMALS) + 0.0


def snap_beat(beat: float) -> float:
    """The nearest grid time, written the way the readers write beats."""
    return tick_to_beat(beat_to_tick(beat))


def tick_error(beat: float) -> float:
    """Distance from ``beat`` to the nearest grid time, in ticks."""
    return abs(beat * TICKS_PER_BEAT - beat_to_tick(beat))


def is_on_grid(beat: float, eps_ticks: float = ON_GRID_EPS) -> bool:
    return tick_error(beat) <= eps_ticks


def grid_tick(beat: float) -> int:
    """``beat_to_tick`` that refuses times off the grid."""
    if not is_on_grid(beat):
        raise OffGridError(f"beat {beat} is {tick_error(beat):.4f} ticks off the 1/24 grid")
    return beat_to_tick(beat)


def quantize_angle(angle: int | float) -> int:
    snapped = round(angle / ANGLE_STEP) * ANGLE_STEP
    return int(max(-ANGLE_MAX, min(ANGLE_MAX, snapped)))


def quantize_frac(value: float) -> float:
    k = max(0, min(FRAC_MAX, round(value * FRAC_STEPS)))
    return k / FRAC_STEPS


def frac_index(value: float) -> int:
    k = round(value * FRAC_STEPS)
    if not 0 <= k <= FRAC_MAX or k / FRAC_STEPS != value:
        raise OffGridError(f"{value} is not a multiple of 1/{FRAC_STEPS} in 0..3")
    return k


def quantize_rotation(degrees: int | float) -> int:
    """Nearest multiple of 15, wrapped into -180..180 (0 means "no rotation")."""
    snapped = int(round(degrees / ROTATION_STEP) * ROTATION_STEP) % 360
    return snapped - 360 if snapped > ROTATION_MAX else snapped


@dataclass(slots=True)
class SnapReport:
    """What ``snap_map`` changed. ``off_grid`` counts objects moved by more than
    the tolerance; ``dropped`` and ``quantized`` count by reason."""

    objects: int = 0
    off_grid: int = 0
    max_error_ticks: float = 0.0
    dropped: Counter[str] = field(default_factory=Counter)
    quantized: Counter[str] = field(default_factory=Counter)

    @property
    def off_grid_ratio(self) -> float:
        return self.off_grid / self.objects if self.objects else 0.0


def _on_cell(x: int, y: int) -> bool:
    return 0 <= x < GRID_COLUMNS and 0 <= y < GRID_ROWS


class Snapper:
    """Snaps times for one object at a time and fills a SnapReport (shared with lighting)."""

    def __init__(self, tolerance_ticks: float) -> None:
        self.tolerance = tolerance_ticks
        self.report = SnapReport()
        self._worst = 0.0

    def begin(self) -> None:
        self.report.objects += 1
        self._worst = 0.0

    def time(self, beat: float) -> float:
        error = tick_error(beat)
        self._worst = max(self._worst, error)
        self.report.max_error_ticks = max(self.report.max_error_ticks, error)
        return snap_beat(beat)

    def end(self) -> None:
        if self._worst > self.tolerance:
            self.report.off_grid += 1

    def drop(self, reason: str) -> None:
        self.report.dropped[reason] += 1

    def quantized(self, name: str, before: float, after: float) -> None:
        if before != after:
            self.report.quantized[name] += 1


def snap_map(
    dmap: DifficultyMap, tolerance_ticks: float = DEFAULT_TOLERANCE_TICKS
) -> tuple[DifficultyMap, SnapReport]:
    """Snap every gameplay object onto the tokenizer grids.

    Times snap to the nearest 1/24 beat (durations snap via their end time).
    Values snap to their grids and are clamped to the vocabulary ranges.
    Objects the vocabulary cannot express at all are dropped: anything before
    beat 0, notes/bombs off the 4x3 grid, walls outside x 0..3 / width 1..4 /
    y 0..2 / height 1..5 or with negative duration, arcs/chains whose tail comes
    before their head, and rotations that snap to 0 degrees. Metadata and the
    lightshow pass through unchanged (see ``lighting.prepare_lightshow``).
    """
    s = Snapper(tolerance_ticks)

    notes: list[Note] = []
    for n in dmap.notes:
        s.begin()
        beat = s.time(n.beat)
        s.end()
        if beat < 0:
            s.drop("before_start")
        elif not _on_cell(n.x, n.y):
            s.drop("note_position")
        else:
            angle = quantize_angle(n.angle_offset)
            s.quantized("angle_offset", n.angle_offset, angle)
            notes.append(replace(n, beat=beat, angle_offset=angle))

    bombs: list[Bomb] = []
    for b in dmap.bombs:
        s.begin()
        beat = s.time(b.beat)
        s.end()
        if beat < 0:
            s.drop("before_start")
        elif not _on_cell(b.x, b.y):
            s.drop("bomb_position")
        else:
            bombs.append(replace(b, beat=beat))

    obstacles: list[Obstacle] = []
    for o in dmap.obstacles:
        s.begin()
        beat = s.time(o.beat)
        end = s.time(o.beat + o.duration)
        s.end()
        if beat < 0:
            s.drop("before_start")
        elif o.duration < 0:
            s.drop("wall_negative_duration")
        elif not (
            0 <= o.x < GRID_COLUMNS
            and 1 <= o.width <= GRID_COLUMNS
            and 0 <= o.y < GRID_ROWS
            and 1 <= o.height <= WALL_MAX_HEIGHT
        ):
            s.drop("wall_shape")
        else:
            duration = tick_to_beat(beat_to_tick(end) - beat_to_tick(beat))
            obstacles.append(replace(o, beat=beat, duration=duration))

    arcs: list[Arc] = []
    for a in dmap.arcs:
        s.begin()
        beat, tail = s.time(a.beat), s.time(a.tail_beat)
        s.end()
        if beat < 0:
            s.drop("before_start")
        elif tail < beat:
            s.drop("arc_reversed")
        elif not (_on_cell(a.x, a.y) and _on_cell(a.tail_x, a.tail_y)):
            s.drop("arc_position")
        else:
            head_mult = quantize_frac(a.head_multiplier)
            tail_mult = quantize_frac(a.tail_multiplier)
            anchor = a.mid_anchor if a.mid_anchor in (0, 1, 2) else 0
            s.quantized("arc_multiplier", a.head_multiplier, head_mult)
            s.quantized("arc_multiplier", a.tail_multiplier, tail_mult)
            s.quantized("arc_mid_anchor", a.mid_anchor, anchor)
            arcs.append(
                replace(
                    a,
                    beat=beat,
                    tail_beat=tail,
                    head_multiplier=head_mult,
                    tail_multiplier=tail_mult,
                    mid_anchor=anchor,
                )
            )

    chains: list[Chain] = []
    for c in dmap.chains:
        s.begin()
        beat, tail = s.time(c.beat), s.time(c.tail_beat)
        s.end()
        if beat < 0:
            s.drop("before_start")
        elif tail < beat:
            s.drop("chain_reversed")
        elif not (_on_cell(c.x, c.y) and _on_cell(c.tail_x, c.tail_y)):
            s.drop("chain_position")
        else:
            slices = max(1, min(MAX_SLICES, c.slice_count))
            squish = quantize_frac(c.squish)
            s.quantized("chain_slices", c.slice_count, slices)
            s.quantized("chain_squish", c.squish, squish)
            chains.append(replace(c, beat=beat, tail_beat=tail, slice_count=slices, squish=squish))

    rotations: list[RotationEvent] = []
    for r in dmap.rotations:
        s.begin()
        beat = s.time(r.beat)
        s.end()
        degrees = quantize_rotation(r.degrees)
        if beat < 0:
            s.drop("before_start")
        elif degrees == 0:
            s.drop("rotation_zero")
        else:
            s.quantized("rotation", r.degrees, degrees)
            rotations.append(replace(r, beat=beat, degrees=degrees))

    snapped = replace(
        dmap,
        notes=notes,
        bombs=bombs,
        obstacles=obstacles,
        arcs=arcs,
        chains=chains,
        rotations=rotations,
    )
    return snapped, s.report
