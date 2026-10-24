"""Arc and chain rules. Unlike the incremental rules in ``state``, these look forward: an
arc's tail sits on a later note. So they run over the whole map, after the notes
(``RuleChecker.check``), and decoding repairs them afterwards instead of masking them
(``generation.postprocess.fix_spans``).

Calibrated on 10,381 arcs and 1,381 chains in 118 human v3 difficulties: every arc head and
tail sits on a same-colour note in its cell with the same direction (or a dot note), one arc
in 10,000 skips a note of its hand, 99% are under 2.8 s long, and chains are at most a
quarter beat long and point their links along the cut.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Sequence

from bsmapper.interfaces import Issue
from bsmapper.rules.config import RulesConfig
from bsmapper.rules.geometry import BASE_ANGLE, cut_vector
from bsmapper.schema import (
    Arc,
    Beatmap,
    Chain,
    Color,
    CutDirection,
    DifficultyMap,
    Note,
    Obstacle,
)

SPAN_RULES = ("arc_anchor", "arc_span", "arc_wall", "chain_anchor")
TICKS = 24  # beat grid resolution (tokenizer, snapping)
PATH_SAMPLES = 8  # points checked along an arc's path for walls


def tick(beat: float) -> int:
    return round(beat * TICKS)


def directions_match(note: CutDirection, end: CutDirection) -> bool:
    """An arc or chain end fits a note with the same direction, or a dot on either side."""
    return note == end or CutDirection.ANY in (note, end)


class NoteIndex:
    """Notes by (tick, colour), and the sorted ticks of each colour."""

    def __init__(self, notes: Sequence[Note]) -> None:
        self.at: dict[tuple[int, Color], list[Note]] = defaultdict(list)
        ticks: dict[Color, set[int]] = defaultdict(set)
        for n in notes:
            self.at[(tick(n.beat), n.color)].append(n)
            ticks[n.color].add(tick(n.beat))
        self.ticks = {c: sorted(ts) for c, ts in ticks.items()}

    def notes(self, beat: float, color: Color) -> list[Note]:
        return self.at.get((tick(beat), color), [])

    def anchor(self, beat: float, color: Color, x: int, y: int, d: CutDirection) -> Note | None:
        """The note an arc/chain end at this cell sits on, if its direction fits."""
        for n in self.notes(beat, color):
            if (n.x, n.y) == (x, y) and directions_match(n.direction, d):
                return n
        return None

    def between(self, color: Color, start: float, end: float) -> int:
        """Ticks with a note of ``color`` strictly between two beats."""
        ts = self.ticks.get(color, [])
        return max(bisect.bisect_left(ts, tick(end)) - bisect.bisect_right(ts, tick(start)), 0)

    def next_tick(self, color: Color, beat: float) -> int | None:
        """The first tick after ``beat`` with a note of ``color``."""
        ts = self.ticks.get(color, [])
        i = bisect.bisect_right(ts, tick(beat))
        return ts[i] if i < len(ts) else None


def arc_hits_wall(arc: Arc, walls: Sequence[Obstacle]) -> bool:
    """Does the arc's path (a straight line between its ends) pass through a wall?"""
    for k in range(1, PATH_SAMPLES):
        f = k / PATH_SAMPLES
        beat = arc.beat + f * (arc.tail_beat - arc.beat)
        x = round(arc.x + f * (arc.tail_x - arc.x))
        y = round(arc.y + f * (arc.tail_y - arc.y))
        for w in walls:
            if (
                w.width > 0
                and w.height > 0
                and w.x <= x < w.x + w.width
                and w.y <= y < w.y + w.height
                and w.beat <= beat < w.beat + w.duration
            ):
                return True
    return False


def chain_points_forward(chain: Chain) -> bool:
    """Chain links must continue the head's cut (a dot head can point anywhere)."""
    angle = BASE_ANGLE.get(chain.direction)
    if angle is None:
        return True
    vx, vy = cut_vector(angle)
    return (chain.tail_x - chain.x) * vx + (chain.tail_y - chain.y) * vy > 0


def arc_problem(
    arc: Arc, index: NoteIndex, beatmap: Beatmap, config: RulesConfig, walls: Sequence[Obstacle]
) -> tuple[str, str] | None:
    """(rule, message) for the first rule ``arc`` breaks, or None."""
    if index.anchor(arc.beat, arc.color, arc.x, arc.y, arc.direction) is None:
        return "arc_anchor", "arc head is not on a matching note"
    if index.anchor(arc.tail_beat, arc.color, arc.tail_x, arc.tail_y, arc.tail_direction) is None:
        return "arc_anchor", "arc tail is not on a matching note"
    if tick(arc.tail_beat) <= tick(arc.beat):
        return "arc_span", "arc tail is not after its head"
    seconds = beatmap.beat_to_seconds(arc.tail_beat) - beatmap.beat_to_seconds(arc.beat)
    if seconds > config.spans.arc_max_s + 1e-9:
        return "arc_span", f"arc is {seconds:.2f}s long (limit {config.spans.arc_max_s:g}s)"
    if index.between(arc.color, arc.beat, arc.tail_beat):
        return "arc_span", "arc skips a note of its hand"
    if arc_hits_wall(arc, walls):
        return "arc_wall", "arc runs through a wall"
    return None


def chain_problem(chain: Chain, index: NoteIndex, config: RulesConfig) -> tuple[str, str] | None:
    """(rule, message) for the first rule ``chain`` breaks, or None."""
    head = index.anchor(chain.beat, chain.color, chain.x, chain.y, chain.direction)
    if head is None or head.direction != chain.direction:
        return "chain_anchor", "chain head is not on a note with its direction"
    length = chain.tail_beat - chain.beat
    if length < 0 or length > config.spans.chain_max_beats + 1e-9:
        return "chain_anchor", f"chain is {length:g} beats long"
    if not chain_points_forward(chain):
        return "chain_anchor", "chain links point against the cut"
    if index.between(chain.color, chain.beat, chain.tail_beat):
        return "chain_anchor", "chain skips a note of its hand"
    return None


def check_spans(dmap: DifficultyMap, beatmap: Beatmap, config: RulesConfig) -> list[Issue]:
    """Issues of every arc and chain in ``dmap``, in time order."""
    index = NoteIndex(dmap.notes)
    found: list[tuple[float, str, str]] = []
    for arc in dmap.arcs:
        problem = arc_problem(arc, index, beatmap, config, dmap.obstacles)
        if problem:
            found.append((arc.beat, *problem))
    for chain in dmap.chains:
        problem = chain_problem(chain, index, config)
        if problem:
            found.append((chain.beat, *problem))
    found.sort(key=lambda f: f[0])
    return [
        Issue(rule, beat, config.severity[rule], message)  # type: ignore[arg-type]
        for beat, rule, message in found
        if config.enabled(rule)
    ]
