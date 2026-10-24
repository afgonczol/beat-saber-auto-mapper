"""Post-processing every generated difficulty goes through, whatever the generator.

1. ``snap_map``: every object onto the 24-ticks-per-beat grid and the value grids.
2. ``dedupe``: drop exact duplicates and objects sharing a cell at the same time.
3. ``fix_with_rules``: replay the map through ``bsmapper.rules`` in time order. An
   object that breaks a hard ("error") rule is repaired when a nearby allowed variant
   exists (notes: another direction, then another cell; rotations: a smaller turn),
   otherwise dropped.
4. ``fix_spans``: re-anchor arcs and chains on the notes left after step 3. An arc joins a
   note to the next note of the same hand; a chain starts on a note and runs along its cut.
   Ends that miss a note are moved onto the nearest note at that time (an arc tail onto
   the hand's next note), and what cannot be fixed is dropped (``bsmapper.rules.spans``).
   After this the checker reports no errors.
5. NJS and start-beat offset from ``njs.JUMP_TABLE`` (NJS and reaction time can be
   overridden).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field

from bsmapper.generation.njs import njs_and_offset
from bsmapper.interfaces import Issue
from bsmapper.rules import MapState, RuleChecker, RulesConfig, ordered_objects
from bsmapper.rules.spans import NoteIndex, arc_problem, chain_problem, tick
from bsmapper.schema import (
    GRID_COLUMNS,
    GRID_ROWS,
    Arc,
    Beatmap,
    Bomb,
    Chain,
    CutDirection,
    DifficultyMap,
    Note,
    Obstacle,
    RotationEvent,
)
from bsmapper.tokenizer import beat_to_tick, snap_map

# Directions to try instead of a note's own, nearest turn first.
_NEIGHBOURS = {
    CutDirection.UP: (CutDirection.UP_LEFT, CutDirection.UP_RIGHT),
    CutDirection.DOWN: (CutDirection.DOWN_LEFT, CutDirection.DOWN_RIGHT),
    CutDirection.LEFT: (CutDirection.UP_LEFT, CutDirection.DOWN_LEFT),
    CutDirection.RIGHT: (CutDirection.UP_RIGHT, CutDirection.DOWN_RIGHT),
    CutDirection.UP_LEFT: (CutDirection.UP, CutDirection.LEFT),
    CutDirection.UP_RIGHT: (CutDirection.UP, CutDirection.RIGHT),
    CutDirection.DOWN_LEFT: (CutDirection.DOWN, CutDirection.LEFT),
    CutDirection.DOWN_RIGHT: (CutDirection.DOWN, CutDirection.RIGHT),
    CutDirection.ANY: (),
}
_OPPOSITE = {
    CutDirection.UP: CutDirection.DOWN,
    CutDirection.DOWN: CutDirection.UP,
    CutDirection.LEFT: CutDirection.RIGHT,
    CutDirection.RIGHT: CutDirection.LEFT,
    CutDirection.UP_LEFT: CutDirection.DOWN_RIGHT,
    CutDirection.DOWN_RIGHT: CutDirection.UP_LEFT,
    CutDirection.UP_RIGHT: CutDirection.DOWN_LEFT,
    CutDirection.DOWN_LEFT: CutDirection.UP_RIGHT,
}


@dataclass(slots=True)
class PostReport:
    off_grid: int = 0  # objects snapping moved by more than the tolerance
    snap_dropped: Counter[str] = field(default_factory=Counter)
    duplicates: int = 0
    repaired: Counter[str] = field(default_factory=Counter)  # by rule
    dropped: Counter[str] = field(default_factory=Counter)  # by rule
    issues: list[Issue] = field(default_factory=list)  # what the checker still reports

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    def summary(self) -> str:
        parts = []
        if self.off_grid:
            parts.append(f"{self.off_grid} snapped")
        if self.duplicates:
            parts.append(f"{self.duplicates} duplicates removed")
        if self.repaired:
            parts.append(f"{sum(self.repaired.values())} repaired")
        if self.dropped:
            parts.append(f"{sum(self.dropped.values())} dropped")
        warnings = len(self.issues) - len(self.errors)
        if warnings:
            parts.append(f"{warnings} warnings")
        return ", ".join(parts) or "clean"


def dedupe(dmap: DifficultyMap) -> tuple[DifficultyMap, int]:
    """Remove exact duplicates and same-tick objects in an occupied cell (first one wins)."""
    removed = 0
    cells: set[tuple[int, int, int]] = set()

    def unique(objs, key):
        nonlocal removed
        seen, out = set(), []
        for obj in objs:
            k = key(obj)
            if k in seen:
                removed += 1
                continue
            seen.add(k)
            out.append(obj)
        return out

    def free_cell(objs):
        nonlocal removed
        out = []
        for obj in sorted(objs, key=lambda o: o.beat):
            k = (beat_to_tick(obj.beat), obj.x, obj.y)
            if k in cells:
                removed += 1
                continue
            cells.add(k)
            out.append(obj)
        return out

    notes = free_cell(dmap.notes)
    bombs = free_cell(dmap.bombs)
    obstacles = unique(dmap.obstacles, lambda o: o)
    arcs = unique(dmap.arcs, lambda o: o)
    chains = unique(dmap.chains, lambda o: o)
    rotations = unique(dmap.rotations, lambda r: beat_to_tick(r.beat))  # one turn per tick
    out = dataclasses.replace(
        dmap,
        notes=notes,
        bombs=bombs,
        obstacles=obstacles,
        arcs=arcs,
        chains=chains,
        rotations=rotations,
    )
    return out, removed


def _note_variants(note: Note) -> list[Note]:
    """Repairs to try for a note, closest first: a 45° turn, the opposite swing (what a
    parity break needs) and its neighbours, then nearby cells."""
    dirs = list(_NEIGHBOURS[note.direction])
    if note.direction in _OPPOSITE:
        flip = _OPPOSITE[note.direction]
        dirs += [flip, *_NEIGHBOURS[flip]]
    out = [dataclasses.replace(note, direction=d) for d in dirs]
    cells = sorted(
        ((x, y) for x in range(GRID_COLUMNS) for y in range(GRID_ROWS)),
        key=lambda c: (abs(c[0] - note.x) + abs(c[1] - note.y), abs(c[1] - note.y)),
    )
    for x, y in cells[1:5]:
        if y == 1 and x in (1, 2):
            continue  # do not repair into a vision-blocking centre cell
        out.append(dataclasses.replace(note, x=x, y=y))
    return out


def _repair(obj, state: MapState):
    if isinstance(obj, Note):
        for variant in _note_variants(obj):
            if state.allowed(variant):
                return variant
    elif isinstance(obj, RotationEvent) and abs(obj.degrees) > 15:
        smaller = dataclasses.replace(obj, degrees=15 if obj.degrees > 0 else -15)
        if state.allowed(smaller):
            return smaller
    return None


def fix_with_rules(
    dmap: DifficultyMap, beatmap: Beatmap, checker: RuleChecker | None = None
) -> tuple[DifficultyMap, Counter[str], Counter[str]]:
    """Repair or drop every object that breaks a hard rule. Returns (map, repaired, dropped)."""
    checker = checker or RuleChecker()
    state = checker.state(dmap, beatmap)
    repaired: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    kept: dict[type, list] = {t: [] for t in (Note, Bomb, Obstacle, Arc, Chain, RotationEvent)}
    for obj in ordered_objects(dmap):
        errors = [i for i in state.check(obj) if i.severity == "error"]
        if errors:
            rule = errors[0].rule
            fixed = _repair(obj, state)
            if fixed is None:
                dropped[rule] += 1
                continue
            repaired[rule] += 1
            obj = fixed
        state.add(obj)
        kept[type(obj)].append(obj)
    out = dataclasses.replace(
        dmap,
        notes=kept[Note],
        bombs=kept[Bomb],
        obstacles=kept[Obstacle],
        arcs=kept[Arc],
        chains=kept[Chain],
        rotations=kept[RotationEvent],
    )
    return out, repaired, dropped


def _nearest(notes: list[Note], x: int, y: int) -> Note | None:
    return min(notes, key=lambda n: (abs(n.x - x) + abs(n.y - y), n.y, n.x), default=None)


def _end_direction(note: Note, own: CutDirection) -> CutDirection:
    """An arc end takes its note's direction; on a dot note it keeps its own."""
    return own if note.direction == CutDirection.ANY else note.direction


def _anchor_arc(arc: Arc, index: NoteIndex) -> Arc | None:
    """``arc`` moved onto a note at its head time and the hand's next note, or None."""
    head = _nearest(index.notes(arc.beat, arc.color), arc.x, arc.y)
    nxt = index.next_tick(arc.color, arc.beat)
    if head is None or nxt is None:
        return None
    tail_beat = nxt / 24
    tail = _nearest(index.notes(tail_beat, arc.color), arc.tail_x, arc.tail_y)
    if tail is None:
        return None
    return dataclasses.replace(
        arc,
        x=head.x,
        y=head.y,
        direction=_end_direction(head, arc.direction),
        tail_beat=tail.beat,
        tail_x=tail.x,
        tail_y=tail.y,
        tail_direction=_end_direction(tail, arc.tail_direction),
    )


def _anchor_chain(chain: Chain, index: NoteIndex) -> Chain | None:
    """``chain`` moved (head and tail together) onto an arrow note at its head time."""
    arrows = [n for n in index.notes(chain.beat, chain.color) if n.direction != CutDirection.ANY]
    head = _nearest(arrows, chain.x, chain.y)
    if head is None:
        return None
    dx, dy = head.x - chain.x, head.y - chain.y
    tail_x, tail_y = chain.tail_x + dx, chain.tail_y + dy
    if not (0 <= tail_x < GRID_COLUMNS and 0 <= tail_y < GRID_ROWS):
        return None
    return dataclasses.replace(
        chain, x=head.x, y=head.y, direction=head.direction, tail_x=tail_x, tail_y=tail_y
    )


def fix_spans(
    dmap: DifficultyMap, beatmap: Beatmap, config: RulesConfig
) -> tuple[DifficultyMap, Counter[str], Counter[str]]:
    """Re-anchor or drop arcs and chains that break a span rule. Returns (map, repaired,
    dropped). At most one arc and one chain start on each note."""
    index = NoteIndex(dmap.notes)
    repaired: Counter[str] = Counter()
    dropped: Counter[str] = Counter()

    def keep(objs, problem, anchor):
        out, heads = [], set()
        for obj in sorted(objs, key=lambda o: o.beat):
            issue = problem(obj)
            if issue is not None:
                fixed = anchor(obj, index)
                if fixed is None or problem(fixed) is not None:
                    dropped[issue[0]] += 1
                    continue
                repaired[issue[0]] += 1
                obj = fixed
            head = (tick(obj.beat), obj.color, obj.x, obj.y)
            if head in heads:
                dropped["duplicate"] += 1
                continue
            heads.add(head)
            out.append(obj)
        return out

    arcs = keep(
        dmap.arcs,
        lambda a: arc_problem(a, index, beatmap, config, dmap.obstacles),
        _anchor_arc,
    )
    chains = keep(dmap.chains, lambda c: chain_problem(c, index, config), _anchor_chain)
    return dataclasses.replace(dmap, arcs=arcs, chains=chains), repaired, dropped


def postprocess(
    dmap: DifficultyMap,
    beatmap: Beatmap,
    checker: RuleChecker | None = None,
    njs: float | None = None,
    reaction_s: float | None = None,
) -> tuple[DifficultyMap, PostReport]:
    """Snap, dedupe, rule-fix and set NJS/offset from the table (``njs`` overrides its NJS)."""
    checker = checker or RuleChecker()
    report = PostReport()
    snapped, snap = snap_map(dmap)
    report.off_grid = snap.off_grid
    report.snap_dropped = Counter(snap.dropped)
    deduped, report.duplicates = dedupe(snapped)
    deduped.njs, deduped.njs_offset = njs_and_offset(
        deduped.difficulty, beatmap.bpm, njs, reaction_s
    )
    fixed, report.repaired, report.dropped = fix_with_rules(deduped, beatmap, checker)
    fixed, span_repaired, span_dropped = fix_spans(fixed, beatmap, checker.config(fixed))
    report.repaired += span_repaired
    report.dropped += span_dropped
    report.issues = checker.check(fixed, beatmap)
    return fixed, report
