"""Post-processing every generated difficulty goes through, whatever the generator.

1. ``snap_map``: every object onto the 24-ticks-per-beat grid and the value grids.
2. ``dedupe``: drop exact duplicates and objects sharing a cell at the same time.
3. ``fix_with_rules``: replay the map through ``bsmapper.rules`` in time order. An
   object that breaks a hard ("error") rule is repaired when a nearby allowed variant
   exists (notes: another direction, then another cell; rotations: a smaller turn),
   otherwise dropped. After this the checker reports no errors.
4. NJS and start-beat offset from ``njs.JUMP_TABLE`` (NJS and reaction time can be
   overridden).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field

from bsmapper.generation.njs import njs_and_offset
from bsmapper.interfaces import Issue
from bsmapper.rules import MapState, RuleChecker, ordered_objects
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
    report.issues = checker.check(fixed, beatmap)
    return fixed, report
