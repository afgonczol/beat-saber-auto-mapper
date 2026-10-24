"""Incremental rule state: the API behind both the checker and the decode masks.

Objects are fed in time order. ``check(obj)`` reports what adding ``obj`` would break
without changing the state; ``add(obj)`` commits it and returns the same issues.
``note_mask`` / ``bomb_mask`` evaluate every candidate at a beat, so the generation
pipeline (WS7) can turn them into token masks for whatever tokenizer it uses.

Every rule only looks backwards in time, so a full-map check is just ``add`` over the
time-sorted objects, and the checker and the masks can never disagree.
"""

from __future__ import annotations

import bisect
import copy
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from bsmapper.interfaces import Issue
from bsmapper.rules.config import RulesConfig, config_for
from bsmapper.rules.geometry import (
    angle_between,
    cut_angle,
    cut_vector,
)
from bsmapper.schema import (
    GRID_COLUMNS,
    GRID_ROWS,
    Arc,
    Beatmap,
    Bomb,
    Chain,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
    Obstacle,
    RotationEvent,
)

EPS_BEAT = 1e-4  # two objects closer than this (in beats) are simultaneous
DEFAULT_NJS = 16.0

FOREHAND, BACKHAND = "forehand", "backhand"
BOTH = frozenset({FOREHAND, BACKHAND})
_FLIP = {FOREHAND: BACKHAND, BACKHAND: FOREHAND}
_HAND = {Color.RED: "left", Color.BLUE: "right"}
_CENTRE = frozenset((1, 2))

MapObject = Note | Bomb | Obstacle | Arc | Chain | RotationEvent

# At equal beats: rotations apply first, walls before the notes they may contain.
_KIND_ORDER = {RotationEvent: 0, Obstacle: 1, Bomb: 2, Note: 3, Arc: 4, Chain: 4}


def ordered_objects(dmap: DifficultyMap) -> list[MapObject]:
    """All gameplay objects of a difficulty in the order ``MapState.add`` expects."""
    objs: list[MapObject] = [
        *dmap.rotations,
        *dmap.obstacles,
        *dmap.bombs,
        *dmap.notes,
        *dmap.arcs,
        *dmap.chains,
    ]
    return sorted(objs, key=lambda o: (o.beat, _KIND_ORDER[type(o)]))


@dataclass(frozen=True, slots=True)
class _Swing:
    """One swing of one hand: a single note, or a stack/slider of same-hand notes."""

    start_beat: float
    last_beat: float
    last_s: float
    last: Note
    angle: float | None  # None while the swing has only dots
    prev_angle: float | None  # direction of the swing before, for the turn check
    before: frozenset[str]  # parities the hand could be in going into this swing
    after: frozenset[str]  # parities this swing can have been hit with
    reset: bool


@dataclass(frozen=True, slots=True)
class _Wall:
    obstacle: Obstacle
    start_s: float
    end_s: float
    end_beat: float
    dodge: frozenset[int] | str | None
    lanes: frozenset[int]  # columns it closes at head height to a standing player


def _dodge_key(o: Obstacle) -> frozenset[int] | str | None:
    """How a wall moves the player: "crouch", the centre columns a standing wall blocks,
    or None if it never reaches the player's head."""
    if o.width <= 0 or o.height <= 0 or o.duration <= 0:
        return None
    centre = frozenset(c for c in (1, 2) if o.x <= c < o.x + o.width)
    if not centre or o.y > 2 or o.y + o.height < 3:
        return None
    return "crouch" if o.y >= 2 else centre


def _head_lanes(o: Obstacle) -> frozenset[int]:
    """Columns a wall closes at head height to a standing player (any duration)."""
    if o.width <= 0 or o.height <= 0 or o.duration <= 0 or o.y >= 2 or o.y + o.height < 3:
        return frozenset()
    return frozenset(c for c in range(GRID_COLUMNS) if o.x <= c < o.x + o.width)


def _inside(o: Obstacle, x: int, y: int, beat: float) -> bool:
    return (
        o.width > 0
        and o.height > 0
        and o.x <= x < o.x + o.width
        and o.y <= y < o.y + o.height
        and o.beat - EPS_BEAT <= beat < o.beat + o.duration - EPS_BEAT
    )


class MapState:
    """Partial map state for one difficulty, checked one object at a time."""

    def __init__(
        self,
        beatmap: Beatmap,
        characteristic: Characteristic | str = Characteristic.STANDARD,
        difficulty: Difficulty | str = Difficulty.EXPERT,
        njs: float | None = None,
        config: RulesConfig | None = None,
        thresholds: Path | str | None = None,
    ) -> None:
        self.beatmap = beatmap
        self.config = config or config_for(difficulty, characteristic, thresholds)
        self.njs = njs if njs and njs > 0 else DEFAULT_NJS
        self.last_beat = -math.inf
        self.notes: list[Note] = []
        self._note_s: list[float] = []
        self.bombs: list[Bomb] = []
        self._bomb_s: list[float] = []
        self._walls: list[_Wall] = []  # walls that can still matter at last_beat
        self._dodge_walls: list[_Wall] = []
        self._head_walls: list[_Wall] = []  # walls closing a centre column at head height
        self._rot_s: list[float] = []
        self._rot_total: list[float] = []
        self._swings: dict[Color, _Swing] = {}
        self._sec: dict[float, float] = {}

    @classmethod
    def for_map(cls, dmap: DifficultyMap, beatmap: Beatmap, **kwargs) -> MapState:
        return cls(beatmap, dmap.characteristic, dmap.difficulty, njs=dmap.njs, **kwargs)

    # --- public API --------------------------------------------------------------

    def seconds(self, beat: float) -> float:
        s = self._sec.get(beat)
        if s is None:
            s = self._sec[beat] = self.beatmap.beat_to_seconds(beat)
        return s

    def check(self, obj: MapObject) -> list[Issue]:
        """Issues that adding ``obj`` now would cause. Does not change the state."""
        return self._evaluate(obj)[0]

    def allowed(self, obj: MapObject, severities: Iterable[str] = ("error",)) -> bool:
        """True if adding ``obj`` breaks no rule of the given severities (hard rules)."""
        sev = set(severities)
        return not any(i.severity in sev for i in self.check(obj))

    def add(self, obj: MapObject) -> list[Issue]:
        """Commit ``obj`` (whether or not it breaks rules) and return its issues."""
        if obj.beat < self.last_beat - EPS_BEAT:
            raise ValueError(
                f"objects must be added in time order ({obj.beat} after {self.last_beat})"
            )
        issues, commit = self._evaluate(obj)
        commit()
        self.last_beat = max(self.last_beat, obj.beat)
        self._prune()
        return issues

    def clone(self) -> MapState:
        """An independent copy, e.g. one per beam or before a tentative ``add``."""
        other = copy.copy(self)
        for name in (
            "notes",
            "_note_s",
            "bombs",
            "_bomb_s",
            "_walls",
            "_dodge_walls",
            "_head_walls",
            "_rot_s",
            "_rot_total",
        ):
            setattr(other, name, list(getattr(self, name)))
        other._swings = dict(self._swings)
        return other

    def extend(self, objs: Iterable[MapObject]) -> list[Issue]:
        issues: list[Issue] = []
        for obj in objs:
            issues += self.add(obj)
        return issues

    def note_mask(
        self, beat: float, color: Color, severities: Iterable[str] = ("error",)
    ) -> np.ndarray:
        """Allowed notes of ``color`` at ``beat``, as a bool array [x, y, direction].

        Shape is (GRID_COLUMNS, GRID_ROWS, 9), indexed by ``CutDirection`` (8 = dot).
        """
        sev = set(severities)
        mask = np.zeros((GRID_COLUMNS, GRID_ROWS, len(CutDirection)), dtype=bool)
        probe = Note(beat, 0, 0, color, CutDirection.ANY)
        if any(i.severity in sev for i in self._note_global(probe)):
            return mask
        for x in range(GRID_COLUMNS):
            for y in range(GRID_ROWS):
                if any(i.severity in sev for i in self._note_position(replace(probe, x=x, y=y))):
                    continue
                for d in CutDirection:
                    note = Note(beat, x, y, color, d)
                    mask[x, y, d] = not any(i.severity in sev for i in self._hand(note)[0])
        return mask

    def candidates_mask(
        self, candidates: Sequence[MapObject | None], severities: Iterable[str] = ("error",)
    ) -> np.ndarray:
        """Bool mask over a list of candidates; ``None`` entries are always allowed.

        A tokenizer can decode each token id it may emit next into the object it would
        place (or None for tokens that place nothing, such as time shifts) and use this
        as its logit mask.
        """
        sev = tuple(severities)
        return np.array([c is None or self.allowed(c, sev) for c in candidates], dtype=bool)

    def bomb_mask(self, beat: float, severities: Iterable[str] = ("error",)) -> np.ndarray:
        """Allowed bomb cells at ``beat``, as a bool array [x, y]."""
        sev = set(severities)
        mask = np.zeros((GRID_COLUMNS, GRID_ROWS), dtype=bool)
        for x in range(GRID_COLUMNS):
            for y in range(GRID_ROWS):
                issues = self.check(Bomb(beat, x, y))
                mask[x, y] = not any(i.severity in sev for i in issues)
        return mask

    # --- dispatch ----------------------------------------------------------------

    def _issue(self, rule: str, beat: float, message: str) -> Issue:
        return Issue(rule, beat, self.config.severity[rule], message)  # type: ignore[arg-type]

    def _on(self, rule: str) -> bool:
        return self.config.enabled(rule)

    def _evaluate(self, obj: MapObject) -> tuple[list[Issue], Callable[[], None]]:
        if isinstance(obj, Note):
            return self._evaluate_note(obj)
        if isinstance(obj, Bomb):
            return self._evaluate_bomb(obj)
        if isinstance(obj, Obstacle):
            return self._evaluate_wall(obj)
        if isinstance(obj, RotationEvent):
            return self._evaluate_rotation(obj)
        return [], lambda: None  # arcs and chains ride on their head/tail notes

    def _prune(self) -> None:
        now = self.last_beat
        self._walls = [w for w in self._walls if w.end_beat >= now - EPS_BEAT]
        horizon = self.seconds(now) - self.config.walls.min_dodge_gap_s
        self._dodge_walls = [w for w in self._dodge_walls if w.end_s >= horizon]
        horizon = self.seconds(now) - self.config.walls.boxin_window_s
        self._head_walls = [w for w in self._head_walls if w.end_s >= horizon]

    # --- notes -------------------------------------------------------------------

    def _evaluate_note(self, note: Note) -> tuple[list[Issue], Callable[[], None]]:
        hand_issues, swing = self._hand(note)
        issues = self._note_global(note) + self._note_position(note) + hand_issues

        def commit() -> None:
            self.notes.append(note)
            self._note_s.append(self.seconds(note.beat))
            if swing is not None:
                self._swings[note.color] = swing

        return issues, commit

    def _note_global(self, note: Note) -> list[Issue]:
        """Rules that depend only on the note's time (density)."""
        if not self._on("nps_ceiling"):
            return []
        d = self.config.density
        t = self.seconds(note.beat)
        start = bisect.bisect_right(self._note_s, t - d.window_s + 1e-9)
        nps = (len(self._note_s) - start + 1) / d.window_s
        if nps > d.max_nps + 1e-9:
            return [
                self._issue(
                    "nps_ceiling",
                    note.beat,
                    f"{nps:.1f} notes/s over {d.window_s:g}s exceeds {d.max_nps:g}",
                )
            ]
        return []

    def _note_position(self, note: Note) -> list[Issue]:
        """Rules that depend on the note's time, cell and colour but not its direction."""
        issues = self._cell_conflicts(note)
        t = self.seconds(note.beat)
        if self._on("note_in_wall"):
            for w in self._walls:
                if _inside(w.obstacle, note.x, note.y, note.beat):
                    issues.append(
                        self._issue(
                            "note_in_wall",
                            note.beat,
                            f"note at ({note.x},{note.y}) is inside a wall from beat "
                            f"{w.obstacle.beat:g}",
                        )
                    )
                    break
        if self._on("vision_block"):
            issues += self._vision(note, t)
        if self._on("crossover"):
            issues += self._crossover(note, t)
        return issues

    def _cell_conflicts(self, obj: Note | Bomb) -> list[Issue]:
        if not (self._on("same_cell") or self._on("overlap")):
            return []
        t = self.seconds(obj.beat)
        window = self.config.overlap.min_distance_m / self.njs
        for objs, times in ((self.notes, self._note_s), (self.bombs, self._bomb_s)):
            start = bisect.bisect_left(times, t - window - 1e-9)
            for other, s in zip(objs[start:], times[start:], strict=True):
                if (other.x, other.y) != (obj.x, obj.y):
                    continue
                if isinstance(obj, Bomb) and isinstance(other, Bomb):
                    continue  # bomb lines may be dense; only notes must stay apart
                if abs(other.beat - obj.beat) < EPS_BEAT:
                    if self._on("same_cell"):
                        return [
                            self._issue(
                                "same_cell",
                                obj.beat,
                                f"two objects at ({obj.x},{obj.y}) at the same time",
                            )
                        ]
                elif self._on("overlap") and t - s < window:
                    return [
                        self._issue(
                            "overlap",
                            obj.beat,
                            f"object at ({obj.x},{obj.y}) is {(t - s) * self.njs:.2f} m behind "
                            f"the one at beat {other.beat:g}",
                        )
                    ]
        return []

    def _vision(self, note: Note, t: float) -> list[Issue]:
        v = self.config.vision
        merge = self.config.parity.merge_window_s
        sources: list[tuple[list, list[float]]] = [(self.notes, self._note_s)]
        if v.bombs_block:
            sources.append((self.bombs, self._bomb_s))
        for objs, times in sources:
            lo = bisect.bisect_left(times, t - v.max_s - 1e-9)
            hi = bisect.bisect_right(times, t - v.min_s + 1e-9)
            for blocker, s in zip(objs[lo:hi], times[lo:hi], strict=True):
                if blocker.y != 1 or blocker.x not in (1, 2):
                    continue
                if isinstance(blocker, Note) and blocker.color == note.color and t - s <= merge:
                    continue  # the same swing (slider or stack), not a blocker
                if (blocker.x == 1 and note.x <= 1) or (blocker.x == 2 and note.x >= 2):
                    return [
                        self._issue(
                            "vision_block",
                            note.beat,
                            f"note at ({note.x},{note.y}) is hidden behind the centre "
                            f"{type(blocker).__name__.lower()} at beat {blocker.beat:g}",
                        )
                    ]
        return []

    def _crossover(self, note: Note, t: float) -> list[Issue]:
        cfg = self.config.crossover
        window = cfg.window_s
        start = bisect.bisect_left(self._note_s, t - window - 1e-9)
        for other, s in zip(self.notes[start:], self._note_s[start:], strict=True):
            if other.color == note.color:
                continue
            if t - s > window + 1e-9 and abs(other.beat - note.beat) >= EPS_BEAT:
                continue
            red, blue = (note, other) if note.color == Color.RED else (other, note)
            if red.x - blue.x >= cfg.min_columns:
                return [
                    self._issue(
                        "crossover",
                        note.beat,
                        f"red note in column {red.x} is right of blue in column {blue.x}",
                    )
                ]
        return []

    # --- hands: parity and swing speed ---------------------------------------------

    def _hand(self, note: Note) -> tuple[list[Issue], _Swing | None]:
        """Parity / swing issues for ``note`` and the hand's swing state after it."""
        p = self.config.parity
        t = self.seconds(note.beat)
        cut = cut_angle(note)
        last = self._swings.get(note.color)
        hand = _HAND.get(note.color, "?")
        issues: list[Issue] = []

        if last is not None and t - last.last_s <= p.merge_window_s + 1e-9:
            if self._mergeable(last, note, cut):
                swing = replace(last, last_beat=note.beat, last_s=t, last=note)
                if last.angle is None and cut is not None:
                    after, problem = self._parity_step(
                        note.color, last.before, last.prev_angle, cut, last.reset
                    )
                    if problem and self._on("parity"):
                        issues.append(self._issue("parity", note.beat, f"{hand} hand: {problem}"))
                    swing = replace(swing, angle=cut, after=after)
                return issues, swing
            if abs(note.beat - last.last_beat) < EPS_BEAT:
                if self._on("parity"):
                    issues.append(
                        self._issue(
                            "parity",
                            note.beat,
                            f"{hand} hand: notes at the same time need different swings",
                        )
                    )
                return issues, None

        gap = t - last.last_s if last is not None else math.inf
        reset = last is None or gap >= p.reset_gap_s - 1e-6 or self._bomb_reset(last, note)
        before = last.after if last is not None else BOTH
        prev_angle = last.angle if last is not None else None
        after, problem = self._parity_step(
            note.color, before, prev_angle, cut, reset, self._continues(last, note, cut)
        )
        if problem and self._on("parity"):
            issues.append(self._issue("parity", note.beat, f"{hand} hand: {problem}"))

        if last is not None and gap < math.inf:
            sw = self.config.swing
            if self._on("swing_interval") and gap < sw.min_interval_s - 1e-9:
                issues.append(
                    self._issue(
                        "swing_interval",
                        note.beat,
                        f"{hand} hand swings again {gap * 1000:.0f} ms after the last swing "
                        f"(min {sw.min_interval_s * 1000:.0f} ms)",
                    )
                )
            if self._on("swing_speed") and gap > 0:
                speed = self._travel(last, note, cut) / gap
                if speed > sw.max_speed + 1e-9:
                    issues.append(
                        self._issue(
                            "swing_speed",
                            note.beat,
                            f"{hand} hand moves {speed:.0f} cells/s between swings "
                            f"(max {sw.max_speed:g})",
                        )
                    )

        swing = _Swing(
            start_beat=note.beat,
            last_beat=note.beat,
            last_s=t,
            last=note,
            angle=cut,
            prev_angle=prev_angle,
            before=before,
            after=after,
            reset=reset,
        )
        return issues, swing

    def _parity_step(
        self,
        color: Color,
        before: frozenset[str],
        prev_angle: float | None,
        cut: float | None,
        reset: bool,
        continuation: bool = False,
    ) -> tuple[frozenset[str], str | None]:
        """Parities the new swing can be hit with, and what is wrong if none fits."""
        p = self.config.parity
        if cut is None:
            if p.dots_flip_parity and not reset:
                return frozenset(_FLIP[x] for x in before), None
            return BOTH, None
        # Inward is RIGHT (+) for the left hand, LEFT (-) for the right hand.
        centre = p.inward_bias_deg if color == Color.RED else -p.inward_bias_deg
        fits = frozenset(
            x
            for x, dev in (
                (FOREHAND, angle_between(cut, centre)),
                (BACKHAND, angle_between(cut, centre + 180.0)),
            )
            if dev <= p.max_wrist_deg + 1e-6
        )
        if reset:
            return fits, None
        cands = frozenset(_FLIP[x] for x in before)
        if prev_angle is not None:
            turn = angle_between(prev_angle, cut)
            if turn <= p.keep_max_turn_deg + 1e-6 or continuation:
                cands |= before  # a curve: the hand may keep its parity
            if turn < p.min_direction_change_deg - 1e-6 and not continuation:
                return fits, f"swing turns only {turn:.0f}° from the previous one (parity break)"
        after = fits & cands
        if not after:
            need = " or ".join(sorted(cands))
            return fits, f"cut needs a reset, the next swing should be a {need}"
        return after, None

    def _continues(self, last: _Swing | None, note: Note, cut: float | None) -> bool:
        """A slightly turned swing further along the last one's path (a slider with a gap)."""
        p = self.config.parity
        if not p.allow_continuation or last is None or last.angle is None or cut is None:
            return False
        if angle_between(last.angle, cut) > p.slider_max_turn_deg + 1e-6:
            return False
        vx, vy = cut_vector(last.angle)
        return (note.x - last.last.x) * vx + (note.y - last.last.y) * vy > 0.5

    def _mergeable(self, last: _Swing, note: Note, cut: float | None) -> bool:
        """Does ``note`` continue the hand's current swing (stack, window or slider)?"""
        prev = last.last
        prev_cut = cut_angle(prev)
        if (
            prev_cut is not None
            and cut is not None
            and angle_between(prev_cut, cut) > self.config.parity.slider_max_turn_deg + 1e-6
        ):
            return False
        if abs(note.beat - prev.beat) < EPS_BEAT:
            return True  # a stack: the order within one beat is arbitrary
        ref = cut if cut is not None else last.angle
        if ref is None:
            return True
        vx, vy = cut_vector(ref)
        return (note.x - prev.x) * vx + (note.y - prev.y) * vy >= -1e-6

    def _bomb_reset(self, last: _Swing, note: Note) -> bool:
        cols = self.config.parity.bomb_reset_columns
        i = bisect.bisect_right(self._bomb_s, last.last_s + 1e-9)
        return any(
            last.last_beat + EPS_BEAT < b.beat < note.beat - EPS_BEAT
            and abs(b.x - last.last.x) <= cols
            for b in self.bombs[i:]
        )

    @staticmethod
    def _travel(last: _Swing, note: Note, cut: float | None) -> float:
        """Saber travel from the end of the last swing to the start of this one."""
        ex, ey = float(last.last.x), float(last.last.y)
        if last.angle is not None:
            vx, vy = cut_vector(last.angle)
            ex, ey = ex + 0.5 * vx, ey + 0.5 * vy
        sx, sy = float(note.x), float(note.y)
        if cut is not None:
            vx, vy = cut_vector(cut)
            sx, sy = sx - 0.5 * vx, sy - 0.5 * vy
        return math.hypot(sx - ex, sy - ey)

    # --- bombs, walls, rotations ----------------------------------------------------

    def _evaluate_bomb(self, bomb: Bomb) -> tuple[list[Issue], Callable[[], None]]:
        issues = self._cell_conflicts(bomb)

        def commit() -> None:
            self.bombs.append(bomb)
            self._bomb_s.append(self.seconds(bomb.beat))

        return issues, commit

    def _evaluate_wall(self, o: Obstacle) -> tuple[list[Issue], Callable[[], None]]:
        cfg = self.config.walls
        t = self.seconds(o.beat)
        end_beat = o.beat + o.duration
        end_s = self.seconds(end_beat)
        dodge = _dodge_key(o) if end_s - t >= cfg.min_dodge_duration_s - 1e-9 else None
        wall = _Wall(o, t, end_s, end_beat, dodge, _head_lanes(o))
        issues: list[Issue] = []
        visible = o.width > 0 and o.height > 0
        if visible and self._on("wall_duration") and wall.end_s - t < cfg.min_duration_s - 1e-9:
            issues.append(
                self._issue(
                    "wall_duration",
                    o.beat,
                    f"wall lasts {(wall.end_s - t) * 1000:.0f} ms "
                    f"(min {cfg.min_duration_s * 1000:.0f} ms)",
                )
            )
        dodge_gap = False
        if wall.dodge is not None and self._on("dodge_gap"):
            for prev in self._dodge_walls:
                gap = t - prev.end_s
                if prev.dodge != wall.dodge and -1e-9 <= gap < cfg.min_dodge_gap_s - 1e-9:
                    issues.append(
                        self._issue(
                            "dodge_gap",
                            o.beat,
                            f"dodge wall starts {gap * 1000:.0f} ms after a different one "
                            f"(min {cfg.min_dodge_gap_s * 1000:.0f} ms)",
                        )
                    )
                    dodge_gap = True
                    break
        centre = wall.lanes & _CENTRE
        if centre and not dodge_gap and self._on("wall_boxin"):
            boxed = "wall closes both centre columns" if centre == _CENTRE else None
            for prev in self._head_walls if boxed is None else ():
                if t - prev.end_s < cfg.boxin_window_s - 1e-9 and prev.lanes & _CENTRE != centre:
                    boxed = (
                        f"walls close both centre columns within "
                        f"{cfg.boxin_window_s * 1000:.0f} ms (with the wall from beat "
                        f"{prev.obstacle.beat:g})"
                    )
                    break
            if boxed:
                issues.append(self._issue("wall_boxin", o.beat, boxed))
        if self._on("note_in_wall"):
            # Only notes at the wall's own start beat can already be placed.
            for note in reversed(self.notes):
                if note.beat < o.beat - EPS_BEAT:
                    break
                if _inside(o, note.x, note.y, note.beat):
                    issues.append(
                        self._issue(
                            "note_in_wall",
                            o.beat,
                            f"wall covers the note at ({note.x},{note.y}) beat {note.beat:g}",
                        )
                    )
                    break

        def commit() -> None:
            if visible and o.duration > 0:
                self._walls.append(wall)
            if wall.dodge is not None:
                self._dodge_walls.append(wall)
            if wall.lanes & _CENTRE:
                self._head_walls.append(wall)

        return issues, commit

    def _evaluate_rotation(self, ev: RotationEvent) -> tuple[list[Issue], Callable[[], None]]:
        cfg = self.config.rotation
        t = self.seconds(ev.beat)
        total = (self._rot_total[-1] if self._rot_total else 0.0) + ev.degrees
        issues: list[Issue] = []
        if self._on("rotation_step") and abs(ev.degrees) > cfg.max_step_deg + 1e-6:
            issues.append(
                self._issue(
                    "rotation_step",
                    ev.beat,
                    f"one rotation of {ev.degrees:+d}° (max {cfg.max_step_deg:g}°)",
                )
            )
        if self._on("rotation_whiplash"):
            start = bisect.bisect_left(self._rot_s, t - cfg.window_s - 1e-9)
            window = [*self._rot_total[start:], total]
            window.append(self._rot_total[start - 1] if start > 0 else 0.0)
            swing = max(window) - min(window)
            if swing > cfg.max_swing_deg + 1e-6:
                issues.append(
                    self._issue(
                        "rotation_whiplash",
                        ev.beat,
                        f"rotation swings {swing:.0f}° within {cfg.window_s:g}s "
                        f"(max {cfg.max_swing_deg:g}°)",
                    )
                )
        if (
            self._on("rotation_range")
            and cfg.max_abs_deg is not None
            and abs(total) > cfg.max_abs_deg + 1e-6
        ):
            issues.append(
                self._issue(
                    "rotation_range",
                    ev.beat,
                    f"rotation reaches {total:+.0f}° (limit ±{cfg.max_abs_deg:g}°)",
                )
            )

        def commit() -> None:
            self._rot_s.append(t)
            self._rot_total.append(total)

        return issues, commit
