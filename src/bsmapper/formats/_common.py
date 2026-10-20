"""Helpers shared by the per-version readers and writers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from bsmapper.formats.report import ReadReport
from bsmapper.schema import (
    GRID_COLUMNS,
    GRID_ROWS,
    Arc,
    BasicEvent,
    Bomb,
    BpmChange,
    Chain,
    Color,
    ColorBoostEvent,
    CutDirection,
    EventBoxGroup,
    Lightshow,
    Note,
    Obstacle,
    RotationEvent,
)

# Beats are rounded to this many decimals on read and write. It removes float
# serialisation noise (10.000000001) without touching any real grid position.
BEAT_DECIMALS = 6

# v2 rotation event values 0..7, and the legacy basic event types that are not lights.
ROTATION_STEPS = (-60, -45, -30, -15, 15, 30, 45, 60)
EVENT_BOOST = 5
EVENT_EARLY_ROTATION = 14
EVENT_LATE_ROTATION = 15
EVENT_BPM = 100

ME_THRESHOLD = 1000  # Mapping Extensions encodes precise values as >= 1000
LEGACY_CHROMA_VALUE = 2_000_000_000  # Chroma 1.x packed RGB into the event value

# Object customData keys (underscores stripped) that only Noodle Extensions understands.
NOODLE_OBJECT_KEYS = frozenset(
    {
        "position",
        "coordinates",
        "rotation",
        "worldRotation",
        "localRotation",
        "scale",
        "size",
        "track",
        "animation",
        "noteJumpMovementSpeed",
        "noteJumpStartBeatOffset",
        "cutDirection",
        "fake",
        "interactable",
        "uninteractable",
        "flip",
        "disableNoteGravity",
        "disableNoteLook",
        "disableSpawnEffect",
        "spawnEffect",
        "link",
    }
)
NOODLE_DIFFICULTY_KEYS = frozenset(
    {"fakeColorNotes", "fakeBombNotes", "fakeObstacles", "fakeBurstSliders"}
)
NOODLE_EVENT_TYPES = frozenset(
    {"AnimateTrack", "AssignPathAnimation", "AssignTrackParent", "AssignPlayerToTrack"}
)
CHROMA_DIFFICULTY_KEYS = frozenset({"environment", "materials"})


# --- Number handling ----------------------------------------------------------


def beat(value: Any) -> float:
    """Parse a beat/float value, rounding away precision noise."""
    return round(float(value), BEAT_DECIMALS) + 0.0  # + 0.0 turns -0.0 into 0.0


def as_int(value: Any) -> int:
    return int(round(float(value)))


def out(value: float) -> int | float:
    """Format a float for JSON output: rounded, and integral values written as ints."""
    rounded = round(float(value), BEAT_DECIMALS) + 0.0
    return int(rounded) if rounded.is_integer() else rounded


def first(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """The value of the first key that is present and not null."""
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return default


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def strip_underscores(data: Mapping[str, Any]) -> dict[str, Any]:
    """v2 custom data keys start with '_'; the canonical form uses the v3 names."""
    return {key.lstrip("_"): value for key, value in data.items()}


def with_underscores(data: Mapping[str, Any]) -> dict[str, Any]:
    return {f"_{key}": value for key, value in data.items()}


def string_list(value: Any) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


# --- Parsed difficulty --------------------------------------------------------


@dataclass(slots=True)
class ParsedDifficulty:
    """Everything one difficulty file contains, before Info.dat metadata is attached."""

    version: str = ""
    notes: list[Note] = field(default_factory=list)
    bombs: list[Bomb] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    chains: list[Chain] = field(default_factory=list)
    rotations: list[RotationEvent] = field(default_factory=list)
    # None for v4 beatmap files, whose lighting lives in a separate lightshow file.
    lightshow: Lightshow | None = None
    # Official BPM changes, in this file's own (tempo-warped) beat space.
    bpm_changes: list[BpmChange] = field(default_factory=list)
    # Legacy MMA2/ChroMapper ``_BPMChanges``: editor-only, object times stay in base-BPM beats.
    editor_bpm_changes: list[BpmChange] = field(default_factory=list)

    def sort(self) -> None:
        for items in (
            self.notes,
            self.bombs,
            self.obstacles,
            self.arcs,
            self.chains,
            self.rotations,
            self.bpm_changes,
            self.editor_bpm_changes,
        ):
            items.sort(key=lambda o: o.beat)
        if self.lightshow is not None:
            sort_lightshow(self.lightshow)

    def remap(self, fn: Callable[[float], float]) -> None:
        """Move every time through ``fn`` (old beat -> new beat)."""
        self.notes = [replace(n, beat=beat(fn(n.beat))) for n in self.notes]
        self.bombs = [replace(b, beat=beat(fn(b.beat))) for b in self.bombs]
        self.obstacles = [
            replace(
                o,
                beat=beat(fn(o.beat)),
                duration=beat(fn(o.beat + o.duration) - fn(o.beat)),
            )
            for o in self.obstacles
        ]
        self.arcs = [
            replace(a, beat=beat(fn(a.beat)), tail_beat=beat(fn(a.tail_beat))) for a in self.arcs
        ]
        self.chains = [
            replace(c, beat=beat(fn(c.beat)), tail_beat=beat(fn(c.tail_beat))) for c in self.chains
        ]
        self.rotations = [replace(r, beat=beat(fn(r.beat))) for r in self.rotations]
        if self.lightshow is not None:
            self.lightshow = remap_lightshow(self.lightshow, fn)


def sort_lightshow(ls: Lightshow) -> None:
    ls.basic_events.sort(key=lambda e: e.beat)
    ls.boost_events.sort(key=lambda e: e.beat)
    ls.event_box_groups.sort(key=lambda e: e.beat)


def remap_lightshow(ls: Lightshow, fn: Callable[[float], float]) -> Lightshow:
    return Lightshow(
        basic_events=[replace(e, beat=beat(fn(e.beat))) for e in ls.basic_events],
        boost_events=[replace(e, beat=beat(fn(e.beat))) for e in ls.boost_events],
        event_box_groups=[replace(e, beat=beat(fn(e.beat))) for e in ls.event_box_groups],
        uses_chroma=ls.uses_chroma,
    )


# --- Validated object construction -------------------------------------------


def _in_grid(x: int, y: int) -> bool:
    return 0 <= x < GRID_COLUMNS and 0 <= y < GRID_ROWS


def _mapping_extensions(report: ReadReport, *values: int) -> bool:
    if any(abs(v) >= ME_THRESHOLD for v in values):
        report.mapping_extensions = True
        return True
    return False


def _valid_direction(d: int) -> bool:
    return 0 <= d <= CutDirection.ANY


def _off_grid(report: ReadReport, kind: str, *positions: tuple[int, int]) -> bool:
    """Off-grid positions can only be placed with Mapping Extensions."""
    if all(_in_grid(x, y) for x, y in positions):
        return False
    report.mapping_extensions = True
    report.drop(f"off_grid_{kind}")
    return True


def make_note(report: ReadReport, b: float, x: int, y: int, c: int, d: int, a: int = 0):
    if _mapping_extensions(report, x, y, d):
        report.drop("mapping_extensions_note")
        return None
    if _off_grid(report, "note", (x, y)):
        return None
    if c not in (0, 1) or not _valid_direction(d):
        report.drop("invalid_note")
        return None
    return Note(b, x, y, Color(c), CutDirection(d), a)


def make_bomb(report: ReadReport, b: float, x: int, y: int):
    if _mapping_extensions(report, x, y):
        report.drop("mapping_extensions_bomb")
        return None
    if _off_grid(report, "bomb", (x, y)):
        return None
    return Bomb(b, x, y)


def make_obstacle(report: ReadReport, b: float, d: float, x: int, y: int, w: int, h: int):
    if _mapping_extensions(report, x, y, w, h):
        report.drop("mapping_extensions_obstacle")
        return None
    return Obstacle(b, d, x, y, w, h)


def make_arc(
    report: ReadReport,
    c: int,
    b: float,
    x: int,
    y: int,
    d: int,
    mu: float,
    tb: float,
    tx: int,
    ty: int,
    tc: int,
    tmu: float,
    m: int,
):
    if _mapping_extensions(report, x, y, d, tx, ty, tc):
        report.drop("mapping_extensions_arc")
        return None
    if _off_grid(report, "arc", (x, y), (tx, ty)):
        return None
    if c not in (0, 1) or not (_valid_direction(d) and _valid_direction(tc)):
        report.drop("invalid_arc")
        return None
    return Arc(b, Color(c), x, y, CutDirection(d), tb, tx, ty, CutDirection(tc), mu, tmu, m)


def make_chain(
    report: ReadReport,
    c: int,
    b: float,
    x: int,
    y: int,
    d: int,
    tb: float,
    tx: int,
    ty: int,
    sc: int,
    s: float,
):
    if _mapping_extensions(report, x, y, d, tx, ty):
        report.drop("mapping_extensions_chain")
        return None
    if _off_grid(report, "chain", (x, y), (tx, ty)):
        return None
    if c not in (0, 1) or not _valid_direction(d) or sc < 1:
        report.drop("invalid_chain")
        return None
    # A squish of 0 crashes the game; treat it as unset.
    return Chain(b, Color(c), x, y, CutDirection(d), tb, tx, ty, sc, s if s > 0 else 1.0)


# --- Custom data --------------------------------------------------------------


def check_object_custom(report: ReadReport, custom: Any) -> None:
    if not isinstance(custom, dict) or not custom:
        return
    keys = set(strip_underscores(custom))
    if keys & NOODLE_OBJECT_KEYS:
        report.noodle = True
    if "color" in keys:
        report.chroma = True
    report.drop("object_custom_data")


def check_difficulty_custom(report: ReadReport, custom: Any) -> None:
    if not isinstance(custom, dict):
        return
    data = strip_underscores(custom)
    if NOODLE_DIFFICULTY_KEYS & {k for k, v in data.items() if v}:
        report.noodle = True
    if CHROMA_DIFFICULTY_KEYS & {k for k, v in data.items() if v}:
        report.chroma = True
    for event in as_list(data.get("customEvents")):
        kind = first(as_dict(event), "_type", "t", "type")
        if kind in NOODLE_EVENT_TYPES:
            report.noodle = True
        elif kind is not None:
            report.chroma = True


def split_chroma(custom: Any) -> tuple[tuple[float, float, float, float] | None, dict | None]:
    """Split event customData into a Chroma RGBA colour and the remaining keys."""
    if not isinstance(custom, dict) or not custom:
        return None, None
    data = strip_underscores(custom)
    raw = data.pop("color", None)
    color = None
    if isinstance(raw, list) and len(raw) >= 3:
        try:
            rgba = [float(v) for v in raw[:4]]
        except (TypeError, ValueError):
            rgba = []
        if rgba:
            rgba += [1.0] * (4 - len(rgba))
            color = (rgba[0], rgba[1], rgba[2], rgba[3])
    return color, (data or None)


def chroma_custom(event: BasicEvent, underscore: bool) -> dict | None:
    data: dict[str, Any] = {}
    if event.color is not None:
        data["color"] = [out(v) for v in event.color]
    if event.custom_data:
        data.update(event.custom_data)
    if not data:
        return None
    return with_underscores(data) if underscore else data


def rotation_degrees(report: ReadReport, value: int) -> int | None:
    if 0 <= value < len(ROTATION_STEPS):
        return ROTATION_STEPS[value]
    if 1000 <= value <= 1720:  # Mapping Extensions precise rotation
        report.mapping_extensions = True
        return value - 1360
    return None


def route_basic_event(
    parsed: ParsedDifficulty,
    report: ReadReport,
    b: float,
    et: int,
    value: int,
    float_value: float,
    custom: Any,
) -> None:
    """Sort a classic event into lights, colour boost, rotation or BPM change."""
    ls = parsed.lightshow
    assert ls is not None
    if et == EVENT_BOOST:
        ls.boost_events.append(ColorBoostEvent(b, value == 1))
    elif et in (EVENT_EARLY_ROTATION, EVENT_LATE_ROTATION):
        degrees = rotation_degrees(report, value)
        if degrees is None:
            report.drop("invalid_rotation")
        else:
            parsed.rotations.append(RotationEvent(b, degrees, early=et == EVENT_EARLY_ROTATION))
    elif et == EVENT_BPM:
        if float_value > 0:
            parsed.bpm_changes.append(BpmChange(b, float_value))
    elif value >= LEGACY_CHROMA_VALUE:
        report.chroma = True
        report.drop("legacy_chroma_event")
    else:
        color, rest = split_chroma(custom)
        ls.basic_events.append(BasicEvent(b, et, value, float_value, color, rest))
        if color is not None or rest:
            ls.uses_chroma = True
            report.chroma = True


# --- Event box groups -----------------------------------------------------------
#
# ``EventBoxGroup.boxes`` holds event boxes in the v3.3 JSON shape, normalised so
# every key is present. The one difference from v3 files: fx boxes keep their
# float events inline under "l" instead of indices into ``_fxEventsCollection``.

FILTER_KEYS = ("c", "f", "p", "t", "r", "n", "s", "l", "d")
FILTER_FLOATS = frozenset({"l"})

# kind -> (box keys, box float keys, events key, event keys, event float keys)
BOX_SPECS: dict[
    str, tuple[tuple[str, ...], frozenset[str], str, tuple[str, ...], frozenset[str]]
] = {
    "color": (
        ("w", "d", "r", "t", "b", "i"),
        frozenset({"w", "r"}),
        "e",
        ("b", "i", "c", "s", "f", "sb", "sf"),
        frozenset({"b", "s", "sb"}),
    ),
    "rotation": (
        ("w", "d", "s", "t", "b", "i", "a", "r"),
        frozenset({"w", "s"}),
        "l",
        ("b", "p", "e", "r", "o", "l"),
        frozenset({"b", "r"}),
    ),
    "translation": (
        ("w", "d", "s", "t", "b", "i", "a", "r"),
        frozenset({"w", "s"}),
        "l",
        ("b", "p", "e", "t"),
        frozenset({"b", "t"}),
    ),
    "fx": (
        ("w", "d", "s", "t", "b", "i"),
        frozenset({"w", "s"}),
        "l",
        ("b", "p", "i", "v"),
        frozenset({"b", "v"}),
    ),
}


def _canon(data: Any, keys: tuple[str, ...], floats: frozenset[str]) -> dict[str, int | float]:
    data = as_dict(data)
    result: dict[str, int | float] = {}
    for key in keys:
        value = data.get(key, 0)
        try:
            result[key] = beat(value) if key in floats else as_int(value)
        except (TypeError, ValueError):
            result[key] = 0
    return result


def canon_box(kind: str, box: Any, fx_events: list[Any] | None = None) -> dict[str, Any]:
    """Normalise one event box. ``fx_events`` resolves v3 fx index references."""
    keys, floats, events_key, event_keys, event_floats = BOX_SPECS[kind]
    box = as_dict(box)
    result: dict[str, Any] = {"f": _canon(box.get("f"), FILTER_KEYS, FILTER_FLOATS)}
    result.update(_canon(box, keys, floats))
    events = as_list(box.get(events_key))
    if fx_events is not None:
        events = [fx_events[i] for i in events if isinstance(i, int) and 0 <= i < len(fx_events)]
    result[events_key] = [_canon(e, event_keys, event_floats) for e in events]
    return result


def make_box_group(kind: str, raw: Any, fx_events: list[Any] | None = None) -> EventBoxGroup:
    raw = as_dict(raw)
    boxes = tuple(canon_box(kind, box, fx_events) for box in as_list(raw.get("e")))
    return EventBoxGroup(beat(raw.get("b", 0)), as_int(raw.get("g", 0)), kind, boxes)  # type: ignore[arg-type]
