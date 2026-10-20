"""v4 beatmap, lightshow and audio data files.

v4 splits objects (``colorNotes``) from their deduplicated data (``colorNotesData``),
moves lighting into a lightshow file shared across difficulties, and stores BPM
changes as sample-index regions in the audio data file (``BPMInfo.dat``).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from bsmapper.formats._common import (
    BOX_SPECS,
    FILTER_FLOATS,
    FILTER_KEYS,
    ParsedDifficulty,
    _canon,
    as_dict,
    as_int,
    as_list,
    beat,
    canon_box,
    check_difficulty_custom,
    check_object_custom,
    chroma_custom,
    make_arc,
    make_bomb,
    make_chain,
    make_note,
    make_obstacle,
    out,
    route_basic_event,
    sort_lightshow,
)
from bsmapper.formats.report import ReadReport
from bsmapper.schema import (
    Beatmap,
    BpmChange,
    ColorBoostEvent,
    DifficultyMap,
    EventBoxGroup,
    Lightshow,
    RotationEvent,
)

log = logging.getLogger(__name__)

VERSION = "4.0.0"
DEFAULT_FREQUENCY = 44100

KIND_TYPES = {"color": 1, "rotation": 2, "translation": 3, "fx": 4}
TYPE_KINDS = {v: k for k, v in KIND_TYPES.items()}
BOX_COLLECTIONS = {
    "color": ("lightColorEventBoxes", "lightColorEvents"),
    "rotation": ("lightRotationEventBoxes", "lightRotationEvents"),
    "translation": ("lightTranslationEventBoxes", "lightTranslationEvents"),
    "fx": ("fxEventBoxes", "floatFxEvents"),
}


class _Pool:
    """Deduplicating list for the v4 ``*Data`` collections."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}

    def add(self, item: dict[str, Any]) -> int:
        key = json.dumps(item, sort_keys=True)
        if key not in self._index:
            self._index[key] = len(self.items)
            self.items.append(item)
        return self._index[key]


def _ref(collection: list[Any], index: Any) -> dict[str, Any] | None:
    try:
        i = as_int(index)
    except (TypeError, ValueError):
        return None
    if 0 <= i < len(collection) and isinstance(collection[i], dict):
        return collection[i]
    return None


# --- Beatmap file ---------------------------------------------------------------


def parse_beatmap(data: dict[str, Any], report: ReadReport) -> ParsedDifficulty:
    parsed = ParsedDifficulty(version=str(data.get("version", VERSION)))
    lanes: list[tuple[float, int]] = []  # (beat, rotation lane) of every object

    def g(raw: dict, key: str, default: Any = 0) -> int:
        return as_int(raw.get(key, default))

    def f(raw: dict, key: str, default: Any = 0) -> float:
        return beat(raw.get(key, default))

    def collect(
        key: str, data_key: str, build, target: list, beat_key: str = "b", ref_key: str = "i"
    ):
        pool = as_list(data.get(data_key))
        for raw in as_list(data.get(key)):
            if not isinstance(raw, dict):
                report.drop("malformed")
                continue
            ref = _ref(pool, raw.get(ref_key, 0))
            if ref is None:
                report.drop("malformed")
                continue
            try:
                obj = build(raw, ref)
                b, lane = f(raw, beat_key), g(raw, "r" if beat_key == "b" else "hr")
            except (TypeError, ValueError):
                report.drop("malformed")
                continue
            if obj is None:  # already counted by the make_* validator
                continue
            check_object_custom(report, raw.get("customData"))
            target.append(obj)
            lanes.append((b, lane))

    notes_data = as_list(data.get("colorNotesData"))
    collect(
        "colorNotes",
        "colorNotesData",
        lambda r, d: make_note(
            report, f(r, "b"), g(d, "x"), g(d, "y"), g(d, "c"), g(d, "d"), g(d, "a")
        ),
        parsed.notes,
    )
    collect(
        "bombNotes",
        "bombNotesData",
        lambda r, d: make_bomb(report, f(r, "b"), g(d, "x"), g(d, "y")),
        parsed.bombs,
    )
    collect(
        "obstacles",
        "obstaclesData",
        lambda r, d: make_obstacle(
            report, f(r, "b"), f(d, "d"), g(d, "x"), g(d, "y"), g(d, "w"), g(d, "h")
        ),
        parsed.obstacles,
    )

    def build_arc(r: dict, d: dict):
        head, tail = _ref(notes_data, r.get("hi", 0)), _ref(notes_data, r.get("ti", 0))
        if head is None or tail is None:
            report.drop("malformed")
            return None
        return make_arc(
            report,
            g(head, "c"),
            f(r, "hb"),
            g(head, "x"),
            g(head, "y"),
            g(head, "d"),
            f(d, "m", 1),
            f(r, "tb"),
            g(tail, "x"),
            g(tail, "y"),
            g(tail, "d"),
            f(d, "tm", 1),
            g(d, "a"),
        )

    def build_chain(r: dict, d: dict):
        head = _ref(notes_data, r.get("i", 0))
        if head is None:
            report.drop("malformed")
            return None
        return make_chain(
            report,
            g(head, "c"),
            f(r, "hb"),
            g(head, "x"),
            g(head, "y"),
            g(head, "d"),
            f(r, "tb"),
            g(d, "tx"),
            g(d, "ty"),
            g(d, "c", 1),
            f(d, "s", 1),
        )

    collect("arcs", "arcsData", build_arc, parsed.arcs, beat_key="hb", ref_key="ai")
    collect("chains", "chainsData", build_chain, parsed.chains, beat_key="hb", ref_key="ci")

    rotation_data = as_list(data.get("spawnRotationsData"))
    for raw in as_list(data.get("spawnRotations")):
        ref = _ref(rotation_data, as_dict(raw).get("i", 0))
        if ref is None:
            report.drop("malformed")
            continue
        parsed.rotations.append(RotationEvent(f(raw, "b"), g(ref, "r"), early=g(ref, "t") == 0))
    if not parsed.rotations:
        parsed.rotations = _rotations_from_lanes(lanes)

    report.drop("njs_event", len(as_list(data.get("njsEvents"))))
    check_difficulty_custom(report, data.get("customData"))
    parsed.sort()
    return parsed


def _rotations_from_lanes(lanes: list[tuple[float, int]]) -> list[RotationEvent]:
    """Recover relative rotation events from per-object absolute lanes."""
    events = []
    current = 0
    for b, lane in sorted(lanes, key=lambda item: item[0]):
        if lane != current:
            events.append(RotationEvent(b, lane - current, early=True))
            current = lane
    return events


def _lane_at(rotations: list[RotationEvent], b: float) -> int:
    """Absolute lane of an object at beat ``b``: early events at ``b`` apply, late ones don't."""
    return sum(r.degrees for r in rotations if r.beat < b or (r.beat == b and r.early))


def write_beatmap(dmap: DifficultyMap) -> dict:
    notes, bombs, walls = _Pool(), _Pool(), _Pool()
    arcs_data, chains_data = _Pool(), _Pool()
    rotations = sorted(dmap.rotations, key=lambda r: r.beat)

    def lane(b: float) -> int:
        return _lane_at(rotations, b) if rotations else 0

    def note_index(x: int, y: int, c: int, d: int, a: int = 0) -> int:
        return notes.add({"x": x, "y": y, "c": c, "d": d, "a": a})

    color_notes = [
        {
            "b": out(n.beat),
            "r": lane(n.beat),
            "i": note_index(n.x, n.y, int(n.color), int(n.direction), n.angle_offset),
        }
        for n in dmap.notes
    ]
    bomb_notes = [
        {"b": out(b.beat), "r": lane(b.beat), "i": bombs.add({"x": b.x, "y": b.y})}
        for b in dmap.bombs
    ]
    obstacles = [
        {
            "b": out(o.beat),
            "r": lane(o.beat),
            "i": walls.add({"d": out(o.duration), "x": o.x, "y": o.y, "w": o.width, "h": o.height}),
        }
        for o in dmap.obstacles
    ]
    arcs = [
        {
            "hb": out(a.beat),
            "tb": out(a.tail_beat),
            "hr": lane(a.beat),
            "tr": lane(a.tail_beat),
            "hi": note_index(a.x, a.y, int(a.color), int(a.direction)),
            "ti": note_index(a.tail_x, a.tail_y, int(a.color), int(a.tail_direction)),
            "ai": arcs_data.add(
                {"m": out(a.head_multiplier), "tm": out(a.tail_multiplier), "a": a.mid_anchor}
            ),
        }
        for a in dmap.arcs
    ]
    chains = [
        {
            "hb": out(c.beat),
            "tb": out(c.tail_beat),
            "hr": lane(c.beat),
            "tr": lane(c.tail_beat),
            "i": note_index(c.x, c.y, int(c.color), int(c.direction)),
            "ci": chains_data.add(
                {"tx": c.tail_x, "ty": c.tail_y, "c": c.slice_count, "s": out(c.squish)}
            ),
        }
        for c in dmap.chains
    ]
    return {
        "version": VERSION,
        "colorNotes": color_notes,
        "colorNotesData": notes.items,
        "bombNotes": bomb_notes,
        "bombNotesData": bombs.items,
        "obstacles": obstacles,
        "obstaclesData": walls.items,
        "arcs": arcs,
        "arcsData": arcs_data.items,
        "chains": chains,
        "chainsData": chains_data.items,
        "spawnRotations": [],
        "spawnRotationsData": [],
    }


# --- Event box conversion (canonical v3 shape <-> v4) ----------------------------


def _box_to_v4(kind: str, box: dict[str, Any]) -> dict[str, Any]:
    result = {"w": box["w"], "d": box["d"], "s": box["r" if kind == "color" else "s"]}
    result.update({"t": box["t"], "b": box["b"], "e": box["i"]})
    if kind in ("rotation", "translation"):
        result["a"], result["f"] = box["a"], box["r"]
    return result


def _box_from_v4(kind: str, box: dict[str, Any]) -> dict[str, Any]:
    result = {"w": box.get("w", 0), "d": box.get("d", 0)}
    result["r" if kind == "color" else "s"] = box.get("s", 0)
    result.update({"t": box.get("t", 0), "b": box.get("b", 0), "i": box.get("e", 0)})
    if kind in ("rotation", "translation"):
        result["a"], result["r"] = box.get("a", 0), box.get("f", 0)
    return result


def _event_to_v4(kind: str, event: dict[str, Any]) -> dict[str, Any]:
    if kind == "color":
        # v3 transition: 0 instant, 1 transition, 2 extend. v4: p 0 transition / 1 extend,
        # with easing -1 (none) meaning an instant change.
        transition = event["i"]
        return {
            "p": 1 if transition == 2 else 0,
            "e": 0 if transition == 1 else -1,
            "c": event["c"],
            "b": event["s"],
            "f": event["f"],
            "sb": event["sb"],
            "sf": event["sf"],
        }
    if kind == "rotation":
        return {"p": event["p"], "e": event["e"], "r": event["r"], "d": event["o"], "l": event["l"]}
    if kind == "translation":
        return {"p": event["p"], "e": event["e"], "t": event["t"]}
    return {"p": event["p"], "e": event["i"], "v": event["v"]}


def _event_from_v4(kind: str, offset: Any, event: dict[str, Any]) -> dict[str, Any]:
    if kind == "color":
        p, e = as_int(event.get("p", 0)), as_int(event.get("e", -1))
        transition = 2 if p == 1 else (0 if e == -1 else 1)
        return {
            "b": offset,
            "i": transition,
            "c": event.get("c", 0),
            "s": event.get("b", 0),
            "f": event.get("f", 0),
            "sb": event.get("sb", 0),
            "sf": event.get("sf", 0),
        }
    if kind == "rotation":
        return {
            "b": offset,
            "p": event.get("p", 0),
            "e": event.get("e", 0),
            "r": event.get("r", 0),
            "o": event.get("d", 0),
            "l": event.get("l", 0),
        }
    if kind == "translation":
        return {"b": offset, "p": event.get("p", 0), "e": event.get("e", 0), "t": event.get("t", 0)}
    return {"b": offset, "p": event.get("p", 0), "i": event.get("e", 0), "v": event.get("v", 0)}


# --- Lightshow file -------------------------------------------------------------


def parse_lightshow(data: dict[str, Any], report: ReadReport) -> Lightshow:
    scratch = ParsedDifficulty(lightshow=Lightshow())
    ls = scratch.lightshow
    assert ls is not None

    events_data = as_list(data.get("basicEventsData"))
    for raw in as_list(data.get("basicEvents")):
        ref = _ref(events_data, as_dict(raw).get("i", 0))
        if ref is None:
            report.drop("malformed")
            continue
        try:
            route_basic_event(
                scratch,
                report,
                beat(raw.get("b", 0)),
                as_int(ref.get("t", 0)),
                as_int(ref.get("i", 0)),
                beat(ref.get("f", 1)),
                raw.get("customData", ref.get("customData")),
            )
        except (TypeError, ValueError):
            report.drop("malformed")
    # Legacy rotation/BPM events have no meaning in a shared v4 lightshow.
    report.drop("legacy_event_in_lightshow", len(scratch.rotations) + len(scratch.bpm_changes))

    boost_data = as_list(data.get("colorBoostEventsData"))
    for raw in as_list(data.get("colorBoostEvents")):
        ref = _ref(boost_data, as_dict(raw).get("i", 0))
        if ref is None:
            report.drop("malformed")
            continue
        ls.boost_events.append(ColorBoostEvent(beat(raw.get("b", 0)), bool(ref.get("b", False))))

    filters = as_list(data.get("indexFilters"))
    for raw in as_list(data.get("eventBoxGroups")):
        raw = as_dict(raw)
        kind = TYPE_KINDS.get(as_int(raw.get("t", 0)))
        if kind is None:
            report.drop("unknown_event_box_group")
            continue
        box_pool = as_list(data.get(BOX_COLLECTIONS[kind][0]))
        event_pool = as_list(data.get(BOX_COLLECTIONS[kind][1]))
        events_key = BOX_SPECS[kind][2]
        boxes = []
        for entry in as_list(raw.get("e")):
            entry = as_dict(entry)
            box = _ref(box_pool, entry.get("e", 0))
            filt = _ref(filters, entry.get("f", 0))
            if box is None or filt is None:
                report.drop("malformed")
                continue
            v3_box: dict[str, Any] = {"f": filt, **_box_from_v4(kind, box)}
            v3_box[events_key] = []
            for item in as_list(entry.get("l")):
                event = _ref(event_pool, as_dict(item).get("i", 0))
                if event is None:
                    report.drop("malformed")
                    continue
                v3_box[events_key].append(_event_from_v4(kind, as_dict(item).get("b", 0), event))
            boxes.append(canon_box(kind, v3_box))
        ls.event_box_groups.append(
            EventBoxGroup(beat(raw.get("b", 0)), as_int(raw.get("g", 0)), kind, tuple(boxes))  # type: ignore[arg-type]
        )

    report.drop("waypoint", len(as_list(data.get("waypoints"))))
    sort_lightshow(ls)
    return ls


def write_lightshow(ls: Lightshow) -> dict:
    basic_data, boost_data, filters = _Pool(), _Pool(), _Pool()
    box_pools = {kind: (_Pool(), _Pool()) for kind in BOX_COLLECTIONS}

    basic = []
    for e in ls.basic_events:
        event: dict[str, Any] = {
            "b": out(e.beat),
            "i": basic_data.add({"t": e.type, "i": e.value, "f": out(e.float_value)}),
        }
        custom = chroma_custom(e, underscore=False)
        if custom:
            event["customData"] = custom
        basic.append(event)

    groups = []
    for group in ls.event_box_groups:
        boxes, events = box_pools[group.kind]
        events_key = BOX_SPECS[group.kind][2]
        entries = []
        for box in group.boxes:
            entries.append(
                {
                    "f": filters.add(_canon(box["f"], FILTER_KEYS, FILTER_FLOATS)),
                    "e": boxes.add(_box_to_v4(group.kind, box)),
                    "l": [
                        {"b": event["b"], "i": events.add(_event_to_v4(group.kind, event))}
                        for event in box[events_key]
                    ],
                }
            )
        groups.append(
            {"b": out(group.beat), "g": group.group_id, "t": KIND_TYPES[group.kind], "e": entries}
        )

    result: dict[str, Any] = {
        "version": VERSION,
        "basicEvents": basic,
        "basicEventsData": basic_data.items,
        "colorBoostEvents": [
            {"b": out(e.beat), "i": boost_data.add({"b": int(e.on)})} for e in ls.boost_events
        ],
        "colorBoostEventsData": boost_data.items,
        "waypoints": [],
        "waypointsData": [],
        "basicEventTypesWithKeywords": {"d": []},
        "eventBoxGroups": groups,
        "indexFilters": filters.items,
    }
    for kind, (box_key, event_key) in BOX_COLLECTIONS.items():
        result[box_key] = box_pools[kind][0].items
        result[event_key] = box_pools[kind][1].items
    result["useNormalEventsAsCompatibleEvents"] = True
    return result


# --- Audio data (BPM regions) ------------------------------------------------------


def _region_bpm(beats: float, samples: int, frequency: float) -> float:
    """BPM of a region, preferring the simplest value that reproduces its sample count."""
    raw = beats * 60.0 * frequency / samples
    for decimals in range(4):
        candidate = round(raw, decimals)
        if candidate > 0 and abs(beats * 60.0 / candidate * frequency - samples) <= 1.0:
            return candidate
    return round(raw, 3)


def parse_audio_data(data: dict[str, Any], base_bpm: float, report: ReadReport) -> list[BpmChange]:
    """BPM changes from v4 ``bpmData`` (or v2 BPMInfo ``_regions``)."""
    frequency = float(data.get("songFrequency", data.get("_songFrequency", DEFAULT_FREQUENCY)))
    regions = as_list(data.get("bpmData", data.get("_regions")))
    changes = []
    for raw in regions:
        raw = as_dict(raw)
        try:
            si = as_int(raw.get("si", raw.get("_startSampleIndex", 0)))
            ei = as_int(raw.get("ei", raw.get("_endSampleIndex", 0)))
            sb = beat(raw.get("sb", raw.get("_startBeat", 0)))
            eb = beat(raw.get("eb", raw.get("_endBeat", 0)))
        except (TypeError, ValueError):
            report.drop("malformed")
            continue
        if ei <= si or eb <= sb:
            continue
        changes.append(BpmChange(sb, _region_bpm(eb - sb, ei - si, frequency)))
    if changes and changes[0].beat <= 0 and abs(changes[0].bpm - base_bpm) < 1e-3:
        changes = changes[1:]
    return changes


def write_audio_data(
    beatmap: Beatmap, duration_s: float, frequency: int = DEFAULT_FREQUENCY
) -> dict:
    total = max(1, round(duration_s * frequency))
    starts = sorted({0.0, *(c.beat for c in beatmap.bpm_changes if c.beat > 0)})
    end_beat = max(beatmap.seconds_to_beat(duration_s), starts[-1] + 1.0)
    bounds = [*starts, end_beat]
    regions = []
    for sb, eb in zip(bounds, bounds[1:], strict=False):
        si = round(beatmap.beat_to_seconds(sb) * frequency)
        ei = round(beatmap.beat_to_seconds(eb) * frequency)
        regions.append({"si": si, "ei": ei, "sb": out(sb), "eb": out(eb)})
    return {
        "version": VERSION,
        "songChecksum": "",
        "songSampleCount": max(total, regions[-1]["ei"]),
        "songFrequency": frequency,
        "bpmData": regions,
        "lufsData": [],
    }
