"""v3 difficulty files (``version`` 3.x, abbreviated keys)."""

from __future__ import annotations

from typing import Any

from bsmapper.formats._common import (
    ParsedDifficulty,
    as_dict,
    as_int,
    as_list,
    beat,
    check_difficulty_custom,
    check_object_custom,
    chroma_custom,
    first,
    make_arc,
    make_bomb,
    make_box_group,
    make_chain,
    make_note,
    make_obstacle,
    out,
    route_basic_event,
)
from bsmapper.formats.report import ReadReport
from bsmapper.schema import (
    BpmChange,
    ColorBoostEvent,
    DifficultyMap,
    EventBoxGroup,
    Lightshow,
    RotationEvent,
)

VERSION = "3.3.0"
GROUP_COLLECTIONS = {
    "color": "lightColorEventBoxGroups",
    "rotation": "lightRotationEventBoxGroups",
    "translation": "lightTranslationEventBoxGroups",
    "fx": "vfxEventBoxGroups",
}


def parse(data: dict[str, Any], report: ReadReport) -> ParsedDifficulty:
    parsed = ParsedDifficulty(version=str(data.get("version", "3.0.0")), lightshow=Lightshow())
    ls = parsed.lightshow
    assert ls is not None

    def g(raw: dict, key: str, default: Any = 0) -> int:
        return as_int(raw.get(key, default))

    def f(raw: dict, key: str, default: Any = 0) -> float:
        return beat(raw.get(key, default))

    def collect(key: str, build, target: list | None = None) -> None:
        for raw in as_list(data.get(key)):
            if not isinstance(raw, dict) or "b" not in raw:
                report.drop("malformed")
                continue
            try:
                obj = build(raw)
            except (TypeError, ValueError):
                report.drop("malformed")
                continue
            if key != "basicBeatmapEvents":  # event customData is Chroma, handled on routing
                check_object_custom(report, raw.get("customData"))
            if obj is not None and target is not None:
                target.append(obj)

    collect(
        "colorNotes",
        lambda r: make_note(
            report, f(r, "b"), g(r, "x"), g(r, "y"), g(r, "c"), g(r, "d"), g(r, "a")
        ),
        parsed.notes,
    )
    collect("bombNotes", lambda r: make_bomb(report, f(r, "b"), g(r, "x"), g(r, "y")), parsed.bombs)
    collect(
        "obstacles",
        lambda r: make_obstacle(
            report, f(r, "b"), f(r, "d"), g(r, "x"), g(r, "y"), g(r, "w"), g(r, "h")
        ),
        parsed.obstacles,
    )
    collect(
        "sliders",
        lambda r: make_arc(
            report,
            g(r, "c"),
            f(r, "b"),
            g(r, "x"),
            g(r, "y"),
            g(r, "d"),
            f(r, "mu", 1),
            f(r, "tb"),
            g(r, "tx"),
            g(r, "ty"),
            g(r, "tc"),
            f(r, "tmu", 1),
            g(r, "m"),
        ),
        parsed.arcs,
    )
    collect(
        "burstSliders",
        lambda r: make_chain(
            report,
            g(r, "c"),
            f(r, "b"),
            g(r, "x"),
            g(r, "y"),
            g(r, "d"),
            f(r, "tb"),
            g(r, "tx"),
            g(r, "ty"),
            g(r, "sc", 1),
            f(r, "s", 1),
        ),
        parsed.chains,
    )
    collect(
        "rotationEvents",
        lambda r: RotationEvent(f(r, "b"), g(r, "r"), early=g(r, "e") == 0),
        parsed.rotations,
    )
    collect(
        "bpmEvents",
        lambda r: BpmChange(f(r, "b"), f(r, "m")) if f(r, "m") > 0 else None,
        parsed.bpm_changes,
    )
    collect(
        "basicBeatmapEvents",
        lambda r: route_basic_event(
            parsed, report, f(r, "b"), g(r, "et"), g(r, "i"), f(r, "f", 1), r.get("customData")
        ),
    )
    collect(
        "colorBoostBeatmapEvents",
        lambda r: ColorBoostEvent(f(r, "b"), bool(r.get("o", False))),
        ls.boost_events,
    )

    fx_collection = as_dict(first(data, "_fxEventsCollection", "_fxEventsCollections", default={}))
    fx_events = resolve_fx_events(
        as_list(fx_collection.get("_fl")), as_list(fx_collection.get("_il"))
    )
    for kind, key in GROUP_COLLECTIONS.items():
        for raw in as_list(data.get(key)):
            try:
                ls.event_box_groups.append(
                    make_box_group(kind, raw, fx_events if kind == "fx" else None)
                )
            except (TypeError, ValueError):
                report.drop("malformed")

    report.drop("waypoint", len(as_list(data.get("waypoints"))))
    check_difficulty_custom(report, data.get("customData"))
    custom = as_dict(data.get("customData"))
    for raw in as_list(first(custom, "BPMChanges", "bpmChanges")):
        raw = as_dict(raw)
        try:
            change = BpmChange(
                beat(first(raw, "b", "_time")), float(first(raw, "m", "_BPM", "_bpm"))
            )
        except (TypeError, ValueError):
            report.drop("malformed")
            continue
        if change.bpm > 0:
            parsed.editor_bpm_changes.append(change)

    parsed.sort()
    return parsed


def resolve_fx_events(float_events: list[Any], int_events: list[Any]) -> list[Any]:
    """The fx events a box index can refer to.

    Whether a vfx box reads ``_fl`` or ``_il`` depends on the environment's light group,
    which the file does not say. Most groups are float groups, so ``_fl`` wins; an index
    only ``_il`` covers falls back to that int event, read as a float one.
    """
    events = list(float_events)
    for raw in int_events[len(events) :]:
        raw = as_dict(raw)
        events.append({"b": raw.get("b", 0), "p": raw.get("p", 0), "i": 0, "v": raw.get("v", 0)})
    return events


def int_fx_events(fx_events: list[dict]) -> list[dict]:
    """``_il`` mirroring ``_fl`` index for index, so every vfx box index is valid whether
    the environment's group is a float or an int group (the game crashes on a missing
    ``_il`` index, e.g. in TheWeekndEnvironment)."""
    return [
        {"b": event["b"], "p": int(event.get("p", 0)), "v": round(event.get("v", 0))}
        for event in fx_events
    ]


def box_group_json(group: EventBoxGroup, fx_events: list[dict]) -> dict[str, Any]:
    """One event box group in v3 JSON. Inline fx events are moved to ``fx_events``."""
    boxes = []
    for box in group.boxes:
        box = dict(box)
        if group.kind == "fx":
            indices = []
            for event in box.get("l", []):
                indices.append(len(fx_events))
                fx_events.append(dict(event))
            box["l"] = indices
        boxes.append(box)
    return {"b": out(group.beat), "g": group.group_id, "e": boxes}


def write(dmap: DifficultyMap, lightshow: Lightshow, bpm_changes: list[BpmChange]) -> dict:
    basic = []
    for e in lightshow.basic_events:
        event: dict[str, Any] = {
            "b": out(e.beat),
            "et": e.type,
            "i": e.value,
            "f": out(e.float_value),
        }
        custom = chroma_custom(e, underscore=False)
        if custom:
            event["customData"] = custom
        basic.append(event)

    fx_events: list[dict] = []
    groups: dict[str, list[dict]] = {key: [] for key in GROUP_COLLECTIONS.values()}
    for group in lightshow.event_box_groups:
        groups[GROUP_COLLECTIONS[group.kind]].append(box_group_json(group, fx_events))

    return {
        "version": VERSION,
        "bpmEvents": [{"b": out(c.beat), "m": out(c.bpm)} for c in bpm_changes],
        "rotationEvents": [
            {"b": out(r.beat), "e": 0 if r.early else 1, "r": r.degrees} for r in dmap.rotations
        ],
        "colorNotes": [
            {
                "b": out(n.beat),
                "x": n.x,
                "y": n.y,
                "c": int(n.color),
                "d": int(n.direction),
                "a": n.angle_offset,
            }
            for n in dmap.notes
        ],
        "bombNotes": [{"b": out(b.beat), "x": b.x, "y": b.y} for b in dmap.bombs],
        "obstacles": [
            {
                "b": out(o.beat),
                "d": out(o.duration),
                "x": o.x,
                "y": o.y,
                "w": o.width,
                "h": o.height,
            }
            for o in dmap.obstacles
        ],
        "sliders": [
            {
                "c": int(a.color),
                "b": out(a.beat),
                "x": a.x,
                "y": a.y,
                "d": int(a.direction),
                "mu": out(a.head_multiplier),
                "tb": out(a.tail_beat),
                "tx": a.tail_x,
                "ty": a.tail_y,
                "tc": int(a.tail_direction),
                "tmu": out(a.tail_multiplier),
                "m": a.mid_anchor,
            }
            for a in dmap.arcs
        ],
        "burstSliders": [
            {
                "c": int(c.color),
                "b": out(c.beat),
                "x": c.x,
                "y": c.y,
                "d": int(c.direction),
                "tb": out(c.tail_beat),
                "tx": c.tail_x,
                "ty": c.tail_y,
                "sc": c.slice_count,
                "s": out(c.squish),
            }
            for c in dmap.chains
        ],
        "waypoints": [],
        "basicBeatmapEvents": basic,
        "colorBoostBeatmapEvents": [{"b": out(e.beat), "o": e.on} for e in lightshow.boost_events],
        **groups,
        "basicEventTypesWithKeywords": {"d": []},
        "_fxEventsCollection": {"_fl": fx_events, "_il": int_fx_events(fx_events)},
        "useNormalEventsAsCompatibleEvents": True,
    }
