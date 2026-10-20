"""v2 difficulty files (``_version`` 2.x, underscore keys)."""

from __future__ import annotations

import logging
from typing import Any

from bsmapper.formats._common import (
    EVENT_BOOST,
    EVENT_BPM,
    EVENT_EARLY_ROTATION,
    EVENT_LATE_ROTATION,
    ROTATION_STEPS,
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
    make_note,
    make_obstacle,
    out,
    route_basic_event,
)
from bsmapper.formats.report import ReadReport
from bsmapper.schema import BpmChange, DifficultyMap, Lightshow, Obstacle

log = logging.getLogger(__name__)

VERSION = "2.6.0"
NOTE_BOMB = 3
# Legacy wall types: 0 = full height, 1 = crouch, 2 = explicit layer/height (2.6.0).
WALL_FULL = (0, 5)
WALL_CROUCH = (2, 3)


def parse(data: dict[str, Any], report: ReadReport) -> ParsedDifficulty:
    parsed = ParsedDifficulty(version=str(data.get("_version", "2.0.0")), lightshow=Lightshow())

    for raw in as_list(data.get("_notes")):
        try:
            b = beat(raw["_time"])
            x, y = as_int(raw.get("_lineIndex", 0)), as_int(raw.get("_lineLayer", 0))
            kind, d = as_int(raw.get("_type", 0)), as_int(raw.get("_cutDirection", 0))
        except (KeyError, TypeError, ValueError):
            report.drop("malformed")
            continue
        check_object_custom(report, raw.get("_customData"))
        if kind == NOTE_BOMB:
            bomb = make_bomb(report, b, x, y)
            if bomb:
                parsed.bombs.append(bomb)
        elif kind in (0, 1):
            note = make_note(report, b, x, y, kind, d)
            if note:
                parsed.notes.append(note)
        else:
            report.drop("invalid_note")

    for raw in as_list(data.get("_obstacles")):
        try:
            b, dur = beat(raw["_time"]), beat(raw.get("_duration", 0))
            x, w = as_int(raw.get("_lineIndex", 0)), as_int(raw.get("_width", 1))
            kind = as_int(raw.get("_type", 0))
            if kind == 0:
                y, h = WALL_FULL
            elif kind == 1:
                y, h = WALL_CROUCH
            elif kind == 2 or kind >= 1000:
                y, h = as_int(raw.get("_lineLayer", 0)), as_int(raw.get("_height", 5))
            else:
                report.drop("invalid_obstacle")
                continue
        except (KeyError, TypeError, ValueError):
            report.drop("malformed")
            continue
        check_object_custom(report, raw.get("_customData"))
        if kind >= 1000:  # Mapping Extensions precision wall
            report.mapping_extensions = True
            report.drop("mapping_extensions_obstacle")
            continue
        obstacle = make_obstacle(report, b, dur, x, y, w, h)
        if obstacle:
            parsed.obstacles.append(obstacle)

    for raw in as_list(data.get("_sliders")):
        try:
            arc = make_arc(
                report,
                as_int(raw.get("_colorType", 0)),
                beat(raw["_headTime"]),
                as_int(raw.get("_headLineIndex", 0)),
                as_int(raw.get("_headLineLayer", 0)),
                as_int(raw.get("_headCutDirection", 0)),
                beat(raw.get("_headControlPointLengthMultiplier", 1)),
                beat(raw["_tailTime"]),
                as_int(raw.get("_tailLineIndex", 0)),
                as_int(raw.get("_tailLineLayer", 0)),
                as_int(raw.get("_tailCutDirection", 0)),
                beat(raw.get("_tailControlPointLengthMultiplier", 1)),
                as_int(raw.get("_sliderMidAnchorMode", 0)),
            )
        except (KeyError, TypeError, ValueError):
            report.drop("malformed")
            continue
        check_object_custom(report, raw.get("_customData"))
        if arc:
            parsed.arcs.append(arc)

    for raw in as_list(data.get("_events")):
        try:
            b, et = beat(raw["_time"]), as_int(raw["_type"])
            value = as_int(raw.get("_value", 0))
            float_value = beat(raw.get("_floatValue", 1))  # absent before 2.5.0, means 1
        except (KeyError, TypeError, ValueError):
            report.drop("malformed")
            continue
        route_basic_event(parsed, report, b, et, value, float_value, raw.get("_customData"))

    report.drop("waypoint", len(as_list(data.get("_waypoints"))))

    custom = as_dict(data.get("_customData"))
    check_difficulty_custom(report, custom)
    for raw in as_list(first(custom, "_BPMChanges", "_bpmChanges", "_BpmChanges")):
        raw = as_dict(raw)
        try:
            change = BpmChange(beat(raw["_time"]), float(first(raw, "_BPM", "_bpm", "_Bpm")))
        except (KeyError, TypeError, ValueError):
            report.drop("malformed")
            continue
        if change.bpm > 0:
            parsed.editor_bpm_changes.append(change)

    parsed.sort()
    return parsed


def _wall_kind(o: Obstacle) -> int:
    if (o.y, o.height) == WALL_FULL:
        return 0
    if (o.y, o.height) == WALL_CROUCH:
        return 1
    return 2


def write(dmap: DifficultyMap, lightshow: Lightshow, bpm_changes: list[BpmChange]) -> dict:
    if dmap.chains:
        log.warning("v2 has no chains; dropping %d from %s", len(dmap.chains), dmap.difficulty)
    if any(n.angle_offset for n in dmap.notes):
        log.warning("v2 has no note angle offsets; they are dropped")
    if lightshow.event_box_groups:
        log.warning(
            "v2 has no group lighting; dropping %d event box groups",
            len(lightshow.event_box_groups),
        )

    notes = [
        {
            "_time": out(n.beat),
            "_lineIndex": n.x,
            "_lineLayer": n.y,
            "_type": int(n.color),
            "_cutDirection": int(n.direction),
        }
        for n in dmap.notes
    ] + [
        {
            "_time": out(b.beat),
            "_lineIndex": b.x,
            "_lineLayer": b.y,
            "_type": NOTE_BOMB,
            "_cutDirection": 0,
        }
        for b in dmap.bombs
    ]
    notes.sort(key=lambda n: n["_time"])

    obstacles = []
    for o in dmap.obstacles:
        kind = _wall_kind(o)
        wall: dict[str, Any] = {
            "_time": out(o.beat),
            "_type": kind,
            "_lineIndex": o.x,
            "_duration": out(o.duration),
            "_width": o.width,
        }
        if kind == 2:
            wall["_lineLayer"], wall["_height"] = o.y, o.height
        obstacles.append(wall)

    events: list[dict[str, Any]] = []
    for e in lightshow.basic_events:
        event: dict[str, Any] = {
            "_time": out(e.beat),
            "_type": e.type,
            "_value": e.value,
            "_floatValue": out(e.float_value),
        }
        custom = chroma_custom(e, underscore=True)
        if custom:
            event["_customData"] = custom
        events.append(event)
    for e in lightshow.boost_events:
        events.append(
            {"_time": out(e.beat), "_type": EVENT_BOOST, "_value": int(e.on), "_floatValue": 0}
        )
    for r in dmap.rotations:
        et = EVENT_EARLY_ROTATION if r.early else EVENT_LATE_ROTATION
        for step in _rotation_steps(r.degrees):
            events.append({"_time": out(r.beat), "_type": et, "_value": step, "_floatValue": 0})
    for c in bpm_changes:
        events.append(
            {"_time": out(c.beat), "_type": EVENT_BPM, "_value": 0, "_floatValue": out(c.bpm)}
        )
    events.sort(key=lambda e: e["_time"])

    sliders = [
        {
            "_colorType": int(a.color),
            "_headTime": out(a.beat),
            "_headLineIndex": a.x,
            "_headLineLayer": a.y,
            "_headCutDirection": int(a.direction),
            "_headControlPointLengthMultiplier": out(a.head_multiplier),
            "_tailTime": out(a.tail_beat),
            "_tailLineIndex": a.tail_x,
            "_tailLineLayer": a.tail_y,
            "_tailCutDirection": int(a.tail_direction),
            "_tailControlPointLengthMultiplier": out(a.tail_multiplier),
            "_sliderMidAnchorMode": a.mid_anchor,
        }
        for a in dmap.arcs
    ]

    return {
        "_version": VERSION,
        "_notes": notes,
        "_sliders": sliders,
        "_obstacles": obstacles,
        "_events": events,
        "_waypoints": [],
    }


def _rotation_steps(degrees: int) -> list[int]:
    """Encode a rotation as v2 event values, splitting anything beyond +-60 degrees."""
    steps = []
    sign = 1 if degrees > 0 else -1
    remaining = abs(degrees)
    while remaining >= 15:
        chunk = min(60, remaining - remaining % 15)
        steps.append(ROTATION_STEPS.index(sign * chunk))
        remaining -= chunk
    if remaining:
        log.warning("v2 rotations are multiples of 15; dropping %d degrees", sign * remaining)
    return steps
