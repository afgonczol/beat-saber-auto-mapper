from __future__ import annotations

from bsmapper.generation import dedupe, fix_with_rules, postprocess
from bsmapper.rules import RuleChecker
from bsmapper.schema import (
    Beatmap,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
    RotationEvent,
)

D, U = CutDirection.DOWN, CutDirection.UP
R, B = Color.RED, Color.BLUE


def beatmap() -> Beatmap:
    return Beatmap(song_name="t", bpm=120.0)


def dmap(*notes: Note, mode=Characteristic.STANDARD, **kw) -> DifficultyMap:
    return DifficultyMap(mode, Difficulty.EXPERT, notes=list(notes), **kw)


def errors(d: DifficultyMap) -> list:
    return [i for i in RuleChecker().check(d, beatmap()) if i.severity == "error"]


def test_snap_and_dedupe() -> None:
    m = dmap(
        Note(4.0004, 1, 0, R, D),
        Note(4.0, 1, 0, B, D),  # same cell, same tick as the red one: dropped
        Note(5.0, 2, 0, B, D),
        Note(5.0, 2, 0, B, D),
    )
    out, report = postprocess(m, beatmap())
    assert [n.beat for n in out.notes] == [4.0, 5.0]
    assert report.duplicates == 2
    assert report.errors == []


def test_parity_break_is_repaired() -> None:
    # Two downs in a row on the same hand, half a beat apart: a parity break.
    m = dmap(Note(4.0, 1, 0, R, D), Note(4.5, 1, 0, R, D))
    assert errors(m)
    out, repaired, dropped = fix_with_rules(m, beatmap())
    assert len(out.notes) == 2 and not dropped
    assert sum(repaired.values()) == 1
    assert errors(out) == []


def test_unfixable_objects_are_dropped() -> None:
    # Far too many notes for Expert's NPS ceiling: the overflow goes.
    notes = [Note(4 + i / 8, i % 4, i % 3, Color(i % 2), CutDirection.ANY) for i in range(200)]
    out, report = postprocess(dmap(*notes), beatmap())
    assert report.dropped
    assert report.errors == []
    assert len(out.notes) < 200


def test_rotation_whiplash_is_softened() -> None:
    rots = [RotationEvent(4.0, 90), RotationEvent(4.5, -105)]  # 105° back within 0.25 s
    m = dmap(mode=Characteristic.DEGREE_360, rotations=rots)
    assert errors(m)
    out, repaired, dropped = fix_with_rules(m, beatmap())
    assert out.rotations == [RotationEvent(4.0, 90), RotationEvent(4.5, -15)]
    assert repaired == {"rotation_whiplash": 1} and not dropped
    assert errors(out) == []


def test_postprocess_sets_njs_from_table() -> None:
    out, _ = postprocess(dmap(Note(4.0, 1, 0, R, D)), beatmap())
    assert out.njs == 16.0 and out.njs_offset != 0.0
    out, _ = postprocess(dmap(Note(4.0, 1, 0, R, D)), beatmap(), njs=20.0)
    assert out.njs == 20.0


def test_dedupe_one_rotation_per_tick() -> None:
    m = dmap(mode=Characteristic.DEGREE_360, rotations=[RotationEvent(4, 15), RotationEvent(4, 30)])
    out, removed = dedupe(m)
    assert removed == 1 and out.rotations == [RotationEvent(4, 15)]
