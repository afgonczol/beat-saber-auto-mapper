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
    Obstacle,
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


def issues(d: DifficultyMap) -> list:
    return RuleChecker().check(d, beatmap())


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


def test_overlapping_walls_in_the_same_columns_merge() -> None:
    walls = [
        Obstacle(164.0, 0.5, 0, 0, 2, 5),
        Obstacle(164.125, 0.5, 0, 0, 2, 5),  # overlaps the first: one wall to 164.625
        Obstacle(164.0, 0.5, 0, 0, 2, 5),  # exact duplicate
        Obstacle(164.25, 0.25, 0, 2, 2, 3),  # same columns, overlapping height
        Obstacle(165.0, 1.0, 0, 0, 2, 5),  # starts after the merged wall ends
        Obstacle(164.0, 1.0, 2, 0, 2, 5),  # other columns
    ]
    out, removed = dedupe(dmap(obstacles=walls))
    assert removed == 3
    assert out.obstacles == [
        Obstacle(164.0, 0.625, 0, 0, 2, 5),
        Obstacle(164.0, 1.0, 2, 0, 2, 5),
        Obstacle(165.0, 1.0, 0, 0, 2, 5),
    ]


def test_wall_boxin_drops_the_later_wall() -> None:
    # Stronger Expert playtest: lanes 3, 2 and 1 closed within a quarter beat.
    walls = [
        Obstacle(172.0, 1.0, 3, 0, 1, 5),
        Obstacle(172.125, 1 / 24, 2, 0, 1, 5),
        Obstacle(172.25, 1 / 24, 1, 0, 1, 5),
    ]
    out, report = postprocess(dmap(obstacles=walls), beatmap())
    assert [(o.beat, o.x) for o in out.obstacles] == [(172.0, 3), (172.125, 2)]
    assert report.dropped == {"wall_boxin": 1} and report.errors == []


def test_parity_break_is_repaired() -> None:
    # Two downs in a row on the same hand, half a beat apart: a parity break.
    m = dmap(Note(4.0, 1, 0, R, D), Note(4.5, 1, 0, R, D))
    assert errors(m)
    out, repaired, dropped = fix_with_rules(m, beatmap())
    assert len(out.notes) == 2 and not dropped
    assert sum(repaired.values()) == 1
    assert errors(out) == []


def test_misaligned_stack_is_repaired() -> None:
    # Maneater Expert playtest: two reds side by side, both pointing up.
    m = dmap(Note(57.0, 2, 2, R, U), Note(57.0, 3, 2, R, U))
    assert {i.rule for i in errors(m)} == {"stack_alignment"}
    out, report = postprocess(m, beatmap())
    assert report.errors == []
    assert "stack_alignment" not in {i.rule for i in issues(out)}


def test_unfixable_objects_are_dropped() -> None:
    # Far too many notes for Expert's NPS ceiling: the overflow goes.
    notes = [Note(4 + i / 8, i % 4, i % 3, Color(i % 2), CutDirection.ANY) for i in range(200)]
    out, report = postprocess(dmap(*notes), beatmap())
    assert report.dropped
    assert report.errors == []
    assert len(out.notes) < 200


def test_rotation_whiplash_is_softened() -> None:
    # +60, -60, -45 within 0.375 s: a 105° swing. The largest allowed last turn is -30.
    rots = [RotationEvent(4.0, 60), RotationEvent(4.5, -60), RotationEvent(4.75, -45)]
    m = dmap(mode=Characteristic.DEGREE_360, rotations=rots)
    assert errors(m)
    out, repaired, dropped = fix_with_rules(m, beatmap())
    assert out.rotations == [*rots[:2], RotationEvent(4.75, -30)]
    assert repaired == {"rotation_whiplash": 1} and not dropped
    assert errors(out) == []


def test_rotation_step_is_shrunk() -> None:
    m = dmap(mode=Characteristic.DEGREE_90, rotations=[RotationEvent(4.0, 90)])
    out, repaired, _ = fix_with_rules(m, beatmap())
    assert out.rotations == [RotationEvent(4.0, 45)]  # 60 would leave 90°'s ±45° range
    assert repaired == {"rotation_step": 1}


def test_postprocess_sets_njs_from_table() -> None:
    out, _ = postprocess(dmap(Note(4.0, 1, 0, R, D)), beatmap())
    assert out.njs == 16.0 and out.njs_offset != 0.0
    out, _ = postprocess(dmap(Note(4.0, 1, 0, R, D)), beatmap(), njs=20.0)
    assert out.njs == 20.0


def test_dedupe_one_rotation_per_tick() -> None:
    m = dmap(mode=Characteristic.DEGREE_360, rotations=[RotationEvent(4, 15), RotationEvent(4, 30)])
    out, removed = dedupe(m)
    assert removed == 1 and out.rotations == [RotationEvent(4, 15)]


# --- arcs and chains ----------------------------------------------------------------------

UP = CutDirection.UP


def test_object_flags_turn_arcs_on_from_hard() -> None:
    from bsmapper.generation.density import object_flags

    def flags(diff, mode=Characteristic.STANDARD):
        f = object_flags(diff, mode)
        return f["allow_arcs"], f["allow_chains"]

    assert flags(Difficulty.NORMAL) == (False, False)
    assert flags(Difficulty.HARD) == (True, False)
    assert flags(Difficulty.EXPERT) == (True, True)
    assert flags(Difficulty.EXPERT_PLUS) == (True, True)
    assert flags(Difficulty.NORMAL, Characteristic.ONE_SABER) == (True, False)  # 66% of maps
    assert flags(Difficulty.EXPERT, Characteristic.NO_ARROWS) == (False, False)


def test_configured_objects(tmp_path) -> None:
    import pytest

    from bsmapper.generation.density import configured_objects

    cfg = tmp_path / "config.toml"
    assert configured_objects(cfg) == {}
    cfg.write_text("[generate]\narcs = true\nchains = false\n", encoding="utf-8")
    assert configured_objects(cfg) == {"arcs": True, "chains": False}
    cfg.write_text("[generate]\narcs = 'yes'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="true or false"):
        configured_objects(cfg)


def _spans(*notes: Note, arcs=(), chains=()) -> DifficultyMap:
    return dmap(*notes, arcs=list(arcs), chains=list(chains))


def test_arc_tail_in_the_air_moves_to_the_hands_next_note() -> None:
    from bsmapper.generation.postprocess import fix_spans
    from bsmapper.schema import Arc

    notes = (Note(4, 1, 0, R, D), Note(5, 0, 1, R, UP), Note(6, 1, 0, R, D))
    # Tail at 4.5 (no note) and in the wrong cell; it should land on the 5.0 note.
    arc = Arc(4, R, 1, 0, D, 4.5, 1, 1, UP)
    m = _spans(*notes, arcs=[arc])
    assert [i.rule for i in issues(m)] == ["arc_anchor"]
    out, repaired, dropped = fix_spans(m, beatmap(), RuleChecker().config(m))
    assert out.arcs == [Arc(4, R, 1, 0, D, 5.0, 0, 1, UP)]
    assert repaired == {"arc_anchor": 1} and not dropped
    assert issues(out) == []


def test_arc_skipping_a_note_is_shortened_and_head_snaps_to_its_note() -> None:
    from bsmapper.generation.postprocess import fix_spans
    from bsmapper.schema import Arc

    notes = (Note(4, 1, 0, R, D), Note(5, 1, 1, R, UP), Note(6, 1, 0, R, D))
    arc = Arc(4, R, 1, 0, D, 6, 1, 0, D)  # runs through the 5.0 swing
    out, repaired, _ = fix_spans(
        _spans(*notes, arcs=[arc]), beatmap(), RuleChecker().config(dmap())
    )
    assert out.arcs == [Arc(4, R, 1, 0, D, 5.0, 1, 1, UP)] and repaired == {"arc_span": 1}
    head_off = Arc(4, R, 2, 1, CutDirection.LEFT, 5, 1, 1, UP)
    out, _, _ = fix_spans(_spans(*notes, arcs=[head_off]), beatmap(), RuleChecker().config(dmap()))
    assert out.arcs == [Arc(4, R, 1, 0, D, 5, 1, 1, UP)]


def test_unanchorable_spans_and_duplicates_are_dropped() -> None:
    from bsmapper.generation.postprocess import fix_spans
    from bsmapper.schema import Arc, Chain

    notes = (Note(4, 1, 2, R, D), Note(5, 1, 1, R, UP))
    no_head = Arc(4.5, R, 1, 0, D, 5, 1, 1, UP)
    blue = Arc(4, B, 1, 2, D, 5, 1, 1, UP)  # no blue notes at all
    twice = [Arc(4, R, 1, 2, D, 5, 1, 1, UP)] * 2
    backwards = Chain(4, R, 1, 2, D, 4.125, 1, 2, 3)  # links don't move along the cut
    good = Chain(4, R, 1, 2, D, 4.125, 1, 0, 3)
    m = _spans(*notes, arcs=[no_head, blue, *twice], chains=[backwards, good])
    out, _, dropped = fix_spans(m, beatmap(), RuleChecker().config(m))
    assert out.arcs == twice[:1] and out.chains == [good]
    assert dropped == {"arc_anchor": 2, "duplicate": 1, "chain_anchor": 1}
    assert issues(out) == []


def test_chain_moves_with_its_head_note() -> None:
    from bsmapper.generation.postprocess import fix_spans
    from bsmapper.schema import Chain

    notes = (Note(4, 2, 2, R, D),)
    chain = Chain(4, R, 1, 2, CutDirection.DOWN_LEFT, 4.125, 1, 0, 3)  # one cell off
    out, repaired, _ = fix_spans(
        _spans(*notes, chains=[chain]), beatmap(), RuleChecker().config(dmap())
    )
    assert out.chains == [Chain(4, R, 2, 2, D, 4.125, 2, 0, 3)] and repaired == {"chain_anchor": 1}


def test_postprocess_reports_span_repairs() -> None:
    from bsmapper.schema import Arc

    notes = (Note(4, 1, 0, R, D), Note(5, 1, 1, R, UP))
    out, report = postprocess(_spans(*notes, arcs=[Arc(4, R, 1, 0, D, 4.5, 1, 1, UP)]), beatmap())
    assert report.repaired == {"arc_anchor": 1} and report.issues == []
    assert out.arcs[0].tail_beat == 5.0
