"""Arc and chain rules (``bsmapper.rules.spans``). 120 BPM, so one beat is 0.5 s."""

from bsmapper.rules import RuleChecker
from bsmapper.schema import (
    Arc,
    Beatmap,
    Chain,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
    Obstacle,
)

R, B = Color.RED, Color.BLUE
D = CutDirection

# Red down at beat 0 (x 1, y 0), red up at beat 1 (x 1, y 1).
NOTES = [Note(0, 1, 0, R, D.DOWN), Note(1, 1, 1, R, D.UP)]
ARC = Arc(0, R, 1, 0, D.DOWN, 1, 1, 1, D.UP)


def rules_of(notes=NOTES, arcs=(), chains=(), obstacles=()):
    dmap = DifficultyMap(
        Characteristic.STANDARD,
        Difficulty.EXPERT,
        notes=list(notes),
        arcs=list(arcs),
        chains=list(chains),
        obstacles=list(obstacles),
    )
    return [i.rule for i in RuleChecker().check(dmap, Beatmap("t", 120.0))]


def test_arc_between_consecutive_notes_is_clean():
    assert rules_of(arcs=[ARC]) == []


def test_arc_end_on_a_dot_note_is_clean():
    notes = [NOTES[0], Note(1, 1, 1, R, D.ANY)]
    assert rules_of(notes, arcs=[ARC]) == []


def test_arc_head_off_its_note():
    assert rules_of(arcs=[Arc(0, R, 2, 0, D.DOWN, 1, 1, 1, D.UP)]) == ["arc_anchor"]


def test_arc_on_other_colour_or_wrong_direction():
    assert rules_of(arcs=[Arc(0, B, 1, 0, D.DOWN, 1, 1, 1, D.UP)]) == ["arc_anchor"]
    assert rules_of(arcs=[Arc(0, R, 1, 0, D.DOWN, 1, 1, 1, D.LEFT)]) == ["arc_anchor"]


def test_arc_tail_in_empty_air():
    assert rules_of(arcs=[Arc(0, R, 1, 0, D.DOWN, 1.5, 1, 1, D.UP)]) == ["arc_anchor"]


def test_arc_skipping_a_note_of_its_hand():
    notes = [*NOTES, Note(2, 1, 0, R, D.DOWN)]
    assert rules_of(notes, arcs=[Arc(0, R, 1, 0, D.DOWN, 2, 1, 0, D.DOWN)]) == ["arc_span"]
    # The other hand's notes in between don't matter.
    notes = [*NOTES, Note(0.5, 2, 0, B, D.DOWN)]
    assert rules_of(notes, arcs=[ARC]) == []


def test_arc_too_long():
    notes = [Note(0, 1, 0, R, D.DOWN), Note(8, 1, 1, R, D.UP)]  # 4 s apart
    assert rules_of(notes, arcs=[Arc(0, R, 1, 0, D.DOWN, 8, 1, 1, D.UP)]) == ["arc_span"]


def test_arc_through_a_wall():
    wall = Obstacle(0.25, 0.25, 0, 0, 2, 3)  # columns 0-1, full height, beats 0.25-0.5
    assert rules_of(arcs=[ARC], obstacles=[wall]) == ["arc_wall"]
    assert rules_of(arcs=[ARC], obstacles=[Obstacle(0.25, 0.5, 3, 0, 1, 3)]) == []


def test_chain_along_the_cut_is_clean():
    # Down cut at (1, 2); links run down to (1, 0).
    notes = [Note(0, 1, 2, R, D.DOWN)]
    assert rules_of(notes, chains=[Chain(0, R, 1, 2, D.DOWN, 0.125, 1, 0, 4)]) == []


def test_bad_chains():
    notes = [Note(0, 1, 2, R, D.DOWN), Note(0.0625, 2, 0, R, D.DOWN)]
    up_links = Chain(0, R, 1, 0, D.UP, 0.125, 1, 2, 4)  # no note at (1, 0)
    backwards = Chain(0, R, 1, 2, D.DOWN, 0.125, 1, 2, 4)  # no movement along the cut
    long = Chain(0, R, 1, 2, D.DOWN, 1, 1, 0, 4)
    skip = Chain(0, R, 1, 2, D.DOWN, 0.125, 1, 0, 4)  # the 0.0625 note is in the way
    assert rules_of(notes[:1], chains=[up_links]) == ["chain_anchor"]
    assert rules_of(notes[:1], chains=[backwards]) == ["chain_anchor"]
    assert rules_of(notes[:1], chains=[long]) == ["chain_anchor"]
    assert rules_of(notes, chains=[skip]) == ["chain_anchor"]


def test_span_rules_can_be_turned_off():
    raw = {"default": {"severity": {"arc_anchor": "off"}}}
    dmap = DifficultyMap(
        Characteristic.STANDARD,
        Difficulty.EXPERT,
        notes=list(NOTES),
        arcs=[Arc(0, R, 2, 0, D.DOWN, 1, 1, 1, D.UP)],
    )
    assert RuleChecker(raw=raw).check(dmap, Beatmap("t", 120.0)) == []
