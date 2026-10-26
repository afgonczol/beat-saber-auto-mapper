"""Hand-made known-bad patterns, each next to a close variant that must stay clean.

All maps run at 120 BPM, so one beat is 0.5 s.
"""

import pytest

from bsmapper.rules import RuleChecker
from bsmapper.schema import (
    Beatmap,
    Bomb,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
    Obstacle,
    RotationEvent,
)

R, B = Color.RED, Color.BLUE
D = CutDirection


def red(beat, x, y, d):
    return Note(beat, x, y, R, d)


def blue(beat, x, y, d):
    return Note(beat, x, y, B, d)


def rules_of(
    notes=(),
    bombs=(),
    obstacles=(),
    rotations=(),
    difficulty=Difficulty.EXPERT,
    characteristic=Characteristic.STANDARD,
    njs=16.0,
):
    dmap = DifficultyMap(
        characteristic,
        difficulty,
        njs=njs,
        notes=list(notes),
        bombs=list(bombs),
        obstacles=list(obstacles),
        rotations=list(rotations),
    )
    return {i.rule for i in RuleChecker().check(dmap, Beatmap("t", 120.0))}


# --- parity ------------------------------------------------------------------------


def test_parity_down_down_flagged_down_up_clean():
    assert "parity" in rules_of([red(0, 1, 0, D.DOWN), red(0.5, 0, 1, D.DOWN)])
    assert rules_of([red(0, 1, 0, D.DOWN), red(0.5, 1, 1, D.UP)]) == set()


def test_parity_45_degree_repeat_is_a_break():
    assert "parity" in rules_of([blue(0, 2, 0, D.DOWN), blue(0.5, 3, 1, D.DOWN_RIGHT)])


def test_parity_left_left_is_a_break():
    assert "parity" in rules_of([blue(0, 3, 1, D.LEFT), blue(0.5, 3, 1, D.LEFT)])


def test_parity_time_gap_resets():
    assert rules_of([red(0, 1, 0, D.DOWN), red(1.5, 1, 0, D.DOWN)]) == set()


def test_parity_bomb_reset():
    notes = [red(0, 1, 0, D.DOWN), red(1, 1, 0, D.DOWN)]
    assert "parity" in rules_of(notes)
    assert rules_of(notes, bombs=[Bomb(0.5, 1, 0)]) == set()


def test_parity_far_bomb_does_not_reset():
    notes = [red(0, 0, 0, D.DOWN), red(1, 0, 0, D.DOWN)]
    assert "parity" in rules_of(notes, bombs=[Bomb(0.5, 3, 0)])


def test_parity_stack_and_slider_are_one_swing():
    stack = [red(0, 0, 0, D.DOWN), red(0, 0, 1, D.DOWN), red(0.5, 1, 1, D.UP)]
    slider = [red(0, 1, 2, D.DOWN), red(0.0625, 1, 1, D.DOWN), red(0.125, 1, 0, D.DOWN)]
    assert rules_of(stack) == set()
    assert rules_of([*slider, red(0.75, 1, 1, D.UP)]) == set()


def test_stack_alignment():
    # The Maneater Expert playtest case: two reds side by side, both pointing up.
    assert rules_of([red(0, 2, 2, D.UP), red(0, 3, 2, D.UP)]) == {"stack_alignment"}
    assert rules_of([red(0, 0, 0, D.DOWN), red(0, 1, 1, D.DOWN)]) == {"stack_alignment"}  # 45°
    assert rules_of([red(0, 0, 1, D.DOWN), red(0, 1, 0, D.DOWN)]) == {"stack_alignment"}
    assert rules_of([red(0, 2, 2, D.LEFT), red(0, 3, 2, D.LEFT)]) == set()
    assert rules_of([red(0, 0, 1, D.DOWN_LEFT), red(0, 1, 2, D.DOWN_LEFT)]) == set()
    assert rules_of([red(0, 0, 0, D.DOWN), red(0, 1, 2, D.DOWN)]) == set()  # knight, 26.6°
    assert rules_of([red(0, 0, 0, D.ANY), red(0, 1, 0, D.ANY)]) == set()
    slider = [red(0, 1, 2, D.DOWN), red(0.0625, 2, 2, D.DOWN)]  # sideways "slider"
    assert rules_of(slider) == {"stack_alignment"}


def test_parity_same_time_opposite_directions():
    assert "parity" in rules_of([red(0, 0, 0, D.DOWN), red(0, 0, 2, D.UP)])


def test_parity_90_degree_turns_are_flexible():
    # DOWN -> LEFT -> DOWN and UP -> LEFT -> DOWN both occur in well-made maps.
    assert rules_of([red(0, 1, 0, D.DOWN), red(0.5, 0, 1, D.LEFT), red(1, 1, 0, D.DOWN)]) == set()
    assert rules_of([red(0, 1, 0, D.UP), red(0.5, 0, 1, D.LEFT), red(1, 1, 0, D.DOWN)]) == set()


def test_parity_rolling_spiral_is_flagged():
    # Each step turns 135°, so the wrist keeps rolling until UP needs a forehand.
    notes = [
        blue(0, 2, 2, D.DOWN),
        blue(0.5, 2, 0, D.UP_RIGHT),
        blue(1.0, 3, 2, D.LEFT),
        blue(1.5, 2, 2, D.DOWN_RIGHT),
        blue(2.0, 3, 0, D.UP),
    ]
    assert "parity" in rules_of(notes)
    assert "parity" not in rules_of(notes[:4])


def test_dots_are_wildcards():
    notes = [red(0, 1, 0, D.DOWN), red(0.5, 0, 1, D.ANY), red(1.0, 1, 0, D.DOWN)]
    assert rules_of(notes) == set()


# --- cells, walls, vision, crossovers ------------------------------------------------


def test_same_cell_and_overlap():
    assert "same_cell" in rules_of([red(0, 1, 0, D.DOWN)], bombs=[Bomb(0, 1, 0)])
    assert "overlap" in rules_of([red(0, 0, 0, D.DOWN), blue(0.05, 0, 0, D.DOWN)])
    # 0.25 beat = 0.125 s * 16 m/s = 2 m apart: fine.
    assert "overlap" not in rules_of([red(0, 0, 0, D.DOWN), blue(0.25, 0, 0, D.DOWN)])


def test_bomb_lines_may_be_dense():
    bombs = [Bomb(i * 0.05, 1, 0) for i in range(5)]
    assert rules_of(bombs=bombs) == set()


def test_note_in_wall():
    wall = Obstacle(0, 2, 0, 0, 1, 5)
    assert "note_in_wall" in rules_of([red(1, 0, 2, D.DOWN)], obstacles=[wall])
    assert rules_of([red(1, 1, 0, D.DOWN)], obstacles=[wall]) == set()
    assert rules_of([red(2, 0, 0, D.DOWN)], obstacles=[wall]) == set()  # wall has ended
    crouch = Obstacle(0, 2, 0, 2, 4, 3)
    assert "note_in_wall" in rules_of([blue(1, 3, 2, D.UP)], obstacles=[crouch])
    assert rules_of([blue(1, 3, 1, D.UP)], obstacles=[crouch]) == set()


def test_vision_block_is_a_warning():
    notes = [red(0, 1, 1, D.DOWN), blue(0.25, 1, 0, D.DOWN)]
    dmap = DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT, notes=notes)
    issues = RuleChecker().check(dmap, Beatmap("t", 120.0))
    assert [(i.rule, i.severity) for i in issues] == [("vision_block", "warning")]


def test_crossover():
    assert "crossover" in rules_of([red(0, 3, 0, D.DOWN), blue(0, 1, 0, D.DOWN)])
    # A one-column crossover is a common, deliberate pattern.
    assert rules_of([red(0, 2, 0, D.DOWN), blue(0, 1, 2, D.DOWN)]) == set()


# --- density and speed ----------------------------------------------------------------


def _stream(count, step, difficulty):
    notes = []
    for i in range(count):
        color = Color(i % 2)
        d = D.DOWN if (i // 2) % 2 == 0 else D.UP
        notes.append(Note(i * step, 1 + color, 0, color, d))
    return rules_of(notes, difficulty=difficulty, njs=12)


def test_nps_ceiling_depends_on_difficulty():
    # 1/4-beat alternating stream = 8 notes/s.
    assert "nps_ceiling" in _stream(32, 0.25, Difficulty.NORMAL)
    assert "nps_ceiling" not in _stream(32, 0.25, Difficulty.EXPERT)


def test_swing_interval():
    fast = [red(0, 1, 0, D.DOWN), red(0.0625, 1, 1, D.UP)]
    assert "swing_interval" in rules_of(fast)
    assert "swing_interval" not in rules_of([red(0, 1, 0, D.DOWN), red(0.25, 1, 1, D.UP)])
    # 0.25 beat (125 ms) is fine on Expert but too fast on Easy.
    easy = rules_of([red(0, 1, 0, D.DOWN), red(0.25, 1, 1, D.UP)], difficulty=Difficulty.EASY)
    assert "swing_interval" in easy


def test_swing_speed_is_a_warning():
    jump = [red(0, 0, 2, D.DOWN), red(0.25, 3, 2, D.UP)]
    assert "swing_speed" in rules_of(jump, difficulty=Difficulty.EASY)


# --- walls and rotations ------------------------------------------------------------


def test_dodge_gap():
    left = Obstacle(0, 1, 1, 0, 1, 5)
    assert "dodge_gap" in rules_of(obstacles=[left, Obstacle(1.1, 1, 2, 0, 1, 5)])
    assert rules_of(obstacles=[left, Obstacle(1.1, 1, 1, 0, 1, 5)]) == set()  # same side
    assert rules_of(obstacles=[left, Obstacle(2, 1, 2, 0, 1, 5)]) == set()  # 0.5 s to move


def test_wall_boxin():
    flick = 1 / 24  # 21 ms at 120 BPM: too short to be a dodge wall
    # The Stronger Expert playtest case: an outer wall, then centre-lane flicks on both sides.
    walls = [Obstacle(0, 1, 3, 0, 1, 5), Obstacle(0.125, flick, 2, 0, 1, 5)]
    assert rules_of(obstacles=walls) == set()
    assert rules_of(obstacles=[*walls, Obstacle(0.25, flick, 1, 0, 1, 5)]) == {"wall_boxin"}
    assert rules_of(obstacles=[Obstacle(0, 1, 1, 0, 2, 5)]) == {"wall_boxin"}  # one wide wall
    assert rules_of(obstacles=[Obstacle(0, 1, 0, 2, 4, 3)]) == set()  # crouch
    assert rules_of(obstacles=[Obstacle(0, 1, 0, 0, 1, 5), Obstacle(0, 1, 3, 0, 1, 5)]) == set()
    side = [Obstacle(0, flick, 2, 0, 1, 5), Obstacle(0.25, flick, 2, 0, 1, 5)]
    assert rules_of(obstacles=side) == set()
    apart = [Obstacle(0, flick, 2, 0, 1, 5), Obstacle(2, flick, 1, 0, 1, 5)]
    assert rules_of(obstacles=apart) == set()


def test_crouch_ceiling_from_segments_is_one_dodge():
    walls = [Obstacle(b, 1, x, 2, 1, 3) for b in (0, 1) for x in range(4)]
    assert rules_of(obstacles=walls) == set()


def test_wall_duration():
    assert "wall_duration" in rules_of(obstacles=[Obstacle(0, 0.01, 0, 0, 1, 5)])
    assert rules_of(obstacles=[Obstacle(0, 0.25, 0, 0, 1, 5)]) == set()


def test_rotation_whiplash():
    whiplash = [RotationEvent(0, 60), RotationEvent(0.5, -60), RotationEvent(1, -60)]
    assert "rotation_whiplash" in rules_of(
        rotations=whiplash, characteristic=Characteristic.DEGREE_360
    )
    steady = [RotationEvent(b, 15) for b in range(0, 16, 2)]
    assert rules_of(rotations=steady, characteristic=Characteristic.DEGREE_360) == set()


def test_rotation_step():
    rots = [RotationEvent(0, 90)]
    assert "rotation_step" in rules_of(rotations=rots, characteristic=Characteristic.DEGREE_360)
    rots = [RotationEvent(0, 60), RotationEvent(4, -45)]
    assert rules_of(rotations=rots, characteristic=Characteristic.DEGREE_360) == set()


def test_rotation_range_only_on_90_degree():
    turns = [RotationEvent(0, 30), RotationEvent(4, 30)]
    assert "rotation_range" in rules_of(rotations=turns, characteristic=Characteristic.DEGREE_90)
    assert rules_of(rotations=turns, characteristic=Characteristic.DEGREE_360) == set()


@pytest.mark.parametrize("difficulty", list(Difficulty))
def test_slow_alternating_map_is_clean_everywhere(difficulty):
    notes = []
    for i in range(32):
        d = D.DOWN if i % 2 == 0 else D.UP
        notes += [red(i * 2, 1, 0, d), blue(i * 2 + 1, 2, 0, d)]
    assert rules_of(notes, difficulty=difficulty) == set()
