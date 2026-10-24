"""The incremental API that WS7 turns into decode masks."""

from pathlib import Path

import numpy as np
import pytest

from bsmapper.formats import read_map_report
from bsmapper.rules import MapState, RuleChecker, ordered_objects
from bsmapper.rules.spans import SPAN_RULES
from bsmapper.schema import (
    GRID_COLUMNS,
    GRID_ROWS,
    Beatmap,
    Bomb,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    Note,
    Obstacle,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "maps"
D = CutDirection


@pytest.fixture
def state():
    return MapState(Beatmap("t", 120.0), Characteristic.STANDARD, Difficulty.EXPERT, njs=16)


def test_check_does_not_change_state(state):
    state.add(Note(0, 1, 0, Color.RED, D.DOWN))
    bad = Note(0.5, 0, 1, Color.RED, D.DOWN)
    first = state.check(bad)
    assert [i.rule for i in first] == ["parity"]
    assert state.check(bad) == first
    assert len(state.notes) == 1
    assert state.add(bad) == first
    assert len(state.notes) == 2


def test_allowed_ignores_warnings_by_default(state):
    state.add(Bomb(0, 1, 1))
    hidden = Note(0.25, 0, 0, Color.RED, D.DOWN)
    assert state.allowed(hidden)
    assert not state.allowed(hidden, severities=("error", "warning"))


def test_out_of_order_add_raises(state):
    state.add(Note(4, 1, 0, Color.RED, D.DOWN))
    with pytest.raises(ValueError):
        state.add(Note(2, 1, 0, Color.RED, D.DOWN))


def test_note_mask_shape_and_parity(state):
    state.add(Note(0, 1, 0, Color.RED, D.DOWN))
    mask = state.note_mask(0.5, Color.RED)
    assert mask.shape == (GRID_COLUMNS, GRID_ROWS, 9)
    assert mask.dtype == bool
    assert not mask[0, 1, D.DOWN]
    assert mask[1, 1, D.UP]
    assert mask[:, :, D.ANY].all()  # dots are always parity-safe


def test_note_mask_blocks_occupied_cells_and_crossovers(state):
    state.add(Note(0, 1, 0, Color.BLUE, D.DOWN))
    mask = state.note_mask(0, Color.RED)
    assert not mask[1, 0].any()  # same cell as the blue note
    assert not mask[3].any()  # two columns right of blue: crossed hands
    assert mask[0, 0].any() and mask[2, 2].any()


def test_note_mask_inside_wall(state):
    state.add(Obstacle(0, 4, 0, 0, 1, 5))
    mask = state.note_mask(1, Color.RED)
    assert not mask[0].any()
    assert mask[1].any()


def test_note_mask_all_false_over_density_ceiling():
    state = MapState(Beatmap("t", 120.0), Characteristic.STANDARD, Difficulty.EASY, njs=10)
    for i in range(12):
        color = Color(i % 2)
        state.add(Note(i * 0.25, 1 + color, 0, color, D.DOWN if (i // 2) % 2 == 0 else D.UP))
    assert not state.note_mask(3.0, Color.RED).any()


def test_note_mask_agrees_with_allowed(state):
    state.add(Note(0, 1, 0, Color.RED, D.DOWN))
    state.add(Note(0, 2, 0, Color.BLUE, D.DOWN))
    state.add(Bomb(0.25, 0, 2))
    mask = state.note_mask(0.5, Color.RED)
    for x in range(GRID_COLUMNS):
        for y in range(GRID_ROWS):
            for d in D:
                assert mask[x, y, d] == state.allowed(Note(0.5, x, y, Color.RED, d))


def test_bomb_mask(state):
    state.add(Note(0, 1, 0, Color.RED, D.DOWN))
    mask = state.bomb_mask(0)
    assert mask.shape == (GRID_COLUMNS, GRID_ROWS)
    assert not mask[1, 0] and mask[0, 0]


def test_candidates_mask(state):
    state.add(Note(0, 1, 0, Color.RED, D.DOWN))
    candidates = [None, Note(0.5, 0, 1, Color.RED, D.DOWN), Note(0.5, 1, 1, Color.RED, D.UP)]
    assert state.candidates_mask(candidates).tolist() == [True, False, True]


def test_clone_is_independent(state):
    state.add(Note(0, 1, 0, Color.RED, D.DOWN))
    beam = state.clone()
    beam.add(Note(0.5, 1, 1, Color.RED, D.UP))
    assert len(state.notes) == 1
    # The original still expects an upswing; the clone now expects a downswing.
    down, up = Note(1, 1, 0, Color.RED, D.DOWN), Note(1, 1, 1, Color.RED, D.UP)
    assert not state.allowed(down) and state.allowed(up)
    assert beam.allowed(down) and not beam.allowed(up)


@pytest.mark.parametrize("name", ["v2", "v3", "v4", "bad"])
def test_checker_equals_incremental_replay(name):
    beatmap, _ = read_map_report(FIXTURES / name)
    checker = RuleChecker()
    for dmap in beatmap.difficulties:
        state = checker.state(dmap, beatmap)
        replay = []
        for obj in ordered_objects(dmap):
            replay += state.add(obj)
        # Arc and chain rules look forward, so only the whole-map check has them.
        full = [i for i in checker.check(dmap, beatmap) if i.rule not in SPAN_RULES]
        assert set(full) <= set(replay)
        assert {i.rule for i in full} == {i.rule for i in replay}


def test_masks_are_numpy_bool(state):
    assert isinstance(state.note_mask(0, Color.BLUE), np.ndarray)
