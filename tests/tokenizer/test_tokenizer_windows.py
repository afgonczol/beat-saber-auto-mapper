"""16-beat windows and previous-window tail context."""

import pytest

from bsmapper.schema import (
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
    Obstacle,
)
from bsmapper.tokenizer import (
    BOS,
    EOS,
    NOTE_GRAMMAR,
    NOTE_VOCAB,
    PREFIX_LEN,
    SEP,
    Conditioning,
    MapTokenizer,
    TokenWindow,
    tail_context,
)

V = NOTE_VOCAB


def _note(beat: float) -> Note:
    return Note(beat, 1, 0, Color.RED, CutDirection.DOWN)


def _map(*objects) -> DifficultyMap:
    dmap = DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT)
    for obj in objects:
        (dmap.notes if isinstance(obj, Note) else dmap.obstacles).append(obj)
    return dmap


def test_window_layout():
    dmap = _map(_note(1.0), _note(17.5))
    windows = MapTokenizer().encode(dmap, Conditioning.from_map(dmap, nps=4.2, style=0.9))
    assert [w.index for w in windows] == [0, 1]
    first = windows[0].input_ids()
    assert first[0] == BOS and first[-1] == EOS
    assert V.names(first[1 : 1 + PREFIX_LEN]) == [
        "CHAR_STANDARD", "DIFF_EXPERT", "NPS_4", "STYLE_4",
        "BOMBS_OFF", "WALLS_OFF", "ARCS_OFF", "CHAINS_OFF",
    ]  # fmt: skip
    assert first[1 + PREFIX_LEN] == SEP  # window 0 has no context
    assert V.names(windows[0].body) == ["TIME_24", "RED", "POS_1_0", "DIR_DOWN"]
    # Window 1: TIME is relative to its own start (17.5 beats -> tick 36).
    assert V.names(windows[1].body) == ["TIME_36", "RED", "POS_1_0", "DIR_DOWN"]
    # ... and its context is window 0's tail, with times relative to window 0.
    assert V.names(windows[1].context) == ["TIME_24", "RED", "POS_1_0", "DIR_DOWN"]
    ids = windows[1].input_ids()
    assert ids[windows[1].body_offset :] == [*windows[1].body, EOS]


def test_objects_on_window_edges():
    windows = MapTokenizer().encode(_map(_note(15.958333), _note(16.0), _note(32.0)))
    assert [V.names(w.body)[0] for w in windows] == ["TIME_383", "TIME_0", "TIME_0"]


def test_long_wall_belongs_to_its_start_window():
    wall = Obstacle(14.0, 40.0, 0, 0, 1, 5)
    tok = MapTokenizer()
    windows = tok.encode(_map(wall))
    assert len(windows) == 1
    assert V.names(windows[0].body)[-4:] == ["DUR_16", "DUR_16", "DUR_8", "DURT_0"]
    assert tok.decode(windows).obstacles == [wall]


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("context_events", [0, 1, 5, 16])
def test_context_is_tail_of_previous_body(seed, context_events, random_map):
    windows = MapTokenizer(context_events).encode(random_map(seed))
    assert windows[0].context == []
    for prev, window in zip(windows, windows[1:], strict=False):
        assert window.context == tail_context(prev.body, NOTE_GRAMMAR, context_events)
        n_events = len(NOTE_GRAMMAR.parse(window.context, terminator=SEP))
        assert n_events == min(context_events, len(NOTE_GRAMMAR.parse(prev.body)))


@pytest.mark.parametrize("seed", range(10))
def test_from_ids_inverts_input_ids(seed, random_map):
    for window in MapTokenizer().encode(random_map(seed)):
        assert TokenWindow.from_ids(window.input_ids(), window.index, PREFIX_LEN) == window


def test_too_few_windows_is_an_error():
    with pytest.raises(ValueError):
        MapTokenizer().encode(_map(_note(40.0)), n_windows=2)
