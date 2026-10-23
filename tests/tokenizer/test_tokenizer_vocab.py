"""Vocabularies, conditioning prefix and the grid/snap utilities."""

import pytest

from bsmapper.interfaces import GenerationRequest
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
    RotationEvent,
)
from bsmapper.tokenizer import (
    BOS,
    EOS,
    LIGHT_VOCAB,
    NOTE_VOCAB,
    PAD,
    SEP,
    Conditioning,
    beat_to_tick,
    is_on_grid,
    notes_per_second,
    nps_bucket,
    snap_beat,
    snap_map,
    style_bucket,
    tick_to_beat,
)

# --- vocabularies ---------------------------------------------------------------


@pytest.mark.parametrize("vocab", [NOTE_VOCAB, LIGHT_VOCAB])
def test_vocab_basics(vocab):
    assert [vocab.name(i) for i in (PAD, BOS, EOS, SEP)] == ["PAD", "BOS", "EOS", "SEP"]
    names = [vocab.name(i) for i in range(len(vocab))]
    assert len(set(names)) == len(names)
    assert vocab.from_names(names) == list(range(len(vocab)))
    for i in range(len(vocab)):
        assert vocab.id(vocab.group(i), vocab.value(i)) == i
    assert len(vocab.ids("TIME")) == 384


def test_note_vocab_size_and_contents():
    v = NOTE_VOCAB
    assert 550 <= len(v) <= 700  # PLAN §5 WS6 budgets ~600 tokens
    assert len(v.ids("POS")) == 12 and len(v.ids("DIR")) == 9
    assert {v.name(i) for i in v.ids("ROT")} >= {"ROT_+15", "ROT_-60", "ROT_+180"}
    assert len(v.ids("STYLE")) == 6  # STYLE_0..4 + STYLE_NONE
    with pytest.raises(KeyError):
        v.id("POS", (4, 0))


# --- conditioning ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("style", "bucket"), [(0.0, 0), (0.19, 0), (0.2, 1), (0.5, 2), (0.99, 4), (1.0, 4)]
)
def test_style_buckets(style, bucket):
    assert style_bucket(style) == bucket


@pytest.mark.parametrize(("nps", "bucket"), [(0.0, 0), (0.99, 0), (1.0, 1), (5.5, 5), (20, 12)])
def test_nps_buckets(nps, bucket):
    assert nps_bucket(nps) == bucket


def test_prefix_round_trip_and_dropout():
    cond = Conditioning(
        Characteristic.DEGREE_360, Difficulty.HARD, nps=5.5, style=0.3, bombs=False, chains=False
    )
    back = Conditioning.from_prefix(cond.prefix_ids())
    assert back.prefix_ids() == cond.prefix_ids()
    assert (back.characteristic, back.difficulty, back.bombs, back.chains) == (
        Characteristic.DEGREE_360, Difficulty.HARD, False, False,
    )  # fmt: skip
    dropped = cond.drop_style().drop_nps().prefix_ids()
    assert NOTE_VOCAB.names(dropped[2:4]) == ["NPS_NONE", "STYLE_NONE"]
    with pytest.raises(ValueError):
        Conditioning.from_prefix(dropped[::-1])


def test_conditioning_from_request():
    request = GenerationRequest(
        Characteristic.ONE_SABER, Difficulty.EXPERT_PLUS, style=0.8, allow_arcs=False
    )
    cond = Conditioning.from_request(request)
    assert NOTE_VOCAB.names(cond.prefix_ids()) == [
        "CHAR_ONE_SABER", "DIFF_EXPERT_PLUS", "NPS_NONE", "STYLE_4",
        "BOMBS_ON", "WALLS_ON", "ARCS_OFF", "CHAINS_ON", "INTENSITY_NONE",
    ]  # fmt: skip


def test_notes_per_second():
    beatmap = Beatmap("t", bpm=120)
    notes = [Note(b, 0, 0, Color.RED, CutDirection.DOWN) for b in (0.0, 4.0, 8.0)]
    dmap = DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT, notes=notes)
    assert notes_per_second(dmap, beatmap) == pytest.approx(3 / 4.0)  # 8 beats = 4 s
    assert notes_per_second(dmap, beatmap, duration_s=10) == pytest.approx(0.3)
    assert notes_per_second(DifficultyMap("Standard", "Expert"), beatmap) == 0.0


# --- grid and snapping ------------------------------------------------------------


def test_tick_conversions():
    assert beat_to_tick(7.333333) == 176
    assert tick_to_beat(176) == 7.333333
    assert tick_to_beat(4) == 0.166667
    assert snap_beat(1.01) == 1.0
    assert snap_beat(-0.0001) == 0.0 and str(snap_beat(-0.0001)) == "0.0"
    assert is_on_grid(34.166667) and not is_on_grid(1.01)


def test_snap_map_reports_and_fixes():
    dmap = DifficultyMap(
        Characteristic.DEGREE_360,
        Difficulty.EXPERT,
        notes=[
            Note(1.001, 0, 0, Color.RED, CutDirection.DOWN),  # small error: snapped silently
            Note(2.02, 1, 0, Color.RED, CutDirection.DOWN),  # 0.48 ticks: off grid
            Note(3.0, 1, 3, Color.RED, CutDirection.DOWN),  # row 3 does not exist
            Note(4.0, 1, 1, Color.BLUE, CutDirection.UP, angle_offset=22),
        ],
        obstacles=[
            Obstacle(1.0, 0.99, 0, 0, 1, 5),
            Obstacle(1.0, -1.0, 0, 0, 1, 5),  # negative duration
            Obstacle(1.0, 1.0, 0, 0, 1, 6),  # too tall
        ],
        arcs=[
            Arc(1.0, Color.RED, 0, 0, CutDirection.DOWN, 2.0, 1, 1, CutDirection.UP, 1.03, 0.5),
            Arc(2.0, Color.RED, 0, 0, CutDirection.DOWN, 1.0, 1, 1, CutDirection.UP),
        ],
        chains=[Chain(1.0, Color.BLUE, 3, 0, CutDirection.DOWN, 1.25, 3, 2, 99, 0.81)],
        rotations=[RotationEvent(1.0, 20), RotationEvent(2.0, 5), RotationEvent(3.0, 270)],
    )
    snapped, report = snap_map(dmap)
    assert [n.beat for n in snapped.notes] == [1.0, 2.0, 4.0]
    assert snapped.notes[-1].angle_offset == 20
    assert [(o.beat, o.duration) for o in snapped.obstacles] == [(1.0, 1.0)]
    assert snapped.arcs[0].head_multiplier == 1.05
    assert (snapped.chains[0].slice_count, snapped.chains[0].squish) == (32, 0.8)
    assert [r.degrees for r in snapped.rotations] == [15, -90]
    assert report.objects == 13
    assert report.off_grid == 2  # the 2.02 note and the wall ending at 1.99
    assert report.max_error_ticks == pytest.approx(0.48)
    assert report.dropped == {
        "note_position": 1,
        "wall_negative_duration": 1,
        "wall_shape": 1,
        "arc_reversed": 1,
        "rotation_zero": 1,
    }
    assert report.quantized == {
        "angle_offset": 1,
        "arc_multiplier": 1,
        "chain_slices": 1,
        "chain_squish": 1,
        "rotation": 2,
    }
    assert report.off_grid_ratio == pytest.approx(2 / 13)
