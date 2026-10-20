import pytest

from bsmapper.schema import Beatmap, BpmChange, Difficulty


def test_constant_tempo_round_trip():
    bm = Beatmap(song_name="t", bpm=120)
    assert bm.beat_to_seconds(4) == pytest.approx(2.0)
    assert bm.seconds_to_beat(2.0) == pytest.approx(4.0)


def test_bpm_change():
    bm = Beatmap(song_name="t", bpm=120, bpm_changes=[BpmChange(beat=8, bpm=60)])
    assert bm.beat_to_seconds(8) == pytest.approx(4.0)
    assert bm.beat_to_seconds(10) == pytest.approx(6.0)
    for beat in (0, 3.5, 8, 12.25):
        assert bm.seconds_to_beat(bm.beat_to_seconds(beat)) == pytest.approx(beat)


def test_difficulty_rank():
    assert [d.rank for d in Difficulty] == [1, 3, 5, 7, 9]
