import numpy as np
import pytest

from bsmapper.audio.features import beat_synchronous
from bsmapper.audio.grid import BeatGrid, fit_beat_grid
from bsmapper.schema import Beatmap, BpmChange

TICK = 1 / 24


def noisy_beats(bpm, offset, n, jitter=0.008, seed=0):
    rng = np.random.default_rng(seed)
    t = offset + np.arange(n) * 60.0 / bpm
    return t + rng.normal(0, jitter, n)


def test_constant_tempo_snaps_and_finds_offset():
    beats = noisy_beats(128.0, 0.37, 400)
    downbeats = beats[2::4]  # bar phase: beat 2 of the tracked beats is a downbeat
    grid = fit_beat_grid(beats, downbeats).grid
    assert grid.bpm == 128
    assert grid.bpm_changes == []
    assert grid.beats_per_bar == 4
    # Beat 0 is the first downbeat at or after 0 s: 0.37 + 2 beats.
    assert grid.offset_s == pytest.approx(0.37 + 2 * 60 / 128, abs=0.003)


def test_missed_and_extra_beats_do_not_shift_the_grid():
    beats = noisy_beats(100.0, 1.0, 300)
    beats = np.delete(beats, [10, 11, 50, 200])  # missed beats
    beats = np.sort(np.append(beats, [beats[100] + 0.31, beats[150] + 0.2]))  # extra beats
    grid = fit_beat_grid(beats).grid
    assert grid.bpm == 100 and grid.bpm_changes == []
    pos = (grid.beat_to_seconds(np.arange(250)) - 1.0) / 0.6  # on the true grid, in beats
    assert np.max(np.abs(pos - np.round(pos))) < 0.01


def test_fractional_bpm_is_not_rounded_into_drift():
    beats = noisy_beats(127.35, 0.1, 500, jitter=0.005)
    grid = fit_beat_grid(beats, beats[::4]).grid
    assert grid.bpm == pytest.approx(127.35, abs=0.01)
    t = grid.beat_to_seconds(np.arange(500))
    assert np.max(np.abs(t - (0.1 + np.arange(500) * 60 / 127.35))) < 0.006


def test_tempo_change_is_detected():
    first = 0.5 + np.arange(64) * 0.5  # 120 BPM
    second = first[-1] + np.arange(1, 129) * 60 / 140  # then 140 BPM
    beats = np.concatenate([first, second]) + np.random.default_rng(1).normal(0, 0.004, 192)
    fit = fit_beat_grid(beats, beats[::4])
    grid = fit.grid
    assert grid.bpm == pytest.approx(120, abs=0.05)
    assert len(grid.bpm_changes) >= 1
    assert grid.bpm_changes[-1].bpm == pytest.approx(140, abs=0.1)
    truth = np.concatenate([first, second])
    k0 = round(grid.seconds_to_beat(truth[0]).item())
    got = grid.beat_to_seconds(np.arange(192) + k0)
    assert np.percentile(np.abs(got - truth), 95) < 0.02


def test_constant_tempo_without_changes_when_disabled():
    beats = noisy_beats(150, 0.0, 200)
    grid = fit_beat_grid(beats, allow_tempo_changes=False).grid
    assert grid.bpm == 150 and grid.bpm_changes == []
    assert grid.offset_s == pytest.approx(0.0, abs=0.003)


def test_three_four_time():
    beats = noisy_beats(90, 0.2, 180)
    grid = fit_beat_grid(beats, beats[1::3]).grid
    assert grid.beats_per_bar == 3
    assert grid.offset_s == pytest.approx(0.2 + 60 / 90, abs=0.003)


def test_too_few_beats():
    with pytest.raises(ValueError):
        fit_beat_grid(np.array([0.0, 0.5, 1.0]))


def test_grid_matches_schema_timing():
    changes = [BpmChange(beat=16, bpm=150), BpmChange(beat=40, bpm=90)]
    grid = BeatGrid(bpm=120, offset_s=0.25, bpm_changes=changes)
    bm = Beatmap(song_name="t", bpm=120, bpm_changes=changes)
    beats = np.array([0, 3.5, 16, 17.25, 40, 63.9])
    want = np.array([bm.beat_to_seconds(b) for b in beats]) + 0.25
    np.testing.assert_allclose(grid.beat_to_seconds(beats), want)
    np.testing.assert_allclose(grid.seconds_to_beat(want), beats)


def test_beat_times_and_lead_in():
    grid = BeatGrid(bpm=120, offset_s=0.5)
    np.testing.assert_allclose(grid.beat_times(2.6), [0.5, 1.0, 1.5, 2.0, 2.5])
    np.testing.assert_allclose(grid.downbeat_times(4.6), [0.5, 2.5, 4.5])
    assert grid.lead_in_s() == pytest.approx(1.5)  # pad 1.5 s so the previous bar starts at 0
    assert BeatGrid(bpm=120, offset_s=0.0).lead_in_s() == 0.0


def test_beat_synchronous_resampling_follows_the_grid():
    frame_rate = 100.0
    frames = np.arange(1000, dtype=np.float64)
    feats = {"t": frames / frame_rate, "two": np.column_stack([frames, -frames])}
    grid = BeatGrid(bpm=125, offset_s=0.3, bpm_changes=[BpmChange(beat=4, bpm=100)])
    sync = beat_synchronous(feats, frame_rate, grid.beat_to_seconds, n_beats=8, ticks_per_beat=24)
    assert sync["t"].shape == (8 * 24,)
    assert sync["two"].shape == (8 * 24, 2)
    np.testing.assert_allclose(sync["t"], grid.beat_to_seconds(np.arange(192) * TICK), atol=1e-5)
    np.testing.assert_allclose(sync["two"][:, 1], -sync["two"][:, 0])
