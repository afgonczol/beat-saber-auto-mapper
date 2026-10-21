"""Map grid, tech score and song grouping, without running the whole builder."""

from __future__ import annotations

import numpy as np
import pytest

from bsmapper.audio.grid import BeatGrid
from bsmapper.data.mapgrid import (
    MapTiming,
    audio_windows,
    grid_drift,
    grid_phase,
    map_grid_features,
    off_grid_ratio,
    onset_alignment,
    shift_map,
)
from bsmapper.data.songs import (
    bit_error_rate,
    fingerprint,
    group_songs,
    match_fingerprints,
    metadata_key,
    normalize_text,
    split_for,
)
from bsmapper.data.tech import (
    FEATURES,
    TechCalibration,
    calibrate_by_difficulty,
    feature_matrix,
    spearman,
    style_by_difficulty,
    tech_features,
)
from bsmapper.schema import (
    BpmChange,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
)

# --- map grid ------------------------------------------------------------------------


def test_timing_with_bpm_change_offset_and_phase():
    t = MapTiming(120.0, (BpmChange(8.0, 60.0),), song_time_offset=0.5)
    assert float(t.seconds(0)) == pytest.approx(0.5)
    assert float(t.seconds(8)) == pytest.approx(4.5)
    assert float(t.seconds(10)) == pytest.approx(6.5)
    assert float(t.beats(6.5)) == pytest.approx(10)
    shifted = MapTiming(120.0, song_time_offset=0.0, phase=0.25)
    assert float(shifted.seconds(0)) == pytest.approx(0.125)
    assert audio_windows(MapTiming(120.0), 40.0) == 5  # 80 beats


def test_grid_phase_removes_constant_offset_only():
    beats = np.arange(64) * 0.5
    assert grid_phase(beats) == (0.0, 0.0, 0.0)
    phase, before, after = grid_phase(beats + 0.07)
    assert phase == pytest.approx(0.07, abs=1e-6)
    assert before == 1.0 and after == 0.0
    free = np.random.default_rng(0).uniform(0, 64, 200)  # free-timed: no single phase
    phase, before, after = grid_phase(free)
    assert phase == 0.0 and before == after > 0.7
    assert off_grid_ratio(np.array([0.0, 1 / 24, 1 / 48])) == pytest.approx(1 / 3)


def test_shift_map_moves_every_object():
    dm = DifficultyMap(
        Characteristic.STANDARD,
        Difficulty.EXPERT,
        notes=[Note(1.0, 0, 0, Color.RED, CutDirection.DOWN)],
    )
    assert shift_map(dm, -0.25).notes[0].beat == 0.75
    assert shift_map(dm, 0) is dm


def test_features_follow_the_map_grid():
    frame_rate = 100.0
    n = 1000  # 10 s
    onset = np.zeros(n, dtype=np.float32)
    onset[50::50] = 1.0  # a click every 0.5 s from 0.5 s: 120 BPM, beat 0 at 0 s
    feats = {
        "mel": np.zeros((n, 80), np.float32),
        "onset_strength": onset,
        "spectral_flux": onset,
        "rms": np.ones(n, np.float32),
    }
    timing = MapTiming(120.0)
    out = map_grid_features(feats, frame_rate, timing, 10.0, n_windows=2)
    assert out.shape == (2 * 384, 83)
    on = out[:, 80].astype(np.float32)
    assert np.all(on[24:480:24] == 1.0)  # every beat from beat 1 to the end of the audio
    assert on[12] == 0.0
    assert np.all(out[481:, :80] == -80) and np.all(out[481:, 82] == 0)  # past 10 s
    al = onset_alignment(on, [b for b in range(1, 20)])
    assert al["onset_best_shift"] == 0 and al["onset_best_ratio"] > 5


def test_grid_drift_relations():
    timing = MapTiming(120.0)
    same = grid_drift(BeatGrid(120.0, 0.0), timing, 30.0)
    assert same.relation == "same" and same.median_abs_ticks == pytest.approx(0, abs=1e-9)
    tick = 0.5 / 24
    late = grid_drift(BeatGrid(120.0, 0.8 * tick), timing, 30.0)
    assert late.median_ticks == pytest.approx(0.8) and late.within_1_tick == 1.0
    double = grid_drift(BeatGrid(240.0, 0.25), timing, 30.0)
    assert double.relation == "double" and double.p95_abs_ticks == pytest.approx(0, abs=1e-9)
    assert grid_drift(BeatGrid(97.0, 0.0), timing, 30.0).relation == "other"


# --- tech score ----------------------------------------------------------------------


def _hand_map(directions, step=0.5, color_both=True) -> DifficultyMap:
    notes = []
    for i, d in enumerate(directions):
        beat = 1 + i * step
        notes.append(Note(beat, 1, 0, Color.RED, d))
        if color_both:
            notes.append(Note(beat, 2, 0, Color.BLUE, d))
    return DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT, notes=notes)


def test_flowy_vs_technical_features():
    down, up = CutDirection.DOWN, CutDirection.UP
    flow = tech_features(_hand_map([down, up] * 16))
    assert flow["angle_change"] == 0 and flow["reset_rate"] == 0
    assert flow["rhythm_irregularity"] == 0 and flow["dot_ratio"] == 0
    assert flow["crossover_rate"] == 0 and flow["odd_snap_ratio"] == 0
    resets = tech_features(_hand_map([down] * 32))
    assert resets["angle_change"] == 1 and resets["reset_rate"] == 1
    sideways = tech_features(_hand_map([CutDirection.LEFT, CutDirection.UP] * 16))
    assert sideways["angle_change"] == pytest.approx(0.5)
    dots = tech_features(_hand_map([CutDirection.ANY] * 32))
    assert dots["dot_ratio"] == 1 and dots["angle_change"] == 0
    triplets = tech_features(_hand_map([down, up] * 16, step=1 / 3))
    assert triplets["odd_snap_ratio"] > 0.6
    assert tech_features(_hand_map([down, up])) is None


def test_slider_notes_merge_into_one_swing():
    notes = [Note(1 + k / 16, 1, 2 - k % 3, Color.RED, CutDirection.DOWN) for k in range(4)]
    notes += [Note(2 + i * 0.5, 1, 0, Color.RED, d)
              for i, d in enumerate([CutDirection.UP, CutDirection.DOWN] * 8)]  # fmt: skip
    f = tech_features(DifficultyMap(Characteristic.STANDARD, Difficulty.HARD, notes=notes))
    assert f["reset_rate"] == 0  # the 4-note slider is one DOWN swing, then UP


def test_calibration_recovers_a_linear_label():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, (300, len(FEATURES)))
    y = 2 * x[:, 0] + x[:, 1] - 0.5 * x[:, 3] + rng.normal(0, 0.05, 300)
    y[:20] = np.nan  # unrated
    cal = TechCalibration.fit(x, y)
    assert cal.n == 280 and cal.r_cv > 0.95
    assert cal.weights[0] > cal.weights[1] > 0 > cal.weights[3]
    assert np.allclose(cal.predict(x[20:]), y[20:], atol=0.3)
    with pytest.raises(ValueError):
        TechCalibration.fit(x[:5], y[20:25])
    unc = TechCalibration.uncalibrated(x)
    assert spearman(unc.predict(x), x.mean(axis=1) / x.std(axis=0).mean()) > 0.9
    assert feature_matrix([None, dict.fromkeys(FEATURES, 1.0)]).shape == (2, len(FEATURES))


def test_calibration_per_difficulty_ignores_the_rating_offset_between_difficulties():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, (400, len(FEATURES)))
    diffs = np.array(["Expert"] * 200 + ["Easy"] * 200)
    # Easy ratings are much lower, and a different feature drives them.
    rating = np.where(diffs == "Expert", 5 + 2 * x[:, 0], 1 + 0.5 * x[:, 2])
    rating = rating + rng.normal(0, 0.02, 400)
    scores, cals = calibrate_by_difficulty(x, rating, diffs)
    assert set(cals) == {"Expert", "Easy"}
    for d, f in (("Expert", 0), ("Easy", 2)):
        m = diffs == d
        assert cals[d].spearman_cv > 0.95
        assert int(np.argmax(cals[d].weights)) == f
        assert abs(scores[m].mean()) < 0.1  # z units within the difficulty
    few, cals = calibrate_by_difficulty(x[:5], rating[:5], diffs[:5])
    assert cals["Expert"].n == 0 and np.isfinite(few).all()  # equal-weight fallback


def test_style_is_a_rank_within_each_difficulty():
    scores = np.array([5.0, 1.0, 3.0, 100.0, -1.0, np.nan])
    diffs = ["Expert", "Expert", "Expert", "Easy", "Easy", "Easy"]
    style = style_by_difficulty(scores, diffs)
    assert style[:3].tolist() == pytest.approx([5 / 6, 1 / 6, 3 / 6])
    assert style[3:5].tolist() == pytest.approx([3 / 4, 1 / 4])
    assert np.isnan(style[5])


# --- songs ---------------------------------------------------------------------------


def _music(seed: int, seconds: float = 40.0) -> np.ndarray:
    """Log-mel of a seeded melody: gliding tones with decaying notes, like real music."""
    pytest.importorskip("librosa")
    from bsmapper.audio.features import compute_frame_features

    rng = np.random.default_rng(seed)
    sr = 22050
    n = int(seconds * sr)
    y = np.zeros(n)
    t = np.arange(n) / sr
    for start in np.cumsum(rng.uniform(0.1, 0.4, int(seconds * 5))):
        i = int(start * sr)
        if i >= n:
            break
        seg = t[: min(int(0.5 * sr), n - i)]
        for f in rng.uniform(100, 4000, 3):
            y[i : i + len(seg)] += np.sin(2 * np.pi * f * seg) * np.exp(-6 * seg)
    return compute_frame_features(y.astype(np.float32))["mel"]


def test_fingerprints_match_the_same_audio_only():
    a = _music(0)
    trimmed = a[137:] + np.random.default_rng(9).normal(0, 0.5, a[137:].shape)  # 1.37 s later
    other = _music(1)
    fps = [fingerprint(a), fingerprint(trimmed), fingerprint(other)]
    assert all(fp.dtype == np.uint32 for fp in fps)
    matches = match_fingerprints(fps)
    assert [(m.a, m.b) for m in matches] == [(0, 1)]
    assert matches[0].ber < 0.3 and abs(matches[0].offset - 137 // 4) <= 1
    assert bit_error_rate(fps[0], fps[2], 0)[0] > 0.4


def test_song_groups_use_fingerprints_and_metadata():
    a = fingerprint(_music(0))
    groups = group_songs(
        keys=["k1", "k2", "k3", "k4", "k5"],
        fingerprints=[a, a.copy(), None, None, None],
        titles=["Song A", "A different title", "Hello (TV Size)", "hello", "hello"],
        artists=["X", "Y", "Band feat. Z", "band", "band"],
        durations_s=[40, 40, 90, 91.5, 200],
    )
    assert groups.group == ["k1", "k1", "k3", "k3", "k5"]
    assert groups.fingerprint_links == 1 and groups.metadata_links == 1


def test_text_normalization_and_splits():
    assert normalize_text("Levitating (feat. DaBaby) [TV Size]") == "levitating"
    assert normalize_text("Dua Lipa featuring DaBaby") == "dua lipa"
    assert metadata_key("", "artist") is None
    splits = [split_for(f"group{i}") for i in range(4000)]
    assert split_for("group1") == split_for("group1")
    share = {s: splits.count(s) / len(splits) for s in ("train", "val", "test")}
    assert share["train"] == pytest.approx(0.9, abs=0.02)
    assert share["val"] == pytest.approx(0.05, abs=0.015)
