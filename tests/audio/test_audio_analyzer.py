"""Synthetic click-track tests: they need librosa + soundfile but no model, network or GPU."""

import shutil

import numpy as np
import pytest

pytest.importorskip("librosa")
sf = pytest.importorskip("soundfile")

from bsmapper.audio import SAMPLE_RATE, Analyzer, load_audio  # noqa: E402
from bsmapper.audio.features import compute_frame_features  # noqa: E402

BPM = 128.0
OFFSET = 0.37
TICK_S = 60 / BPM / 24


def click_track(bpm=BPM, offset=OFFSET, duration=48.0, sr=SAMPLE_RATE, loud_from=None, seed=0):
    """Kick on downbeats, hi clicks on the other beats; optionally a loud noisy second half."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(sr * duration), np.float32)
    n = np.arange(int(0.04 * sr))
    kick = np.sin(2 * np.pi * 70 * n / sr) * np.exp(-n / (0.012 * sr))
    click = 0.5 * np.sin(2 * np.pi * 1500 * n / sr) * np.exp(-n / (0.003 * sr))
    for i, t in enumerate(np.arange(offset, duration, 60 / bpm)):
        s = int(round(t * sr))
        hit = kick if i % 4 == 0 else click
        if loud_from is not None and t >= loud_from:
            hit = 1.6 * hit + 0.3 * rng.standard_normal(len(n)) * np.exp(-n / (0.02 * sr))
        y[s : s + len(n)] += hit[: len(y) - s].astype(np.float32)
    y += 0.002 * rng.standard_normal(len(y)).astype(np.float32)
    return y / np.abs(y).max() * 0.8


@pytest.fixture(scope="module")
def result():
    y = click_track(loud_from=24.0 + OFFSET)
    return Analyzer(tracker="librosa").analyze_signal(y)


def test_click_track_grid(result):
    grid = result.grid
    assert grid.bpm == BPM
    assert grid.bpm_changes == []
    assert grid.beats_per_bar == 4
    # The kick sits on the first beat, so beat 0 is at the first click.
    assert abs(grid.offset_s - OFFSET) <= TICK_S


def test_contract_invariants(result):
    a = result.analysis
    assert a.duration_s == pytest.approx(48.0)
    assert a.bpm == BPM
    assert a.beat_times[0] == pytest.approx(result.grid.offset_s)
    np.testing.assert_allclose(np.diff(a.beat_times), 60 / BPM)
    np.testing.assert_allclose(a.downbeat_times, a.beat_times[::4])
    assert a.beat_times[-1] < a.duration_s
    assert a.ticks_per_beat == 24

    n_frames = a.features["mel"].shape[0]
    assert a.frame_rate == pytest.approx(SAMPLE_RATE / 220)
    assert n_frames == pytest.approx(a.duration_s * a.frame_rate, abs=2)
    for name in ("mel", "onset_strength", "spectral_flux", "rms"):
        assert a.features[name].shape[0] == n_frames
        assert a.beat_features[name].shape[0] == len(a.beat_times) * 24
    assert a.features["mel"].shape[1] == 80


def test_onsets_land_on_tick_zero(result):
    a = result.analysis
    onset = a.beat_features["onset_strength"].reshape(len(a.beat_times), 24)
    # Averaged over beats, the onset peak is within one tick of the beat.
    peak = int(np.argmax(onset.mean(axis=0)))
    assert min(peak, 24 - peak) <= 1


def test_sections_follow_the_energy_jump(result):
    sections = result.analysis.sections
    assert sections[0].start_s == 0.0
    assert sections[-1].end_s == pytest.approx(48.0)
    for s, nxt in zip(sections, sections[1:], strict=False):
        assert s.end_s == nxt.start_s
    assert all(0.0 <= s.energy <= 1.0 for s in sections)
    assert len(sections) >= 2
    starts = np.array([s.start_s for s in sections[1:]])
    assert np.min(np.abs(starts - (24.0 + OFFSET))) <= 2 * 60 / BPM * 4  # within 2 bars
    assert sections[-1].energy > sections[0].energy


def test_frame_features_are_normalized():
    y = click_track(duration=8.0)
    feats = compute_frame_features(y)
    for name in ("onset_strength", "spectral_flux", "rms"):
        assert feats[name].max() == pytest.approx(1.0)
        assert feats[name].min() >= 0.0
    assert feats["mel"].max() == pytest.approx(0.0, abs=1e-4)
    assert feats["mel"].min() >= -80.0


def test_decode_resamples_to_mono(tmp_path):
    y = click_track(duration=4.0)
    stereo = np.stack([y, y], axis=1)
    sf.write(tmp_path / "song.wav", np.repeat(stereo, 2, axis=0), 44100)
    out = load_audio(tmp_path / "song.wav")
    assert out.ndim == 1 and out.dtype == np.float32
    assert len(out) == pytest.approx(len(y), abs=2)


def test_decode_egg_without_ffmpeg(tmp_path, monkeypatch):
    y = click_track(duration=4.0)
    sf.write(tmp_path / "song.ogg", y, SAMPLE_RATE, format="OGG", subtype="VORBIS")
    (tmp_path / "song.ogg").rename(tmp_path / "song.egg")
    monkeypatch.setattr(shutil, "which", lambda _: None)
    out = load_audio(tmp_path / "song.egg")
    assert len(out) == pytest.approx(len(y), abs=2)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_with_ffmpeg(tmp_path):
    y = click_track(duration=4.0)
    sf.write(tmp_path / "song.flac", y, 48000)  # wrong rate on purpose: ffmpeg resamples
    out = load_audio(tmp_path / "song.flac")
    assert len(out) == pytest.approx(len(y) * SAMPLE_RATE / 48000, abs=2)
