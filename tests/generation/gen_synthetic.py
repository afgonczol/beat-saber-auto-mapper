"""Synthetic songs for the generation tests: no audio, just an ``AudioAnalysis``.

A steady tempo with onsets on every beat (strong), every half beat (medium) and, in the
loud middle section, every quarter beat (weak). The first ``silence_s`` seconds are silent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bsmapper.audio import AnalysisResult, BeatGrid
from bsmapper.audio.grid import GridFit
from bsmapper.interfaces import AudioAnalysis, Section

BPM = 120.0
TPB = 24
FRAME_RATE = 100.0


def make_analysis(
    duration: float = 60.0,
    bpm: float = BPM,
    offset_s: float = 0.0,
    silence_s: float = 2.0,
) -> AudioAnalysis:
    """An analysis whose grid beat 0 is at ``offset_s`` (0 = already in song time)."""
    period = 60.0 / bpm
    beat_times = np.arange(offset_s, duration, period)
    n_ticks = len(beat_times) * TPB
    tick_s = offset_s + np.arange(n_ticks) * period / TPB
    third = duration / 3

    onset = np.zeros(n_ticks, dtype=np.float32)
    onset[::TPB] = 1.0
    onset[TPB // 2 :: TPB] = 0.5
    loud = (tick_s >= third) & (tick_s < 2 * third)
    quarter = np.zeros(n_ticks, dtype=bool)
    quarter[TPB // 4 :: TPB // 2] = True
    onset[loud & quarter] = 0.3
    rms = np.where(tick_s < silence_s, 0.0, 0.6).astype(np.float32)

    n_frames = int(duration * FRAME_RATE)
    frame_t = np.arange(n_frames) / FRAME_RATE
    tick_of_frame = np.clip(np.round((frame_t - offset_s) / period * TPB).astype(int), 0, None)
    tick_of_frame = np.minimum(tick_of_frame, n_ticks - 1)
    frame_onset = np.where(
        np.abs((frame_t - offset_s) / period * TPB - tick_of_frame) < 0.5, onset[tick_of_frame], 0
    )
    sections = [
        Section(0.0, third, "intro", 0.3),
        Section(third, 2 * third, "drop", 1.0),
        Section(2 * third, duration, "outro", 0.4),
    ]
    return AudioAnalysis(
        duration_s=duration,
        bpm=bpm,
        beat_times=beat_times,
        downbeat_times=beat_times[::4],
        sections=sections,
        features={
            "onset_strength": frame_onset.astype(np.float32),
            "rms": np.where(frame_t < silence_s, 0.0, 0.6).astype(np.float32),
        },
        frame_rate=FRAME_RATE,
        beat_features={"onset_strength": onset, "rms": rms},
        ticks_per_beat=TPB,
    )


class FakeAnalyzer:
    """Stands in for ``bsmapper.audio.Analyzer``: returns a synthetic analysis."""

    def __init__(self, duration: float = 60.0, bpm: float = BPM, offset_s: float = 0.0) -> None:
        self.duration, self.bpm, self.offset_s = duration, bpm, offset_s
        self.calls = 0

    def analyze_full(self, path: Path) -> AnalysisResult:
        self.calls += 1
        analysis = make_analysis(self.duration, self.bpm, self.offset_s)
        grid = BeatGrid(self.bpm, self.offset_s)
        fit = GridFit(grid, np.zeros(len(analysis.beat_times)), 0.03)
        beats = analysis.beat_times
        return AnalysisResult(analysis, grid, fit, beats, beats[::4], "fake")
