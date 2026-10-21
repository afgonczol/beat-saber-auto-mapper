"""Synthetic analyses: a 120 BPM song (2 beats per second) with four labelled sections."""

from __future__ import annotations

import numpy as np

from bsmapper.interfaces import AudioAnalysis, Section

BPM = 120.0
DURATION = 64.0  # seconds = 128 beats
FRAME_RATE = 100.0
SECTIONS = [  # seconds; beats are twice these
    ("intro", 0.0, 16.0, 0.3),
    ("verse", 16.0, 32.0, 0.55),
    ("drop", 32.0, 48.0, 1.0),
    ("outro", 48.0, 64.0, 0.35),
]


def onset_envelope(duration: float = DURATION, frame_rate: float = FRAME_RATE) -> np.ndarray:
    """Onsets on every eighth note: strong on beats, weak off the beat."""
    env = np.zeros(int(duration * frame_rate))
    for t in np.arange(0.0, duration, 60.0 / BPM / 2):
        i = int(round(t * frame_rate))
        if i < len(env):
            on_beat = abs((t * BPM / 60.0) - round(t * BPM / 60.0)) < 1e-9
            env[i] = 1.0 if on_beat else 0.4
            if i + 1 < len(env):
                env[i + 1] = env[i] * 0.5  # a short decay, not a plateau
    return env


def make_analysis(
    sections=SECTIONS, onsets: bool = True, duration: float = DURATION
) -> AudioAnalysis:
    beats = np.arange(0.0, duration, 60.0 / BPM)
    features = {"onset_strength": onset_envelope(duration)} if onsets else {}
    return AudioAnalysis(
        duration_s=duration,
        bpm=BPM,
        beat_times=beats,
        downbeat_times=beats[::4],
        sections=[Section(s, e, label, energy) for label, s, e, energy in sections],
        features=features,
        frame_rate=FRAME_RATE,
    )
