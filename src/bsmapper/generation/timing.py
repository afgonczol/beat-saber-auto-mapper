"""Song time: the analysis as the packaged ``song.egg`` hears it.

The analyzer numbers beats from the first downbeat in the audio (grid beat 0 at
``BeatGrid.offset_s``), but Beat Saber puts map beat 0 at the start of the audio file.
The pipeline pads the song with ``BeatGrid.lead_in_s()`` of silence so the bar before
grid beat 0 starts at 0 s; map beat = grid beat + ``beats_per_bar`` (see WS2's notes).

``song_time_analysis`` shifts an ``AudioAnalysis`` the same way, so generators can treat
``beat_times[i]`` as map beat ``i`` and row ``j`` of ``beat_features`` as map beat
``j / ticks_per_beat`` without knowing about the padding.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from bsmapper.interfaces import AudioAnalysis, Section
from bsmapper.schema import Beatmap, BpmChange

LEAD_IN_EPS_S = 1e-6


def map_beat_shift(lead_in_s: float, beats_per_bar: int) -> int:
    """Beats to add to a grid beat to get the map beat, for a given padding."""
    return beats_per_bar if lead_in_s > LEAD_IN_EPS_S else 0


def _pad_rows(arr: np.ndarray, n: int) -> np.ndarray:
    """Prepend ``n`` rows of the array's minimum (silence for every feature we compute)."""
    if n <= 0 or arr.shape[0] == 0:
        return arr
    fill = np.full((n, *arr.shape[1:]), arr.min(axis=0), dtype=arr.dtype)
    return np.concatenate([fill, arr], axis=0)


def song_time_analysis(
    analysis: AudioAnalysis, lead_in_s: float, beats_per_bar: int = 4
) -> AudioAnalysis:
    """``analysis`` shifted by ``lead_in_s`` of prepended silence.

    The ``beats_per_bar`` beats of the lead-in bar are added in front of ``beat_times``
    (and its downbeat in front of ``downbeat_times``), so ``beat_times[i]`` is map beat i.
    Frame and beat-synchronous features get silent rows in front; the first section is
    stretched back to 0 s.
    """
    if lead_in_s <= LEAD_IN_EPS_S:
        return analysis
    period = 60.0 / analysis.bpm
    first = float(analysis.beat_times[0]) if len(analysis.beat_times) else 0.0
    pre = first + lead_in_s - period * np.arange(beats_per_bar, 0, -1, dtype=np.float64)
    beat_times = np.concatenate([pre, np.asarray(analysis.beat_times) + lead_in_s])
    downbeats = np.concatenate([pre[:1], np.asarray(analysis.downbeat_times) + lead_in_s])

    sections = [
        Section(s.start_s + lead_in_s, s.end_s + lead_in_s, s.label, s.energy)
        for s in sorted(analysis.sections, key=lambda s: s.start_s)
    ]
    if sections:
        sections[0].start_s = 0.0

    n_frames = int(round(lead_in_s * analysis.frame_rate))
    n_ticks = beats_per_bar * analysis.ticks_per_beat
    return dataclasses.replace(
        analysis,
        duration_s=analysis.duration_s + lead_in_s,
        beat_times=beat_times,
        downbeat_times=downbeats,
        sections=sections,
        features={k: _pad_rows(v, n_frames) for k, v in analysis.features.items()},
        beat_features={k: _pad_rows(v, n_ticks) for k, v in analysis.beat_features.items()},
    )


def tempo_beatmap(analysis: AudioAnalysis, tolerance_bpm: float = 0.01) -> Beatmap:
    """An empty ``Beatmap`` whose tempo map reproduces a song-time analysis's beats.

    ``NoteGenerator.generate`` only gets the analysis, but the rules engine needs a
    ``Beatmap`` to turn beats into seconds. Beat times from ``BeatGrid`` are piecewise
    linear, so one ``BpmChange`` per tempo segment rebuilds them exactly.
    """
    times = np.asarray(analysis.beat_times, dtype=np.float64)
    if len(times) < 2:
        return Beatmap(song_name="", bpm=analysis.bpm)
    bpms = 60.0 / np.diff(times)
    beatmap = Beatmap(song_name="", bpm=round(float(bpms[0]), 4))
    current = bpms[0]
    for i, bpm in enumerate(bpms[1:], start=1):
        if abs(bpm - current) > tolerance_bpm:
            beatmap.bpm_changes.append(BpmChange(float(i), round(float(bpm), 4)))
            current = bpm
    return beatmap


def seconds_to_beats(analysis: AudioAnalysis, seconds: float | np.ndarray) -> np.ndarray:
    """Map beats of song-time ``seconds``, interpolated along ``beat_times``."""
    times = np.asarray(analysis.beat_times, dtype=np.float64)
    t = np.asarray(seconds, dtype=np.float64)
    if len(times) < 2:
        return t * analysis.bpm / 60.0
    beats = np.interp(t, times, np.arange(len(times), dtype=np.float64))
    period_end = times[-1] - times[-2]
    beyond = t > times[-1]
    return np.where(beyond, len(times) - 1 + (t - times[-1]) / period_end, beats)
