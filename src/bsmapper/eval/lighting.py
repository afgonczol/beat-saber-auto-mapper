"""Lightshow metrics on vanilla basic events (one lightshow per song).

Generated lightshows are rendered in the ``vanilla`` style for evaluation, and human
lightshows are the dataset's basic events (v3 box groups are not stored), so both sides
use the same event vocabulary. Songs whose human map has no basic light events only
contribute to the generated side.

- ``events_per_bar``: light events (types 0-4) per 4 beats of song
- ``onset_precision``: share of light-event ticks within ``ONSET_TOLERANCE`` ticks of an
  audio onset peak; ``onset_lift``: that over the share of all ticks (first to last event)
  near a peak, so 1 = no better than chance; ``onset_recall``: share of strong onset
  peaks with a light event there. Peaks are local maxima of the onset column above its
  ``PEAK_QUANTILE`` quantile (about 1.6 per beat; 75% of human note onsets sit on one,
  against 37% of random ticks); strong peaks are above ``STRONG_QUANTILE``
- ``on_beat`` / ``on_downbeat``: share of light-event ticks on a beat / on a bar line
- ``section_response``: share of section boundaries (WS2 sections on the features) with
  a ring spin or zoom, a boost toggle or a light colour change within one beat;
  ``bar_response`` is the same share at the other bar lines and ``section_lift`` the
  ratio (dense human shows change colour almost everywhere, so look at the lift)
- ``section_energy_r``: Spearman correlation of light density per section with the
  section's energy
"""

from __future__ import annotations

import math

import numpy as np

from bsmapper.data.mapgrid import N_MELS
from bsmapper.data.tech import spearman
from bsmapper.interfaces import AudioAnalysis
from bsmapper.lighting.events import LIGHT_TYPES, RING_SPIN, RING_ZOOM, decode_value
from bsmapper.schema import Beatmap, Lightshow
from bsmapper.tokenizer import TICKS_PER_BEAT

ONSET_TOLERANCE = 2  # ticks
PEAK_HALF_WIDTH = 3  # ticks: an onset peak is the maximum of +-3 ticks
PEAK_QUANTILE = 0.9  # a peak is above this quantile of the song's onset strength ...
STRONG_QUANTILE = 0.97  # ... and a strong one, which should get a light, above this
SECTION_WINDOW = 1.0  # beats either side of a boundary


def onset_peaks(onset: np.ndarray, quantile: float = PEAK_QUANTILE) -> np.ndarray:
    """Ticks of local onset-strength maxima above the song's ``quantile``."""
    x = np.asarray(onset, dtype=np.float64)
    if len(x) < 3:
        return np.empty(0, dtype=np.int64)
    minimum = max(float(np.quantile(x, quantile)), 1e-6)
    w = PEAK_HALF_WIDTH
    padded = np.pad(x, w, mode="constant", constant_values=-np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * w + 1)
    is_peak = (x >= windows.max(axis=1)) & (x >= minimum)
    return np.nonzero(is_peak)[0]


def near(ticks: np.ndarray, targets: np.ndarray, tolerance: int) -> np.ndarray:
    """For each tick, is a target within ``tolerance``?"""
    if not len(targets) or not len(ticks):
        return np.zeros(len(ticks), dtype=bool)
    if len(targets) == 1:
        return np.abs(targets[0] - ticks) <= tolerance
    idx = np.clip(np.searchsorted(targets, ticks), 1, len(targets) - 1)
    d = np.minimum(np.abs(targets[idx] - ticks), np.abs(targets[idx - 1] - ticks))
    return d <= tolerance


def lighting_metrics(
    lightshow: Lightshow | None,
    features: np.ndarray,
    analysis: AudioAnalysis,
    beatmap: Beatmap,
) -> dict[str, float] | None:
    """The metrics above for one lightshow, or None when it has no light events."""
    if lightshow is None:
        return None
    events = sorted(lightshow.basic_events, key=lambda e: e.beat)
    lights = [e for e in events if e.type in LIGHT_TYPES]
    if not lights:
        return None
    n_beats = max(beatmap.seconds_to_beat(analysis.duration_s), 1.0)
    ticks = np.unique(np.round(np.array([e.beat for e in lights]) * TICKS_PER_BEAT)).astype(
        np.int64
    )
    onset = features[:, N_MELS]
    peaks = onset_peaks(onset)
    strong = onset_peaks(onset, STRONG_QUANTILE)
    precision = float(near(ticks, peaks, ONSET_TOLERANCE).mean())
    chance = float(near(np.arange(ticks[0], ticks[-1] + 1), peaks, ONSET_TOLERANCE).mean())
    out = {
        "events_per_bar": len(lights) / (n_beats / 4.0),
        "onset_precision": precision,
        "onset_lift": precision / chance if chance > 0 else float("nan"),
        "onset_recall": float(near(strong, ticks, ONSET_TOLERANCE).mean())
        if len(strong)
        else float("nan"),
        "on_beat": float(np.mean(ticks % TICKS_PER_BEAT == 0)),
        "on_downbeat": float(np.mean(ticks % (4 * TICKS_PER_BEAT) == 0)),
    }

    # Section boundaries: a big change (spin/zoom, boost, colour change) near each one.
    bounds = [beatmap.seconds_to_beat(s.start_s) for s in analysis.sections[1:]]
    changes = [e.beat for e in events if e.type in (RING_SPIN, RING_ZOOM)]
    changes += [b.beat for b in lightshow.boost_events]
    last_color: dict[int, object] = {}
    for e in lights:
        dec = decode_value(e.value)
        if dec is None:
            continue
        if e.type in last_color and last_color[e.type] != dec[0]:
            changes.append(e.beat)
        last_color[e.type] = dec[0]
    changes_arr = np.sort(np.array(changes, dtype=np.float64))

    def answered(beats) -> float:
        if not len(beats):
            return float("nan")
        if not len(changes_arr):
            return 0.0
        return float(np.mean([np.min(np.abs(changes_arr - b)) <= SECTION_WINDOW for b in beats]))

    out["section_response"] = answered(bounds)
    bars = [b for b in np.arange(4.0, n_beats - 4.0, 4.0) if all(abs(b - x) > 2 for x in bounds)]
    out["bar_response"] = answered(bars)
    out["section_lift"] = (
        out["section_response"] / out["bar_response"]
        if out["bar_response"] and math.isfinite(out["section_response"])
        else float("nan")
    )
    secs = analysis.sections
    if len(secs) >= 3:
        beats = np.array([e.beat for e in lights])
        density, energy = [], []
        for s in secs:
            a, b = beatmap.seconds_to_beat(s.start_s), beatmap.seconds_to_beat(s.end_s)
            if b - a < 4:
                continue
            density.append(np.sum((beats >= a) & (beats < b)) / (b - a))
            energy.append(s.energy)
        out["section_energy_r"] = (
            spearman(np.array(density), np.array(energy)) if len(density) >= 3 else float("nan")
        )
    else:
        out["section_energy_r"] = float("nan")
    return out
