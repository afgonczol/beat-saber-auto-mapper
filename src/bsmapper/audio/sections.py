"""Section segmentation (Foote novelty on a beat-level self-similarity matrix).

Boundaries are snapped to bar lines, since lighting palette changes and
density shifts should land on downbeats. Labels are a heuristic from energy
and position; downstream code should lean on ``energy`` and treat the label as
a hint.
"""

from __future__ import annotations

import numpy as np

from bsmapper.interfaces import Section


def per_beat_mean(arr: np.ndarray, frame_rate: float, beat_times: np.ndarray) -> np.ndarray:
    """Mean of a frame feature over each beat ``[beat_times[i], beat_times[i+1])``."""
    n = arr.shape[0]
    starts = np.clip(np.round(np.asarray(beat_times) * frame_rate).astype(np.int64), 0, n - 1)
    last = starts[-1] + (int(np.median(np.diff(starts))) if len(starts) > 1 else 1)
    ends = np.maximum(np.append(starts[1:], min(last, n)), starts + 1)
    csum = np.concatenate([np.zeros((1,) + arr.shape[1:]), np.cumsum(arr, axis=0)])
    counts = (ends - starts).reshape((-1,) + (1,) * (arr.ndim - 1))
    return ((csum[ends] - csum[starts]) / counts).astype(np.float32)


def _novelty(feats: np.ndarray, half_width: int) -> np.ndarray:
    x = feats - feats.mean(axis=0)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    ssm = x @ x.T
    w = half_width
    g = np.exp(-0.5 * (np.linspace(-2, 2, 2 * w) ** 2))
    sign = np.sign(np.arange(2 * w) - w + 0.5)
    kernel = np.outer(g, g) * np.outer(sign, sign)
    n = len(ssm)
    padded = np.pad(ssm, w, mode="edge")
    nov = np.array([np.sum(padded[i : i + 2 * w, i : i + 2 * w] * kernel) for i in range(n)])
    return np.maximum(nov, 0.0)


def segment_sections(
    features: dict[str, np.ndarray],
    frame_rate: float,
    beat_times: np.ndarray,
    duration_s: float,
    beats_per_bar: int = 4,
    min_bars: int = 4,
    kernel_bars: int = 4,
) -> list[Section]:
    """Split the song into sections with 0..1 energy relative to the song."""
    n_beats = len(beat_times)
    rms = features["rms"]
    onset = features["onset_strength"]
    if n_beats < 2 * min_bars * beats_per_bar:
        bounds = [0]
    else:
        mel = per_beat_mean(features["mel"], frame_rate, beat_times)
        # Timbre (mel shape) plus loudness and rhythmic activity, standardized.
        x = np.column_stack(
            [
                mel,
                per_beat_mean(rms, frame_rate, beat_times) * 40,
                per_beat_mean(onset, frame_rate, beat_times) * 40,
            ]
        )
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
        nov = _novelty(x, kernel_bars * beats_per_bar // 2)
        # Score each bar line by the strongest novelty within half a bar of it.
        bars = np.arange(0, n_beats, beats_per_bar)
        half = beats_per_bar // 2
        score = np.array([nov[max(b - half, 0) : b + half + 1].max() for b in bars])
        threshold = score.mean() + 0.5 * score.std()
        min_gap = min_bars
        chosen: list[int] = []
        for i in np.argsort(-score):
            if score[i] < threshold:
                break
            if i < min_gap or len(bars) - i < min_gap:
                continue
            if all(abs(i - j) >= min_gap for j in chosen):
                chosen.append(int(i))
        bounds = [0] + sorted(int(bars[i]) for i in chosen)

    starts = [0.0] + [float(beat_times[b]) for b in bounds[1:]]
    ends = starts[1:] + [float(duration_s)]
    loud = [_span_mean(rms, frame_rate, s, e) for s, e in zip(starts, ends, strict=True)]
    busy = [_span_mean(onset, frame_rate, s, e) for s, e in zip(starts, ends, strict=True)]
    loud_n = np.array(loud) / (max(loud) or 1.0)
    busy_n = np.array(busy) / (max(busy) or 1.0)
    energy = 0.7 * loud_n + 0.3 * busy_n
    energy = energy / (energy.max() or 1.0)
    labels = _label(energy)
    return [
        Section(start_s=s, end_s=e, label=lab, energy=float(en))
        for s, e, lab, en in zip(starts, ends, labels, energy, strict=True)
    ]


def _span_mean(arr: np.ndarray, frame_rate: float, start_s: float, end_s: float) -> float:
    a = int(start_s * frame_rate)
    b = max(int(end_s * frame_rate), a + 1)
    seg = arr[a:b]
    return float(seg.mean()) if seg.size else 0.0


def _label(energy: np.ndarray) -> list[str]:
    n = len(energy)
    if n == 1:
        return ["verse"]
    hi = np.percentile(energy, 75)
    labels = []
    for i, e in enumerate(energy):
        prev = energy[i - 1] if i else e
        if i == 0 and e < hi:
            labels.append("intro")
        elif i == n - 1 and e < hi:
            labels.append("outro")
        elif e >= hi and e >= 0.8:
            labels.append("drop" if prev < 0.6 * e else "chorus")
        elif e < 0.5:
            labels.append("break")
        else:
            labels.append("verse")
    return labels
