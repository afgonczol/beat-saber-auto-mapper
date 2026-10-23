"""Pattern reuse on repeated music: does a map repeat itself where the song does?

Human mappers usually map a repeated chorus with the same pattern, its mirror, or at
least the same rhythm. This module finds repeated audio and classifies what a map does
there.

1. **Repeated audio.** The song is cut into ``SEGMENT_BEATS``-beat segments starting on
   every bar line (``HOP_BEATS``). Each segment is described by its per-beat mel spectrum
   (z-scored over the song) and its onset envelope at 1/8-beat resolution. A segment
   repeats an earlier, non-overlapping one when the cosine similarity of the two
   descriptors is at least ``SIMILARITY``; each segment keeps only its best earlier
   match. Silent segments are skipped. The pairs depend on the audio only, so every
   system is scored on the same pairs.
2. **What the map does** on each pair, comparing the notes of both segments at the same
   offset from the segment start (F1 of the two note sets, ``MATCH_F1`` or more counts):
   - ``exact``: same notes (tick, colour, cell, direction)
   - ``mirrored``: same notes after a left-right mirror (colours swap, x -> 3 - x,
     left/right directions swap)
   - ``same_rhythm``: same onset ticks, different notes
   - ``different``: anything else (pairs where neither segment has notes are left out)

Shares of the four classes are compared with the human maps' shares.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from bsmapper.data.mapgrid import N_MELS
from bsmapper.schema import CutDirection, DifficultyMap
from bsmapper.tokenizer import TICKS_PER_BEAT

SEGMENT_BEATS = 8
HOP_BEATS = 4
SIMILARITY = 0.85
MATCH_F1 = 0.8
SILENT_RMS = 0.02
ONSET_STEPS = 8  # onset envelope samples per beat
CLASSES = ("exact", "mirrored", "same_rhythm", "different")

_MIRROR_DIR = {
    CutDirection.LEFT: CutDirection.RIGHT,
    CutDirection.RIGHT: CutDirection.LEFT,
    CutDirection.UP_LEFT: CutDirection.UP_RIGHT,
    CutDirection.UP_RIGHT: CutDirection.UP_LEFT,
    CutDirection.DOWN_LEFT: CutDirection.DOWN_RIGHT,
    CutDirection.DOWN_RIGHT: CutDirection.DOWN_LEFT,
}


def segment_descriptors(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(segment start beats, unit descriptors) from a (ticks, 83) feature block."""
    n_beats = features.shape[0] // TICKS_PER_BEAT
    if n_beats < SEGMENT_BEATS * 2:
        return np.empty(0, dtype=np.int64), np.empty((0, 0))
    usable = features[: n_beats * TICKS_PER_BEAT].astype(np.float64)
    beats = usable.reshape(n_beats, TICKS_PER_BEAT, -1)
    mel = beats[:, :, :N_MELS].mean(axis=1)
    mel = (mel - mel.mean(axis=0)) / (mel.std(axis=0) + 1e-6)
    step = TICKS_PER_BEAT // ONSET_STEPS
    onset = beats[:, :, N_MELS].reshape(n_beats, ONSET_STEPS, step).mean(axis=2)
    onset = (onset - onset.mean()) / (onset.std() + 1e-6)
    rms = beats[:, :, N_MELS + 2].mean(axis=1)
    starts, descs = [], []
    for s in range(0, n_beats - SEGMENT_BEATS + 1, HOP_BEATS):
        seg = slice(s, s + SEGMENT_BEATS)
        if np.median(rms[seg]) < SILENT_RMS:
            continue
        v = np.concatenate([mel[seg].ravel(), 2.0 * onset[seg].ravel()])
        v = v - v.mean()
        norm = np.linalg.norm(v)
        if norm > 0:
            starts.append(s)
            descs.append(v / norm)
    if not descs:
        return np.empty(0, dtype=np.int64), np.empty((0, 0))
    return np.array(starts, dtype=np.int64), np.stack(descs)


def repeated_pairs(features: np.ndarray, threshold: float = SIMILARITY) -> list[tuple[int, int]]:
    """(earlier start beat, later start beat) for every repeated segment."""
    starts, descs = segment_descriptors(features)
    if not len(starts):
        return []
    sim = descs @ descs.T
    pairs = []
    for j in range(len(starts)):
        earlier = np.nonzero(starts + SEGMENT_BEATS <= starts[j])[0]
        if not len(earlier):
            continue
        best = earlier[np.argmax(sim[earlier, j])]
        if sim[best, j] >= threshold:
            pairs.append((int(starts[best]), int(starts[j])))
    return pairs


def _segment(dmap: DifficultyMap, start: int) -> set[tuple[int, int, int, int, int]]:
    lo, hi = start * TICKS_PER_BEAT, (start + SEGMENT_BEATS) * TICKS_PER_BEAT
    out = set()
    for n in dmap.notes:
        t = round(n.beat * TICKS_PER_BEAT)
        if lo <= t < hi:
            out.add((t - lo, int(n.color), n.x, n.y, int(n.direction)))
    return out


def _mirror(notes: set[tuple[int, int, int, int, int]]) -> set[tuple[int, int, int, int, int]]:
    return {
        (t, 1 - c, 3 - x, y, int(_MIRROR_DIR.get(CutDirection(d), CutDirection(d))))
        for t, c, x, y, d in notes
    }


def _f1(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


def classify(dmap: DifficultyMap, earlier: int, later: int) -> str | None:
    """The class of one repeated pair, or None when neither segment has notes."""
    a, b = _segment(dmap, earlier), _segment(dmap, later)
    if not a and not b:
        return None
    if _f1(a, b) >= MATCH_F1:
        return "exact"
    if _f1(_mirror(a), b) >= MATCH_F1:
        return "mirrored"
    if _f1({n[0] for n in a}, {n[0] for n in b}) >= MATCH_F1:
        return "same_rhythm"
    return "different"


def repetition_counts(dmap: DifficultyMap, pairs: list[tuple[int, int]]) -> Counter:
    counts: Counter = Counter({c: 0 for c in CLASSES})
    for a, b in pairs:
        c = classify(dmap, a, b)
        if c is not None:
            counts[c] += 1
    return counts


def shares(counts: Counter) -> dict[str, float]:
    total = sum(counts[c] for c in CLASSES)
    return {c: (counts[c] / total if total else float("nan")) for c in CLASSES}


def share_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Total variation distance between two class-share dicts (0 = same, 1 = disjoint)."""
    if any(not np.isfinite(a[c]) or not np.isfinite(b[c]) for c in CLASSES):
        return float("nan")
    return 0.5 * sum(abs(a[c] - b[c]) for c in CLASSES)
