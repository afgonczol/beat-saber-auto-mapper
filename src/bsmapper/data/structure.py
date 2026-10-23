"""Song structure: which earlier part of a song a 16-beat window repeats.

Human mappers reuse patterns when the music repeats (a second chorus, a repeated
drop), often mirrored. note_v1 learns this from a *reference window*: for each
16-beat window ``k`` the most similar earlier 16-beat span of the same song, found
from beat-synchronous audio features. Its notes become extra decoder context
(``REF ... REF_END``, see ``tokenizer.windows``).

Per beat (``structure_features``)::

    chroma (12)   pitch classes, from ``beat_chroma`` (harmony and melody)
    mel    (20)   80 log-mel bins averaged in groups of 4 (timbre, loudness)
    onset  (8)    the beat's onset strength in 8 slots of 3 ticks (rhythm)
    rms    (1)    loudness, used to skip silence

Candidate spans start a whole number of bars (4 beats) before the window and end
at or before its start, so a candidate keeps the window's position in the bar and
only uses music (and, at generation time, notes) that already exists. The span
score is the mean over its 16 beats of a weighted cosine similarity per feature
group, the diagonal of a self-similarity matrix. ``find_references`` returns the
best candidate per window with its score; training and generation keep it when the
score clears ``DEFAULT_REF_THRESHOLD`` (tuned on dataset v1, see docs/DATASET.md).

``repetition_kind`` and ``repetition_profile`` classify how a map's notes in a
window relate to its reference: an exact copy, a mirror (colours and x flipped),
the same rhythm with other notes, or different. WS10 uses them on generated maps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from bsmapper.schema import Color, CutDirection, DifficultyMap
from bsmapper.tokenizer.grid import TICKS_PER_BEAT, WINDOW_BEATS, WINDOW_TICKS

N_CHROMA = 12
MEL_GROUP = 4  # 80 mel bins -> 20 bands
ONSET_SLOTS = 8  # 24 ticks -> 8 slots of 3
STRUCTURE_DIM = N_CHROMA + 80 // MEL_GROUP + ONSET_SLOTS + 1
GROUPS = {
    "chroma": slice(0, N_CHROMA),
    "mel": slice(N_CHROMA, N_CHROMA + 80 // MEL_GROUP),
    "onset": slice(STRUCTURE_DIM - 1 - ONSET_SLOTS, STRUCTURE_DIM - 1),
}
RMS_COL = STRUCTURE_DIM - 1
GROUP_WEIGHTS = {"chroma": 1.0, "mel": 1.0, "onset": 1.0}
BAR_BEATS = 4
MIN_GAP_BARS = WINDOW_BEATS // BAR_BEATS  # the span must end at or before the window start
DEFAULT_REF_THRESHOLD = 0.85  # tuned on dataset v1 (docs/DATASET.md, "Structure")
SILENT_RMS = 0.02  # beats quieter than this (scaled to the song's max) count as silence

CHROMA_N_FFT = 4096
CHROMA_HOP = 1024


def beat_chroma(
    y: np.ndarray,
    sr: int,
    beat_seconds: Callable[[np.ndarray], np.ndarray],
    n_beats: int,
) -> np.ndarray:
    """(n_beats, 12) mean chroma per beat; ``beat_seconds`` maps beats to audio time.

    Beats outside the audio are zero.
    """
    chroma = frame_chroma(y, sr)
    return per_beat_mean(chroma, sr / CHROMA_HOP, beat_seconds, n_beats).astype(np.float32)


def frame_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    """(frames, 12) chroma at ``sr / CHROMA_HOP`` frames per second."""
    import librosa

    return librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=CHROMA_N_FFT, hop_length=CHROMA_HOP, center=True
    ).T.astype(np.float32)


def per_beat_mean(
    frames: np.ndarray,
    frame_rate: float,
    beat_seconds: Callable[[np.ndarray], np.ndarray],
    n_beats: int,
) -> np.ndarray:
    """Mean of frame rows over each beat (the nearest frame if a beat holds none)."""
    edges = np.asarray(beat_seconds(np.arange(n_beats + 1, dtype=np.float64))) * frame_rate
    lo = np.clip(np.ceil(edges[:-1]).astype(np.int64), 0, len(frames))
    hi = np.clip(np.ceil(edges[1:]).astype(np.int64), 0, len(frames))
    csum = np.concatenate([np.zeros((1, frames.shape[1])), np.cumsum(frames, axis=0)])
    count = (hi - lo)[:, None]
    out = (csum[hi] - csum[lo]) / np.maximum(count, 1)
    nearest = np.clip(np.round(edges[:-1]).astype(np.int64), 0, max(len(frames) - 1, 0))
    empty = (count[:, 0] == 0) & (edges[:-1] >= 0) & (edges[:-1] < len(frames))
    if len(frames):
        out[empty] = frames[nearest[empty]]
    return out


def structure_features(tick_features: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """(n_beats, STRUCTURE_DIM) per-beat features from the 83-column tick features
    (mel 0..79, onset, flux, rms; 24 rows per beat) and ``beat_chroma``."""
    n_beats = min(len(tick_features) // TICKS_PER_BEAT, len(chroma))
    t = np.asarray(tick_features[: n_beats * TICKS_PER_BEAT], dtype=np.float32)
    t = t.reshape(n_beats, TICKS_PER_BEAT, -1)
    mel = t[:, :, :80].mean(axis=1).reshape(n_beats, -1, MEL_GROUP).mean(axis=2)
    onset = t[:, :, 80].reshape(n_beats, ONSET_SLOTS, -1).mean(axis=2)
    rms = t[:, :, 82].mean(axis=1, keepdims=True)
    return np.concatenate([chroma[:n_beats], mel, onset, rms], axis=1).astype(np.float32)


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(norm > 1e-6, x / np.maximum(norm, 1e-6), 0.0)


def beat_similarity(struct: np.ndarray) -> np.ndarray:
    """(n_beats, n_beats) weighted cosine similarity of the per-beat features.

    Mel bands are z-scored over the song's audible beats first, so the score
    compares timbre relative to the song rather than overall loudness. Silent
    beats have similarity 0 to everything.
    """
    s = np.asarray(struct, dtype=np.float64)
    audible = s[:, RMS_COL] >= SILENT_RMS
    total = np.zeros((len(s), len(s)))
    weight = 0.0
    for name, cols in GROUPS.items():
        x = s[:, cols].copy()
        if name == "mel" and audible.sum() > 1:
            mu, sd = x[audible].mean(axis=0), x[audible].std(axis=0)
            x = (x - mu) / np.maximum(sd, 1e-3)
        x[~audible] = 0.0
        u = _unit(x)
        total += GROUP_WEIGHTS[name] * (u @ u.T)
        weight += GROUP_WEIGHTS[name]
    return total / weight


@dataclass(frozen=True, slots=True)
class References:
    """Best earlier span per window: ``start`` in ticks (-1 = none) and its score."""

    start: np.ndarray  # int64 (n_windows,)
    score: np.ndarray  # float32 (n_windows,), NaN where there is no candidate

    def kept(self, threshold: float = DEFAULT_REF_THRESHOLD) -> np.ndarray:
        """Start ticks with the ones below ``threshold`` set to -1."""
        ok = (self.start >= 0) & (np.nan_to_num(self.score, nan=-1.0) >= threshold)
        return np.where(ok, self.start, -1)

    def ref_start(self, k: int, threshold: float = DEFAULT_REF_THRESHOLD) -> int | None:
        if k < 0 or k >= len(self.start):
            return None
        s = int(self.start[k])
        return s if s >= 0 and float(self.score[k]) >= threshold else None


def find_references(struct: np.ndarray, n_windows: int) -> References:
    """The most similar earlier, bar-shifted 16-beat span for each window."""
    start = np.full(n_windows, -1, dtype=np.int64)
    score = np.full(n_windows, np.nan, dtype=np.float32)
    n_beats = len(struct)
    if n_beats < 2 * WINDOW_BEATS:
        return References(start, score)
    sim = beat_similarity(struct)
    idx = np.arange(WINDOW_BEATS)
    for k in range(1, n_windows):
        a = k * WINDOW_BEATS
        if a + WINDOW_BEATS > n_beats:
            break
        b = a - BAR_BEATS * np.arange(MIN_GAP_BARS, a // BAR_BEATS + 1)
        if not len(b):
            continue
        scores = sim[a + idx[None, :], b[:, None] + idx[None, :]].mean(axis=1)
        best = int(np.argmax(scores))  # ties: the nearest earlier span
        start[k] = int(b[best]) * TICKS_PER_BEAT
        score[k] = scores[best]
    return References(start, score)


# --- how a map's notes relate to the reference ----------------------------------------

REPETITION_KINDS = ("copy", "mirror", "rhythm", "different")
MATCH_F1 = 0.8  # share of notes that must agree to count as a copy / mirror / same rhythm

_MIRROR_DIR = {
    CutDirection.UP: CutDirection.UP,
    CutDirection.DOWN: CutDirection.DOWN,
    CutDirection.LEFT: CutDirection.RIGHT,
    CutDirection.RIGHT: CutDirection.LEFT,
    CutDirection.UP_LEFT: CutDirection.UP_RIGHT,
    CutDirection.UP_RIGHT: CutDirection.UP_LEFT,
    CutDirection.DOWN_LEFT: CutDirection.DOWN_RIGHT,
    CutDirection.DOWN_RIGHT: CutDirection.DOWN_LEFT,
    CutDirection.ANY: CutDirection.ANY,
}

NoteKey = tuple[int, int, int, int, int]  # tick in span, color, x, y, direction


def span_notes(dmap: DifficultyMap, start_tick: int) -> list[NoteKey]:
    """Colour notes in [start, start + 16 beats), as (tick in span, color, x, y, dir)."""
    lo, hi = start_tick, start_tick + WINDOW_TICKS
    out = []
    for n in dmap.notes:
        t = int(round(n.beat * TICKS_PER_BEAT))
        if lo <= t < hi:
            out.append((t - lo, int(n.color), n.x, n.y, int(n.direction)))
    return out


def mirror_notes(notes: Sequence[NoteKey]) -> list[NoteKey]:
    red, blue = int(Color.RED), int(Color.BLUE)
    return [
        (t, blue if c == red else red, 3 - x, y, int(_MIRROR_DIR[CutDirection(d)]))
        for t, c, x, y, d in notes
    ]


def _f1(a: Sequence, b: Sequence) -> float:
    """F1 of two multisets (1.0 when both are empty)."""
    if not a and not b:
        return 1.0
    from collections import Counter

    common = sum((Counter(a) & Counter(b)).values())
    return 2 * common / (len(a) + len(b))


def repetition_scores(window: Sequence[NoteKey], ref: Sequence[NoteKey]) -> dict[str, float]:
    """F1 agreement of a window's notes with its reference: exact, mirrored, rhythm."""
    return {
        "copy": _f1(window, ref),
        "mirror": _f1(window, mirror_notes(ref)),
        "rhythm": _f1(sorted({n[0] for n in window}), sorted({n[0] for n in ref})),
    }


def repetition_kind(window: Sequence[NoteKey], ref: Sequence[NoteKey]) -> str | None:
    """``copy``, ``mirror``, ``rhythm`` or ``different``; None when both are empty.

    A window counts as a copy (or mirror) when at least ``MATCH_F1`` of the notes
    agree exactly (after mirroring), as the same rhythm when the note times agree that
    well. A symmetric pattern that is both a copy and a mirror counts as a copy.
    """
    if not window and not ref:
        return None
    s = repetition_scores(window, ref)
    if s["copy"] >= MATCH_F1:
        return "copy"
    if s["mirror"] >= MATCH_F1:
        return "mirror"
    if s["rhythm"] >= MATCH_F1:
        return "rhythm"
    return "different"


def repetition_profile(dmap: DifficultyMap, ref_starts: Sequence[int | None]) -> dict[str, float]:
    """Share of matched windows of each kind, for a human or generated map.

    ``ref_starts[k]`` is window ``k``'s reference start tick (None or -1 = no
    reference), e.g. ``find_references(...).kept()``. The result has a share per
    ``REPETITION_KINDS`` entry and ``matched`` (windows counted).
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for k, s in enumerate(ref_starts):
        if s is None or s < 0:
            continue
        kind = repetition_kind(span_notes(dmap, k * WINDOW_TICKS), span_notes(dmap, int(s)))
        if kind is not None:
            counts[kind] += 1
    n = sum(counts.values())
    out = {kind: counts[kind] / n if n else float("nan") for kind in REPETITION_KINDS}
    out["matched"] = float(n)
    return out


# --- the same on token windows (dataset build, training data) ------------------------


def event_notes(events) -> list[NoteKey]:
    """NoteKeys of parsed note events (``Grammar.parse`` output, ticks as given)."""
    return [
        (e.tick, int(e.head), e.values[0][0], e.values[0][1], int(e.values[1]))
        for e in events
        if e.kind == "NOTE"
    ]


def window_repetition(
    tokens: np.ndarray, offsets: np.ndarray, refs: References
) -> list[tuple[int, float, str | None, dict[str, float]]]:
    """(window, score, kind, F1 scores) for every window with a reference candidate,
    from one difficulty's stored window bodies."""
    from bsmapper.tokenizer import NOTE_GRAMMAR, span_tokens

    bodies = [tokens[offsets[k] : offsets[k + 1]].astype(np.int64).tolist()
              for k in range(len(offsets) - 1)]  # fmt: skip
    out = []
    for k, (s, score) in enumerate(zip(refs.start, refs.score, strict=True)):
        if s < 0 or k >= len(bodies):
            continue
        window = event_notes(NOTE_GRAMMAR.parse(bodies[k]))
        ref = event_notes(NOTE_GRAMMAR.parse(span_tokens(bodies[:k], int(s), NOTE_GRAMMAR, 10**6)))
        kind = repetition_kind(window, ref)
        out.append((k, float(score), kind, repetition_scores(window, ref)))
    return out
