"""Fit a mappable beat grid (BPM, offset, BPM changes) to tracked beat times.

Beat trackers return one timestamp per beat, with jitter, missed beats and the
odd extra one. A map needs the opposite: a single BPM (or a few BPM changes)
and an offset, so that every tick of the grid lands on the music. We fit a
straight line ``t = offset + k * 60 / bpm`` through the beats, fall back to a
continuous piecewise-linear fit when the tempo drifts, snap the BPM to a round
value when that costs nothing, and put beat 0 on the first downbeat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bsmapper.schema import BpmChange


@dataclass(slots=True)
class BeatGrid:
    """A tempo map anchored in the audio.

    Beat 0 sits at ``offset_s`` seconds and is a downbeat; ``0 <= offset_s`` is
    less than one bar. ``bpm_changes`` use the same beat numbering and follow the
    semantics of ``schema.Beatmap.bpm_changes``.
    """

    bpm: float
    offset_s: float
    bpm_changes: list[BpmChange] = field(default_factory=list)
    beats_per_bar: int = 4

    def _segments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(start_beat, start_seconds, bpm) arrays, one entry per constant-tempo segment."""
        starts, secs, bpms = [0.0], [self.offset_s], [self.bpm]
        for c in sorted(self.bpm_changes, key=lambda c: c.beat):
            if c.beat <= 0:
                bpms[0] = c.bpm
                continue
            secs.append(secs[-1] + (c.beat - starts[-1]) * 60.0 / bpms[-1])
            starts.append(c.beat)
            bpms.append(c.bpm)
        return np.array(starts), np.array(secs), np.array(bpms)

    def beat_to_seconds(self, beats: float | np.ndarray) -> np.ndarray:
        starts, secs, bpms = self._segments()
        b = np.asarray(beats, dtype=np.float64)
        i = np.clip(np.searchsorted(starts, b, side="right") - 1, 0, None)
        return secs[i] + (b - starts[i]) * 60.0 / bpms[i]

    def seconds_to_beat(self, seconds: float | np.ndarray) -> np.ndarray:
        starts, secs, bpms = self._segments()
        t = np.asarray(seconds, dtype=np.float64)
        i = np.clip(np.searchsorted(secs, t, side="right") - 1, 0, None)
        return starts[i] + (t - secs[i]) * bpms[i] / 60.0

    def n_beats(self, duration_s: float) -> int:
        """Number of whole beats from beat 0 that start before ``duration_s``."""
        return max(int(np.ceil(float(self.seconds_to_beat(duration_s)) - 1e-9)), 0)

    def beat_times(self, duration_s: float) -> np.ndarray:
        return self.beat_to_seconds(np.arange(self.n_beats(duration_s), dtype=np.float64))

    def downbeat_times(self, duration_s: float) -> np.ndarray:
        return self.beat_times(duration_s)[:: self.beats_per_bar]

    def lead_in_s(self) -> float:
        """Silence to prepend to the audio so the downbeat *before* beat 0 lands at 0 s.

        Beat Saber puts beat 0 at the start of the audio file. Padding by this
        amount (and adding ``beats_per_bar`` to every beat when the padding is
        nonzero) keeps the map on an integer grid; it is 0 when beat 0 is at 0 s.
        """
        bar = self.beats_per_bar * 60.0 / self.bpm
        return (bar - self.offset_s) % bar if self.offset_s > 1e-6 else 0.0


# --- fitting -------------------------------------------------------------------


def _beat_indices(beats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer beat numbers for tracked beats, absorbing missed and extra beats."""
    ibis = np.diff(beats)
    period = float(np.median(ibis))
    keep, ks = [0], [0]
    recent: list[float] = []
    for i in range(1, len(beats)):
        local = float(np.median(recent[-8:])) if len(recent) >= 4 else period
        frac = (beats[i] - beats[keep[-1]]) / local
        steps = int(round(frac))
        if steps < 1 or abs(frac - steps) > 0.25:
            continue  # extra or off-beat detection between two grid beats
        if steps == 1 and abs(frac - 1) < 0.15:
            recent.append(beats[i] - beats[keep[-1]])
        keep.append(i)
        ks.append(ks[-1] + steps)
    return np.array(ks, dtype=np.float64), beats[np.array(keep)]


def _robust_line(
    k: np.ndarray, t: np.ndarray, anchor: tuple[float, float] | None = None
) -> tuple[float, float, np.ndarray]:
    """Least-squares ``t = a + p*k`` with outlier rejection. Returns (a, p, inlier mask)."""
    mask = np.ones(len(k), dtype=bool)
    a = p = 0.0
    for _ in range(4):
        kk, tt = k[mask], t[mask]
        if anchor is None:
            p, a = np.polyfit(kk, tt, 1)
        else:
            k0, t0 = anchor
            dk = kk - k0
            p = float(np.dot(dk, tt - t0) / max(np.dot(dk, dk), 1e-12))
            a = t0 - p * k0
        r = np.abs(t - (a + p * k))
        mad = float(np.median(r[mask])) * 1.4826
        new = r <= max(3 * mad, 0.015)
        if new.sum() < max(4, 0.5 * len(k)) or np.array_equal(new, mask):
            break
        mask = new
    return float(a), float(p), mask


@dataclass(slots=True)
class GridFit:
    grid: BeatGrid
    residuals_s: np.ndarray  # each tracked beat minus its nearest grid beat
    tol_s: float

    @property
    def coverage(self) -> float:
        """Share of tracked beats within ``tol_s`` of a grid beat."""
        return float(np.mean(np.abs(self.residuals_s) <= self.tol_s))

    @property
    def median_error_s(self) -> float:
        return float(np.median(np.abs(self.residuals_s)))


def _snap_bpm(bpm: float) -> list[float]:
    """Round BPM candidates, coarsest first."""
    cands = [round(bpm), round(bpm * 2) / 2, round(bpm * 10) / 10, round(bpm * 100) / 100]
    return list(dict.fromkeys(c for c in cands if c > 0))


def _comb_fit(beats: np.ndarray, tol_s: float, search: float = 0.03) -> tuple[float, float]:
    """Constant tempo ``t = a + k*p`` that puts the most tracked beats on the grid.

    A brute-force search over period and phase, so sections where the tracker
    loses the phase (breakdowns), switches tempo octave or misses beats only
    cost their own beats instead of dragging a least-squares fit.
    """
    ibis = np.diff(beats)
    p0 = float(np.median(ibis))
    span = max(float(beats[-1] - beats[0]), p0)
    step = tol_s * p0 / (4 * span)  # drift of at most tol/4 between neighbouring candidates
    periods = np.arange(p0 * (1 - search), p0 * (1 + search), step)
    bins = 96
    phase = (beats[None, :] / periods[:, None]) % 1.0
    hist = np.zeros((len(periods), bins))
    np.add.at(hist, (np.arange(len(periods))[:, None], (phase * bins).astype(np.int64) % bins), 1)
    h = max(1, int(round(tol_s / p0 * bins)))
    win = sum(np.roll(hist, s, axis=1) for s in range(-h, h + 1))
    i, j = np.unravel_index(np.argmax(win), win.shape)
    p = float(periods[i])
    a = (j + 0.5) / bins * p
    for _ in range(3):  # least-squares refinement on the inliers
        k = np.round((beats - a) / p)
        inl = np.abs(beats - (a + k * p)) <= tol_s
        if inl.sum() < 4:
            break
        p, a = (float(v) for v in np.polyfit(k[inl], beats[inl], 1))
    return a, p


def _nearest_residuals(knots: list[tuple[float, float, float]], t: np.ndarray) -> np.ndarray:
    pos = _invert_knots(knots, t)
    return t - _eval_knots(knots, np.round(pos))


def fit_beat_grid(
    beats: np.ndarray,
    downbeats: np.ndarray | None = None,
    *,
    tol_s: float = 0.025,
    allow_tempo_changes: bool = True,
    min_coverage: float = 0.8,
    min_segment_beats: int = 32,
    snap: bool = True,
    snap_tol_s: float = 0.005,
) -> GridFit:
    """Fit a :class:`BeatGrid` to tracked beat (and optional downbeat) times in seconds.

    A single tempo is preferred. BPM changes are only introduced when fewer than
    ``min_coverage`` of the tracked beats sit on the constant grid and a
    piecewise fit covers clearly more of them.
    """
    beats = np.unique(np.asarray(beats, dtype=np.float64))
    if len(beats) < 4:
        raise ValueError(f"need at least 4 beats to fit a grid, got {len(beats)}")

    # 1. One tempo for the whole song.
    a, p = _comb_fit(beats, tol_s)
    knots: list[tuple[float, float, float]] = [(0.0, a, 60.0 / p)]  # (k, t at k, bpm after)
    if snap:
        # Take the roundest BPM whose line stays within ``snap_tol_s`` of the free fit
        # over the whole song, so rounding never adds audible drift.
        k = np.round((beats - a) / p)
        inl = np.abs(beats - (a + k * p)) <= tol_s
        span = np.array([k[inl].min(), k[inl].max()])
        for bpm in _snap_bpm(60.0 / p):
            ps = 60.0 / bpm
            a_s = float(np.median(beats[inl] - ps * k[inl]))
            if np.max(np.abs((a_s + ps * span) - (a + p * span))) <= snap_tol_s:
                knots = [(0.0, a_s, bpm)]
                break
    resid = _nearest_residuals(knots, beats)
    coverage = float(np.mean(np.abs(resid) <= tol_s))

    # 2. Tempo drift: a continuous piecewise-linear fit, kept only if it clearly helps.
    if allow_tempo_changes and coverage < min_coverage and len(beats) >= 2 * min_segment_beats:
        k_seq, t_seq = _beat_indices(beats)
        pw_knots = _piecewise_fit(k_seq, t_seq, tol_s, min_segment_beats, snap)
        pw_resid = _nearest_residuals(pw_knots, beats)
        if float(np.mean(np.abs(pw_resid) <= tol_s)) >= coverage + 0.15:
            knots, resid = pw_knots, pw_resid

    return GridFit(grid=_anchor_on_downbeat(knots, downbeats), residuals_s=resid, tol_s=tol_s)


def _eval_knots(knots: list[tuple[float, float, float]], k: np.ndarray) -> np.ndarray:
    ks = np.array([kn[0] for kn in knots])
    ts = np.array([kn[1] for kn in knots])
    ps = 60.0 / np.array([kn[2] for kn in knots])
    i = np.clip(np.searchsorted(ks, k, side="right") - 1, 0, None)
    return ts[i] + (k - ks[i]) * ps[i]


def _piecewise_fit(
    k: np.ndarray, t: np.ndarray, tol_s: float, min_len: int, snap: bool
) -> list[tuple[float, float, float]]:
    knots: list[tuple[float, float, float]] = []
    start, anchor = 0, None
    n = len(k)
    while start < n - 1:
        end = min(start + min_len, n)
        a, p, _ = _robust_line(k[start:end], t[start:end], anchor)
        # Extend while the newest beats still sit on the line.
        while end < n:
            a2, p2, _ = _robust_line(k[start : end + 1], t[start : end + 1], anchor)
            lo = max(end - 3, start)
            tail = np.abs(t[lo : end + 1] - (a2 + p2 * k[lo : end + 1]))
            if float(np.median(tail)) > tol_s:
                break
            a, p, end = a2, p2, end + 1
        bpm = 60.0 / p
        if snap and anchor is not None:
            # Keep tempo changes readable: snap when the segment's error barely moves.
            for cand in _snap_bpm(bpm):
                if abs(60.0 / cand - p) * (k[end - 1] - k[start]) < tol_s / 2:
                    bpm = cand
                    break
        k0 = float(k[start]) if anchor is None else anchor[0]
        t0 = a + p * k0 if anchor is None else anchor[1]
        if knots and abs(knots[-1][2] - bpm) < 1e-3:
            pass  # same tempo as the previous segment: just extend it
        else:
            knots.append((k0, t0, bpm))
        if end >= n:
            break
        # The next segment starts where this one's last kept beat sits on the line.
        cut = max(end - 2, start + 1)
        anchor = (float(k[cut]), t0 + (k[cut] - k0) * 60.0 / bpm)
        start = cut
    return knots


def _anchor_on_downbeat(
    knots: list[tuple[float, float, float]], downbeats: np.ndarray | None
) -> BeatGrid:
    beats_per_bar, phase = 4, 0
    if downbeats is not None and len(downbeats) >= 2:
        # Map each downbeat to the nearest grid beat number.
        k_of = _invert_knots(knots, np.asarray(downbeats, dtype=np.float64))
        kd = np.round(k_of)
        kd = kd[np.abs(k_of - kd) < 0.25].astype(np.int64)
        if len(kd) >= 2:
            gaps = np.diff(np.unique(kd))
            gaps = gaps[(gaps >= 2) & (gaps <= 8)]
            if gaps.size:
                vals, counts = np.unique(gaps, return_counts=True)
                beats_per_bar = 3 if vals[np.argmax(counts)] == 3 else 4
            vals, counts = np.unique(kd % beats_per_bar, return_counts=True)
            phase = int(vals[np.argmax(counts)])

    # Beat 0 = the first downbeat at or after 0 s (extrapolating before the first tracked beat).
    k_at_zero = float(_invert_knots(knots, np.array([0.0]))[0])
    k0 = np.ceil(k_at_zero - 1e-6)
    k0 += (phase - k0) % beats_per_bar
    offset = float(_eval_knots(knots, np.array([k0]))[0])
    seg = int(np.searchsorted([kn[0] for kn in knots], k0, side="right")) - 1
    bpm0 = knots[max(seg, 0)][2]
    changes = [BpmChange(beat=float(kn[0] - k0), bpm=kn[2]) for kn in knots[1:] if kn[0] > k0]
    # A change before beat 0 just sets the starting tempo.
    return BeatGrid(bpm=bpm0, offset_s=offset, bpm_changes=changes, beats_per_bar=beats_per_bar)


def refine_offset(
    grid: BeatGrid,
    envelope: np.ndarray,
    frame_rate: float,
    *,
    search_s: float = 0.04,
    latency_s: float = 0.0,
) -> tuple[BeatGrid, float]:
    """Shift the whole grid to where the onset envelope, summed over all beats, peaks.

    Trackers have a style-dependent bias of a few to tens of milliseconds, but
    the onset envelope at the grid beats, averaged over hundreds of beats, pins
    the phase down to a few ms. ``latency_s`` is the envelope's own delay
    (calibrated against human maps, see ``docs/audio.md``). Returns the new grid
    and the applied shift in seconds; the shift never moves beat 0 before 0 s.
    """
    duration = len(envelope) / frame_rate
    beats = grid.beat_times(duration)
    beats = beats[(beats > search_s) & (beats < duration - search_s)]
    if len(beats) < 8:
        return grid, 0.0
    shifts = np.arange(-search_s, search_s + 1e-9, 0.001)
    frames = np.arange(len(envelope))
    score = np.array([np.interp((beats + s) * frame_rate, frames, envelope).sum() for s in shifts])
    i = int(np.argmax(score))
    best = float(shifts[i])
    if 0 < i < len(shifts) - 1:  # parabolic peak for sub-millisecond precision
        y0, y1, y2 = score[i - 1 : i + 2]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            best += 0.5 * (y0 - y2) / denom * 0.001
    shift = float(best - latency_s)
    shifted = BeatGrid(grid.bpm, grid.offset_s + shift, list(grid.bpm_changes), grid.beats_per_bar)
    if shifted.offset_s >= 0:
        return shifted, shift
    # Keep beat 0 at or after 0 s: renumber so beat 0 is the next downbeat.
    bpb = grid.beats_per_bar
    bpm = grid.bpm
    changes = []
    for c in sorted(grid.bpm_changes, key=lambda c: c.beat):
        if c.beat - bpb <= 0:
            bpm = c.bpm
        else:
            changes.append(BpmChange(c.beat - bpb, c.bpm))
    offset = float(shifted.beat_to_seconds(bpb))
    return BeatGrid(bpm, offset, changes, bpb), shift


def _invert_knots(knots: list[tuple[float, float, float]], times: np.ndarray) -> np.ndarray:
    ks = np.array([kn[0] for kn in knots])
    ts = np.array([kn[1] for kn in knots])
    ps = 60.0 / np.array([kn[2] for kn in knots])
    i = np.clip(np.searchsorted(ts, times, side="right") - 1, 0, None)
    return ks[i] + (times - ts[i]) / ps[i]
