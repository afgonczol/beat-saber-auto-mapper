"""Reference windows (song structure) and the repetition metric."""

import numpy as np
import pytest

from bsmapper.data.structure import (
    STRUCTURE_DIM,
    find_references,
    mirror_notes,
    per_beat_mean,
    repetition_kind,
    repetition_profile,
    span_notes,
    structure_features,
    window_repetition,
)
from bsmapper.schema import Characteristic, Color, CutDirection, Difficulty, DifficultyMap, Note
from bsmapper.tokenizer import MapTokenizer


def _song(sections: str, beats_per_section: int = 16, seed: int = 0) -> np.ndarray:
    """Per-beat structure features for a song made of labelled sections, e.g. "ABAB"."""
    rng = np.random.default_rng(seed)
    looks = {c: rng.normal(size=(beats_per_section, STRUCTURE_DIM)) for c in set(sections)}
    rows = []
    for c in sections:
        x = looks[c] + 0.05 * rng.normal(size=(beats_per_section, STRUCTURE_DIM))
        x[:, -1] = 0.5  # audible
        rows.append(x)
    return np.concatenate(rows).astype(np.float32)


def test_finds_the_repeated_section():
    refs = find_references(_song("ABCAB"), 5)
    assert refs.start[0] == -1 and np.isnan(refs.score[0])
    assert refs.start[3] == 0 and refs.start[4] == 16 * 24  # A -> A, B -> B
    assert refs.score[3] > 0.9 and refs.score[4] > 0.9
    assert refs.score[2] < 0.5  # C has no match
    kept = refs.kept(0.8)
    assert kept[2] == -1 and kept[3] == 0
    assert refs.ref_start(3, 0.8) == 0 and refs.ref_start(2, 0.8) is None


def test_bar_shifted_repeat():
    # The repeat starts 2 bars later than a window boundary would.
    x = _song("ABCD", seed=1)
    x = np.concatenate([x[:40], x[:24], x[40:]])  # beats 40..63 repeat beats 0..23
    refs = find_references(x, len(x) // 16)
    assert refs.start[3] == 8 * 24  # window 3 = beats 48..63 = beats 8..23
    assert refs.score[3] > 0.9


def test_silence_does_not_match():
    x = _song("AAAA")
    x[:, -1] = 0.0  # silent
    assert np.nanmax(find_references(x, 4).score) == 0.0


def test_structure_features_and_beat_means():
    ticks = np.zeros((48, 83), dtype=np.float32)
    ticks[:24, :80] = -10.0
    ticks[24:, 80] = 1.0
    ticks[:, 82] = 0.25
    f = structure_features(ticks, np.ones((2, 12), dtype=np.float32))
    assert f.shape == (2, STRUCTURE_DIM)
    assert f[0, 12] == pytest.approx(-10.0) and f[1, 12] == 0.0
    assert f[1, 32:40].tolist() == [1.0] * 8 and f[0, -1] == pytest.approx(0.25)
    frames = np.arange(10, dtype=np.float64)[:, None]
    m = per_beat_mean(frames, 2.0, lambda b: np.asarray(b) * 2.0, 2)  # 4 frames per beat
    assert m[:, 0].tolist() == [1.5, 5.5]


def _n(beat, x, color=Color.RED, d=CutDirection.DOWN_LEFT):
    return Note(beat, x, 0, color, d)


def test_repetition_kinds():
    ref = [(0, 0, 0, 0, int(CutDirection.DOWN_LEFT)), (24, 1, 3, 0, int(CutDirection.UP))]
    assert repetition_kind(ref, ref) == "copy"
    mirrored = mirror_notes(ref)
    assert mirrored == [(0, 1, 3, 0, int(CutDirection.DOWN_RIGHT)), (24, 0, 0, 0, 0)]
    assert repetition_kind(mirrored, ref) == "mirror"
    rhythm = [(0, 0, 2, 1, 1), (24, 1, 1, 2, 0)]
    assert repetition_kind(rhythm, ref) == "rhythm"
    assert repetition_kind([(12, 0, 0, 0, 0)], ref) == "different"
    assert repetition_kind([], []) is None


def test_repetition_profile_on_maps():
    notes = [_n(1, 0), _n(2, 3, Color.BLUE, CutDirection.UP)]  # window 0
    notes += [_n(17, 3, Color.BLUE, CutDirection.DOWN_RIGHT), _n(18, 0, Color.RED, CutDirection.UP)]
    notes += [_n(33, 0), _n(34, 3, Color.BLUE, CutDirection.UP)]  # window 2: copy of 0
    dmap = DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT, notes=notes)
    assert span_notes(dmap, 16 * 24)[0] == (24, 1, 3, 0, int(CutDirection.DOWN_RIGHT))
    prof = repetition_profile(dmap, [None, 0, 0])
    assert prof["matched"] == 2 and prof["copy"] == 0.5 and prof["mirror"] == 0.5
    # The token-level version used by the dataset stats agrees.
    windows = MapTokenizer().encode(dmap, n_windows=3)
    tokens = np.array([t for w in windows for t in w.body], dtype=np.int16)
    offsets = np.cumsum([0, *(len(w.body) for w in windows)])
    refs = find_references(_song("AAA"), 3)
    refs.start[:] = [-1, 0, 0]
    kinds = [k for _, _, k, _ in window_repetition(tokens, offsets, refs)]
    assert kinds == ["mirror", "copy"]
