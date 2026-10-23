"""Tokenizer v2 (note_v1): INTENSITY prefix, reference blocks, v1 compatibility."""

import hashlib

import pytest

from bsmapper.interfaces import GenerationRequest
from bsmapper.schema import Characteristic, Color, CutDirection, Difficulty, DifficultyMap, Note
from bsmapper.tokenizer import (
    BOS,
    EOS,
    NOTE_GRAMMAR,
    NOTE_VOCAB,
    NOTE_VOCAB_SIZES,
    REF,
    REF_END,
    SEP,
    Conditioning,
    MapTokenizer,
    TokenWindow,
    intensity_bucket,
    prefix_len,
    span_tokens,
    version_for_vocab,
)

V = NOTE_VOCAB


def test_v1_ids_are_a_prefix_of_v2():
    assert NOTE_VOCAB_SIZES == {1: 655, 2: 663}
    assert len(V) == 663
    # Every v1 id keeps its name (the hash of the 655 v1 names as note_v0 was trained).
    names = "|".join(V.names(range(655))).encode()
    assert hashlib.sha1(names).hexdigest() == "269ea5736cd23e427721439d13ff040d070f58c3"
    assert V.names(range(655, 663)) == [
        "INTENSITY_0", "INTENSITY_1", "INTENSITY_2", "INTENSITY_3", "INTENSITY_4",
        "INTENSITY_NONE", "REF", "REF_END",
    ]  # fmt: skip
    assert version_for_vocab(655) == 1 and version_for_vocab(663) == 2
    with pytest.raises(ValueError):
        version_for_vocab(600)


def test_intensity_buckets():
    assert [intensity_bucket(x) for x in (0.0, 0.19, 0.2, 0.5, 0.6, 0.89, 0.9, 1.0)] == [
        0, 0, 1, 2, 3, 3, 4, 4,
    ]  # fmt: skip


def test_prefix_versions():
    request = GenerationRequest(Characteristic.STANDARD, Difficulty.EXPERT, intensity=0.95)
    cond = Conditioning.from_request(request)
    v2, v1 = cond.prefix_ids(), cond.prefix_ids(version=1)
    assert len(v1) == prefix_len(1) == 8 and len(v2) == prefix_len(2) == 9
    assert v2[:8] == v1 and V.name(v2[8]) == "INTENSITY_4"
    assert max(v1) < NOTE_VOCAB_SIZES[1]
    assert V.name(cond.drop_intensity().prefix_ids()[8]) == "INTENSITY_NONE"
    back = Conditioning.from_prefix(v2)
    assert back.intensity == pytest.approx(0.95) and back.prefix_ids() == v2
    assert Conditioning.from_prefix(v1).intensity is None


def _map(*beats_cols):
    notes = [Note(b, x, 0, Color(x % 2), CutDirection.DOWN) for b, x in beats_cols]
    return DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT, notes=notes)


def test_span_tokens_crosses_windows():
    dmap = _map((2.0, 0), (14.0, 1), (17.0, 2), (30.0, 3), (33.0, 0))
    windows = MapTokenizer().encode(dmap)
    bodies = [w.body for w in windows[:2]]
    # Span beats 12..28 (ticks 288..672): the notes at 14 and 17, times relative to 12.
    ids = span_tokens(bodies, 12 * 24, NOTE_GRAMMAR)
    events = NOTE_GRAMMAR.parse(ids)
    assert [e.tick for e in events] == [2 * 24, 5 * 24]
    assert [e.values[0] for e in events] == [(1, 0), (2, 0)]
    # A cap cuts at an event boundary.
    capped = span_tokens(bodies, 12 * 24, NOTE_GRAMMAR, max_tokens=5)
    assert len(NOTE_GRAMMAR.parse(capped)) == 1 and len(capped) == 4
    assert span_tokens(bodies, 40 * 24, NOTE_GRAMMAR) == []


def test_window_with_reference_round_trips():
    w = TokenWindow(3, body=[V.id("TIME", 0)], context=[V.id("TIME", 5)], prefix=[10] * 9)
    w.ref = [V.id("TIME", 7)]
    ids = w.input_ids()
    assert ids[:11] == [BOS, *[10] * 9, REF]
    assert ids[11:14] == [V.id("TIME", 7), REF_END, V.id("TIME", 5)]
    assert ids[w.body_offset - 1] == SEP and ids[-1] == EOS
    assert TokenWindow.from_ids(ids, 3, 9) == w
    w.ref = []
    assert TokenWindow.from_ids(w.input_ids(), 3, 9).ref == []
    w.ref = None
    assert REF not in w.input_ids()
