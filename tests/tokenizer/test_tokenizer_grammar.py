"""The incremental parser and its next-token mask (used by WS7 to mask logits)."""

import pytest

from bsmapper.tokenizer import (
    EOS,
    LIGHT_GRAMMAR,
    LIGHT_VOCAB,
    NOTE_GRAMMAR,
    NOTE_VOCAB,
    SEP,
    LightingTokenizer,
    MapTokenizer,
    TokenGrammarError,
)
from bsmapper.tokenizer.grammar import span_ids

V = NOTE_VOCAB


def ids(*names: str) -> list[int]:
    return V.from_names(names)


def _check_mask_allows_every_token(grammar, tokens, terminator):
    parser = grammar.parser(terminator)
    for token in tokens:
        assert parser.allowed_mask()[token], grammar.vocab.name(token)
        parser.feed(token)
    assert parser.allowed_mask()[terminator]
    parser.feed(terminator)
    assert parser.done and not parser.allowed_mask().any()
    return parser.events


@pytest.mark.parametrize("seed", range(40))
def test_mask_allows_every_encoded_token(seed, random_map, random_lightshow):
    for window in MapTokenizer().encode(random_map(seed)):
        events = _check_mask_allows_every_token(NOTE_GRAMMAR, window.body, EOS)
        assert len(events) == len(NOTE_GRAMMAR.parse(window.body))
        _check_mask_allows_every_token(NOTE_GRAMMAR, window.context, SEP)
    for window in LightingTokenizer().encode(random_lightshow(seed)):
        _check_mask_allows_every_token(LIGHT_GRAMMAR, window.body, EOS)


def test_mask_at_window_start():
    mask = NOTE_GRAMMAR.parser().allowed_mask()
    allowed = {V.group(i) for i in mask.nonzero()[0]}
    assert allowed == {"TIME", "EOS"}  # no object before the first TIME


def test_time_must_advance():
    parser = NOTE_GRAMMAR.parser()
    parser.feed(V.id("TIME", 10))
    mask = parser.allowed_mask()
    assert not mask[V.id("TIME", 10)] and not mask[V.id("TIME", 3)]
    assert mask[V.id("TIME", 11)] and mask[V.id("NOTE", 0)] and mask[EOS]
    with pytest.raises(TokenGrammarError):
        parser.feed(V.id("TIME", 10))


def test_mask_inside_an_object_only_allows_its_slot():
    parser = NOTE_GRAMMAR.parser()
    for token in ids("TIME_0", "WALL", "WX_0", "WW_1", "WH_0_5"):
        parser.feed(token)
    groups = {V.group(i) for i in parser.allowed_mask().nonzero()[0]}
    assert groups == {"DUR", "DURT"}
    parser.feed(V.id("DUR", 3))  # after DUR_3 only the tick remainder can follow
    groups = {V.group(i) for i in parser.allowed_mask().nonzero()[0]}
    assert groups == {"DURT"}
    assert parser.in_object
    with pytest.raises(TokenGrammarError):
        parser.feed(EOS)


def test_optional_modifier_and_partial_state():
    parser = NOTE_GRAMMAR.parser()
    assert parser.feed(V.id("TIME", 5)) == []
    assert parser.feed(V.id("NOTE", 1)) == []
    assert parser.partial == ("NOTE", 1, ())
    parser.feed(V.id("POS", (2, 1)))
    assert parser.feed(V.id("DIR", 0)) == []  # complete, but an ANGLE may still follow
    assert not parser.in_object
    assert parser.allowed_mask()[V.id("ANGLE", 15)]
    (event,) = parser.feed(V.id("ANGLE", 15))
    assert (event.tick, event.kind, event.values) == (5, "NOTE", ((2, 1), 0, 15))
    assert parser.partial is None


def test_pending_object_is_flushed_by_the_next_token():
    parser = NOTE_GRAMMAR.parser()
    for token in ids("TIME_0", "ROT_+30"):
        parser.feed(token)
    (event,) = parser.feed(V.id("TIME", 4))
    assert (event.kind, event.head, event.values) == ("ROT", 30, (None,))
    assert NOTE_GRAMMAR.parse(ids("TIME_0", "ROT_-15", "LATE"))[0].values == (True,)


def test_light_events_with_no_slots_complete_immediately():
    parser = LIGHT_GRAMMAR.parser()
    parser.feed(LIGHT_VOCAB.id("TIME", 0))
    for token in LIGHT_VOCAB.from_names(["EVT_1", "VAL_3"]):
        parser.feed(token)
    flushed = parser.feed(LIGHT_VOCAB.id("BOOST", True))  # flushes EVT, completes BOOST
    assert [e.kind for e in flushed] == ["EVT", "BOOST"]


@pytest.mark.parametrize(
    "names",
    [
        ["RED", "POS_0_0", "DIR_UP"],  # object before TIME
        ["TIME_0", "POS_0_0"],  # slot token without a head
        ["TIME_0", "RED", "DIR_UP"],  # slots out of order
        ["TIME_0", "BOMB", "POS_0_0", "ANGLE_+15"],  # modifier on the wrong object
        ["TIME_0", "RED", "POS_0_0", "DIR_UP", "ANGLE_+15", "ANGLE_+15"],
        ["TIME_0", "WALL", "WX_0", "WW_1", "WH_0_5", "DUR_2", "DUR_1", "DURT_0"],
        ["TIME_0", "EOS", "TIME_1"],
        ["TIME_0", "CHAR_STANDARD"],
    ],
)
def test_invalid_sequences_raise(names):
    with pytest.raises(TokenGrammarError):
        NOTE_GRAMMAR.parse(ids(*names))


def test_truncated_object_raises():
    with pytest.raises(TokenGrammarError):
        NOTE_GRAMMAR.parse(ids("TIME_0", "ARC", "RED"))


@pytest.mark.parametrize("ticks", [0, 1, 23, 24, 25, 383, 384, 24 * 16, 24 * 33 + 5, 10_000])
def test_spans_are_canonical_and_parse_back(ticks):
    tokens = span_ids(V, ticks)
    assert sum(V.group(t) == "DURT" for t in tokens) == 1
    body = [V.id("TIME", 0), *ids("WALL", "WX_0", "WW_1", "WH_0_5"), *tokens]
    (event,) = NOTE_GRAMMAR.parse(body)
    assert event.values[-1] == ticks
