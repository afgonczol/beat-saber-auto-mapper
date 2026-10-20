from datetime import UTC, datetime

import pytest

from bsmapper.data.filters import FilterConfig, MapRecord, check, gold_reasons


def rec(doc, **over) -> MapRecord:
    r = MapRecord.from_api(doc("abc", **over))
    assert r is not None
    return r


def test_parse_basic(doc):
    r = rec(doc)
    assert r.key == "abc"
    assert r.votes == 210
    assert r.song_name == "Song abc"
    assert r.diffs[0].characteristic == "Standard"
    assert r.diffs[0].parity_resets == 2
    assert r.download_url.endswith(".zip")
    assert check(r, FilterConfig()) is None


def test_picks_latest_published_version(doc):
    d = doc("abc")
    old = dict(d["versions"][0], hash="old", createdAt="2022-01-01T00:00:00Z")
    draft = dict(d["versions"][0], hash="draft", state="Unpublished", createdAt="2024-01-01Z")
    d["versions"] = [old, d["versions"][0], draft]
    r = MapRecord.from_api(d)
    assert r is not None and r.hash == d["versions"][1]["hash"]


def test_no_published_version(doc):
    d = doc("abc")
    d["versions"][0]["state"] = "Unpublished"
    assert MapRecord.from_api(d) is None


def test_dict_round_trip(doc):
    r = rec(doc, ranked=True)
    assert MapRecord.from_dict(r.to_dict()) == r


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"automapper": True}, "ai"),
        ({"declaredAi": "Uploader"}, "ai"),
        ({"declaredAi": "SageScore"}, "ai"),
        ({"score": 0.7}, "score"),
        ({"upvotes": 5, "downvotes": 1}, "votes"),
        ({"uploaded": "2020-06-01T00:00:00Z"}, "too_old"),
        ({"duration": 10}, "duration"),
        ({"duration": 1200}, "duration"),
        ({"diff": {"ne": True}}, "mods"),
        ({"diff": {"me": True}}, "mods"),
    ],
)
def test_rejections(doc, over, reason):
    assert check(rec(doc, **over), FilterConfig()) == reason


def test_filters_are_configurable(doc):
    cfg = FilterConfig(
        min_score=0.5,
        min_votes=1,
        uploaded_after=datetime(2019, 1, 1, tzinfo=UTC),
        exclude_ai=False,
        exclude_noodle=False,
    )
    for over in ({"automapper": True}, {"score": 0.6}, {"uploaded": "2020-01-01Z"}):
        assert check(rec(doc, **over), cfg) is None
    assert check(rec(doc, diff={"ne": True}), cfg) is None


def test_search_params_leave_automapper_unset():
    # null = "no AI" on BeatSaver; false would mean "only AI".
    params = FilterConfig().search_params()
    assert "automapper" not in params
    assert params["order"] == "Rating"
    assert params["from"] == "2021-01-01T00:00:00Z"


def test_gold_reasons(doc):
    assert gold_reasons(rec(doc)) == []
    assert gold_reasons(rec(doc, ranked=True)) == ["scoresaber_ranked"]
    assert gold_reasons(rec(doc, blRanked=True)) == ["beatleader_ranked"]
    assert gold_reasons(rec(doc, curatedAt="2024-01-01T00:00:00Z")) == ["beatsaver_curated"]
    assert gold_reasons(rec(doc), frozenset({"abc"})) == ["curated_list"]
