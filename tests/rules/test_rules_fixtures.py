"""Known-good fixture maps are clean, the known-bad one is flagged rule by rule."""

from pathlib import Path

import pytest

from bsmapper.formats import read_map_report
from bsmapper.rules import RuleChecker
from bsmapper.rules.config import load_raw

FIXTURES = Path(__file__).parent.parent / "fixtures" / "maps"
GOOD = ["v2", "v2_modded", "v3", "v4"]


@pytest.fixture(scope="module")
def checker():
    return RuleChecker()


@pytest.mark.parametrize("name", GOOD)
def test_good_fixtures_have_no_errors(name):
    # The format fixtures' chains test the readers: their links run off the grid edge.
    checker = RuleChecker(raw={**load_raw(), "default": _chains_off()})
    beatmap, _ = read_map_report(FIXTURES / name)
    for dmap in beatmap.difficulties:
        errors = [i for i in checker.check(dmap, beatmap) if i.severity == "error"]
        assert errors == [], f"{name} {dmap.characteristic}/{dmap.difficulty}: {errors}"


def _chains_off() -> dict:
    default = load_raw()["default"]
    return {**default, "severity": {**default["severity"], "chain_anchor": "off"}}


def _issues(checker, characteristic, difficulty):
    beatmap, _ = read_map_report(FIXTURES / "bad")
    dmap = next(
        d
        for d in beatmap.difficulties
        if d.characteristic == characteristic and d.difficulty == difficulty
    )
    return {(i.rule, i.beat) for i in checker.check(dmap, beatmap)}


def test_bad_fixture_expert_plus(checker):
    found = _issues(checker, "Standard", "ExpertPlus")
    expected = {
        ("parity", 4.5),
        ("same_cell", 10.0),
        ("note_in_wall", 17.0),
        ("crossover", 22.0),
        ("vision_block", 28.25),
        ("swing_interval", 40.0625),
        ("swing_speed", 40.0625),
        ("dodge_gap", 47.1),
        ("wall_duration", 52.0),
        ("overlap", 58.05),
    }
    assert found == expected


def test_bad_fixture_density(checker):
    assert {rule for rule, _ in _issues(checker, "Standard", "Normal")} == {"nps_ceiling"}


def test_bad_fixture_rotation(checker):
    assert _issues(checker, "360Degree", "Expert") == {("rotation_whiplash", 5.0)}


def test_density_reported_once_per_passage(checker):
    found = _issues(checker, "Standard", "Normal")
    assert len(found) == 1
