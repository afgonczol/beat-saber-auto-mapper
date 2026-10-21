from __future__ import annotations

import pytest
from gen_synthetic import make_analysis

from bsmapper.generation import STUB_LABEL, StubGenerator, tempo_beatmap
from bsmapper.interfaces import GenerationRequest
from bsmapper.rules import RuleChecker
from bsmapper.schema import Characteristic, Color, CutDirection, Difficulty
from bsmapper.tokenizer import notes_per_second

DIFFS = sorted(Difficulty, key=lambda d: d.rank)
DOWNS = (CutDirection.DOWN, CutDirection.DOWN_LEFT, CutDirection.DOWN_RIGHT)


@pytest.fixture(scope="module")
def analysis():
    return make_analysis()


@pytest.fixture(scope="module")
def generator():
    return StubGenerator()


def gen(generator, analysis, mode=Characteristic.STANDARD, diff=Difficulty.EXPERT, **kw):
    return generator.generate(analysis, GenerationRequest(mode, diff, **kw))


def test_density_rises_with_difficulty_and_passes_rules(generator, analysis) -> None:
    beatmap = tempo_beatmap(analysis)
    nps = []
    for diff in DIFFS:
        dmap = gen(generator, analysis, diff=diff)
        assert dmap.label.startswith(STUB_LABEL) and dmap.label != STUB_LABEL
        assert dmap.notes, diff
        errors = [i for i in RuleChecker().check(dmap, beatmap) if i.severity == "error"]
        assert errors == [], (diff, errors[:3])
        nps.append(notes_per_second(dmap, beatmap))
    assert nps == sorted(nps) and len(set(nps)) == len(nps), nps


def test_no_notes_in_silence_or_past_the_end(generator, analysis) -> None:
    dmap = gen(generator, analysis, diff=Difficulty.EXPERT_PLUS)
    first_s = min(n.beat for n in dmap.notes) * 60 / analysis.bpm
    last_s = max(n.beat for n in dmap.notes) * 60 / analysis.bpm
    assert first_s >= 2.0  # the synthetic song is silent for 2 s
    assert last_s <= analysis.duration_s - 1.0


def test_parity_alternates_per_hand(generator, analysis) -> None:
    dmap = gen(generator, analysis, diff=Difficulty.NORMAL, style=0.0)
    for color in Color:
        dirs = [n.direction for n in dmap.notes if n.color == color]
        downs = [d in DOWNS for d in dirs]
        flips = sum(a != b for a, b in zip(downs, downs[1:], strict=False))
        assert flips >= 0.9 * (len(downs) - 1)


def test_seed_and_style_change_the_map(generator, analysis) -> None:
    a = gen(generator, analysis, seed=1)
    assert a.notes == gen(generator, analysis, seed=1).notes
    assert a.notes != gen(generator, analysis, seed=2).notes
    straight = (CutDirection.UP, CutDirection.DOWN)

    def diagonal_share(style: float) -> float:
        notes = gen(generator, analysis, style=style).notes
        return sum(n.direction not in straight for n in notes) / len(notes)

    assert diagonal_share(1.0) > diagonal_share(0.0) + 0.15


def test_target_nps_is_respected(generator, analysis) -> None:
    beatmap = tempo_beatmap(analysis)
    dmap = gen(generator, analysis, diff=Difficulty.EXPERT_PLUS, target_nps=2.0)
    assert notes_per_second(dmap, beatmap) == pytest.approx(2.0, rel=0.15)


def test_modes(generator, analysis) -> None:
    one = gen(generator, analysis, Characteristic.ONE_SABER)
    assert one.notes and {n.color for n in one.notes} == {Color.BLUE}
    dots = gen(generator, analysis, Characteristic.NO_ARROWS)
    assert {n.direction for n in dots.notes} == {CutDirection.ANY}
    assert gen(generator, analysis, Characteristic.LIGHTSHOW).notes == []
    assert gen(generator, analysis).rotations == []


@pytest.mark.parametrize("mode", [Characteristic.DEGREE_90, Characteristic.DEGREE_360])
def test_rotations(generator, analysis, mode) -> None:
    dmap = gen(generator, analysis, mode, Difficulty.EXPERT)
    assert len(dmap.rotations) >= 5
    total, peak = 0, 0
    for r in dmap.rotations:
        assert r.degrees % 15 == 0 and r.degrees != 0
        total += r.degrees
        peak = max(peak, abs(total))
    if mode == Characteristic.DEGREE_90:
        assert peak <= 45
    issues = RuleChecker().check(dmap, tempo_beatmap(analysis))
    assert [i for i in issues if i.severity == "error"] == []
