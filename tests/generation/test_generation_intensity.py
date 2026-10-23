from __future__ import annotations

import pytest
from gen_synthetic import make_analysis

from bsmapper.generation import StubGenerator, tempo_beatmap
from bsmapper.generation.density import target_nps
from bsmapper.generation.intensity import (
    HUMAN_PEAK_NPS_P95,
    HUMAN_SWING_INTERVAL_P5,
    IntensityChecker,
    checker_for,
    configured_intensity,
    plan,
    relaxed_rules,
)
from bsmapper.interfaces import GenerationRequest
from bsmapper.rules import RuleChecker, config_for
from bsmapper.schema import Characteristic, Difficulty, DifficultyMap
from bsmapper.tokenizer import notes_per_second

S = Characteristic.STANDARD
DIFFS = sorted(Difficulty, key=lambda d: d.rank)


@pytest.mark.parametrize("diff", DIFFS)
def test_half_intensity_is_the_old_default_and_targets_rise(diff) -> None:
    assert plan(diff, S, 120.0, 0.5).wanted_nps == pytest.approx(
        target_nps(diff, S, 120.0), abs=0.01
    )
    assert plan(diff, S, 120.0, None).wanted_nps == plan(diff, S, 120.0, 0.5).wanted_nps
    wanted = [plan(diff, S, 120.0, i).wanted_nps for i in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert wanted == sorted(wanted) and wanted[0] < wanted[-1]
    assert plan(diff, S, 120.0, 1.0).percentile == pytest.approx(0.9)


def test_shift_above_075() -> None:
    assert plan(Difficulty.EXPERT, S, 120.0, 0.75).condition_as == Difficulty.EXPERT
    shifted = plan(Difficulty.EXPERT, S, 120.0, 0.8)
    assert shifted.condition_as == Difficulty.EXPERT_PLUS and shifted.label == Difficulty.EXPERT
    assert shifted.shifted
    assert plan(Difficulty.EXPERT_PLUS, S, 120.0, 1.0).condition_as == Difficulty.EXPERT_PLUS
    assert (
        plan(Difficulty.HARD, S, 120.0, 1.0, model_intensity=True).condition_as == Difficulty.HARD
    )


def test_rule_ceilings_follow_the_shift_up_to_the_human_p95() -> None:
    base = config_for(Difficulty.HARD, S)
    assert relaxed_rules(base, Difficulty.HARD, 0.5) is base
    up = relaxed_rules(base, Difficulty.HARD, 0.9)
    assert up.density.max_nps == max(base.density.max_nps, HUMAN_PEAK_NPS_P95[Difficulty.EXPERT])
    assert up.swing.min_interval_s == min(
        base.swing.min_interval_s, HUMAN_SWING_INTERVAL_P5[Difficulty.EXPERT]
    )
    top = config_for(Difficulty.EXPERT_PLUS, S)
    assert relaxed_rules(top, Difficulty.EXPERT_PLUS, 1.0) is top
    checker = RuleChecker()
    assert checker_for(checker, 0.6) is checker
    relaxed = checker_for(checker, 0.9)
    assert isinstance(relaxed, IntensityChecker)
    dmap = DifficultyMap(S, Difficulty.HARD)
    assert relaxed.config(dmap).density.max_nps == up.density.max_nps


def test_stub_gets_denser_with_intensity() -> None:
    analysis = make_analysis()
    beatmap = tempo_beatmap(analysis)
    gen = StubGenerator()

    def nps(level):
        req = GenerationRequest(S, Difficulty.HARD, intensity=level)
        return notes_per_second(gen.generate(analysis, req), beatmap)

    assert nps(0.0) < nps(0.5) < nps(1.0)


def test_configured_intensity(tmp_path) -> None:
    cfg = tmp_path / "config.toml"
    assert configured_intensity(cfg) is None
    cfg.write_text("[generate]\nintensity = 0.8\n", encoding="utf-8")
    assert configured_intensity(cfg) == 0.8
    cfg.write_text("[generate]\nintensity = 3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        configured_intensity(cfg)
