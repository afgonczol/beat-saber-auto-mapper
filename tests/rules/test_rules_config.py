import pytest

from bsmapper.rules import RULES, RuleChecker, config_for
from bsmapper.rules.config import DEFAULT_THRESHOLDS, load_raw
from bsmapper.schema import (
    Beatmap,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
)


def test_yaml_has_every_rule_and_difficulty():
    raw = load_raw()
    assert set(raw["default"]["severity"]) == set(RULES)
    assert set(raw["difficulties"]) == {d.value for d in Difficulty}
    assert DEFAULT_THRESHOLDS.exists()


def test_layers_merge():
    easy = config_for(Difficulty.EASY)
    expert_plus = config_for(Difficulty.EXPERT_PLUS)
    assert easy.density.max_nps < expert_plus.density.max_nps
    assert easy.swing.min_interval_s > expert_plus.swing.min_interval_s
    assert easy.parity == expert_plus.parity  # untouched keys come from default
    assert config_for(Difficulty.HARD, Characteristic.DEGREE_90).rotation.max_abs_deg == 45
    assert config_for(Difficulty.HARD, Characteristic.DEGREE_360).rotation.max_abs_deg is None


def test_unknown_keys_are_rejected():
    raw = load_raw()
    raw["default"]["parity"]["max_wrist"] = 90
    with pytest.raises(ValueError, match="max_wrist"):
        config_for(Difficulty.HARD, raw=raw)
    raw = load_raw()
    raw["default"]["severity"]["parrity"] = "error"
    with pytest.raises(ValueError, match="parrity"):
        config_for(Difficulty.HARD, raw=raw)


def test_custom_yaml_file(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "default:\n  severity: {parity: 'off'}\n"
        "difficulties:\n  Hard:\n    density: {max_nps: 3}\n",
        encoding="utf-8",
    )
    cfg = config_for(Difficulty.HARD, path=path)
    assert cfg.density.max_nps == 3
    assert not cfg.enabled("parity")


def test_severity_off_disables_rule():
    raw = load_raw()
    raw["default"]["severity"]["parity"] = "off"
    notes = [
        Note(0, 1, 0, Color.RED, CutDirection.DOWN),
        Note(0.5, 0, 1, Color.RED, CutDirection.DOWN),
    ]
    dmap = DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT, notes=notes)
    assert RuleChecker(raw=raw).check(dmap, Beatmap("t", 120.0)) == []
    assert [i.rule for i in RuleChecker().check(dmap, Beatmap("t", 120.0))] == ["parity"]
