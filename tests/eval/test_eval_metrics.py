from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from synth import BPM, click_features, human_map, lightshow

from bsmapper.eval import repetition as rep
from bsmapper.eval.lighting import lighting_metrics, near, onset_peaks
from bsmapper.eval.metrics import (
    difficulty_metrics,
    distributions,
    jensen_shannon,
    match_onsets,
    onset_scores,
    percentile_of,
    swing_groups,
)
from bsmapper.eval.reference import HumanReference
from bsmapper.interfaces import AudioAnalysis, Section
from bsmapper.rules import RuleChecker
from bsmapper.schema import (
    Beatmap,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
)

E = Difficulty.EXPERT
S = Characteristic.STANDARD


def shifted(dmap: DifficultyMap, ticks: int) -> DifficultyMap:
    return replace(dmap, notes=[replace(n, beat=n.beat + ticks / 24) for n in dmap.notes])


def test_onset_f1_counts_matches_within_the_tolerance() -> None:
    ref = human_map(E)
    assert onset_scores(ref, ref)["onset_f1_0"] == 1.0
    one = onset_scores(ref, shifted(ref, 1))
    assert one["onset_f1_0"] == 0.0 and one["onset_f1_1"] == 1.0
    three = onset_scores(ref, shifted(ref, 3))
    assert three["onset_f1_2"] == 0.0
    half = replace(ref, notes=ref.notes[::2])  # every other onset: precision 1, recall 0.5
    s = onset_scores(ref, half)
    assert s["onset_p_0"] == 1.0 and s["onset_r_0"] == pytest.approx(0.5, abs=0.01)
    assert onset_scores(ref, DifficultyMap(S, E))["onset_f1_2"] == 0.0


def test_onset_matching_is_one_to_one() -> None:
    ref = np.array([0, 2, 4])
    assert match_onsets(ref, np.array([1]), 1) == 1  # one estimate, two refs in reach
    assert match_onsets(ref, np.array([1, 1, 1, 3]), 1) == 3  # 1-0, 1-2, 3-4
    assert match_onsets(ref, np.array([1, 1, 1, 1]), 1) == 2  # each ref used once


def test_jensen_shannon_bounds() -> None:
    p = Counter({"a": 3, "b": 1})
    assert jensen_shannon(p, p) == pytest.approx(0.0)
    assert jensen_shannon(Counter({"a": 1}), Counter({"b": 1})) == pytest.approx(1.0)
    assert 0 < jensen_shannon(p, Counter({"a": 1, "b": 1})) < 1
    assert math.isnan(jensen_shannon(Counter(), p))


def test_distributions_and_swings() -> None:
    dmap = human_map(E)
    d = distributions(dmap)
    assert sum(d["direction"].values()) == len(dmap.notes)
    assert set(d["direction"]) == {str(int(CutDirection.DOWN)), str(int(CutDirection.UP))}
    assert d["rhythm"] == Counter({"12": len(dmap.notes) - 1})  # half-beat gaps
    assert sum(d["pattern_3gram"].values()) == sum(d["pattern"].values()) - 4  # 2 per hand
    stack = [Note(1.0, 1, 0, Color.RED, CutDirection.DOWN), Note(1.05, 1, 1, Color.RED, 1)]
    assert len(swing_groups([*stack, Note(2.0, 1, 0, Color.RED, 0)], Color.RED)) == 2


def test_difficulty_metrics_on_a_clean_map() -> None:
    dmap = human_map(Difficulty.HARD)
    m = difficulty_metrics(dmap, Beatmap("", BPM), RuleChecker())
    assert m["nps"] == pytest.approx(2.0, rel=0.02)
    assert m["red_share"] == pytest.approx(0.5, abs=0.02)
    assert m["errors_per_100"] == 0.0
    assert m["crossover_rate"] == 0.0
    assert math.isnan(m["tech_score"])  # no calibration given


def test_percentile_and_reference_ladder() -> None:
    assert percentile_of(5.0, [1, 2, 3, 4, 5, 6, 7, 8, 9]) == pytest.approx(50.0, abs=6)
    assert math.isnan(percentile_of(1.0, []))
    rows = []
    for level, diff in enumerate(Difficulty):
        for i in range(30):
            nps = level + 1.0 + (i - 14.5) / 50
            rows.append(
                {
                    "split": "train",
                    "characteristic": "Standard",
                    "difficulty": diff.value,
                    "bpm": 110.0 + i,
                    "nps": nps,
                    "tech_score": float(i),
                }  # fmt: skip
            )
    rows.append({**rows[0], "split": "test", "nps": 99.0})  # never part of the reference
    ref = HumanReference(rows)
    assert ref.nps_percentile(S, E, 120.0, 4.0) == pytest.approx(50.0, abs=5)
    assert ref.equivalent_level(S, 120.0, 4.0) == pytest.approx(3.0, abs=0.05)
    assert ref.step_offset(S, E, 120.0, 3.0) == pytest.approx(-1.0, abs=0.05)
    assert ref.step_offset(S, E, 120.0, 4.5) == pytest.approx(0.5, abs=0.05)
    assert ref.equivalent_level(S, 120.0, 7.0) == pytest.approx(6.0, abs=0.05)  # past Expert+
    # Rare modes fall back to Standard's maps.
    assert ref.nps_percentile(Characteristic.DEGREE_360, E, 120.0, 4.0) == pytest.approx(50, abs=5)
    assert ref.style_percentile(E, 29.5) == 100.0


def test_repeated_audio_pairs_and_classes() -> None:
    feats = click_features()
    pairs = rep.repeated_pairs(feats)
    assert pairs and all((b - a) % 32 == 0 for a, b in pairs)  # A with A, B with B
    human = human_map(E)
    counts = rep.repetition_counts(human, pairs)
    assert counts["exact"] == len(pairs)

    def map_of(notes):
        return DifficultyMap(S, E, notes=notes)

    a = [Note(0.0, 1, 0, Color.RED, CutDirection.LEFT), Note(1.0, 2, 0, Color.BLUE, 1)]
    later = 8.0
    exact = map_of([*a, *(replace(n, beat=n.beat + later) for n in a)])
    assert rep.classify(exact, 0, 8) == "exact"
    mirror = [
        Note(later, 2, 0, Color.BLUE, CutDirection.RIGHT),
        Note(later + 1, 1, 0, Color.RED, CutDirection.DOWN),
    ]
    assert rep.classify(map_of([*a, *mirror]), 0, 8) == "mirrored"
    rhythm = [replace(n, x=0, y=2) for n in mirror]
    assert rep.classify(map_of([*a, *rhythm]), 0, 8) == "same_rhythm"
    other = [Note(later + 0.5, 0, 0, Color.RED, CutDirection.UP)]
    assert rep.classify(map_of([*a, *other]), 0, 8) == "different"
    assert rep.classify(map_of(a), 16, 24) is None  # no notes in either segment
    sh = rep.shares(Counter({"exact": 1, "different": 1}))
    assert sh["exact"] == 0.5 and rep.share_distance(sh, sh) == 0.0


def test_unrelated_audio_has_no_repeats() -> None:
    rng = np.random.default_rng(1)
    feats = click_features()
    feats[:, :80] = rng.uniform(-60, -10, (feats.shape[0] // 24, 80)).repeat(24, axis=0)
    assert rep.repeated_pairs(feats) == []


def _analysis() -> AudioAnalysis:
    beats = np.arange(64) * 60 / BPM
    sections = [Section(0.0, 8.0, "verse", 0.4), Section(8.0, 16.0, "chorus", 1.0)]
    sections += [Section(16.0, 24.0, "verse", 0.4), Section(24.0, 32.0, "chorus", 1.0)]
    return AudioAnalysis(32.0, BPM, beats, beats[::4], sections)


def test_lighting_metrics_reward_lights_on_onsets() -> None:
    feats = click_features()
    peaks = onset_peaks(feats[:, 80])
    assert set(peaks % 24) <= {1, 13}
    assert near(np.array([0, 5]), peaks, 2).tolist() == [True, False]
    on = lighting_metrics(lightshow(), feats, _analysis(), Beatmap("", BPM))
    assert on is not None
    assert on["on_beat"] == 1.0 and on["onset_precision"] == 1.0 and on["onset_lift"] > 1.5
    assert on["events_per_bar"] == pytest.approx(4.0)
    assert on["section_response"] == 1.0  # colour change and ring spin every 16 beats
    off = lightshow()
    off.basic_events = [replace(e, beat=e.beat + 0.25) for e in off.basic_events]
    late = lighting_metrics(off, feats, _analysis(), Beatmap("", BPM))
    assert late["onset_precision"] == 0.0 and late["on_beat"] == 0.0
    assert lighting_metrics(None, feats, _analysis(), Beatmap("", BPM)) is None
