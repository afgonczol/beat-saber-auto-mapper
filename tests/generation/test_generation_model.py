from __future__ import annotations

import math

import numpy as np
import pytest
from gen_synthetic import make_analysis

from bsmapper.generation import ModelGenerator, SamplingConfig, postprocess, sample_token
from bsmapper.generation.timing import tempo_beatmap
from bsmapper.interfaces import GenerationRequest
from bsmapper.rules import RuleChecker
from bsmapper.schema import Characteristic, Color, Difficulty
from bsmapper.tokenizer import EOS, NOTE_VOCAB, WINDOW_BEATS

V = NOTE_VOCAB


class FakeSampler:
    """Random logits shaped like a plausible model: notes likely, small time steps."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.windows: list[float] = []
        base = np.zeros(len(V))
        base[V.ids("TIME")] = -np.arange(len(V.ids("TIME"))) / 6.0
        base[V.ids("NOTE")] += 3.0
        base[V.ids("WALL", "ARC", "CHAIN", "BOMB")] -= 1.0
        base[EOS] = -4.0
        self.base = base

    def encode(self, analysis, start_beat: float, n_beats: int):
        self.windows.append(start_beat)
        return start_beat

    def next_logits(self, memory, input_ids):
        return self.base + self.rng.normal(0.0, 1.0, len(V))


@pytest.fixture(scope="module")
def analysis():
    return make_analysis(duration=24.0)  # 48 beats = 3 windows


def request(**kw) -> GenerationRequest:
    kw.setdefault("allow_obstacles", False)
    kw.setdefault("allow_arcs", False)
    kw.setdefault("allow_chains", False)
    return GenerationRequest(kw.pop("mode", Characteristic.STANDARD), Difficulty.EXPERT, **kw)


def errors(dmap, analysis) -> list:
    issues = RuleChecker().check(dmap, tempo_beatmap(analysis))
    return [i for i in issues if i.severity == "error"]


def test_needs_a_sampler(analysis) -> None:
    with pytest.raises(RuntimeError, match="WS6"):
        ModelGenerator().generate(analysis, request())


def test_decodes_window_by_window_under_rule_masks(analysis) -> None:
    sampler = FakeSampler()
    gen = ModelGenerator(sampler)
    dmap = gen.generate(analysis, request())
    n_windows = math.ceil(len(analysis.beat_times) / WINDOW_BEATS)
    assert sampler.windows == [k * WINDOW_BEATS for k in range(n_windows)]
    assert gen.last_stats.windows == n_windows
    assert gen.last_stats.rule_masked_steps > 0
    assert len(dmap.notes) > 20
    assert not dmap.obstacles and not dmap.arcs and not dmap.chains and not dmap.rotations
    masked_errors = errors(dmap, analysis)

    unmasked = ModelGenerator(FakeSampler(), SamplingConfig(rule_masks=False))
    raw = unmasked.generate(analysis, request())
    assert len(masked_errors) < len(errors(raw, analysis))

    fixed, report = postprocess(dmap, tempo_beatmap(analysis))
    assert report.errors == []
    assert len(fixed.notes) >= 0.8 * len(dmap.notes)


def test_seeded_and_mode_masks(analysis) -> None:
    a = ModelGenerator(FakeSampler(1)).generate(analysis, request(seed=3))
    b = ModelGenerator(FakeSampler(1)).generate(analysis, request(seed=3))
    assert a.notes == b.notes
    one = ModelGenerator(FakeSampler()).generate(
        analysis, request(mode=Characteristic.ONE_SABER, allow_bombs=False)
    )
    assert one.notes and {n.color for n in one.notes} == {Color.BLUE} and not one.bombs


def test_density_control(analysis) -> None:
    sparse = ModelGenerator(FakeSampler(), SamplingConfig(density_gain=6.0))
    dense = ModelGenerator(FakeSampler(), SamplingConfig(density_gain=6.0))
    few = sparse.generate(analysis, request(target_nps=1.0))
    many = dense.generate(analysis, request(target_nps=8.0))
    assert len(few.notes) < len(many.notes)


def test_sample_token() -> None:
    rng = np.random.default_rng(0)
    logits = np.array([0.0, 2.0, 1.0, 5.0])
    mask = np.array([True, True, True, False])
    assert sample_token(logits, mask, rng, temperature=0.0) == 1
    assert {sample_token(logits, mask, rng, top_p=0.5) for _ in range(50)} == {1}
    assert {sample_token(logits, mask, rng, temperature=5.0) for _ in range(200)} == {0, 1, 2}
    with pytest.raises(ValueError):
        sample_token(logits, np.zeros(4, dtype=bool), rng)
