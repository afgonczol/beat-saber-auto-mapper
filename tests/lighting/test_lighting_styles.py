"""WS8b phase 2: the v2 light prompt, lighting styles and section scene plans."""

from __future__ import annotations

import numpy as np
import pytest

from bsmapper.lighting.scenes import DARK_SCENE, SectionSpan, plan_scenes, window_sections
from bsmapper.lighting.styles import (
    STYLE_FEATURE_NAMES,
    STYLE_PROTOTYPES,
    StyleModel,
    fit_softmax,
    is_sequenced,
    kmeans,
    loudness_bin,
    map_style_features,
    name_clusters,
    song_vector,
    window_loudness,
)
from bsmapper.schema import BasicEvent, ColorBoostEvent, Lightshow
from bsmapper.tokenizer import (
    LIGHT_REF,
    LIGHT_VOCAB,
    LIGHT_VOCAB_SIZES,
    WINDOW_TICKS,
    LightingTokenizer,
    LightPrompt,
    light_input_ids,
    light_version_for_vocab,
    window_scene,
)
from bsmapper.tokenizer.vocab import LIGHT_PALETTES

PAL = {name: i for i, name in enumerate(LIGHT_PALETTES)}
BLUE_ON, RED_ON, WHITE_ON = 1, 5, 9


def _body(events: list[BasicEvent]) -> list[int]:
    (window,) = LightingTokenizer(context_events=0).encode(Lightshow(events), n_windows=1)
    return window.body


# --- tokenizer --------------------------------------------------------------------------


def test_light_vocab_v2_appends_to_v1() -> None:
    assert {1: 500, 2: len(LIGHT_VOCAB)} == LIGHT_VOCAB_SIZES
    assert light_version_for_vocab(500) == 1 and light_version_for_vocab(len(LIGHT_VOCAB)) == 2
    with pytest.raises(ValueError):
        light_version_for_vocab(501)
    assert min(LIGHT_VOCAB.ids("LSTYLE", "LEVEL", "PAL", "FLIPS", "REF", "REF_END")) == 500


def test_light_prompt_round_trips() -> None:
    prompt = LightPrompt(style=3, level=4, palette=PAL["RED_LEAD"], flips=1)
    ids = prompt.prefix_ids()
    assert LIGHT_VOCAB.names(ids) == ["LSTYLE_3", "LEVEL_4", "PAL_RED_LEAD", "FLIPS_1"]
    assert LightPrompt.from_prefix(ids) == prompt
    assert LIGHT_VOCAB.names(LightPrompt().prefix_ids()) == [
        "LSTYLE_NONE", "LEVEL_NONE", "PAL_NONE", "FLIPS_NONE"
    ]  # fmt: skip
    assert prompt.drop_style().style is None and prompt.drop_scene().scene == (None, None, None)
    with pytest.raises(ValueError):
        LightPrompt(style=8).prefix_ids()
    ids = light_input_ids(prompt.prefix_ids(), [7], [], [9])
    assert ids[5] == LIGHT_REF and ids[7] == LIGHT_REF + 1  # the light vocabulary's REF


def test_window_scene_counts_level_palette_and_flips() -> None:
    assert window_scene([]) == (0, PAL["DARK"], 0)
    # 32 blue events on two lights, one per half beat: level 2/beat -> bucket 2, no flips.
    blue = [BasicEvent(i / 2, i % 2, BLUE_ON) for i in range(32)]
    assert window_scene(_body(blue)) == (2, PAL["BLUE"], 0)
    # Alternating red and blue on one light: every lit event after the first flips.
    flip = [BasicEvent(i / 2, 0, RED_ON if i % 2 else BLUE_ON) for i in range(32)]
    level, palette, flips = window_scene(_body(flip))
    assert (palette, flips) == (PAL["EVEN"], 3)
    # Mostly white: a white palette; off events count for the level but not the colours.
    white = [BasicEvent(i, 1, WHITE_ON if i % 3 else 0) for i in range(16)]
    assert window_scene(_body(white))[1] == PAL["WHITE"]
    # Ring spins are not lights.
    assert window_scene(_body([BasicEvent(i, 8, 0) for i in range(16)])) == (0, PAL["DARK"], 0)


# --- styles -----------------------------------------------------------------------------


def _show(n_beats: int, step: float, colours=(BLUE_ON, RED_ON)) -> Lightshow:
    events = []
    for i in range(int(n_beats / step)):
        events.append(BasicEvent(i * step, i % 5, colours[i % len(colours)] + 2))  # fades
    return Lightshow(events, [ColorBoostEvent(16.0, True), ColorBoostEvent(32.0, False)])


def _feats(n_windows: int, loud_from: int = 0) -> np.ndarray:
    feats = np.zeros((n_windows * WINDOW_TICKS, 83), dtype=np.float32)
    feats[:, :80] = -40.0
    feats[loud_from * WINDOW_TICKS :, 82] = 0.5
    feats[: loud_from * WINDOW_TICKS, 82] = 0.1
    feats[::24, 80] = 1.0  # an onset on every beat
    return feats


def test_map_style_features_describe_a_show() -> None:
    feats = _feats(8, loud_from=2)
    f = map_style_features(_show(128, 0.5), feats, 120.0)
    assert f is not None
    assert f["rate"] > 1 and f["fade"] == pytest.approx(1.0) and f["stacked"] == 0.0
    assert f["on_beat"] == pytest.approx(0.5)  # every other change is on a beat
    assert f["boost_time"] == pytest.approx(16 / 128)
    assert not is_sequenced(f)
    assert map_style_features(Lightshow([BasicEvent(0.0, 0, 1)]), feats, 120.0) is None
    # Many events of one light on the same ticks: a lightID sequence.
    stacked = Lightshow(
        [BasicEvent(i / 2, 0, 1 + 4 * (i % 2)) for i in range(256) for _ in range(3)]
    )
    g = map_style_features(stacked, feats, 120.0)
    assert g is not None and g["stacked"] == pytest.approx(1.0) and is_sequenced(g)


def test_loudness_and_song_vector() -> None:
    feats = _feats(4, loud_from=2)
    loud = window_loudness(feats, 4)
    assert loud.tolist() == pytest.approx([0.2, 0.2, 1.0, 1.0])
    assert [loudness_bin(x) for x in (0.0, 0.2, 0.5, 0.95)] == [0, 1, 2, 4]
    v = song_vector(feats, 120.0)
    assert v.shape == (18,) and np.isfinite(v).all()


def test_kmeans_softmax_and_names() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, (40, 2))
    b = rng.normal(3, 0.1, (40, 2))
    _, labels = kmeans(np.vstack([a, b]), 2, n_init=3)
    assert len(set(labels[:40])) == 1 and len(set(labels[40:])) == 1 and labels[0] != labels[40]
    x = np.vstack([a, b])
    y = np.array([0] * 40 + [1] * 40)
    w, c = fit_softmax((x - x.mean(0)) / x.std(0), y, 2, steps=300)
    assert ((((x - x.mean(0)) / x.std(0)) @ w + c).argmax(1) == y).all()
    index = {n: i for i, n in enumerate(STYLE_FEATURE_NAMES)}
    centers = np.zeros((6, len(STYLE_FEATURE_NAMES)))
    for i, name in enumerate(["flash", "calm", "pulse", "accent", "scenes"]):
        for feat, weight in STYLE_PROTOTYPES[name].items():
            centers[i, index[feat]] = weight
    assert name_clusters(centers) == ["flash", "calm", "pulse", "accent", "scenes", "style5"]


def _model() -> StyleModel:
    n = len(STYLE_FEATURE_NAMES)
    scenes = {
        "*": {str(b): [[1, PAL["EVEN"], 1, 0.5], [0, PAL["DARK"], 0, 0.5]] for b in range(5)},
        "calm": {
            "4": [[1, PAL["EVEN"], 1, 0.6], [3, PAL["RED"], 0, 0.4]],
            "1": [[1, PAL["BLUE"], 0, 1.0]],
        },
    }
    return StyleModel(
        names=["pulse", "calm"],
        mean=[0.0] * n,
        scale=[1.0] * n,
        centers=[[0.0] * n, [1.0] * n],
        song_mean=[0.0] * 18,
        song_scale=[1.0] * 18,
        coef=[[0.0, 0.0]] * 18,
        intercept=[0.0, 5.0],
        scenes=scenes,
    )


def test_style_model_round_trips_and_picks(tmp_path) -> None:
    m = _model()
    m.save(tmp_path / "s.json")
    back = StyleModel.load(tmp_path / "s.json")
    assert back == m and back.index("calm") == 1
    with pytest.raises(ValueError, match="unknown light style"):
        back.index("disco")
    probs = back.song_probs(np.zeros(18))
    assert probs[1] > 0.99
    assert {back.pick(np.zeros(18), seed=s) for s in range(5)} == {1}
    assert back.scene_table(1, 1) == [((1, PAL["BLUE"], 0), 1.0)]
    assert back.scene_table(1, 2) == back.scene_table(None, 2)  # falls back to all styles


# --- scene plans ------------------------------------------------------------------------


def test_scene_plan_holds_per_section_and_repeats() -> None:
    m = _model()
    loud = [0.01, 0.95, 0.95, 0.3, 0.95, 0.95]
    sections = [
        SectionSpan(0, 16), SectionSpan(16, 48, red_lead=False), SectionSpan(48, 64),
        SectionSpan(64, 96, red_lead=True),
    ]  # fmt: skip
    assert window_sections(sections, 6) == [0, 1, 1, 2, 3, 3]
    refs = [-1, -1, -1, -1, 16 * 24, 32 * 24]  # windows 4-5 repeat windows 1-2
    for seed in range(10):
        plan = plan_scenes(m, 1, loud, sections, refs, seed=seed)
        assert plan[0] == DARK_SCENE  # silence
        assert plan[1] == plan[2] and plan[4] == plan[5]  # one scene per section
        assert plan[4] == plan[1]  # the repeated section reuses its scene
        assert plan[1][0] >= 2 and plan[1][1] != PAL["DARK"]  # loud: never near-dark
        assert plan[1][1] in (PAL["BLUE"], PAL["EVEN"])  # red palettes follow the blue lead
    without_refs = {tuple(plan_scenes(m, 1, loud, sections, None, seed=s)[4]) for s in range(10)}
    assert without_refs == {(3, PAL["RED"], 0)}  # section 3 has a red lead


def test_scene_plan_without_a_scene_gives_no_preference() -> None:
    m = _model()
    m.scenes = {}
    assert plan_scenes(m, 0, [0.9, 0.0], [SectionSpan(0, 32)]) == [None, DARK_SCENE]
