"""Human group-lighting effects (WS8c): library, planning, splicing, environment choice."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from synthetic import make_analysis

from bsmapper.formats import write_map
from bsmapper.lighting import (
    EffectLibrary,
    EffectRenderer,
    RuleLightingGenerator,
    environment_for,
    light_beatmap,
)
from bsmapper.lighting.demo import demo_beatmap
from bsmapper.lighting.effects import (
    BAR_BEATS,
    CARRY_GLIDE,
    EFFECTS_VERSION,
    INDEX_FILE,
    Placement,
    SectionInfo,
    _state_box,
    phrases,
    plan_effects,
    render_effects,
    song_targets,
)
from bsmapper.lighting.effects_build import describe, extract
from bsmapper.lighting.events import Action, LightColor, light_value
from bsmapper.schema import BasicEvent, Lightshow

ALL = {"c": 0, "f": 1, "p": 1, "t": 0, "r": 0, "n": 0, "s": 0, "l": 0.0, "d": 0}
HALF = {**ALL, "p": 2}  # first half of the lights only


def colour_box(c: int, s: float = 1.0, i: int = 0, f: dict | None = None) -> dict:
    event = {"b": 0.0, "i": i, "c": c, "s": s, "f": 0, "sb": 0.0, "sf": 0}
    return {"f": dict(f or ALL), "w": 0.0, "d": 1, "r": 0.0, "t": 1, "b": 0, "i": 0, "e": [event]}


def rotation_box(angle: float, axis: int = 0) -> dict:
    events = [{"b": 0.0, "p": 1, "e": 0, "r": 0.0, "o": 0, "l": 0},
              {"b": 2.0, "p": 0, "e": 3, "r": angle, "o": 0, "l": 1}]  # fmt: skip
    box = {"f": dict(ALL), "w": 0.0, "d": 1, "s": 0.0, "t": 1, "b": 0, "i": 0}
    return {**box, "a": axis, "r": 0, "l": events}


def busy_source(bars: int = 16) -> dict:
    """Colour hits on every beat in groups 0/1 (red / blue), a rotation every bar."""
    groups = []
    for beat in range(int(bars * BAR_BEATS)):
        groups.append([float(beat), "color", beat % 2, [colour_box(beat % 2)]])
        if beat % 4 == 0:
            groups.append([float(beat), "rotation", 2, [rotation_box(15.0)]])
    return {"bpm": 120.0, "bars": bars, "groups": groups, "basic": [], "boost": [[0.0, False]]}


def quiet_source(bars: int = 16) -> dict:
    """One white hit per bar in group 3."""
    groups = [[b * BAR_BEATS, "color", 3, [colour_box(2)]] for b in range(bars)]
    return {"bpm": 120.0, "bars": bars, "groups": groups, "basic": [], "boost": []}


def make_library(root: Path, env: str, sources: dict[str, dict]) -> Path:
    folder = root / env
    folder.mkdir(parents=True)
    index = []
    for key, src in sources.items():
        with gzip.open(folder / f"{key}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(src, f)
        desc = describe(src["groups"], src["basic"], src["bars"])
        index.append({"key": key, "mapper": f"m-{key}", "song": key, "bars": src["bars"], **desc})
    doc = {"version": EFFECTS_VERSION, "environment": env, "maps": index}
    (folder / INDEX_FILE).write_text(json.dumps(doc), encoding="utf-8")
    return folder


@pytest.fixture
def library(tmp_path: Path) -> EffectLibrary:
    root = tmp_path / "fx"
    make_library(root, "LizzoEnvironment", {"busy": busy_source(), "quiet": quiet_source()})
    return EffectLibrary(root / "LizzoEnvironment")


def test_describe_counts_activity_colours_motion_and_rhythm():
    d = describe(
        busy_source(2)["groups"], [[0.5, 0, light_value(LightColor.WHITE, Action.ON), 1.0]], 2
    )
    assert d["act"] == [5, 4]  # 4 colour box groups per bar, plus one basic light event
    assert d["motion"] == [1, 1]
    assert d["red"] == [2, 2] and d["blue"] == [2, 2] and d["white"] == [1, 0]
    assert d["rhythm"][0][0] == 1 and d["rhythm"][0][2] == 1 and d["rhythm"][0][4] == 1


def test_library_ranks_activity_within_the_environment(library: EffectLibrary):
    busy, quiet = library.maps
    assert busy.act.mean() > quiet.act.mean()
    assert quiet.white.max() == 1.0 and busy.white.max() == 0.0
    assert np.allclose(np.linalg.norm(busy.rhythm, axis=1), 1.0)


def test_phrases_cut_sections_into_eight_bar_phrases():
    sections = [SectionInfo(0, 40, 0.3), SectionInfo(40, 124, 0.9)]  # 10 bars, 21 bars
    out = phrases(sections, 31)
    assert [(p.section, p.start_bar, p.bars) for p in out] == [
        (0, 0, 8), (0, 8, 2), (1, 10, 8), (1, 18, 8), (1, 26, 5),
    ]  # fmt: skip
    one_bar_left = phrases([SectionInfo(0, 36, 0.5)], 9)  # 9 bars: the last joins the phrase
    assert [(p.start_bar, p.bars) for p in one_bar_left] == [(0, 9)]


def test_song_targets_blend_busy_bars_with_section_intensity():
    events = [BasicEvent(b * 0.5, 0, 1, 1.0) for b in range(8)]  # 8 hits in bar 0, then none
    act, rhythm = song_targets(Lightshow(basic_events=events), [SectionInfo(0, 16, 0.8)], 4)
    assert act[0] > act[1] and act[1] == act[2]
    assert np.isclose(np.linalg.norm(rhythm[0]), 1.0) and not rhythm[1].any()
    assert rhythm[0][0] > 0 and rhythm[0][2] > 0 and rhythm[0][1] == 0


def test_plan_matches_activity(library: EffectLibrary):
    sections = [SectionInfo(0, 32, 0.0), SectionInfo(32, 64, 1.0)]
    ph = phrases(sections, 16)
    low, high = np.full(16, 0.0), np.full(16, 1.0)
    act = np.concatenate([low[:8], high[8:]])
    plan = plan_effects(library, act, np.zeros((16, 16)), ph, sections, seed=0)
    names = [library.maps[p.source].key for p in plan]
    assert names == ["quiet", "busy"]


def test_repeated_section_replays_its_passages(library: EffectLibrary):
    sections = [SectionInfo(0, 32, 0.9), SectionInfo(32, 48, 0.1), SectionInfo(48, 80, 0.9)]
    ph = phrases(sections, 20)
    act = np.full(20, 0.5)
    plan = plan_effects(library, act, np.zeros((20, 16)), ph, sections, repeat_of=[None, None, 0])
    first, last = plan[0], plan[-1]
    assert (last.source, last.src_bar, last.bars) == (first.source, first.src_bar, first.bars)


def test_plan_is_seeded(library: EffectLibrary):
    sections = [SectionInfo(0, 64, 0.5)]
    ph = phrases(sections, 16)
    act, rhythm = np.full(16, 0.5), np.zeros((16, 16))
    a = plan_effects(library, act, rhythm, ph, sections, seed=3)
    b = plan_effects(library, act, rhythm, ph, sections, seed=3)
    assert a == b


def test_state_box_keeps_the_last_state():
    box = colour_box(1)
    box["e"] += [{"b": 1.0, "i": 1, "c": 0, "s": 0.5, "f": 0, "sb": 0.0, "sf": 0},
                 {"b": 2.0, "i": 2, "c": 0, "s": 0.0, "f": 0, "sb": 0.0, "sf": 0}]  # fmt: skip
    state = _state_box("color", box)
    assert state["e"] == [{"b": 0.0, "i": 0, "c": 0, "s": 0.5, "f": 0, "sb": 0.0, "sf": 0}]
    rot = _state_box("rotation", rotation_box(30.0))
    assert [(e["b"], e["p"], e["r"], e["l"]) for e in rot["l"]] == [
        (0.0, 1, 30.0, 0), (CARRY_GLIDE, 0, 30.0, 0),
    ]  # fmt: skip
    only_extend = colour_box(0, i=2)
    assert _state_box("color", only_extend) is None


def test_render_offsets_passages_and_rebuilds_their_state(library: EffectLibrary):
    busy, quiet = 0, 1
    placements = [Placement(0, 2, quiet, 0), Placement(2, 2, busy, 4, swap=True)]
    ls = render_effects(library, placements, 16.0)
    by_beat = {}
    for g in ls.event_box_groups:
        by_beat.setdefault(g.beat, []).append(g)
    # The busy passage starts at song beat 8 (its own beat 16): its state at the cut
    # (group 0 last hit red at beat 14, group 1 blue at 15, swapped) comes first.
    at8 = {(g.kind, g.group_id): g for g in by_beat[8.0]}
    assert at8[("color", 0)].boxes[0]["e"][0]["c"] == LightColor.BLUE  # red, swapped
    assert at8[("color", 1)].boxes[0]["e"][0]["c"] == LightColor.RED
    assert at8[("rotation", 2)].boxes[0]["l"][-1]["r"] == 15.0
    # The quiet source's white group 3 is not set by the busy source: it is turned off.
    off = at8[("color", 3)].boxes[0]["e"][0]
    assert off["s"] == 0.0 and off["i"] == 0
    assert {g.beat for g in ls.event_box_groups if g.kind == "color"} >= {8.0, 9.0, 15.0}
    assert max(g.beat for g in ls.event_box_groups) == 16.0  # everything off at the end
    assert [(e.beat, e.on) for e in ls.boost_events] == [(8.0, False)]


def test_state_stops_at_a_box_that_sets_every_light():
    from bsmapper.lighting.effects import _Source

    groups = [[0.0, "color", 0, [colour_box(0)]],
              [1.0, "color", 0, [colour_box(1)]],  # every light: earlier ones do not matter
              [2.0, "color", 0, [colour_box(2, f=HALF)]]]  # fmt: skip
    state = _Source({"groups": groups, "basic": [], "boost": []}).light_state(3.0)
    assert [b["e"][0]["c"] for b in state[("color", 0)]] == [1, 2]


def test_environment_is_picked_per_song(tmp_path: Path):
    # A 20-song web UI batch (all seed 0) came out all TheWeekndEnvironment.
    root = tmp_path / "fx"
    envs = ["LizzoEnvironment", "PyroEnvironment", "TheWeekndEnvironment", "RockMixtapeEnvironment"]
    for env in envs:
        make_library(root, env, {"busy": busy_source()})
    effects = EffectRenderer(root, seed=0)
    songs = [f"Song {i} - Artist" for i in range(20)]
    picks = [effects.pick_environment(song) for song in songs]
    assert set(picks) == set(envs)
    assert picks == [effects.pick_environment(song) for song in songs]  # stable per song
    rerolls = {EffectRenderer(root, seed=s).pick_environment(songs[0]) for s in range(20)}
    assert len(rerolls) > 1  # another seed can re-roll a song's environment


def test_renderer_picks_and_renders_an_effect_environment(tmp_path: Path):
    root = tmp_path / "fx"
    sources = {"busy": busy_source(), "quiet": quiet_source()}
    make_library(root, "LizzoEnvironment", sources)
    effects = EffectRenderer(root, seed=1)
    assert effects.supports("LizzoEnvironment") and not effects.supports("WeaveEnvironment")
    assert environment_for("v3", "DefaultEnvironment", None, effects) == "LizzoEnvironment"
    assert environment_for("v3", "WeaveEnvironment", None, effects) == "WeaveEnvironment"
    assert environment_for("vanilla", "DefaultEnvironment", None, effects) == "DefaultEnvironment"

    analysis = make_analysis()
    beatmap = demo_beatmap(analysis)
    beatmap.environment = "DefaultEnvironment"
    light_beatmap(beatmap, analysis, "v3", generator=RuleLightingGenerator(effects=effects))
    assert beatmap.environment == "LizzoEnvironment"
    groups = beatmap.lightshow.event_box_groups
    assert groups and {g.group_id for g in groups} <= {0, 1, 2, 3}
    assert effects.last_sources and effects.last_sources[0][1].startswith("m-")


def test_extract_reads_a_group_lit_map(tmp_path: Path):
    analysis = make_analysis()
    beatmap = demo_beatmap(analysis, environment="WeaveEnvironment")
    folder = write_map(beatmap, tmp_path / "map", "v3", song_duration=analysis.duration_s)
    source, reason = extract(folder)
    assert reason == "" and source is not None
    assert len(source["groups"]) == len(beatmap.lightshow.event_box_groups)
    assert source["bars"] == len(source["desc"]["act"])
    assert sum(source["desc"]["act"]) > 0

    info = json.loads((folder / "Info.dat").read_text(encoding="utf-8"))
    for s in info["_difficultyBeatmapSets"]:
        for b in s["_difficultyBeatmaps"]:
            b["_customData"] = {"_requirements": ["Chroma"]}
    (folder / "Info.dat").write_text(json.dumps(info), encoding="utf-8")
    assert extract(folder) == (None, "chroma or noodle")
