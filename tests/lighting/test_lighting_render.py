import pytest

from bsmapper.formats._common import canon_box
from bsmapper.lighting import (
    DEFAULT_LEGACY_ENVIRONMENT,
    DEFAULT_V3_ENVIRONMENT,
    RuleLightingGenerator,
    available_environments,
    environment_for,
    get_profile,
    light_beatmap,
    load_profiles,
    render_chroma,
    render_v3,
    render_vanilla,
    rule_lightshow,
)
from bsmapper.lighting.events import (
    CENTER_LIGHTS,
    LIGHT_TYPES,
    RING_SPIN,
    VANILLA_TYPES,
    Action,
    LightColor,
    decode_value,
    light_value,
)
from bsmapper.lighting.render import color_events
from bsmapper.schema import BasicEvent, Beatmap, Lightshow

ENVIRONMENTS = ["EDMEnvironment", "TheSecondEnvironment", "WeaveEnvironment"]


# --- profiles ----------------------------------------------------------------------


def test_bundled_profiles():
    assert available_environments() == ENVIRONMENTS
    assert DEFAULT_V3_ENVIRONMENT in load_profiles()
    assert get_profile("DefaultEnvironment") is None


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_profiles_are_consistent(environment):
    profile = get_profile(environment)
    assert profile.validate() == []
    assert set(profile.channels) == set(LIGHT_TYPES)
    assert profile.spin and profile.left_sweep and profile.right_sweep
    # Every colour group is played by some channel or passed through as a basic type.
    played = {gid for c in profile.channels.values() for gid in c.groups}
    assert played == {gid for gid, g in profile.groups.items() if g.color}


def test_profile_validation_reports_problems():
    from dataclasses import replace

    from bsmapper.lighting.profiles import Channel, Motion

    profile = get_profile("EDMEnvironment")
    broken = replace(
        profile,
        channels={0: Channel((12, 99))},
        spin=Motion((2,), axis=2),
    )
    problems = broken.validate()
    assert "channel 0: group 12 has no colour" in problems
    assert "channel 0: unknown group 99" in problems
    assert "spin: group 2 cannot rotate on axis 2" in problems


# --- vanilla -----------------------------------------------------------------------


def test_vanilla_strips_colour_and_unknown_types():
    ls = Lightshow(
        basic_events=[
            BasicEvent(0, 0, 1, color=(1, 0, 0, 1)),
            BasicEvent(1, 40, 1),
            BasicEvent(2, RING_SPIN, 0),
        ]
    )
    out = render_vanilla(ls)
    assert [(e.beat, e.type, e.color) for e in out.basic_events] == [(0, 0, None), (2, 8, None)]
    assert not out.uses_chroma and not out.event_box_groups


def test_vanilla_generator_output(analysis, beatmap):
    ls = RuleLightingGenerator().generate(analysis, beatmap, "vanilla")
    assert ls.basic_events and ls.boost_events
    assert {e.type for e in ls.basic_events} <= VANILLA_TYPES
    assert not ls.event_box_groups and not ls.uses_chroma


# --- v3 ----------------------------------------------------------------------------


def test_color_events_per_action():
    on = color_events(light_value(LightColor.RED, Action.ON), 0.8)
    assert [(e["b"], e["i"], e["c"], e["s"]) for e in on] == [(0.0, 0, 0, 0.8)]
    flash = color_events(light_value(LightColor.BLUE, Action.FLASH), 1.0)
    assert [(e["b"], e["i"], e["c"], e["s"]) for e in flash] == [(0, 0, 1, 1.2), (0.25, 1, 1, 1.0)]
    fade = color_events(light_value(LightColor.WHITE, Action.FADE), 1.0)
    assert [(e["i"], e["c"], e["s"]) for e in fade] == [(0, 2, 1.2), (1, 2, 0.0)]
    transition = color_events(light_value(LightColor.RED, Action.TRANSITION), 0.5)
    assert [(e["i"], e["s"]) for e in transition] == [(1, 0.5)]
    assert [e["s"] for e in color_events(0, 1.0)] == [0.0]


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_v3_render(analysis, beatmap, environment):
    profile = get_profile(environment)
    plan = rule_lightshow(analysis, beatmap)
    ls = render_v3(plan.lightshow, profile)

    assert ls.boost_events == plan.lightshow.boost_events
    assert {e.type for e in ls.basic_events} <= profile.basic_types
    kinds = {g.kind for g in ls.event_box_groups}
    assert kinds == {"color", "rotation"}
    for group in ls.event_box_groups:
        info = profile.groups[group.group_id]
        if group.kind == "color":
            assert info.color
        else:
            assert all(info.rotation[box["a"]] for box in group.boxes)
        # Boxes are already in the canonical shape the formats package reads back.
        for box in group.boxes:
            assert canon_box(group.kind, box) == box
    keys = [(g.beat, g.kind, g.group_id) for g in ls.event_box_groups]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)

    # Every light event lands on all of its channel's groups.
    color_keys = {(g.beat, g.group_id) for g in ls.event_box_groups if g.kind == "color"}
    for e in plan.lightshow.basic_events:
        if e.type in LIGHT_TYPES:
            for gid in profile.channels[e.type].groups:
                assert (e.beat, gid) in color_keys

    # Ring spins turn the spin groups at each section start.
    spin_keys = {(g.beat, g.group_id) for g in ls.event_box_groups if g.kind == "rotation"}
    for start in (0, 32, 64, 96):
        for gid in profile.spin.groups:
            assert (start, gid) in spin_keys


def test_v3_spins_alternate_direction_and_sweeps_mirror(analysis, beatmap):
    profile = get_profile("WeaveEnvironment")
    ls = render_v3(rule_lightshow(analysis, beatmap).lightshow, profile)
    spin_gid = profile.spin.groups[0]
    spins = [g for g in ls.event_box_groups if g.kind == "rotation" and g.group_id == spin_gid]
    directions = [g.boxes[0]["l"][-1]["o"] for g in spins if g.boxes[0]["l"][-1]["l"] == 1]
    assert directions[:4] == [1, 2, 1, 2]
    left = [g for g in ls.event_box_groups if g.group_id == profile.left_sweep.groups[0]]
    right = [g for g in ls.event_box_groups if g.group_id == profile.right_sweep.groups[0]]
    left_rot = [g for g in left if g.kind == "rotation"]
    right_rot = [g for g in right if g.kind == "rotation"]
    assert left_rot and len(left_rot) == len(right_rot)
    assert {g.boxes[0]["r"] for g in left_rot} == {0}
    assert {g.boxes[0]["r"] for g in right_rot} == {1}
    # Sweeps get faster in the drop (64..96) than in the verse (32..64).
    verse = [g.beat for g in left_rot if 32 <= g.beat < 64]
    drop = [g.beat for g in left_rot if 64 <= g.beat < 96]
    assert len(drop) > len(verse)


def test_v3_generator_falls_back_to_vanilla_without_profile(analysis, beatmap):
    ls = RuleLightingGenerator().generate(analysis, beatmap, "v3")  # DefaultEnvironment
    assert not ls.event_box_groups
    assert ls.basic_events


def test_unknown_style_is_rejected(analysis, beatmap):
    with pytest.raises(ValueError):
        RuleLightingGenerator().generate(analysis, beatmap, "neon")  # type: ignore[arg-type]


# --- Chroma -----------------------------------------------------------------------


def test_chroma_colours_from_section_palettes(analysis, beatmap):
    plan = rule_lightshow(analysis, beatmap)
    ls = render_chroma(plan.lightshow, plan.sections)
    assert ls.uses_chroma
    assert not ls.event_box_groups
    for e in ls.basic_events:
        decoded = decode_value(e.value) if e.type in LIGHT_TYPES else None
        if decoded is None:
            assert e.color is None
            continue
        section = next(s for s in reversed(plan.sections) if s.start <= e.beat)
        rgb = {
            LightColor.RED: section.palette.left,
            LightColor.BLUE: section.palette.right,
            LightColor.WHITE: section.palette.accent,
        }[decoded[0]]
        assert e.color == pytest.approx((*rgb, 1.0), abs=1e-3)


def test_chroma_gradients_on_centre_lights(analysis, beatmap):
    plan = rule_lightshow(analysis, beatmap)
    ls = render_chroma(plan.lightshow, plan.sections)
    center = [e for e in ls.basic_events if e.type == CENTER_LIGHTS]
    gradients = [
        e for e in center if decode_value(e.value) and decode_value(e.value)[1] == Action.TRANSITION
    ]
    # One gradient per section, ending just before the next section (or the song end).
    assert [e.beat for e in gradients] == [31.75, 63.75, 95.75, 127.75]
    for section, gradient in zip(plan.sections, gradients, strict=True):
        assert gradient.color == pytest.approx((*section.palette.right, 1.0), abs=1e-3) or (
            gradient.color == pytest.approx((*section.palette.left, 1.0), abs=1e-3)
        )
        start = next(e for e in center if e.beat == section.start)
        assert decode_value(gradient.value)[0] != decode_value(start.value)[0]
    beats = [e.beat for e in ls.basic_events]
    assert beats == sorted(beats)


# --- environments ------------------------------------------------------------------


def test_environment_for():
    assert environment_for("v3", "DefaultEnvironment") == DEFAULT_V3_ENVIRONMENT
    assert environment_for("v3", "EDMEnvironment") == "EDMEnvironment"
    assert environment_for("v3", "DefaultEnvironment", "TheSecondEnvironment") == (
        "TheSecondEnvironment"
    )
    assert environment_for("v3", "EDMEnvironment", "NiceEnvironment") == "EDMEnvironment"
    assert environment_for("vanilla", "WeaveEnvironment") == DEFAULT_LEGACY_ENVIRONMENT
    assert environment_for("chroma", "BigMirrorEnvironment") == "BigMirrorEnvironment"
    assert environment_for("chroma", "DefaultEnvironment", "NiceEnvironment") == "NiceEnvironment"


@pytest.mark.parametrize(
    ("style", "environment"),
    [
        ("v3", "WeaveEnvironment"),
        ("vanilla", "DefaultEnvironment"),
        ("chroma", "DefaultEnvironment"),
    ],
)
def test_light_beatmap_sets_environment_and_lightshow(analysis, style, environment):
    beatmap = Beatmap("x", 120.0, environment="WeaveEnvironment" if style != "v3" else "Nope")
    light_beatmap(beatmap, analysis, style)
    assert beatmap.environment == environment
    assert bool(beatmap.lightshow.event_box_groups) == (style == "v3")
    assert beatmap.lightshow.uses_chroma == (style == "chroma")
