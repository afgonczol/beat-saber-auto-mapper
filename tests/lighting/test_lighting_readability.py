from dataclasses import replace

import numpy as np
import pytest

from bsmapper.lighting import get_profile, render_v3
from bsmapper.lighting.events import (
    BACK_LASERS,
    CENTER_LIGHTS,
    LEFT_LASER_SPEED,
    LEFT_LASERS,
    RIGHT_LASER_SPEED,
    RING_LIGHTS,
    RING_SPIN,
    Action,
    LightColor,
    light_value,
)
from bsmapper.lighting.metrics import (
    constant_tempo,
    motion_stats,
    pattern_repeat,
    span_stats,
    together_share,
    visible_changes,
)
from bsmapper.lighting.readability import (
    ReadabilityConfig,
    ReadabilityMask,
    limit_lightshow,
    moment_priority,
    section_intensity,
)
from bsmapper.models.sampling import StepState
from bsmapper.schema import BasicEvent, ColorBoostEvent, Lightshow
from bsmapper.tokenizer import LIGHT_GRAMMAR, LIGHT_VOCAB

B2S = constant_tempo(120.0)  # 0.5 s per beat
RED_FADE = light_value(LightColor.RED, Action.FADE)
BLUE_FADE = light_value(LightColor.BLUE, Action.FADE)
RED_ON = light_value(LightColor.RED, Action.ON)
BLUE_ON = light_value(LightColor.BLUE, Action.ON)
PEAK = section_intensity([0.0], [1.0])
CALM = section_intensity([0.0], [0.0])


def ev(beat, type_, value, float_value=1.0):
    return BasicEvent(beat, type_, value, float_value)


# --- metrics -----------------------------------------------------------------------


def test_visible_changes_skip_repeats_and_count_pulses():
    ls = Lightshow(
        [
            ev(0, RING_LIGHTS, RED_ON),
            ev(1, RING_LIGHTS, RED_ON),  # same state: invisible
            ev(2, RING_LIGHTS, RED_FADE),  # pulse: visible
            ev(3, RING_LIGHTS, RED_FADE),  # pulse again: visible
            ev(4, RING_LIGHTS, RED_ON, 0.0),  # brightness 0 after the fade: still dark
            ev(5, RING_LIGHTS, RED_ON),  # on again
            ev(6, RING_LIGHTS, RED_ON, 0.4),  # big brightness step: visible
            ev(7, RING_LIGHTS, RED_ON, 0.0),  # brightness 0: off
        ]
    )
    changes = visible_changes(ls)[RING_LIGHTS]
    assert [c.beat for c in changes] == [0, 2, 3, 5, 6, 7]
    assert [c.lit for c in changes] == [True, True, True, True, True, False]


def test_color_switches_count_across_fades():
    # Red and blue fades a beat apart read as colour flips even though each fades out.
    ls = Lightshow([ev(b, BACK_LASERS, RED_FADE if b % 2 else BLUE_FADE) for b in range(8)])
    st = span_stats(ls, 0, 8, B2S)
    assert st.color_switches_per_s == pytest.approx(7 / 4.0)
    assert st.changes_per_s == pytest.approx(8 / 4.0)
    assert st.fade_share == 1.0
    assert st.hold_median_s == pytest.approx(0.5)


def test_span_stats_strobe_and_moments():
    events = [ev(i / 24, CENTER_LIGHTS, RED_FADE) for i in range(0, 48, 2)]  # every 1/12 beat
    events.append(ev(0, LEFT_LASERS, RED_ON))
    st = span_stats(Lightshow(events), 0, 2, B2S)
    assert st.strobe_per_s > 20  # 42 ms holds
    assert st.short_hold_share == 1.0
    assert st.active_ticks_per_s == pytest.approx(24 / 1.0)


def test_pattern_repeat_and_together_share():
    bar = [
        ev(0, BACK_LASERS, RED_FADE),
        ev(1, RING_LIGHTS, BLUE_FADE),
        ev(2, LEFT_LASERS, RED_FADE),
    ]
    repeated = Lightshow([replace(e, beat=e.beat + 4 * k) for k in range(4) for e in bar])
    assert pattern_repeat(visible_changes(repeated), 0, 16) == pytest.approx(1.0, abs=0.01)
    rng = np.random.default_rng(0)
    noise = Lightshow(
        sorted(
            (
                ev(float(t) / 4, int(c), RED_FADE if r else BLUE_FADE)
                for t, c, r in zip(
                    rng.integers(0, 64, 40),
                    rng.integers(0, 5, 40),
                    rng.integers(0, 2, 40),
                    strict=True,
                )
            ),
            key=lambda e: e.beat,
        )
    )
    assert pattern_repeat(visible_changes(noise), 0, 16) < 0.3
    together = Lightshow([ev(0, t, RED_FADE) for t in range(5)] + [ev(1, BACK_LASERS, RED_ON)])
    assert together_share(visible_changes(together), 0, 4) == pytest.approx(5 / 6)


def test_motion_stats_flags_whips():
    profile = get_profile("WeaveEnvironment")
    slow = render_v3(Lightshow([ev(0, RING_SPIN, 0), ev(8, RING_SPIN, 0)]), profile)
    assert motion_stats(slow, 0, 16, B2S).whips_per_s == 0
    # A hand-made whip: a full turn in a quarter beat.
    from bsmapper.lighting.render import _turn, rotation_box
    from bsmapper.schema import EventBoxGroup

    box = rotation_box(profile.spin, _turn(0.25, 0.0, loops=1), 0.0)
    whip = Lightshow(
        event_box_groups=[EventBoxGroup(4.0, profile.spin.groups[0], "rotation", (box,))]
    )
    st = motion_stats(whip, 0, 16, B2S)
    assert st.whips_per_s > 0 and st.max_deg_per_s == pytest.approx(360 / 0.125)


# --- post-filter -------------------------------------------------------------------


def test_priority_order():
    assert [moment_priority(b) for b in (0, 1, 1.5, 1.25, 4)] == [0, 1, 2, 3, 0]


def test_hold_keeps_the_downbeat_over_an_earlier_offbeat():
    cfg = ReadabilityConfig(min_hold_s=(0.3, 0.3))  # 0.6 beat at 120 BPM
    ls = Lightshow([ev(3.75, BACK_LASERS, RED_FADE), ev(4.0, BACK_LASERS, RED_FADE)])
    out, report = limit_lightshow(ls, B2S, PEAK, cfg)
    assert [e.beat for e in out.basic_events] == [4.0]
    assert report.dropped["hold"] == 1


def test_color_hold_recolours_quick_flips():
    beats = [0, 1, 2, 3, 4, 5]
    ls = Lightshow([ev(b, RING_LIGHTS, BLUE_FADE if b % 2 else RED_FADE) for b in beats])
    out, report = limit_lightshow(ls, B2S, PEAK)  # peak: 2-beat colour hold
    values = [e.value for e in out.basic_events]
    # Red holds until beat 2; the blue at beat 3 is the first flip allowed, then blue holds.
    assert values == [RED_FADE] * 3 + [BLUE_FADE] * 3
    assert report.recolored == 2
    calm, _ = limit_lightshow(ls, B2S, CALM)  # calm: 4 beats
    assert [e.value for e in calm.basic_events] == [RED_FADE] * 5 + [BLUE_FADE]


def test_moment_cap_limits_busy_seconds():
    # 16 moments per second (every 1/8 beat at 120 BPM) on rotating channels.
    events = [ev(i / 8, i % 5, RED_FADE) for i in range(32)]
    out, report = limit_lightshow(Lightshow(events), B2S, CALM)
    beats = sorted({e.beat for e in out.basic_events})
    for b in beats:
        assert sum(b <= x < b + 2.0 for x in beats) <= 4  # 4 per second when calm
    assert 0.0 in beats and 1.0 in beats  # downbeat and beats first
    assert report.dropped["moment_cap"] > 0


def test_motion_gaps_sync_and_boost_blips():
    ls = Lightshow(
        [
            ev(0, RING_SPIN, 0),
            ev(1, RING_SPIN, 0),  # too close
            ev(4, RING_SPIN, 0),
            ev(0, LEFT_LASER_SPEED, 3),
            ev(0.25, RIGHT_LASER_SPEED, 3),  # moves to 0 with the left side
            ev(0.5, LEFT_LASER_SPEED, 5),  # inside the 1-beat gap at peak
            ev(2, LEFT_LASER_SPEED, 7),
            ev(2, 5, 1),  # other types pass through
        ],
        boost_events=[
            ColorBoostEvent(8, True),
            ColorBoostEvent(9, False),  # a one-beat blip: both go
            ColorBoostEvent(16, True),
        ],
    )
    out, report = limit_lightshow(ls, B2S, PEAK)
    spins = [e.beat for e in out.basic_events if e.type == RING_SPIN]
    assert spins == [0, 4]
    speeds = [(e.beat, e.type) for e in out.basic_events if e.type in (12, 13)]
    assert speeds == [(0, LEFT_LASER_SPEED), (0, RIGHT_LASER_SPEED), (2, LEFT_LASER_SPEED)]
    assert any(e.type == 5 for e in out.basic_events)
    assert [(b.beat, b.on) for b in out.boost_events] == [(16, True)]
    assert report.moved == 2 and report.dropped["boost"] == 2  # 0.5 joined 0.25 first


def test_filter_is_identity_on_a_calm_show():
    ls = Lightshow([ev(b, BACK_LASERS, RED_FADE) for b in range(0, 32, 4)])
    out, report = limit_lightshow(ls, B2S, CALM)
    assert out.basic_events == ls.basic_events and report.total() == 0


def test_section_intensity_lookup():
    at = section_intensity([0.0, 16.0, 32.0], [0.2, 0.9, 0.5])
    assert [at(b) for b in (0, 15.9, 16, 40)] == [0.2, 0.2, 0.9, 0.5]
    cfg = ReadabilityConfig()
    assert cfg.at(cfg.min_hold_s, 0.0) == 0.25 and cfg.at(cfg.min_hold_s, 1.0) == 0.12


# --- decode-time mask --------------------------------------------------------------


def _feed(parser, names):
    for name in names:
        group, _, value = name.partition("_")
        parser.feed(LIGHT_VOCAB.id(group, int(value)))


def test_mask_forbids_unused_types_and_respects_gaps_across_windows():
    cfg = ReadabilityConfig(decode_motion_gaps=True, decode_light_holds=True)
    mask = ReadabilityMask(B2S, PEAK, cfg)
    parser = LIGHT_GRAMMAR.parser()
    body: list[int] = []
    _feed(parser, ["TIME_0", "EVT_8", "VAL_0", "TIME_360"])  # a spin at beat 0, now 15
    m = mask(StepState(0, 0, body, parser))
    assert m is not None
    assert not m[LIGHT_VOCAB.id("EVT", 6)]  # extra light channel: no renderer shows it
    assert m[LIGHT_VOCAB.id("EVT", RING_SPIN)]  # 15 beats later: allowed

    # Window 1 starts at beat 16; the spin of window 0 at 15.0 is still inside the gap.
    window0 = [LIGHT_VOCAB.id("TIME", 0), LIGHT_VOCAB.id("EVT", 8), LIGHT_VOCAB.id("VAL", 0)]
    window0 += [LIGHT_VOCAB.id("TIME", 360), LIGHT_VOCAB.id("EVT", 8), LIGHT_VOCAB.id("VAL", 0)]
    mask = ReadabilityMask(B2S, PEAK, cfg)
    p0 = LIGHT_GRAMMAR.parser()
    for t in window0[:-1]:
        p0.feed(t)
    mask(StepState(0, 0, window0, p0))  # sees the body; the last event completes later
    p1 = LIGHT_GRAMMAR.parser()
    _feed(p1, ["TIME_12"])  # beat 16.5
    m = mask(StepState(1, 0, [], p1))
    assert not m[LIGHT_VOCAB.id("EVT", RING_SPIN)]
    assert m[LIGHT_VOCAB.id("EVT", BACK_LASERS)]


def test_mask_default_leaves_lights_and_motion_alone():
    mask = ReadabilityMask(B2S, PEAK)
    parser = LIGHT_GRAMMAR.parser()
    _feed(parser, ["TIME_0", "EVT_0", "VAL_3", "EVT_8", "VAL_0", "TIME_1"])
    m = mask(StepState(0, 0, [], parser))
    for t in (BACK_LASERS, RING_SPIN, LEFT_LASER_SPEED):
        assert m[LIGHT_VOCAB.id("EVT", t)]
    assert not m[LIGHT_VOCAB.id("EVT", 6)]
    _feed(parser, ["EVT_1"])  # inside an object: the value is never masked
    assert mask(StepState(0, 0, [], parser)) is None


# --- renderer: no whips, no sweep restarts -------------------------------------------


def test_close_spins_do_not_whip():
    profile = get_profile("WeaveEnvironment")
    ls = Lightshow([ev(b, RING_SPIN, 0) for b in (0, 4, 5, 8, 9, 12)])  # spin rolls
    out = render_v3(ls, profile)
    st = motion_stats(out, 0, 16, constant_tempo(104.0))
    assert st.whips_per_s == 0
    loops = [
        g.beat
        for g in out.event_box_groups
        if g.kind == "rotation"
        and g.group_id == profile.spin.groups[0]
        and g.boxes[0]["l"][-1]["l"] == 1
    ]
    assert loops == [0, 4, 8, 12]  # spins during a running turn are ignored


def test_sweeps_continue_through_repeated_and_new_speeds():
    profile = get_profile("WeaveEnvironment")
    gid = profile.left_sweep.groups[0]

    def targets(events):
        out = render_v3(Lightshow(events), profile)
        rot = sorted(
            (g for g in out.event_box_groups if g.kind == "rotation" and g.group_id == gid),
            key=lambda g: g.beat,
        )
        return [(g.beat, g.boxes[0]["l"][-1]["r"]) for g in rot]

    base = targets([ev(0, LEFT_LASER_SPEED, 3), ev(16, LEFT_LASER_SPEED, 0)])
    repeated = targets(
        [ev(0, LEFT_LASER_SPEED, 3), ev(3, LEFT_LASER_SPEED, 3), ev(16, LEFT_LASER_SPEED, 0)]
    )
    assert repeated == base  # a repeated speed does not restart the sweep
    changed = targets([ev(0, LEFT_LASER_SPEED, 1), ev(4, LEFT_LASER_SPEED, 7)])
    signs = [np.sign(((a + 180) % 360) - 180) for _, a in changed if a % 360]
    assert all(s1 != s2 for s1, s2 in zip(signs, signs[1:], strict=False))  # keeps alternating


def test_dark_loud_windows_take_the_fallback_lights():
    model = Lightshow(
        [ev(b, BACK_LASERS, RED_FADE) for b in range(16, 32)]  # window 1 lit, 0 and 2 dark
        + [ev(4, RING_SPIN, 0)]
    )
    rules = Lightshow([ev(b, RING_LIGHTS, BLUE_FADE) for b in range(0, 48, 2)])
    at = section_intensity([0.0, 32.0], [0.9, 0.1])  # window 2 is calm: dark is fine
    out, report = limit_lightshow(model, B2S, at, fallback=rules)
    assert report.filled_windows == 1
    lights = [(e.beat, e.type) for e in out.basic_events if e.type < 5]
    assert {t for b, t in lights if b < 16} == {RING_LIGHTS}  # window 0 from the rules
    assert {t for b, t in lights if 16 <= b < 32} == {BACK_LASERS}  # the model's own
    assert not [b for b, _ in lights if b >= 32]
    assert any(e.type == RING_SPIN for e in out.basic_events)  # motion stays the model's
    unfilled, report = limit_lightshow(model, B2S, at)
    assert report.filled_windows == 0
