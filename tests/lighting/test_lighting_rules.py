import numpy as np
import pytest
from synthetic import BPM, make_analysis

from bsmapper.lighting.events import (
    BACK_LASERS,
    CENTER_LIGHTS,
    LASER_SPEED_TYPES,
    LEFT_LASERS,
    LIGHT_TYPES,
    OFF,
    RIGHT_LASERS,
    RING_LIGHTS,
    RING_SPIN,
    RING_ZOOM,
    Action,
    LightColor,
    decode_value,
    light_value,
)
from bsmapper.lighting.palettes import CALM, HIGH, MID, PALETTES, choose_palettes
from bsmapper.lighting.rules import (
    RuleConfig,
    onset_peaks,
    rule_lightshow,
    section_mood,
    thin_hits,
)
from bsmapper.schema import Beatmap


def of_type(events, *types):
    return [e for e in events if e.type in types]


def test_light_values_round_trip():
    assert light_value(LightColor.BLUE, Action.ON) == 1
    assert light_value(LightColor.RED, Action.FLASH) == 6
    assert light_value(LightColor.WHITE, Action.TRANSITION) == 12
    for color in LightColor:
        for action in Action:
            assert decode_value(light_value(color, action)) == (color, action)
    assert decode_value(OFF) is None
    assert decode_value(13) is None


def test_section_moods():
    config = RuleConfig()
    assert section_mood("drop", 0.5, config) == HIGH
    assert section_mood("verse", 0.9, config) == HIGH
    assert section_mood("chorus", 0.6, config) == HIGH
    assert section_mood("intro", 0.5, config) == CALM
    assert section_mood("verse", 0.2, config) == CALM
    assert section_mood("verse", 0.55, config) == MID


def test_palettes_follow_mood_and_never_repeat():
    moods = [CALM, CALM, HIGH, HIGH, HIGH, MID, MID, HIGH]
    palettes = choose_palettes(moods)
    assert [p.mood for p in palettes] == moods
    assert all(a != b for a, b in zip(palettes, palettes[1:], strict=False))
    assert choose_palettes(moods, seed=1) != palettes
    assert {p.mood for p in PALETTES} == {CALM, MID, HIGH}


def test_sections_are_planned_in_beats(analysis, beatmap):
    plan = rule_lightshow(analysis, beatmap)
    assert [(s.start, s.end) for s in plan.sections] == [(0, 32), (32, 64), (64, 96), (96, 128)]
    assert [s.mood for s in plan.sections] == [CALM, MID, HIGH, CALM]
    assert [s.boost for s in plan.sections] == [False, False, True, False]
    # The lead colour alternates from section to section.
    assert [s.primary for s in plan.sections] == [LightColor.RED, LightColor.BLUE] * 2
    assert plan.end_beat == 128


def test_events_are_on_the_tick_grid_sorted_and_in_range(analysis, beatmap):
    events = rule_lightshow(analysis, beatmap).lightshow.basic_events
    beats = [e.beat for e in events]
    assert beats == sorted(beats)
    assert all(0 <= b <= 128 for b in beats)
    assert all(abs(b * 24 - round(b * 24)) < 1e-9 for b in beats)
    assert len({(e.beat, e.type) for e in events}) == len(events)


def test_section_changes_spin_zoom_and_set_laser_speed(analysis, beatmap):
    events = rule_lightshow(analysis, beatmap).lightshow.basic_events
    assert [e.beat for e in of_type(events, RING_SPIN)] == [0, 32, 64, 96]
    # Zoom on the energy jumps into the verse, the drop and back down to the outro.
    assert [e.beat for e in of_type(events, RING_ZOOM)] == [32, 64, 96]
    speeds = {(e.beat, e.value) for e in of_type(events, LASER_SPEED_TYPES[0])}
    assert speeds == {(0, 1), (32, 3), (64, 7), (96, 1), (128, 0)}


def test_back_and_ring_lights_follow_downbeats(analysis, beatmap):
    events = rule_lightshow(analysis, beatmap).lightshow.basic_events
    downbeats = set(range(0, 128, 4))
    back = of_type(events, BACK_LASERS)
    assert {e.beat for e in back if e.value != OFF} <= downbeats
    # Every downbeat of the verse and drop flashes the back lights.
    flashing = {
        e.beat for e in back if decode_value(e.value) and decode_value(e.value)[1] == Action.FLASH
    }
    assert set(range(32, 96, 4)) <= flashing
    # Drops add white on phrase starts and ring hits on the half bar.
    assert decode_value(next(e.value for e in back if e.beat == 64))[0] == LightColor.WHITE
    rings = {e.beat for e in of_type(events, RING_LIGHTS) if e.value != OFF}
    assert {66, 70} <= rings


def test_lasers_follow_onsets_with_mood_dependent_gaps(analysis, beatmap):
    plan = rule_lightshow(analysis, beatmap)
    lasers = of_type(plan.lightshow.basic_events, LEFT_LASERS, RIGHT_LASERS)
    lasers = [e for e in lasers if e.value != OFF]
    onsets = {b / 2 for b in range(0, 256)}  # every eighth note in seconds -> beats
    for section in plan.sections:
        hits = [e for e in lasers if section.start <= e.beat < section.end]
        assert hits, section
        assert {e.beat for e in hits} <= onsets
        gap = RuleConfig().onset_gap[section.mood]
        assert min(np.diff([e.beat for e in hits])) >= gap - 1e-9
        # Sides alternate.
        assert all(a.type != b.type for a, b in zip(hits, hits[1:], strict=False))
    drop = [e for e in lasers if 64 <= e.beat < 96]
    assert len(drop) == 64  # every eighth-note onset of the drop
    calm = [e for e in lasers if 0 <= e.beat < 32]
    assert all(decode_value(e.value)[1] == Action.FADE for e in calm)


def test_boost_covers_the_drop(analysis, beatmap):
    boosts = rule_lightshow(analysis, beatmap).lightshow.boost_events
    assert [(b.beat, b.on) for b in boosts] == [(64, True), (96, False)]


def test_boost_switches_off_at_song_end_after_a_final_drop(beatmap):
    sections = [("verse", 0.0, 32.0, 0.5), ("drop", 32.0, 64.0, 1.0)]
    boosts = rule_lightshow(make_analysis(sections), beatmap).lightshow.boost_events
    assert [(b.beat, b.on) for b in boosts] == [(64, True), (128, False)]


def test_everything_is_off_at_the_end(analysis, beatmap):
    events = rule_lightshow(analysis, beatmap).lightshow.basic_events
    last = {e.type: e for e in events if e.beat == 128}
    for type_ in LIGHT_TYPES:
        assert last[type_].value == OFF
    for type_ in LASER_SPEED_TYPES:
        assert last[type_].value == 0


def test_centre_lights_take_the_section_colour(analysis, beatmap):
    plan = rule_lightshow(analysis, beatmap)
    center = {e.beat: e for e in of_type(plan.lightshow.basic_events, CENTER_LIGHTS)}
    for section in plan.sections:
        assert decode_value(center[section.start].value)[0] == section.primary


def test_time_offset_shifts_everything(analysis, beatmap):
    base = rule_lightshow(analysis, beatmap).lightshow.basic_events
    shifted = rule_lightshow(analysis, beatmap, time_offset_s=2.0).lightshow.basic_events
    # 2 s at 120 BPM is 4 beats; the end-of-song events move too.
    assert [e.beat + 4 for e in base] == [e.beat for e in shifted]


def test_bpm_changes_are_respected(analysis):
    from bsmapper.schema import BpmChange

    beatmap = Beatmap("x", BPM, bpm_changes=[BpmChange(64, 240)])
    plan = rule_lightshow(analysis, beatmap)
    # 32 s is beat 64; after it the tempo doubles, so 48 s is beat 64 + 16 * 4.
    assert [s.start for s in plan.sections] == [0, 32, 64, 128]


def test_without_onsets_lasers_fall_back_to_beats(beatmap):
    plan = rule_lightshow(make_analysis(onsets=False), beatmap)
    lasers = [e for e in of_type(plan.lightshow.basic_events, LEFT_LASERS, RIGHT_LASERS)]
    assert lasers
    assert all(e.beat == int(e.beat) for e in lasers)


def test_without_sections_the_song_is_one_section(beatmap):
    plan = rule_lightshow(make_analysis(sections=[]), beatmap)
    assert len(plan.sections) == 1
    assert plan.sections[0].mood == MID
    assert of_type(plan.lightshow.basic_events, BACK_LASERS)


def test_onset_peaks_and_thinning(analysis):
    peaks = onset_peaks(analysis, 0.0, 2.0)
    assert [round(t, 2) for t, _ in peaks] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    assert [s for _, s in peaks][:2] == pytest.approx([1.0, 0.4])
    hits = [(0.0, 0.5), (0.25, 0.9), (0.5, 0.4), (1.0, 0.8)]
    assert thin_hits(hits, 0.5) == [(0.25, 0.9), (1.0, 0.8)]


def test_intensity_spreads_loud_songs():
    from bsmapper.lighting.rules import intensities

    assert intensities([0.3, 0.55, 1.0, 0.35]) == pytest.approx([0.15, 0.454, 1.0, 0.211], abs=1e-3)
    assert intensities([0.8, 0.8]) == [0.8, 0.8]
    assert intensities([]) == []


def test_loud_song_still_gets_contrast_and_boost(beatmap):
    # A loud master: every section between 0.53 and 1.0 (as on a real pop song).
    sections = [
        ("intro", 0.0, 8.0, 0.64),
        ("chorus", 8.0, 16.0, 1.0),
        ("verse", 16.0, 24.0, 0.74),
        ("chorus", 24.0, 32.0, 0.96),
        ("verse", 32.0, 40.0, 0.53),
        ("outro", 40.0, 48.0, 0.91),
    ]
    plan = rule_lightshow(make_analysis(sections, duration=48.0), beatmap)
    assert [s.mood for s in plan.sections] == [CALM, HIGH, MID, HIGH, CALM, HIGH]
    # Boost on the high sections entered with a jump: both choruses and the outro.
    assert [s.boost for s in plan.sections] == [False, True, False, True, False, True]
    boosts = [(b.beat, b.on) for b in plan.lightshow.boost_events]
    assert boosts == [(16, True), (32, False), (48, True), (64, False), (80, True), (96, False)]
