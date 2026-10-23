from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from gen_synthetic import TPB, make_analysis

from bsmapper.audio import BeatGrid
from bsmapper.generation import JUMP_TABLE, njs_and_offset, reaction_time
from bsmapper.generation.njs import default_hjd
from bsmapper.generation.timing import seconds_to_beats, song_time_analysis, tempo_beatmap
from bsmapper.schema import BpmChange, Difficulty


def test_song_time_puts_map_beats_on_the_padded_audio() -> None:
    grid = BeatGrid(120.0, 0.3)
    original = make_analysis(offset_s=0.3)
    lead = grid.lead_in_s()
    song = song_time_analysis(original, lead, grid.beats_per_bar)
    # map beat i plays at i * 0.5 s in the padded file; grid beat 0 is map beat 4.
    assert song.beat_times[:6] == pytest.approx(np.arange(6) * 0.5)
    assert song.beat_times[4] == pytest.approx(original.beat_times[0] + lead)
    assert song.duration_s == pytest.approx(original.duration_s + lead)
    assert song.downbeat_times[0] == pytest.approx(0.0)
    assert song.sections[0].start_s == 0.0
    assert song.sections[1].start_s == pytest.approx(original.sections[1].start_s + lead)
    for name, rows in song.beat_features.items():
        assert len(rows) == len(original.beat_features[name]) + 4 * TPB
        assert np.array_equal(rows[4 * TPB :], original.beat_features[name])
    n_pad = round(lead * original.frame_rate)
    assert len(song.features["rms"]) == len(original.features["rms"]) + n_pad
    assert song_time_analysis(original, 0.0) is original


def test_tempo_beatmap_rebuilds_tempo_changes() -> None:
    a = make_analysis(duration=20.0)
    times = np.concatenate([np.arange(16) * 0.5, 7.5 + np.arange(1, 17) * 0.4])
    a = dataclasses.replace(a, beat_times=times)
    beatmap = tempo_beatmap(a)
    assert beatmap.bpm == pytest.approx(120.0)
    assert beatmap.bpm_changes == [BpmChange(15.0, 150.0)]  # beat 15 -> 16 takes 0.4 s
    for i, t in enumerate(times):
        assert beatmap.beat_to_seconds(i) == pytest.approx(t)
    assert float(seconds_to_beats(a, 8.3)) == pytest.approx(17.0)


@pytest.mark.parametrize("bpm", [70.0, 120.0, 200.0])
@pytest.mark.parametrize("diff", list(Difficulty))
def test_njs_offset_hits_the_reaction_time(bpm: float, diff: Difficulty) -> None:
    njs, offset = njs_and_offset(diff, bpm)
    assert njs == JUMP_TABLE[diff].njs
    # rounding the offset to 0.05 beats moves the reaction time by < 0.025 beats
    tolerance = 0.026 * 60 / bpm
    target = JUMP_TABLE[diff].reaction_s
    assert reaction_time(njs, offset, bpm) == pytest.approx(target, abs=tolerance)


def test_reaction_time_override() -> None:
    njs, offset = njs_and_offset(Difficulty.EXPERT_PLUS, 104.0, reaction_s=0.9)
    assert njs == 17.0
    assert reaction_time(njs, offset, 104.0) == pytest.approx(0.9, abs=0.026 * 60 / 104)
    # Expert+ defaults: NJS 17, 0.7 s, about 23.8 m of jump distance
    njs, offset = njs_and_offset(Difficulty.EXPERT_PLUS, 120.0)
    assert 2 * njs * reaction_time(njs, offset, 120.0) == pytest.approx(23.8, abs=0.5)


def test_jump_overrides_resolve_in_order(tmp_path) -> None:
    from bsmapper.generation import GenerateOptions, JumpOverride, configured_jumps

    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[jump."Expert+"]\nnjs = 18\nreaction_time = 0.75\n[jump.Hard]\nreaction_time = 0.95\n',
        encoding="utf-8",
    )
    defaults = configured_jumps(cfg)
    assert defaults[Difficulty.EXPERT_PLUS] == JumpOverride(18, 0.75)
    opts = GenerateOptions(
        njs=15.0,
        jump={Difficulty.EXPERT_PLUS: JumpOverride(reaction_s=0.9)},
        jump_defaults=defaults,
    )
    assert opts.jump_for(Difficulty.EXPERT_PLUS) == JumpOverride(15.0, 0.9)
    assert opts.jump_for(Difficulty.HARD) == JumpOverride(15.0, 0.95)
    assert GenerateOptions(jump_defaults=defaults).jump_for(Difficulty.EXPERT_PLUS) == (
        JumpOverride(18, 0.75)
    )
    cfg.write_text("[jump.Insane]\nnjs = 30\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Insane"):
        configured_jumps(cfg)


def test_njs_override_and_game_halving() -> None:
    assert njs_and_offset(Difficulty.EXPERT, 120.0, njs=22.0)[0] == 22.0
    assert default_hjd(10.0, 60.0) == 1.0  # 10 * 1 s * 4 beats > 18 m: halved twice
    assert default_hjd(10.0, 200.0) == 4.0
