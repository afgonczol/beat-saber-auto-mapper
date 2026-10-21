"""The lit demo map, written as a full map folder with bsmapper.formats and read back."""

import json
import os
from pathlib import Path

import pytest
from synthetic import make_analysis

from bsmapper.formats import read_map_report, write_map
from bsmapper.lighting.demo import demo_beatmap, trivial_notes
from bsmapper.schema import Color, CutDirection


def test_trivial_notes_alternate_hands_and_swings():
    notes = trivial_notes(4, 18)
    assert [n.beat for n in notes] == [4, 6, 8, 10, 12, 14, 16, 18]
    assert [n.color for n in notes] == [Color.RED, Color.BLUE] * 4
    red = [n.direction for n in notes if n.color == Color.RED]
    assert red == [CutDirection.DOWN, CutDirection.UP] * 2


@pytest.mark.parametrize("style", ["v3", "vanilla", "chroma"])
@pytest.mark.parametrize("version", ["v3", "v4"])
def test_demo_map_folder_round_trips(tmp_path: Path, style, version):
    analysis = make_analysis()
    beatmap = demo_beatmap(analysis, lead_in_s=0.5, style=style, song_name="Synthetic")
    out = write_map(beatmap, tmp_path / "map", version, song_duration=analysis.duration_s + 0.5)

    info = json.loads((out / "Info.dat").read_text("utf-8"))
    dump = json.dumps(info)
    assert ("Chroma" in dump) == (style == "chroma")
    assert "_requirements" not in dump and '"requirements"' not in dump

    back, report = read_map_report(out)
    assert not report.needs_mods
    assert back.environment == beatmap.environment
    assert len(back.difficulties) == 1
    assert back.difficulties[0].notes == beatmap.difficulties[0].notes
    ls = back.difficulties[0].lightshow or back.lightshow
    assert ls.basic_events == beatmap.lightshow.basic_events
    assert ls.boost_events == beatmap.lightshow.boost_events
    assert ls.event_box_groups == beatmap.lightshow.event_box_groups
    assert ls.uses_chroma == (style == "chroma")


def test_demo_beatmap_shifts_by_one_bar_with_lead_in():
    analysis = make_analysis()
    plain = demo_beatmap(analysis, style="vanilla")
    padded = demo_beatmap(analysis, lead_in_s=2.0, style="vanilla")  # one 4-beat bar at 120 BPM
    assert [e.beat + 4 for e in plain.lightshow.basic_events] == [
        e.beat for e in padded.lightshow.basic_events
    ]
    assert padded.difficulties[0].notes[0].beat == 4


@pytest.mark.slow
def test_demo_on_a_local_song(tmp_path: Path):
    """Set BSMAPPER_DEMO_SONG to an audio file; needs ffmpeg and the audio extra."""
    song = os.environ.get("BSMAPPER_DEMO_SONG")
    if not song:
        pytest.skip("BSMAPPER_DEMO_SONG is not set")
    from bsmapper.lighting.demo import build_demo_map

    out = build_demo_map(Path(song), tmp_path / "map", style="v3")
    back, report = read_map_report(out)
    assert (out / "song.egg").exists()
    assert back.environment == "WeaveEnvironment"
    assert back.lightshow.event_box_groups or back.difficulties[0].lightshow.event_box_groups
