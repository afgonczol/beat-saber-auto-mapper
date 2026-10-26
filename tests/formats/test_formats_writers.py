import json
from pathlib import Path

import pytest

from bsmapper.formats import read_map, write_map
from bsmapper.schema import (
    BasicEvent,
    Beatmap,
    BpmChange,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Lightshow,
    Note,
    Obstacle,
    RotationEvent,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "maps"


def load(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def small_beatmap(**kwargs) -> Beatmap:
    dmap = DifficultyMap(
        Characteristic.STANDARD,
        Difficulty.EXPERT,
        notes=[
            Note(1, 1, 0, Color.RED, CutDirection.DOWN),
            Note(2, 2, 0, Color.BLUE, CutDirection.DOWN),
        ],
        obstacles=[
            Obstacle(4, 1, 0, 0, 1, 5),
            Obstacle(6, 1, 0, 2, 4, 3),
            Obstacle(8, 1, 3, 1, 1, 2),
        ],
    )
    return Beatmap(song_name="Small", bpm=100, difficulties=[dmap], **kwargs)


def test_v3_is_default_and_uses_v2_info(tmp_path):
    write_map(small_beatmap(), tmp_path)
    assert load(tmp_path / "Info.dat")["_version"] == "2.0.0"
    data = load(tmp_path / "ExpertStandard.dat")
    assert data["version"] == "3.3.0"
    assert data["colorNotes"][0] == {"b": 1, "x": 1, "y": 0, "c": 0, "d": 1, "a": 0}
    assert data["bpmEvents"] == []


def test_v2_wall_types(tmp_path):
    write_map(small_beatmap(), tmp_path, "v2")
    walls = load(tmp_path / "ExpertStandard.dat")["_obstacles"]
    assert [w["_type"] for w in walls] == [0, 1, 2]
    assert (walls[2]["_lineLayer"], walls[2]["_height"]) == (1, 2)
    assert "_lineLayer" not in walls[0]


def test_v2_large_rotations_are_split(tmp_path):
    bm = small_beatmap()
    bm.difficulties[0].characteristic = Characteristic.DEGREE_360
    bm.difficulties[0].rotations = [RotationEvent(1, 90)]
    write_map(bm, tmp_path, "v2")
    events = load(tmp_path / "Expert360Degree.dat")["_events"]
    assert [(e["_type"], e["_value"]) for e in events] == [(14, 7), (14, 5)]  # +60, +30
    assert read_map(tmp_path).difficulties[0].rotations == [
        RotationEvent(1, 60),
        RotationEvent(1, 30),
    ]


def test_v3_fx_events_become_indices(tmp_path):
    write_map(read_map(FIXTURES / "v3"), tmp_path, "v3")
    data = load(tmp_path / "ExpertPlusStandard.dat")
    assert data["vfxEventBoxGroups"][0]["e"][0]["l"] == [0, 1]
    assert data["_fxEventsCollection"]["_fl"][1] == {"b": 2, "p": 0, "i": 1, "v": 100}


def test_v3_fx_int_events_mirror_float_events(tmp_path):
    # An int vfx group (e.g. in TheWeekndEnvironment) reads its indices from _il; an
    # empty _il made the game fail to load the map with ArgumentOutOfRangeException.
    write_map(read_map(FIXTURES / "v3"), tmp_path, "v3")
    collection = load(tmp_path / "ExpertPlusStandard.dat")["_fxEventsCollection"]
    assert len(collection["_il"]) == len(collection["_fl"]) > 0
    assert collection["_il"][1] == {"b": 2, "p": 0, "v": 100}
    assert all(isinstance(e["v"], int) for e in collection["_il"])


def test_v3_fx_index_only_in_int_events_is_read(tmp_path):
    src = FIXTURES / "v3"
    for f in src.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    path = tmp_path / "ExpertPlus.dat"
    data = load(path)
    data["_fxEventsCollection"]["_il"] = [{"b": 0, "p": 0, "v": 0}] * len(
        data["_fxEventsCollection"]["_fl"]
    ) + [{"b": 4, "p": 0, "v": 1}]
    data["vfxEventBoxGroups"][0]["e"][0]["l"].append(len(data["_fxEventsCollection"]["_fl"]))
    path.write_text(json.dumps(data), "utf-8")
    dmap = next(d for d in read_map(tmp_path).difficulties if d.difficulty.name == "EXPERT_PLUS")
    fx = next(g for g in dmap.lightshow.event_box_groups if g.kind == "fx")
    assert fx.boxes[0]["l"][-1] == {"b": 4.0, "p": 0, "i": 0, "v": 1.0}


def test_v4_layout(tmp_path):
    write_map(read_map(FIXTURES / "v3"), tmp_path, "v4", song_duration=30)
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {
        "Info.dat",
        "BPMInfo.dat",
        "Lightshow.dat",
        "ExpertStandard.dat",
        "ExpertPlusStandard.dat",
        "Normal90Degree.dat",
        "Normal90Degree.lightshow.dat",
    }
    info = load(tmp_path / "Info.dat")
    assert info["version"] == "4.0.0"
    assert info["audio"]["songDuration"] == 30
    assert info["environmentNames"] == ["WeaveEnvironment", "GlassDesertEnvironment"]
    files = {
        (d["difficulty"], d["lightshowDataFilename"], d["environmentNameIdx"])
        for d in info["difficultyBeatmaps"]
    }
    assert files == {
        ("Expert", "Lightshow.dat", 0),
        ("ExpertPlus", "Lightshow.dat", 0),
        ("Normal", "Normal90Degree.lightshow.dat", 1),
    }
    # Note data is deduplicated: arc heads/tails reuse colorNotesData entries.
    plus = load(tmp_path / "ExpertPlusStandard.dat")
    assert len(plus["colorNotesData"]) < len(plus["colorNotes"]) + 2 * len(plus["arcs"])
    # Rotations are stored as absolute lanes on the objects.
    normal = load(tmp_path / "Normal90Degree.dat")
    assert [n["r"] for n in normal["colorNotes"]] == [15, 15, 0]


def test_v4_bpm_regions(tmp_path):
    bm = small_beatmap(bpm_changes=[BpmChange(8, 150)])
    write_map(bm, tmp_path, "v4", song_duration=10)
    audio = load(tmp_path / "BPMInfo.dat")
    assert audio["songFrequency"] == 44100
    # 8 beats at 100 BPM = 4.8 s; then 150 BPM until 10 s = 13 more beats.
    assert audio["bpmData"] == [
        {"si": 0, "ei": 211680, "sb": 0, "eb": 8},
        {"si": 211680, "ei": 441000, "sb": 8, "eb": 21},
    ]
    assert read_map(tmp_path).bpm_changes == [BpmChange(8, 150)]


@pytest.mark.parametrize("bpm", [97.5, 128.125, 173.913])
def test_v4_fractional_bpm_survives_sample_rounding(bpm, tmp_path):
    bm = small_beatmap(bpm_changes=[BpmChange(2, bpm), BpmChange(10, 100)])
    write_map(bm, tmp_path, "v4", song_duration=20)
    assert read_map(tmp_path).bpm_changes == bm.bpm_changes


def test_v4_short_bpm_regions_keep_timing(tmp_path):
    """A 1-beat region cannot pin a 3-decimal BPM at sample resolution, but timing holds."""
    bm = small_beatmap(
        bpm_changes=[BpmChange(2, 128.125), BpmChange(3, 173.913), BpmChange(4, 100)]
    )
    write_map(bm, tmp_path, "v4", song_duration=10)
    back = read_map(tmp_path)
    for b in (3, 4, 8):
        assert back.beat_to_seconds(b) == pytest.approx(bm.beat_to_seconds(b), abs=1 / 44100)


def test_chroma_suggestion(tmp_path):
    bm = small_beatmap()
    bm.lightshow = Lightshow(
        basic_events=[BasicEvent(1, 0, 1, color=(1, 0, 0, 1))], uses_chroma=True
    )
    for version in ("v2", "v3", "v4"):
        out = tmp_path / version
        write_map(bm, out, version)
        info = load(out / "Info.dat")
        if version == "v4":
            assert info["difficultyBeatmaps"][0]["customData"]["suggestions"] == ["Chroma"]
        else:
            entry = info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]
            assert entry["_customData"]["_suggestions"] == ["Chroma"]
        assert read_map(out).lightshow == bm.lightshow


def test_unknown_version_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_map(small_beatmap(), tmp_path, "v5")  # type: ignore[arg-type]
