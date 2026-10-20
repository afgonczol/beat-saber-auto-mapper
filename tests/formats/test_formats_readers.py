import json
import shutil
import zipfile
from pathlib import Path

import pytest

from bsmapper.formats import MapFormatError, read_map_report
from bsmapper.formats.reader import editor_to_official, normalize_bpm_changes
from bsmapper.schema import (
    Arc,
    BpmChange,
    Chain,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    Note,
    Obstacle,
    RotationEvent,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "maps"


def dmap_of(beatmap, characteristic, difficulty):
    return next(
        d
        for d in beatmap.difficulties
        if d.characteristic == characteristic and d.difficulty == difficulty
    )


def test_v2_metadata_and_objects():
    bm, report = read_map_report(FIXTURES / "v2")
    assert (bm.song_name, bm.song_sub_name, bm.song_author) == (
        "Fixture Two",
        "v2 test map",
        "Nobody",
    )
    assert (bm.mapper, bm.bpm, bm.environment) == ("hand-written", 120, "BigMirrorEnvironment")
    assert (bm.audio_path, bm.cover_path, bm.preview_start) == ("song.egg", "cover.png", 12)
    assert bm.bpm_changes == [BpmChange(16, 150)]
    assert report.bpm_source == "events"

    expert = dmap_of(bm, Characteristic.STANDARD, Difficulty.EXPERT)
    assert (expert.njs, expert.njs_offset, expert.label) == (16, -0.25, "Fixture Expert")
    assert expert.notes[0] == Note(4, 1, 0, Color.RED, CutDirection.DOWN)
    assert expert.notes[6].beat == 7.333333
    assert len(expert.bombs) == 1
    assert expert.obstacles == [
        Obstacle(10, 2, 0, 0, 1, 5),  # type 0: full height
        Obstacle(13, 1.5, 0, 2, 4, 3),  # type 1: crouch
        Obstacle(16, 1, 3, 1, 1, 2),  # type 2: explicit layer/height
    ]
    assert expert.arcs == [
        Arc(6, Color.RED, 1, 0, CutDirection.DOWN_LEFT, 8, 0, 2, CutDirection.UP_LEFT, 1, 0.5, 1)
    ]
    # The shared lightshow is on the Beatmap when it is the most common one.
    ls = expert.lightshow or bm.lightshow
    assert [e.beat for e in ls.boost_events] == [8, 12]
    assert [e.on for e in ls.boost_events] == [True, False]
    no_float = next(e for e in ls.basic_events if e.type == 12)
    assert no_float.float_value == 1.0  # absent before v2.5.0 means 1
    chroma = next(e for e in ls.basic_events if e.color is not None)
    assert chroma.color == (1.0, 0.5, 0.0, 1.0)
    assert chroma.custom_data == {"lightID": [1, 2, 3]}
    assert ls.uses_chroma and report.chroma


def test_v2_rotations():
    bm, _ = read_map_report(FIXTURES / "v2")
    hard = dmap_of(bm, Characteristic.DEGREE_360, Difficulty.HARD)
    assert hard.rotations == [
        RotationEvent(4, 15, early=True),
        RotationEvent(6, -30, early=True),
        RotationEvent(7, 60, early=False),
    ]


def test_v3_objects_and_noise():
    bm, report = read_map_report(FIXTURES / "v3")
    assert "skipping unsupported difficulty Legacy/Easy" in report.warnings
    assert [(d.characteristic, d.difficulty) for d in bm.difficulties] == [
        (Characteristic.STANDARD, Difficulty.EXPERT),
        (Characteristic.STANDARD, Difficulty.EXPERT_PLUS),
        (Characteristic.DEGREE_90, Difficulty.NORMAL),
    ]
    # {b: 0, m: 128} repeats the Info BPM and is dropped as a no-op.
    assert bm.bpm_changes == [BpmChange(32, 174)]
    plus = dmap_of(bm, Characteristic.STANDARD, Difficulty.EXPERT_PLUS)
    assert plus.notes[2].angle_offset == 15
    assert plus.notes[7].beat == 4.0  # 4.0000000001 is float noise
    assert plus.notes[9].beat == 34.166667
    assert plus.chains == [
        Chain(3.25, Color.BLUE, 3, 0, CutDirection.DOWN_RIGHT, 3.5, 3, 2, 4, 0.8)
    ]


def test_v3_lightshow_sharing_and_boxes():
    bm, _ = read_map_report(FIXTURES / "v3")
    expert = dmap_of(bm, Characteristic.STANDARD, Difficulty.EXPERT)
    plus = dmap_of(bm, Characteristic.STANDARD, Difficulty.EXPERT_PLUS)
    normal = dmap_of(bm, Characteristic.DEGREE_90, Difficulty.NORMAL)
    assert expert.lightshow is None and plus.lightshow is None  # identical -> shared
    assert normal.lightshow is not None and not normal.lightshow.basic_events

    groups = bm.lightshow.event_box_groups
    assert [(g.beat, g.kind) for g in groups] == [
        (2, "color"),
        (2, "rotation"),
        (4, "color"),
        (6, "translation"),
        (6, "fx"),
    ]
    # Missing keys are filled in, so every box has the full v3.3 key set.
    sparse = groups[2].boxes[0]
    assert sparse["f"] == {"c": 0, "f": 2, "p": 0, "t": 2, "r": 0, "n": 0, "s": 0, "l": 0, "d": 0}
    assert sparse["e"][0] == {"b": 0, "i": 0, "c": 0, "s": 1, "f": 0, "sb": 0, "sf": 0}
    # fx events are resolved from _fxEventsCollection and stored inline.
    assert groups[4].boxes[0]["l"] == [
        {"b": 0, "p": 0, "i": 0, "v": 0},
        {"b": 2, "p": 0, "i": 1, "v": 100},
    ]


def test_v3_rotations_and_defaults():
    bm, _ = read_map_report(FIXTURES / "v3")
    normal = dmap_of(bm, Characteristic.DEGREE_90, Difficulty.NORMAL)
    assert normal.rotations == [
        RotationEvent(4, 15, True),
        RotationEvent(6, -30, False),
        RotationEvent(36, 15, True),
    ]
    assert normal.notes[2] == Note(36, 1, 1, Color.RED, CutDirection.UP)  # missing d -> 0


def test_v4_reads_split_files():
    bm, report = read_map_report(FIXTURES / "v4")
    assert report.info_version == "4.0.1"
    assert report.bpm_source == "audio_data"
    assert bm.bpm_changes == [BpmChange(16, 150)]
    assert (bm.environment, bm.mapper, bm.preview_duration) == (
        "TheSecondEnvironment",
        "hand-written",
        8,
    )
    plus = dmap_of(bm, Characteristic.STANDARD, Difficulty.EXPERT_PLUS)
    assert plus.label == "Fixture Plus"
    assert plus.notes[2] == Note(3, 0, 1, Color.RED, CutDirection.LEFT, 30)
    assert plus.arcs == [
        Arc(3, Color.RED, 0, 1, CutDirection.LEFT, 4, 1, 2, CutDirection.UP, 1, 0.75, 2)
    ]
    assert plus.chains == [Chain(3.5, Color.BLUE, 3, 1, CutDirection.RIGHT, 3.75, 3, 2, 3, 0.5)]
    assert plus.lightshow is None  # both difficulties use Lightshow.dat

    ls = bm.lightshow
    assert [(e.type, e.value, e.float_value) for e in ls.basic_events[:2]] == [
        (0, 1, 1),
        (1, 6, 0.8),
    ]
    assert ls.basic_events[3].color == (1, 0, 0, 1)
    assert [(e.beat, e.on) for e in ls.boost_events] == [(8, True), (12, False)]
    color = ls.event_box_groups[0].boxes[0]
    assert [e["i"] for e in color["e"]] == [0, 1, 2]  # instant, transition, extend
    rotation = ls.event_box_groups[1].boxes[0]
    assert rotation["f"]["s"] == 7 and rotation["r"] == 1 and rotation["l"][0]["o"] == 2


def test_v4_rotations_from_lanes():
    bm, _ = read_map_report(FIXTURES / "v4")
    expert = dmap_of(bm, Characteristic.DEGREE_360, Difficulty.EXPERT)
    assert expert.rotations == [
        RotationEvent(4, 15),
        RotationEvent(6, 30),
        RotationEvent(20, -60),
    ]


def test_modded_map_is_flagged_and_cleaned():
    bm, report = read_map_report(FIXTURES / "v2_modded")
    assert report.mapping_extensions and report.noodle and report.chroma
    assert report.needs_mods
    assert report.requirements == ["Mapping Extensions"]
    assert report.dropped["mapping_extensions_note"] == 2
    assert report.dropped["off_grid_note"] == 1
    assert report.dropped["mapping_extensions_obstacle"] == 1
    assert report.dropped["legacy_chroma_event"] == 1
    assert report.dropped["rotation_in_non_rotating_mode"] == 1
    assert "missing difficulty file Missing.dat" in report.warnings
    assert [d.difficulty for d in bm.difficulties] == [Difficulty.EXPERT]


def test_legacy_editor_bpm_changes_are_remapped():
    """MMA2 _BPMChanges leave times in base-BPM beats; the reader warps them."""
    bm, report = read_map_report(FIXTURES / "v2_modded")
    assert report.bpm_source == "custom"
    assert bm.bpm_changes == [BpmChange(8, 200)]
    expert = bm.difficulties[0]
    # File beats 8.5 and 9 at 100 BPM sit 0.3 s and 0.6 s after the change at 200 BPM.
    assert [n.beat for n in expert.notes] == [4, 8, 9, 10, 14]
    assert expert.obstacles == [Obstacle(18, 2, 0, 0, 1, 5)]
    # The audio timing is unchanged by the remap.
    assert bm.beat_to_seconds(14) == pytest.approx(11 * 60 / 100)


def test_editor_to_official():
    assert editor_to_official(100, [BpmChange(8, 200), BpmChange(10, 50)]) == [
        BpmChange(8, 200),
        BpmChange(12, 50),
    ]


def test_normalize_bpm_changes():
    changes = [BpmChange(0, 120), BpmChange(8, 140), BpmChange(8, 150), BpmChange(16, 150)]
    assert normalize_bpm_changes(120, changes) == [BpmChange(8, 150)]


def test_disagreeing_difficulties_are_remapped(tmp_path):
    shutil.copytree(FIXTURES / "v2", tmp_path, dirs_exist_ok=True)
    hard = tmp_path / "Hard360Degree.dat"
    data = json.loads(hard.read_text())
    data["_events"] = [e for e in data["_events"] if e["_type"] != 100]  # no BPM change
    data["_notes"][-1]["_time"] = 18.5  # 9.25 s at a constant 120 BPM
    hard.write_text(json.dumps(data))
    bm, report = read_map_report(tmp_path)
    assert "difficulties disagree on BPM changes; times were remapped" in report.warnings
    remapped = dmap_of(bm, Characteristic.DEGREE_360, Difficulty.HARD).notes[-1].beat
    assert bm.beat_to_seconds(remapped) == pytest.approx(9.25)


def test_read_from_zip_with_nested_folder(tmp_path):
    archive = tmp_path / "map.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in (FIXTURES / "v4").iterdir():
            zf.write(path, f"Some Folder/{path.name.lower()}")
    bm, _ = read_map_report(archive)
    assert bm == read_map_report(FIXTURES / "v4")[0]


def test_missing_info_raises(tmp_path):
    with pytest.raises(MapFormatError):
        read_map_report(tmp_path)
    (tmp_path / "Info.dat").write_text("{not json")
    with pytest.raises(MapFormatError):
        read_map_report(tmp_path)
