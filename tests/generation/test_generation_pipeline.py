from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from gen_synthetic import FakeAnalyzer

from bsmapper.formats import read_map, write_map
from bsmapper.generation import GenerateOptions, Pipeline, replace_difficulty, safe_folder_name
from bsmapper.rules import RuleChecker
from bsmapper.schema import Characteristic, Difficulty

DIFFS = sorted(Difficulty, key=lambda d: d.rank)


def song_file(tmp_path: Path) -> Path:
    path = tmp_path / "Some Artist - Some Song.mp3"
    path.write_bytes(b"not decoded; the fake analyzer ignores it")
    return path


@pytest.mark.parametrize("offset", [0.0, 0.3])
def test_build_lights_and_checks_every_difficulty(tmp_path: Path, offset: float) -> None:
    pipeline = Pipeline(analyzer=FakeAnalyzer(offset_s=offset))
    song = pipeline.analyze(song_file(tmp_path))
    assert (song.title, song.artist) == ("Some Song", "Some Artist")
    if offset:
        assert song.lead_in_s == pytest.approx(2.0 - offset)
        assert song.song_analysis.beat_times[4] == pytest.approx(2.0)

    stages = []
    opts = GenerateOptions(modes=[Characteristic.STANDARD, Characteristic.DEGREE_90])
    beatmap, results = pipeline.build(song, opts, lambda s, f, m: stages.append((s, f)))
    assert [s for s, _ in stages][:1] == ["generate"] and "lighting" in {s for s, _ in stages}
    fractions = [f for _, f in stages]
    assert fractions == sorted(fractions)

    assert beatmap.mapper == "bsmapper (stub)"
    assert beatmap.environment == "WeaveEnvironment"
    assert beatmap.lightshow.event_box_groups
    keys = [(d.characteristic, d.difficulty) for d in beatmap.difficulties]
    assert keys == [(m, d) for m in opts.modes for d in DIFFS]
    for mode in opts.modes:
        nps = [r.nps for r in results if r.dmap.characteristic == mode]
        assert all(a < b for a, b in zip(nps, nps[1:], strict=False)), nps
    for r in results:
        assert r.report.errors == []
        assert not [i for i in RuleChecker().check(r.dmap, beatmap) if i.severity == "error"]
        assert r.dmap.njs > 0
    assert 5.0 <= beatmap.preview_start <= song.song_analysis.duration_s


def test_lower_difficulty_is_regenerated_below_the_harder_one(tmp_path: Path) -> None:
    pipeline = Pipeline(analyzer=FakeAnalyzer())
    # Make Hard as dense as Expert: the pipeline must pull it back under.
    pipeline.generator.profiles = dict(pipeline.generator.profiles)
    pipeline.generator.profiles[Difficulty.HARD] = pipeline.generator.profiles[Difficulty.EXPERT]
    song = pipeline.analyze(song_file(tmp_path))
    opts = GenerateOptions(difficulties=[Difficulty.HARD, Difficulty.EXPERT])
    _, results = pipeline.build(song, opts)
    hard, expert = results
    assert hard.attempts == 2
    assert hard.nps < 0.9 * expert.nps


def test_replace_difficulty_keeps_order(tmp_path: Path) -> None:
    pipeline = Pipeline(analyzer=FakeAnalyzer(duration=20.0))
    song = pipeline.analyze(song_file(tmp_path))
    beatmap, _ = pipeline.build(song, GenerateOptions(difficulties=[Difficulty.EASY]))
    res = pipeline.generate_difficulty(
        song, beatmap, Characteristic.STANDARD, Difficulty.EXPERT, GenerateOptions(), seed=4
    )
    new = replace_difficulty(beatmap, res.dmap)
    assert [d.difficulty for d in new.difficulties] == [Difficulty.EASY, Difficulty.EXPERT]
    assert [d.difficulty for d in beatmap.difficulties] == [Difficulty.EASY]


def test_safe_folder_name() -> None:
    assert safe_folder_name('A/B: "C"?', "D*") == "AB C - D"
    assert safe_folder_name("...", "") == "bsmapper map"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_run_packages_a_padded_zip(tmp_path: Path) -> None:
    import soundfile as sf

    sr = 22050
    t = np.arange(int(sr * 30)) / sr
    clicks = (np.sin(2 * np.pi * 880 * t) * (np.mod(t - 0.3, 0.5) < 0.03)).astype(np.float32)
    audio = tmp_path / "clicks.wav"
    sf.write(audio, clicks, sr)

    pipeline = Pipeline(analyzer=FakeAnalyzer(duration=30.0, offset_s=0.3))
    result = pipeline.run(audio, tmp_path / "out", GenerateOptions())
    assert result.zip_path is not None and result.zip_path.is_file()
    names = set(zipfile.ZipFile(result.zip_path).namelist())
    assert {"Info.dat", "song.egg", "cover.png"} <= names
    assert {f"{d.value}Standard.dat" for d in Difficulty} <= names
    beatmap = read_map(result.zip_path)
    assert len(beatmap.difficulties) == 5
    for dmap in beatmap.difficulties:
        assert not [i for i in RuleChecker().check(dmap, beatmap) if i.severity == "error"]
    import mutagen

    egg_len = mutagen.File(result.map_dir / "song.egg").info.length
    assert egg_len == pytest.approx(30.0 + result.song.lead_in_s, abs=0.05)
    # A rerun with fewer difficulties leaves no stale files behind.
    write_map(beatmap, result.map_dir)
    pipeline.run(audio, tmp_path / "out", GenerateOptions(difficulties=[Difficulty.EASY]))
    assert sorted(p.name for p in result.map_dir.glob("*.dat")) == ["EasyStandard.dat", "Info.dat"]
