from __future__ import annotations

import re
from pathlib import Path

import pytest
from gen_synthetic import FakeAnalyzer
from typer.testing import CliRunner

from bsmapper import cli, generation
from bsmapper.schema import Characteristic, Difficulty

runner = CliRunner()


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path: Path):
    """The real Pipeline with a fake analyzer and packaging that only writes the map files."""
    from bsmapper.formats import write_map

    calls = {}

    class FakePipeline(generation.Pipeline):
        def __init__(self, **kw) -> None:
            super().__init__(analyzer=FakeAnalyzer(duration=20.0))

        def package(self, song, beatmap, map_dir, zip_path=None, version="v3"):
            calls["beatmap"] = beatmap
            write_map(beatmap, map_dir)
            if zip_path:
                zip_path.write_bytes(b"zip")
            return zip_path or map_dir

    def fake_make_pipeline(generator="auto", lighting="auto", checkpoint=None, device=None, **kw):
        calls["setup"] = (generator, lighting, checkpoint)
        pipeline = FakePipeline()
        pipeline.setup = generation.GeneratorSetup("stub", "rules", None, "test")
        return pipeline

    monkeypatch.setattr(generation, "make_pipeline", fake_make_pipeline)  # what the CLI imports
    monkeypatch.setenv("BSMAPPER_CONFIG", str(tmp_path / "config.toml"))
    return calls


def test_parse_lists() -> None:
    assert cli._parse_enum_list("standard, 360 ,90", Characteristic, cli._MODE_ALIASES, "m") == [
        Characteristic.STANDARD,
        Characteristic.DEGREE_360,
        Characteristic.DEGREE_90,
    ]
    assert cli._parse_enum_list("Expert+,easy", Difficulty, cli._DIFF_ALIASES, "d") == [
        Difficulty.EXPERT_PLUS,
        Difficulty.EASY,
    ]
    with pytest.raises(Exception, match="unknown value"):
        cli._parse_enum_list("Insane", Difficulty, cli._DIFF_ALIASES, "d")


def test_generate_and_install(fake_pipeline, tmp_path: Path) -> None:
    song = tmp_path / "Artist - Title.mp3"
    song.write_bytes(b"x")
    levels = tmp_path / "Beat Saber" / "Beat Saber_Data" / "CustomLevels"
    levels.mkdir(parents=True)
    (tmp_path / "config.toml").write_text(
        f"[game]\nbeat_saber_dir = '{(tmp_path / 'Beat Saber').as_posix()}'\n", encoding="utf-8"
    )
    args = ["generate", str(song), "--out", str(tmp_path / "out"), "--modes", "Standard,360"]
    result = runner.invoke(cli.app, [*args, "--difficulties", "Hard,Expert+", "--install"])
    assert result.exit_code == 0, result.output
    assert "ExpertPlus" in result.output and "Installed to" in result.output
    assert (levels / "Title - Artist" / "Info.dat").is_file()
    assert (tmp_path / "out" / "Title - Artist.zip").is_file()
    diffs = [(d.characteristic, d.difficulty) for d in fake_pipeline["beatmap"].difficulties]
    assert (Characteristic.DEGREE_360, Difficulty.EXPERT_PLUS) in diffs and len(diffs) == 4
    assert "Chroma" not in result.output
    assert "errors/100" in result.output and "stub generator" in result.output
    assert fake_pipeline["setup"] == ("auto", "auto", None)


def test_title_and_artist(fake_pipeline, tmp_path: Path) -> None:
    song = tmp_path / "track01 (Official Video).mp3"
    song.write_bytes(b"x")
    out = tmp_path / "out"
    args = ["generate", str(song), "--out", str(out), "--difficulties", "Expert"]
    result = runner.invoke(cli.app, [*args, "--title", "Real", "--artist", "Band"])
    assert result.exit_code == 0, result.output
    beatmap = fake_pipeline["beatmap"]
    assert (beatmap.song_name, beatmap.song_author) == ("Real", "Band")
    assert (out / "Real - Band.zip").is_file()
    other = tmp_path / "other.mp3"
    other.write_bytes(b"x")
    result = runner.invoke(cli.app, ["generate", str(song), str(other), "--title", "Real"])
    assert result.exit_code != 0 and "single song" in result.output


def test_jump_options(fake_pipeline, tmp_path: Path) -> None:
    from bsmapper.generation import JUMP_TABLE

    assert cli._parse_jump_option("0.8", "--reaction-time", 0.3, 2.0) == (0.8, {})
    assert cli._parse_jump_option("Hard=0.9, Expert+=0.75", "x", 0.3, 2.0) == (
        None,
        {Difficulty.HARD: 0.9, Difficulty.EXPERT_PLUS: 0.75},
    )
    with pytest.raises(Exception, match="outside"):
        cli._parse_jump_option("5", "--reaction-time", 0.3, 2.0)
    song = tmp_path / "s.mp3"
    song.write_bytes(b"x")
    args = ["generate", str(song), "--out", str(tmp_path / "out"), "--no-zip"]
    args += ["--difficulties", "Hard,Expert,Expert+", "--reaction-time", "0.9", "--njs", "Hard=14"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    by_diff = {d.difficulty: d for d in fake_pipeline["beatmap"].difficulties}
    bpm = fake_pipeline["beatmap"].bpm
    from bsmapper.generation.njs import reaction_time

    for diff in (Difficulty.EXPERT, Difficulty.EXPERT_PLUS):
        d = by_diff[diff]
        assert reaction_time(d.njs, d.njs_offset, bpm) == pytest.approx(0.9, abs=0.02)
    hard = by_diff[Difficulty.HARD]
    assert hard.njs == 14.0
    rt = reaction_time(hard.njs, hard.njs_offset, bpm)
    assert rt == pytest.approx(JUMP_TABLE[Difficulty.HARD].reaction_s, abs=0.02)


def test_intensity_option_and_config_default(fake_pipeline, tmp_path: Path) -> None:
    song = tmp_path / "s.mp3"
    song.write_bytes(b"x")
    args = ["generate", str(song), "--out", str(tmp_path / "out"), "--no-zip"]
    args += ["--difficulties", "Hard"]
    (tmp_path / "config.toml").write_text("[generate]\nintensity = 0.8\n", encoding="utf-8")
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "Intensity: 0.80" in result.output
    result = runner.invoke(cli.app, [*args, "--intensity", "0.3"])
    assert "Intensity: 0.30, repetition: learned" in result.output
    result = runner.invoke(cli.app, [*args, "--repetition", "0.9"])
    assert "repetition: 0.90" in result.output
    assert runner.invoke(cli.app, [*args, "--intensity", "2"]).exit_code != 0


def test_arcs_and_chains_options_and_config(fake_pipeline, tmp_path: Path) -> None:
    song = tmp_path / "s.mp3"
    song.write_bytes(b"x")
    args = ["generate", str(song), "--out", str(tmp_path / "out"), "--no-zip"]
    args += ["--difficulties", "Hard"]
    result = runner.invoke(cli.app, args)
    assert "Arcs: on from Hard, chains: on from Expert" in result.output
    (tmp_path / "config.toml").write_text("[generate]\narcs = false\n", encoding="utf-8")
    result = runner.invoke(cli.app, args)
    assert "Arcs: off, chains: on from Expert" in result.output
    result = runner.invoke(cli.app, [*args, "--arcs", "--no-chains"])
    assert "Arcs: on, chains: off" in result.output
    (tmp_path / "config.toml").write_text("[generate]\narcs = 1\n", encoding="utf-8")
    assert runner.invoke(cli.app, args).exit_code != 0


def test_chroma_warning(fake_pipeline, tmp_path: Path) -> None:
    song = tmp_path / "s.mp3"
    song.write_bytes(b"x")
    out = ["--out", str(tmp_path / "out"), "--difficulties", "Easy", "--no-zip"]
    result = runner.invoke(cli.app, ["generate", str(song), "--lighting", "chroma", *out])
    assert result.exit_code == 0, result.output
    assert "Chroma.dll is missing" in result.output
    assert fake_pipeline["beatmap"].lightshow.uses_chroma

    bad = runner.invoke(cli.app, ["generate", str(song), "--lighting", "neon", *out])
    assert bad.exit_code != 0


def test_generate_batch_folder(fake_pipeline, monkeypatch, tmp_path: Path) -> None:
    """A folder of songs: one model load, a failure does not stop the rest, summary at the end."""
    jobs = pytest.importorskip("bsmapper.server.jobs")
    assert cli.AUDIO_SUFFIXES == jobs.AUDIO_SUFFIXES  # the CLI and the web UI take the same files
    loads = []
    fake_make = generation.make_pipeline
    monkeypatch.setattr(
        generation, "make_pipeline", lambda *a, **kw: loads.append(1) or fake_make()
    )
    analyze = generation.Pipeline.analyze

    def flaky_analyze(self, audio, *args):
        if Path(audio).stem.startswith("Broken"):
            raise ValueError("not a song")
        return analyze(self, audio, *args)

    monkeypatch.setattr(generation.Pipeline, "analyze", flaky_analyze)
    songs = tmp_path / "songs"
    songs.mkdir()
    for name in ("B Artist - Second.ogg", "A Artist - First.mp3", "Broken - Song.wav"):
        (songs / name).write_bytes(b"x")
    (songs / "cover.jpg").write_bytes(b"x")
    extra = tmp_path / "C Artist - Third.flac"
    extra.write_bytes(b"x")
    levels = tmp_path / "levels"
    levels.mkdir()
    (tmp_path / "config.toml").write_text(
        f"[game]\ncustom_levels = '{levels.as_posix()}'\n", encoding="utf-8"
    )
    args = ["generate", str(songs), str(extra), str(songs / "A Artist - First.mp3")]
    args += ["--out", str(tmp_path / "out"), "--difficulties", "Expert", "--install"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1, result.output  # one song failed
    assert len(loads) == 1
    assert "Queued 4 songs" in result.output  # the repeated file counts once
    order = [result.output.index(f"] {n}") for n in ("A Artist", "B Artist", "Broken", "C Artist")]
    assert order == sorted(order)
    assert "Summary: 3 of 4 songs done, 1 failed" in result.output
    assert "ValueError: not a song" in result.output
    for title, artist in (("First", "A Artist"), ("Second", "B Artist"), ("Third", "C Artist")):
        assert (levels / f"{title} - {artist}" / "Info.dat").is_file()
    summary = result.output[result.output.index("Summary:") :]
    assert summary.count(" done ") == 3 and summary.count("FAILED") == 1
    assert re.search(r"1 difficulty, \d+ notes, installed as First - A Artist", summary)


def test_generate_rejects_missing_and_empty_inputs(fake_pipeline, tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["generate", str(tmp_path / "nope.mp3")])
    assert result.exit_code != 0 and "not a file or folder" in result.output
    (tmp_path / "empty").mkdir()
    result = runner.invoke(cli.app, ["generate", str(tmp_path / "empty")])
    assert result.exit_code != 0 and "no audio files" in result.output
