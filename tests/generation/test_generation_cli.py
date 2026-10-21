from __future__ import annotations

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

    monkeypatch.setattr(generation, "Pipeline", FakePipeline)  # what the CLI imports
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
