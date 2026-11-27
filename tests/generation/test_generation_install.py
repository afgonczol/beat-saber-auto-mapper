from __future__ import annotations

from pathlib import Path

import pytest

from bsmapper.formats import write_map
from bsmapper.generation import GameConfig, install_map, load_config, uninstall_map
from bsmapper.generation.install import DEFAULT_BEAT_SABER_DIR
from bsmapper.schema import Beatmap


def make_map(folder: Path, mapper: str) -> Path:
    write_map(Beatmap(song_name="Song", bpm=120.0, mapper=mapper), folder)
    (folder / "song.egg").write_bytes(b"egg")
    return folder


def test_config_defaults_and_overrides(tmp_path: Path, monkeypatch) -> None:
    missing = load_config(tmp_path / "nope.toml")
    assert missing.beat_saber_dir == DEFAULT_BEAT_SABER_DIR
    assert missing.levels_dir == DEFAULT_BEAT_SABER_DIR / "Beat Saber_Data" / "CustomLevels"

    cfg = tmp_path / "config.toml"
    cfg.write_text("[game]\nbeat_saber_dir = 'D:/Games/Beat Saber'\n", encoding="utf-8")
    monkeypatch.setenv("BSMAPPER_CONFIG", str(cfg))
    game = load_config()
    assert game.levels_dir == Path("D:/Games/Beat Saber/Beat Saber_Data/CustomLevels")
    assert game.chroma_dll == Path("D:/Games/Beat Saber/Plugins/Chroma.dll")

    cfg.write_text(
        "[game]\ncustom_levels = 'E:/BS/Beat Saber_Data/CustomLevels'\n", encoding="utf-8"
    )
    game = load_config()
    assert game.levels_dir == Path("E:/BS/Beat Saber_Data/CustomLevels")
    assert game.beat_saber_dir == Path("E:/BS")  # so Plugins/ is found next to it


def test_chroma_detection(tmp_path: Path) -> None:
    game = GameConfig(beat_saber_dir=tmp_path)
    assert not game.chroma_installed()
    (tmp_path / "Plugins").mkdir()
    (tmp_path / "Plugins" / "Chroma.dll").write_bytes(b"")
    assert game.chroma_installed()


def test_install_replaces_our_own_map_but_not_others(tmp_path: Path) -> None:
    levels = tmp_path / "CustomLevels"
    levels.mkdir()
    game = GameConfig(custom_levels=levels)
    src = make_map(tmp_path / "out" / "Song - Artist", "bsmapper (stub)")

    dest = install_map(src, game)
    assert dest == levels / "Song - Artist" and (dest / "Info.dat").is_file()
    (dest / "stale.dat").write_text("{}")
    assert install_map(src, game) == dest  # reinstall replaces it
    assert not (dest / "stale.dat").exists()

    foreign = make_map(levels / "Other", "SomeHuman")
    other_src = make_map(tmp_path / "out" / "Other", "bsmapper")
    installed = install_map(other_src, game)
    assert installed == levels / "Other (2)"
    assert (foreign / "Info.dat").is_file()


def test_install_needs_the_levels_folder(tmp_path: Path) -> None:
    src = make_map(tmp_path / "m", "bsmapper")
    with pytest.raises(FileNotFoundError, match="CustomLevels"):
        install_map(src, GameConfig(custom_levels=tmp_path / "missing"))


def test_uninstall_only_removes_our_maps_in_custom_levels(tmp_path: Path) -> None:
    levels = tmp_path / "CustomLevels"
    levels.mkdir()
    game = GameConfig(custom_levels=levels)
    ours = install_map(make_map(tmp_path / "src" / "Ours", "bsmapper"), game)
    theirs = make_map(levels / "Theirs", "someone")
    outside = make_map(tmp_path / "Outside", "bsmapper")
    assert not uninstall_map(theirs, game) and theirs.is_dir()
    assert not uninstall_map(outside, game) and outside.is_dir()
    assert not uninstall_map(levels / "Missing", game)
    assert uninstall_map(ours, game) and not ours.exists()
