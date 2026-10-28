"""Installing a generated map into the game, and the user config that says where.

``~/.bsmapper/config.toml`` (all keys optional; ``BSMAPPER_CONFIG`` points elsewhere)::

    [game]
    beat_saber_dir = 'C:\\Program Files (x86)\\Steam\\steamapps\\common\\Beat Saber'
    custom_levels = 'D:\\Games\\Beat Saber\\Beat Saber_Data\\CustomLevels'  # overrides the above

``custom_levels`` defaults to ``<beat_saber_dir>/Beat Saber_Data/CustomLevels``. The
Chroma check looks for ``<beat_saber_dir>/Plugins/Chroma.dll``.
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

from bsmapper.formats import MapFormatError, ReadReport
from bsmapper.formats import info as info_mod

DEFAULT_BEAT_SABER_DIR = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber")
CONFIG_ENV = "BSMAPPER_CONFIG"
MAPPER_PREFIX = "bsmapper"  # Info.dat author of every map we write


def config_path() -> Path:
    env = os.environ.get(CONFIG_ENV)
    return Path(env) if env else Path.home() / ".bsmapper" / "config.toml"


@dataclass(frozen=True, slots=True)
class GameConfig:
    beat_saber_dir: Path = DEFAULT_BEAT_SABER_DIR
    custom_levels: Path | None = None  # None: derived from beat_saber_dir

    @property
    def levels_dir(self) -> Path:
        if self.custom_levels is not None:
            return self.custom_levels
        return self.beat_saber_dir / "Beat Saber_Data" / "CustomLevels"

    @property
    def chroma_dll(self) -> Path:
        return self.beat_saber_dir / "Plugins" / "Chroma.dll"

    def chroma_installed(self) -> bool:
        return self.chroma_dll.is_file()


def load_config(path: Path | None = None) -> GameConfig:
    """Read the ``[game]`` table of the config file; missing file or keys give defaults."""
    path = path or config_path()
    if not path.is_file():
        return GameConfig()
    with path.open("rb") as f:
        data = tomllib.load(f)
    game = data.get("game", {})
    if not isinstance(game, dict):
        raise ValueError(f"{path}: [game] must be a table")
    bs_dir = Path(game["beat_saber_dir"]) if game.get("beat_saber_dir") else DEFAULT_BEAT_SABER_DIR
    levels = Path(game["custom_levels"]) if game.get("custom_levels") else None
    # CustomLevels sits in <game>/Beat Saber_Data/, so Plugins/ can be found from it.
    derive = levels is not None and not game.get("beat_saber_dir")
    if derive and levels.parent.name == "Beat Saber_Data":
        bs_dir = levels.parent.parent
    return GameConfig(bs_dir, levels)


def _is_ours(map_dir: Path) -> bool:
    """Is ``map_dir`` a map bsmapper wrote (its Info.dat author starts with "bsmapper")?"""
    info = map_dir / "Info.dat"
    if not info.is_file():
        return False
    try:
        data = json.loads(info.read_text(encoding="utf-8"))
        parsed = info_mod.parse(data, ReadReport())
    except (OSError, ValueError, MapFormatError):
        return False
    return parsed.beatmap.mapper.startswith(MAPPER_PREFIX)


def install_map(map_dir: Path, config: GameConfig | None = None) -> Path:
    """Copy a map folder into CustomLevels and return the installed folder.

    A previous install of a bsmapper map with the same folder name is replaced; a
    folder of the same name that is not ours is left alone and we install next to it.
    """
    config = config or load_config()
    levels = config.levels_dir
    if not levels.is_dir():
        raise FileNotFoundError(
            f"CustomLevels folder not found: {levels}. Set [game] custom_levels in {config_path()}"
        )
    dest = levels / map_dir.name
    n = 2
    while dest.exists() and not _is_ours(dest):
        dest = levels / f"{map_dir.name} ({n})"
        n += 1
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(map_dir, dest)
    return dest


def uninstall_map(installed: Path, config: GameConfig | None = None) -> bool:
    """Delete an installed map folder if it is a bsmapper map directly inside CustomLevels.

    Returns whether it was deleted. Maps someone else made are never touched.
    """
    config = config or load_config()
    installed = Path(installed)
    try:
        inside = installed.resolve().parent == config.levels_dir.resolve()
    except OSError:
        return False
    if not inside or not _is_ours(installed):
        return False
    shutil.rmtree(installed)
    return True
