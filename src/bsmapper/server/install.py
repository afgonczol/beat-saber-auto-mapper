"""Copy a job's finished map into Beat Saber's CustomLevels (``~/.bsmapper/config.toml``).

Used by ``POST /api/jobs/{id}/install`` and by jobs queued with "install when done".
"""

from __future__ import annotations

import shutil
from pathlib import Path

from bsmapper.generation import install_map, load_config, safe_folder_name, uninstall_map
from bsmapper.generation.install import config_path
from bsmapper.server.models import InstallResult
from bsmapper.server.runner import JobInput, JobResult

CHROMA_MISSING = (
    "This map uses Chroma lighting, but Chroma.dll was not found in {dll}. Without the "
    "Chroma mod the lights play in the default colours."
)


class MapNotReady(RuntimeError):
    """The job has no packaged map folder to install."""


class InstallUnavailable(RuntimeError):
    """CustomLevels was not found or the config file is invalid."""


def install_job(job: JobInput, result: JobResult) -> InstallResult:
    """Install ``result`` (the map in ``job.work_dir/map``); again replaces the earlier copy."""
    map_dir = job.work_dir / "map"
    if not (map_dir / "Info.dat").is_file():
        raise MapNotReady("map not ready")
    try:
        config = load_config()
    except (OSError, ValueError) as exc:
        raise InstallUnavailable(f"bad config {config_path()}: {exc}") from exc
    beatmap = result.beatmap
    # install_map names the installed folder after the source folder, so stage a copy.
    name = safe_folder_name(beatmap.song_name, beatmap.song_author)
    staged = job.work_dir / "install" / name
    shutil.rmtree(staged.parent, ignore_errors=True)
    try:
        shutil.copytree(map_dir, staged)
        dest = install_map(staged, config)
    except FileNotFoundError as exc:
        raise InstallUnavailable(str(exc)) from exc
    finally:
        shutil.rmtree(staged.parent, ignore_errors=True)
    warnings = []
    shows = [beatmap.lightshow, *(d.lightshow for d in beatmap.difficulties)]
    chroma = job.options.lighting == "chroma" or any(s is not None and s.uses_chroma for s in shows)
    if chroma and not config.chroma_installed():
        warnings.append(CHROMA_MISSING.format(dll=config.chroma_dll.parent))
    return InstallResult(path=str(dest), warnings=warnings)


def uninstall_path(path: str) -> None:
    """Delete an earlier install of a job's map (only a bsmapper map inside CustomLevels)."""
    uninstall_map(Path(path))
