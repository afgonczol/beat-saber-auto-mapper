"""Rename the song of a finished job: Info.dat, the zip's name and the map in memory.

Runners write the map folder to ``work_dir/map`` (see ``install``), so this works for any
runner without generating anything again.
"""

from __future__ import annotations

import dataclasses

from bsmapper.formats import set_song_info, zip_map
from bsmapper.generation import safe_folder_name
from bsmapper.server.install import MapNotReady
from bsmapper.server.runner import JobInput, JobResult


def retitle(job: JobInput, result: JobResult, title: str, artist: str) -> JobResult:
    """``result`` with the song called ``title`` by ``artist``, its folder and zip updated."""
    map_dir = job.work_dir / "map"
    if not (map_dir / "Info.dat").is_file():
        raise MapNotReady("map not ready")
    set_song_info(map_dir, title, artist)
    zip_path = result.zip_path.with_name(f"{safe_folder_name(title, artist)}.zip")
    if result.zip_path != zip_path:
        result.zip_path.unlink(missing_ok=True)
    zip_map(map_dir, zip_path)
    beatmap = dataclasses.replace(result.beatmap, song_name=title, song_author=artist)
    return dataclasses.replace(result, beatmap=beatmap, zip_path=zip_path)
