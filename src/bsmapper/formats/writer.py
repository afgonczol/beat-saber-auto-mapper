"""Write a ``schema.Beatmap`` as a v2, v3 or v4 map folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bsmapper.formats import info as info_mod
from bsmapper.formats import v2, v3, v4
from bsmapper.formats.packaging import audio_duration
from bsmapper.interfaces import FormatVersion
from bsmapper.schema import Beatmap, DifficultyMap, Lightshow

INFO_FILENAME = "Info.dat"
LIGHTSHOW_FILENAME = "Lightshow.dat"
AUDIO_DATA_FILENAME = "BPMInfo.dat"


def dump_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def difficulty_filename(dmap: DifficultyMap, suffix: str = ".dat") -> str:
    return f"{dmap.difficulty.value}{dmap.characteristic.value}{suffix}"


def estimate_duration(beatmap: Beatmap) -> float:
    """Song length in seconds from the last timed object, when the audio is unavailable."""
    last = 0.0
    lightshows = [beatmap.lightshow] + [d.lightshow for d in beatmap.difficulties if d.lightshow]
    for dmap in beatmap.difficulties:
        for o in dmap.obstacles:
            last = max(last, o.beat + o.duration)
        for group in (dmap.notes, dmap.bombs, dmap.rotations):
            last = max([last, *(o.beat for o in group)])
        for group in (dmap.arcs, dmap.chains):
            last = max([last, *(o.tail_beat for o in group)])
    for ls in lightshows:
        for group in (ls.basic_events, ls.boost_events, ls.event_box_groups):
            last = max([last, *(e.beat for e in group)])
    return beatmap.beat_to_seconds(last) + 1.0


def write_map(
    beatmap: Beatmap,
    out_dir: Path | str,
    version: FormatVersion = "v3",
    *,
    song_duration: float | None = None,
) -> Path:
    """Write Info.dat and the difficulty files into ``out_dir`` and return it.

    v2/v3 copy the effective lightshow into every difficulty file. v4 writes the shared
    ``Beatmap.lightshow`` once as Lightshow.dat, plus one file per difficulty that has
    its own lightshow, and a BPMInfo.dat audio data file. Audio and cover files are not
    touched here; see ``packaging``.
    """
    if version not in ("v2", "v3", "v4"):
        raise ValueError(f"unknown format version {version!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[info_mod.InfoEntry] = []
    shared_used = False
    for dmap in beatmap.difficulties:
        lightshow: Lightshow = dmap.lightshow if dmap.lightshow is not None else beatmap.lightshow
        filename = difficulty_filename(dmap)
        entry = info_mod.InfoEntry(dmap, filename, uses_chroma=lightshow.uses_chroma)
        if version == "v2":
            data = v2.write(dmap, lightshow, beatmap.bpm_changes)
        elif version == "v3":
            data = v3.write(dmap, lightshow, beatmap.bpm_changes)
        else:
            data = v4.write_beatmap(dmap)
            if dmap.lightshow is None:
                entry.lightshow_filename = LIGHTSHOW_FILENAME
                shared_used = True
            else:
                entry.lightshow_filename = difficulty_filename(dmap, ".lightshow.dat")
                dump_json(v4.write_lightshow(dmap.lightshow), out_dir / entry.lightshow_filename)
        dump_json(data, out_dir / filename)
        entries.append(entry)

    if version == "v4":
        if shared_used:
            dump_json(v4.write_lightshow(beatmap.lightshow), out_dir / LIGHTSHOW_FILENAME)
        if song_duration is None:
            measured = audio_duration(out_dir / info_mod.basename(beatmap.audio_path))
            song_duration = measured if measured else estimate_duration(beatmap)
        dump_json(v4.write_audio_data(beatmap, song_duration), out_dir / AUDIO_DATA_FILENAME)
        info = info_mod.write_v4(beatmap, entries, song_duration, AUDIO_DATA_FILENAME)
    else:
        info = info_mod.write_v2(beatmap, entries)
    dump_json(info, out_dir / INFO_FILENAME)
    return out_dir


def set_song_info(map_dir: Path | str, title: str, artist: str) -> None:
    """Change the song title and artist in a map folder's Info.dat (v2 or v4), in place.

    Only those two fields change; the difficulty files are not rewritten.
    """
    path = Path(map_dir) / INFO_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    if info_mod.is_v4(data):
        song = data.setdefault("song", {})
        song["title"], song["author"] = title, artist
    else:
        data["_songName"], data["_songAuthorName"] = title, artist
    dump_json(data, path)


class MapFolderWriter:
    """``interfaces.MapWriter``: v3 by default, v2 and v4 on request."""

    def write(self, beatmap: Beatmap, out_dir: Path, version: FormatVersion = "v3") -> Path:
        return write_map(beatmap, out_dir, version)
