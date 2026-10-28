"""WS1: v2/v3/v4 map readers and writers, Info.dat, packaging and validation.

beatmap, report = read_map_report("path/to/map_or.zip")
write_map(beatmap, "out/MySong", version="v3")
package_map(beatmap, Path("song.mp3"), Path("out/MySong"), zip_path=Path("out/MySong.zip"))
"""

from bsmapper.formats.packaging import (
    SongTags,
    audio_duration,
    convert_to_egg,
    extract_cover,
    package_map,
    read_tags,
    write_placeholder_cover,
    zip_map,
)
from bsmapper.formats.reader import MapFolderReader, read_map, read_map_report
from bsmapper.formats.report import MapFormatError, ReadReport
from bsmapper.formats.writer import MapFolderWriter, set_song_info, write_map

__all__ = [
    "MapFolderReader",
    "MapFolderWriter",
    "MapFormatError",
    "ReadReport",
    "SongTags",
    "audio_duration",
    "convert_to_egg",
    "extract_cover",
    "package_map",
    "read_map",
    "read_map_report",
    "read_tags",
    "set_song_info",
    "write_map",
    "write_placeholder_cover",
    "zip_map",
]
