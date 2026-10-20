import shutil
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path

import pytest

from bsmapper.formats import (
    audio_duration,
    package_map,
    read_map,
    read_tags,
    write_placeholder_cover,
    zip_map,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "maps"


def test_placeholder_cover_is_valid_png(tmp_path):
    path = write_placeholder_cover(tmp_path / "cover.png", "My Song", size=32)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (32, 32)
    idat_len = struct.unpack(">I", data[33:37])[0]
    pixels = zlib.decompress(data[41 : 41 + idat_len])
    assert len(pixels) == 32 * (1 + 32 * 3)
    # Same seed, same image; different seed, different colours.
    again = write_placeholder_cover(tmp_path / "again.png", "My Song", size=32)
    other = write_placeholder_cover(tmp_path / "other.png", "Other", size=32)
    assert again.read_bytes() == data != other.read_bytes()


def test_read_tags_falls_back_to_filename(tmp_path):
    path = tmp_path / "Some Artist - Some Title.mp3"
    path.write_bytes(b"not audio")
    assert read_tags(path).artist == "Some Artist"
    assert read_tags(path).title == "Some Title"
    plain = tmp_path / "plain.wav"
    plain.write_bytes(b"")
    assert (read_tags(plain).title, read_tags(plain).artist) == ("plain", "")
    assert audio_duration(plain) is None


def test_zip_map_is_flat(tmp_path):
    archive = zip_map(FIXTURES / "v3", tmp_path / "out" / "map.zip")
    with zipfile.ZipFile(archive) as zf:
        assert sorted(zf.namelist()) == sorted(p.name for p in (FIXTURES / "v3").iterdir())
    assert read_map(archive) == read_map(FIXTURES / "v3")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
@pytest.mark.parametrize("version", ["v3", "v4"])
def test_package_map_end_to_end(version, tmp_path):
    source = tmp_path / "Test Artist - Test Song.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            str(source),
        ],
        check=True,
    )
    beatmap = read_map(FIXTURES / "v3")
    beatmap.song_name = beatmap.song_author = ""
    out = tmp_path / "map"
    archive = package_map(beatmap, source, out, version=version, zip_path=tmp_path / "map.zip")

    assert (out / "song.egg").read_bytes()[:4] == b"OggS"
    assert audio_duration(out / "song.egg") == pytest.approx(3, abs=0.1)
    assert (out / "cover.png").is_file()
    packaged = read_map(archive)
    assert (packaged.song_name, packaged.song_author) == ("Test Song", "Test Artist")
    assert (packaged.audio_path, packaged.cover_path) == ("song.egg", "cover.png")
    expected = read_map(FIXTURES / "v3").difficulties
    # v4 turns the late rotation of the 90Degree fixture into an early one; compare notes there.
    assert [d.notes for d in packaged.difficulties] == [d.notes for d in expected]
    if version == "v3":
        assert packaged.difficulties == expected
