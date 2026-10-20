"""Turn a ``Beatmap`` plus a source audio file into a playable map folder and zip.

Steps: convert audio to ``song.egg`` (OGG Vorbis, via ffmpeg), pick a cover (given,
embedded in the audio file, or a generated placeholder), fill missing song
metadata from the audio tags, write the map files and zip the folder.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import shutil
import struct
import subprocess
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

from bsmapper.interfaces import FormatVersion
from bsmapper.schema import Beatmap

AUDIO_FILENAME = "song.egg"
COVER_STEM = "cover"
PLACEHOLDER_SIZE = 256


@dataclass(frozen=True, slots=True)
class SongTags:
    title: str = ""
    artist: str = ""


def _mutagen_file(path: Path, easy: bool = False):
    try:
        import mutagen
    except ImportError:  # pragma: no cover - mutagen is a core dependency
        return None
    try:
        return mutagen.File(path, easy=easy)
    except Exception:  # mutagen raises many error types for unreadable files
        return None


def audio_duration(path: Path) -> float | None:
    """Length of an audio file in seconds, or None if it is missing or unreadable."""
    if not path.is_file():
        return None
    audio = _mutagen_file(path)
    length = getattr(getattr(audio, "info", None), "length", None)
    return float(length) if length else None


def read_tags(path: Path) -> SongTags:
    """Title and artist from the file's tags, falling back to an "Artist - Title" filename."""
    title = artist = ""
    audio = _mutagen_file(path, easy=True)
    tags = getattr(audio, "tags", None)
    if tags:
        title = str((tags.get("title") or [""])[0]).strip()
        artist = str((tags.get("artist") or [""])[0]).strip()
    if not title:
        stem = path.stem
        if " - " in stem and not artist:
            artist, title = (part.strip() for part in stem.split(" - ", 1))
        else:
            title = stem
    return SongTags(title, artist)


def _is_vorbis(path: Path) -> bool:
    try:
        from mutagen.oggvorbis import OggVorbis

        OggVorbis(path)
        return True
    except Exception:
        return False


def convert_to_egg(src: Path, dst: Path, quality: int = 6, ffmpeg: str = "ffmpeg") -> Path:
    """Convert any audio file to OGG Vorbis at ``dst``. Vorbis input is copied as-is."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in (".ogg", ".egg") and _is_vorbis(src):
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return dst
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-map_metadata",
        "-1",
        "-c:a",
        "libvorbis",
        "-q:a",
        str(quality),
        "-f",
        "ogg",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found; install it or pass ffmpeg=<path>") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr.strip()}") from exc
    return dst


def _embedded_picture(path: Path) -> tuple[bytes, str] | None:
    """(image bytes, mime type) of the first picture embedded in an audio file."""
    audio = _mutagen_file(path)
    if audio is None:
        return None
    pictures = getattr(audio, "pictures", None)  # FLAC
    if pictures:
        return pictures[0].data, pictures[0].mime
    tags = getattr(audio, "tags", None)
    if not tags:
        return None
    if hasattr(tags, "getall"):  # ID3
        frames = tags.getall("APIC")
        if frames:
            return frames[0].data, frames[0].mime
    covers = tags.get("covr") if hasattr(tags, "get") else None  # MP4
    if covers:
        from mutagen.mp4 import MP4Cover

        cover = covers[0]
        mime = "image/png" if cover.imageformat == MP4Cover.FORMAT_PNG else "image/jpeg"
        return bytes(cover), mime
    blocks = tags.get("metadata_block_picture") if hasattr(tags, "get") else None  # Ogg
    if blocks:
        from mutagen.flac import Picture

        picture = Picture(base64.b64decode(blocks[0]))
        return picture.data, picture.mime
    return None


def extract_cover(audio: Path, out_dir: Path) -> Path | None:
    """Save the cover art embedded in ``audio`` into ``out_dir``, if there is one."""
    found = _embedded_picture(audio)
    if found is None or not found[0]:
        return None
    data, mime = found
    suffix = ".png" if "png" in (mime or "").lower() or data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
    dst = out_dir / f"{COVER_STEM}{suffix}"
    dst.write_bytes(data)
    return dst


def write_placeholder_cover(path: Path, seed: str, size: int = PLACEHOLDER_SIZE) -> Path:
    """Write a square gradient PNG whose colours are derived from ``seed``."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    start, end = digest[0:3], digest[3:6]
    rows = []
    for y in range(size):
        row = bytearray([0])  # PNG filter type "none"
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            row += bytes(int(a + (b - a) * t) for a, b in zip(start, end, strict=True))
        rows.append(bytes(row))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


def zip_map(map_dir: Path, zip_path: Path) -> Path:
    """Zip the files of a map folder, flat, the way BeatSaver and the game expect."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(map_dir.iterdir()):
            if path.is_file() and path.resolve() != zip_path.resolve():
                zf.write(path, path.name)
    return zip_path


def package_map(
    beatmap: Beatmap,
    audio: Path,
    out_dir: Path,
    *,
    version: FormatVersion = "v3",
    cover: Path | None = None,
    zip_path: Path | None = None,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Build a complete map folder in ``out_dir``; returns the zip if ``zip_path`` is set.

    The input ``beatmap`` is not modified. Empty song name/author are filled from tags.
    """
    from bsmapper.formats.writer import write_map  # writer imports this module

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    egg = convert_to_egg(Path(audio), out_dir / AUDIO_FILENAME, ffmpeg=ffmpeg)

    if cover is not None:
        cover_dst = out_dir / f"{COVER_STEM}{Path(cover).suffix.lower() or '.png'}"
        if Path(cover).resolve() != cover_dst.resolve():
            shutil.copyfile(cover, cover_dst)
    else:
        cover_dst = extract_cover(Path(audio), out_dir) or write_placeholder_cover(
            out_dir / f"{COVER_STEM}.png", beatmap.song_name or Path(audio).stem
        )

    tags = read_tags(Path(audio))
    beatmap = dataclasses.replace(
        beatmap,
        audio_path=AUDIO_FILENAME,
        cover_path=cover_dst.name,
        song_name=beatmap.song_name or tags.title,
        song_author=beatmap.song_author or tags.artist,
    )
    write_map(beatmap, out_dir, version, song_duration=audio_duration(egg))
    if zip_path is not None:
        return zip_map(out_dir, Path(zip_path))
    return out_dir
