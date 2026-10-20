"""Read map files from a folder or straight from a downloaded zip."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from bsmapper.formats.report import MapFormatError

INFO_NAME = "info.dat"


def parse_json(raw: bytes) -> Any:
    # utf-8-sig strips the BOM some editors write.
    return json.loads(raw.decode("utf-8-sig"))


class MapSource:
    """A map folder or zip. File lookups are case-insensitive, as they are on Windows."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._zip: zipfile.ZipFile | None = None
        self._files: dict[str, str] = {}  # lower-case name -> real name/member
        if self.path.is_dir():
            self._files = {p.name.lower(): p.name for p in self.path.iterdir() if p.is_file()}
        elif zipfile.is_zipfile(self.path):
            self._zip = zipfile.ZipFile(self.path)
            members = [m for m in self._zip.namelist() if not m.endswith("/")]
            infos = [m for m in members if PurePosixPath(m).name.lower() == INFO_NAME]
            if not infos:
                raise MapFormatError(f"{self.path} has no Info.dat")
            # Some zips wrap the map in a folder; use the shallowest Info.dat.
            prefix = str(PurePosixPath(min(infos, key=lambda m: m.count("/"))).parent)
            prefix = "" if prefix == "." else prefix + "/"
            for member in members:
                if member.startswith(prefix) and "/" not in member[len(prefix) :]:
                    self._files[member[len(prefix) :].lower()] = member
        else:
            raise MapFormatError(f"{self.path} is neither a map folder nor a zip")

    def __enter__(self) -> MapSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def find(self, name: str) -> str | None:
        return self._files.get(name.replace("\\", "/").lower())

    def read_bytes(self, name: str) -> bytes:
        real = self.find(name)
        if real is None:
            raise FileNotFoundError(name)
        if self._zip is not None:
            return self._zip.read(real)
        return (self.path / real).read_bytes()

    def load_json(self, name: str) -> Any:
        return parse_json(self.read_bytes(name))
