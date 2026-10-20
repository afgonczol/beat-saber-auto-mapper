"""Scrape output on disk: `manifest.jsonl` (one line per downloaded map) and `state.json`.

The manifest is append-only and is written only after a zip is safely in place, so it is
the source of truth for what has been downloaded. A torn last line (from a crash mid-write)
is ignored on load. `state.json` holds the search cursor and per-map rejection reasons.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bsmapper.data.filters import MapRecord

MANIFEST_NAME = "manifest.jsonl"
STATE_NAME = "state.json"
ZIP_DIR = "zips"


@dataclass(slots=True)
class ManifestEntry:
    record: MapRecord
    zip_path: str  # relative to the output directory, forward slashes
    bytes: int
    gold: bool
    gold_reasons: list[str]
    downloaded_at: str

    def to_json(self) -> str:
        d = self.record.to_dict()
        d.update(
            zip_path=self.zip_path,
            bytes=self.bytes,
            gold=self.gold,
            gold_reasons=self.gold_reasons,
            downloaded_at=self.downloaded_at,
        )
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> ManifestEntry:
        d = json.loads(line)
        extra = {
            k: d.pop(k) for k in ("zip_path", "bytes", "gold", "gold_reasons", "downloaded_at")
        }
        return cls(record=MapRecord.from_dict(d), **extra)


def read_manifest(out_dir: Path) -> list[ManifestEntry]:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        return []
    entries: dict[str, ManifestEntry] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = ManifestEntry.from_json(line)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # torn or foreign line
            entries[e.record.key] = e  # last write wins
    return list(entries.values())


def append_manifest(out_dir: Path, entry: ManifestEntry) -> None:
    path = out_dir / MANIFEST_NAME
    needs_newline = path.exists() and path.stat().st_size > 0 and not _ends_with_newline(path)
    with path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")  # isolate a torn previous line
        f.write(entry.to_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


@dataclass(slots=True)
class ScrapeState:
    query: dict[str, Any] = field(default_factory=dict)
    next_page: int = 0
    exhausted: bool = False
    rejected: dict[str, str] = field(default_factory=dict)  # map key -> reason
    updated_at: str = ""

    @classmethod
    def load(cls, out_dir: Path) -> ScrapeState:
        path = out_dir / STATE_NAME
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        return cls(
            query=d.get("query", {}),
            next_page=int(d.get("next_page", 0)),
            exhausted=bool(d.get("exhausted", False)),
            rejected=dict(d.get("rejected", {})),
            updated_at=str(d.get("updated_at", "")),
        )

    def save(self, out_dir: Path) -> None:
        self.updated_at = now_iso()
        path = out_dir / STATE_NAME
        tmp = path.with_name(path.name + ".tmp")
        payload = {
            "query": self.query,
            "next_page": self.next_page,
            "exhausted": self.exhausted,
            "rejected": self.rejected,
            "updated_at": self.updated_at,
        }
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(path)


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) == b"\n"
