"""Scrape output on disk: `manifest.jsonl` (one line per downloaded map) and `state.json`.

The manifest is append-only and is written only after a zip is safely in place, so it is
the source of truth for what has been downloaded. A torn last line (from a crash mid-write)
is ignored on load. `state/<query key>.json` holds one search query's cursor and per-map
rejection reasons, so several queries (say the main one and a characteristic-targeted one)
can share an output directory without resetting each other.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bsmapper.data.filters import MapRecord

MANIFEST_NAME = "manifest.jsonl"
STATE_DIR = "state"
LEGACY_STATE_NAME = "state.json"  # single cursor, from before per-query states
ZIP_DIR = "zips"


@dataclass(slots=True)
class ManifestEntry:
    record: MapRecord
    zip_path: str  # relative to the output directory, forward slashes
    bytes: int
    gold: bool
    gold_reasons: list[str]
    downloaded_at: str
    beatleader_at: str | None = None  # when BeatLeader was asked; None = not yet

    def to_json(self) -> str:
        d = self.record.to_dict()
        d.update(
            zip_path=self.zip_path,
            bytes=self.bytes,
            gold=self.gold,
            gold_reasons=self.gold_reasons,
            downloaded_at=self.downloaded_at,
            beatleader_at=self.beatleader_at,
        )
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> ManifestEntry:
        d = json.loads(line)
        extra = {
            k: d.pop(k) for k in ("zip_path", "bytes", "gold", "gold_reasons", "downloaded_at")
        }
        extra["beatleader_at"] = d.pop("beatleader_at", None)
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


def write_manifest(out_dir: Path, entries: list[ManifestEntry]) -> None:
    """Atomically replace the manifest (used to compact it after in-place updates)."""
    path = out_dir / MANIFEST_NAME
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.to_json() + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_manifest(out_dir: Path, entry: ManifestEntry) -> None:
    path = out_dir / MANIFEST_NAME
    needs_newline = path.exists() and path.stat().st_size > 0 and not _ends_with_newline(path)
    with path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")  # isolate a torn previous line
        f.write(entry.to_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


def query_key(query: dict[str, Any]) -> str:
    """A short stable name for a search query (its params, order-independent)."""
    blob = json.dumps(query, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def state_path(out_dir: Path, query: dict[str, Any]) -> Path:
    return out_dir / STATE_DIR / f"{query_key(query)}.json"


@dataclass(slots=True)
class ScrapeState:
    """The cursor of one search query. Each distinct query has its own file."""

    query: dict[str, Any] = field(default_factory=dict)
    next_page: int = 0
    exhausted: bool = False
    rejected: dict[str, str] = field(default_factory=dict)  # map key -> reason
    updated_at: str = ""

    @classmethod
    def load(cls, out_dir: Path, query: dict[str, Any]) -> ScrapeState:
        """The saved state for `query`, or a fresh one at page 0."""
        path = state_path(out_dir, query)
        legacy = out_dir / LEGACY_STATE_NAME
        if not path.exists() and legacy.exists():
            old = cls._read(legacy)
            if old is not None and old.query == query:
                path.parent.mkdir(parents=True, exist_ok=True)
                legacy.replace(path)  # adopt the old single cursor for its own query
        state = cls._read(path) if path.exists() else None
        if state is None or state.query != query:
            return cls(query=query)
        return state

    @classmethod
    def _read(cls, path: Path) -> ScrapeState | None:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cls(
            query=d.get("query", {}),
            next_page=int(d.get("next_page", 0)),
            exhausted=bool(d.get("exhausted", False)),
            rejected=dict(d.get("rejected", {})),
            updated_at=str(d.get("updated_at", "")),
        )

    def save(self, out_dir: Path) -> None:
        self.updated_at = now_iso()
        path = state_path(out_dir, self.query)
        path.parent.mkdir(parents=True, exist_ok=True)
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
