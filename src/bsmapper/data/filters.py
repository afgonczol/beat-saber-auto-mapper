"""Curation filters and the gold-subset tag for BeatSaver maps.

`MapRecord.from_api` turns one search result into a dataclass; `check` decides whether it
goes into the training set; `gold_reasons` says why it is in the higher-weighted gold subset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from bsmapper.schema import Characteristic

# Names BeatSaver's `characteristics` search filter knows. It silently ignores any other
# name (and then returns every map), so FilterConfig rejects unknown ones up front.
SEARCH_CHARACTERISTICS = frozenset({c.value for c in Characteristic} | {"Lawless", "Legacy"})


@dataclass(frozen=True, slots=True)
class BeatLeaderRating:
    """BeatLeader's ratings for one (characteristic, difficulty). Any value may be missing."""

    characteristic: str
    difficulty: str
    status: str | None = None  # unranked / nominated / qualified / ranked / ...
    stars: float | None = None
    tech_rating: float | None = None
    pass_rating: float | None = None
    acc_rating: float | None = None
    predicted_acc: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiffInfo:
    characteristic: str
    difficulty: str
    notes: int = 0
    bombs: int = 0
    obstacles: int = 0
    nps: float = 0.0
    njs: float = 0.0
    offset: float = 0.0
    seconds: float = 0.0
    events: int = 0
    environment: str | None = None
    chroma: bool = False
    noodle: bool = False
    mapping_extensions: bool = False
    cinema: bool = False
    vivify: bool = False
    parity_errors: int = 0
    parity_warns: int = 0
    parity_resets: int = 0
    beatleader: BeatLeaderRating | None = None  # None: BeatLeader has no rating for it

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiffInfo:
        d = dict(d)
        bl = d.pop("beatleader", None)
        return cls(**d, beatleader=BeatLeaderRating(**bl) if bl else None)

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> DiffInfo:
        parity = d.get("paritySummary") or {}
        return cls(
            characteristic=str(d.get("characteristic", "")),
            difficulty=str(d.get("difficulty", "")),
            notes=int(d.get("notes", 0)),
            bombs=int(d.get("bombs", 0)),
            obstacles=int(d.get("obstacles", 0)),
            nps=float(d.get("nps", 0.0)),
            njs=float(d.get("njs", 0.0)),
            offset=float(d.get("offset", 0.0)),
            seconds=float(d.get("seconds", 0.0)),
            events=int(d.get("events", 0)),
            environment=d.get("environment"),
            chroma=bool(d.get("chroma", False)),
            noodle=bool(d.get("ne", False)),
            mapping_extensions=bool(d.get("me", False)),
            cinema=bool(d.get("cinema", False)),
            vivify=bool(d.get("vivify", False)),
            parity_errors=int(parity.get("errors", 0)),
            parity_warns=int(parity.get("warns", 0)),
            parity_resets=int(parity.get("resets", 0)),
        )


@dataclass(frozen=True, slots=True)
class MapRecord:
    """One BeatSaver map, reduced to its latest published version."""

    key: str
    hash: str
    name: str
    song_name: str
    song_sub_name: str
    song_author: str
    level_author: str
    uploader: str
    bpm: float
    duration: int  # seconds
    uploaded: str  # ISO 8601
    score: float
    upvotes: int
    downvotes: int
    automapper: bool
    declared_ai: str
    ranked_scoresaber: bool
    ranked_beatleader: bool
    curated: bool
    download_url: str
    tags: tuple[str, ...] = ()
    diffs: tuple[DiffInfo, ...] = ()

    @property
    def votes(self) -> int:
        return self.upvotes + self.downvotes

    @property
    def uploaded_at(self) -> datetime:
        return _parse_time(self.uploaded)

    @classmethod
    def from_api(cls, doc: dict[str, Any]) -> MapRecord | None:
        """Parse a search result. Returns None if it has no published version."""
        version = _latest_published(doc.get("versions") or [])
        if version is None:
            return None
        meta = doc.get("metadata") or {}
        stats = doc.get("stats") or {}
        return cls(
            key=str(doc["id"]),
            hash=str(version["hash"]).lower(),
            name=str(doc.get("name", "")),
            song_name=str(meta.get("songName", "")),
            song_sub_name=str(meta.get("songSubName", "")),
            song_author=str(meta.get("songAuthorName", "")),
            level_author=str(meta.get("levelAuthorName", "")),
            uploader=str((doc.get("uploader") or {}).get("name", "")),
            bpm=float(meta.get("bpm", 0.0)),
            duration=int(meta.get("duration", 0)),
            uploaded=str(doc.get("uploaded") or doc.get("createdAt") or ""),
            score=float(stats.get("score", 0.0)),
            upvotes=int(stats.get("upvotes", 0)),
            downvotes=int(stats.get("downvotes", 0)),
            automapper=bool(doc.get("automapper", False)),
            declared_ai=str(doc.get("declaredAi", "None")),
            ranked_scoresaber=bool(doc.get("ranked", False)),
            ranked_beatleader=bool(doc.get("blRanked", False)),
            curated=doc.get("curatedAt") is not None or doc.get("curator") is not None,
            download_url=str(version.get("downloadURL", "")),
            tags=tuple(doc.get("tags") or ()),
            diffs=tuple(DiffInfo.from_api(d) for d in version.get("diffs") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MapRecord:
        d = dict(d)
        d["tags"] = tuple(d.get("tags") or ())
        d["diffs"] = tuple(DiffInfo.from_dict(x) for x in d.get("diffs") or ())
        return cls(**d)


@dataclass(slots=True)
class FilterConfig:
    min_score: float = 0.75
    min_votes: int = 30
    uploaded_after: datetime = field(default_factory=lambda: datetime(2021, 1, 1, tzinfo=UTC))
    exclude_ai: bool = True
    exclude_noodle: bool = True  # also drops Mapping Extensions and Vivify maps
    min_duration: int = 30  # seconds
    max_duration: int = 600
    # Only maps with at least one difficulty in one of these characteristics (OR).
    # Empty means any characteristic.
    characteristics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.characteristics = tuple(self.characteristics)
        unknown = sorted(set(self.characteristics) - SEARCH_CHARACTERISTICS)
        if unknown:
            known = ", ".join(sorted(SEARCH_CHARACTERISTICS))
            raise ValueError(f"Unknown characteristic(s) {', '.join(unknown)}; use {known}.")

    def search_params(self) -> dict[str, Any]:
        """Server-side pre-filter for `/search/text`. `check` still re-applies everything.

        `automapper` is left unset on purpose: BeatSaver reads null as "no AI",
        false as "only AI" and true as "both". `characteristics` is a comma-separated
        OR list, and is left out when empty so the untargeted query stays as it was.
        """
        params: dict[str, Any] = {
            "order": "Rating",
            "minRating": self.min_score,
            "minVotes": self.min_votes,
            "from": self.uploaded_after.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "minDuration": self.min_duration,
            "maxDuration": self.max_duration,
        }
        if self.characteristics:
            params["characteristics"] = ",".join(self.characteristics)
        return params

    def wants(self, rec: MapRecord) -> bool:
        """True if the map has a difficulty in one of `characteristics` (always, if unset)."""
        chars = set(self.characteristics)
        return not chars or any(d.characteristic in chars for d in rec.diffs)


def check(rec: MapRecord, cfg: FilterConfig) -> str | None:
    """Return None if the map passes, otherwise a short rejection reason."""
    if cfg.exclude_ai and (rec.automapper or rec.declared_ai not in ("None", "")):
        return "ai"
    if rec.score < cfg.min_score:
        return "score"
    if rec.votes < cfg.min_votes:
        return "votes"
    if not rec.uploaded or rec.uploaded_at < cfg.uploaded_after:
        return "too_old"
    if not cfg.min_duration <= rec.duration <= cfg.max_duration:
        return "duration"
    if not rec.diffs:
        return "no_diffs"
    if not cfg.wants(rec):
        return "characteristic"
    if cfg.exclude_noodle and any(d.noodle or d.mapping_extensions or d.vivify for d in rec.diffs):
        return "mods"
    if not rec.download_url:
        return "no_url"
    return None


def gold_reasons(rec: MapRecord, curated_keys: frozenset[str] = frozenset()) -> list[str]:
    """Why a map belongs to the gold subset (empty if it does not)."""
    reasons = []
    if rec.ranked_scoresaber:
        reasons.append("scoresaber_ranked")
    if rec.ranked_beatleader:
        reasons.append("beatleader_ranked")
    if rec.curated:
        reasons.append("beatsaver_curated")
    if rec.key.lower() in curated_keys:
        reasons.append("curated_list")
    return reasons


def _latest_published(versions: list[dict[str, Any]]) -> dict[str, Any] | None:
    published = [v for v in versions if v.get("state", "Published") == "Published"]
    if not published:
        return None
    return max(published, key=lambda v: str(v.get("createdAt", "")))


def _parse_time(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
