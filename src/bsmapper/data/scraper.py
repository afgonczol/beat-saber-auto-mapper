"""Resumable BeatSaver scrape: search by rating, filter, download zips, append to manifest.

Output layout (under `out_dir`, default `data/raw/`):

    zips/<key>.zip      the map as served by BeatSaver
    manifest.jsonl      one line per downloaded map (metadata, gold tag, BeatLeader ratings,
                        zip path, size)
    state/<query>.json  search cursor + rejection reasons per search query, so a rerun picks
                        up where it stopped
    cache/search/       cached search pages (see BeatSaverClient)
    cache/beatleader/   cached BeatLeader ratings by map hash (see BeatLeaderClient)

Rerunning with the same filters continues from the saved page and skips maps already in the
manifest. Each distinct set of filters has its own cursor, so a characteristic-targeted run
(`FilterConfig.characteristics`) can top up the same directory without moving the main
cursor. Every run appends to the one shared manifest.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from bsmapper.data.beatleader import BeatLeaderClient, attach_ratings
from bsmapper.data.beatsaver import BeatSaverClient, BeatSaverError
from bsmapper.data.filters import FilterConfig, MapRecord, check, gold_reasons
from bsmapper.data.http import HttpError
from bsmapper.data.manifest import (
    ZIP_DIR,
    ManifestEntry,
    ScrapeState,
    append_manifest,
    now_iso,
    query_key,
    read_manifest,
    write_manifest,
)

log = logging.getLogger(__name__)

GB = 1024**3


@dataclass(slots=True)
class ScrapeConfig:
    out_dir: Path = Path("data/raw")
    # Total maps wanted in the manifest. With `filters.characteristics`, only maps that have
    # one of those characteristics count, so a targeted run's limit is independent of the rest.
    limit: int = 1000
    max_bytes: int = 50 * GB  # stop once downloaded zips reach this size
    min_free_bytes: int = 20 * GB  # stop if the disk has less free space than this
    max_pages: int | None = None  # safety stop for a single run
    filters: FilterConfig = field(default_factory=FilterConfig)
    curated_keys: frozenset[str] = frozenset()


@dataclass(slots=True)
class ScrapeResult:
    downloaded: int = 0  # new this run
    total: int = 0  # in the manifest after this run
    matching: int = 0  # of those, the ones `limit` counts (see ScrapeConfig.limit)
    total_bytes: int = 0
    gold: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    rejected: Counter[str] = field(default_factory=Counter)
    stop_reason: str = ""


def load_curated_keys(path: Path) -> frozenset[str]:
    """Read a curated-map list: one BeatSaver key per line, `#` starts a comment."""
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            keys.add(line.lower())
    return frozenset(keys)


@dataclass(slots=True)
class RatingsCoverage:
    maps: int = 0
    maps_with_ratings: int = 0
    diffs: int = 0
    diffs_with_tech: int = 0
    not_looked_up: int = 0


def ratings_coverage(entries: list[ManifestEntry]) -> RatingsCoverage:
    cov = RatingsCoverage(maps=len(entries))
    for e in entries:
        cov.not_looked_up += e.beatleader_at is None
        rated = [d for d in e.record.diffs if d.beatleader and d.beatleader.tech_rating is not None]
        cov.maps_with_ratings += bool(rated)
        cov.diffs += len(e.record.diffs)
        cov.diffs_with_tech += len(rated)
    return cov


def fetch_ratings(bl: BeatLeaderClient, entry: ManifestEntry) -> ManifestEntry:
    """Attach BeatLeader ratings. On a network/API failure, return the entry unchanged
    (still `beatleader_at=None`) so a later run retries; never fail the map."""
    try:
        ratings = bl.ratings(entry.record.hash)
    except HttpError as e:
        log.warning("BeatLeader lookup failed for %s: %s", entry.record.key, e)
        return entry
    return replace(entry, record=attach_ratings(entry.record, ratings), beatleader_at=now_iso())


def backfill_ratings(bl: BeatLeaderClient, out_dir: Path, refresh: bool = False) -> int:
    """Look up ratings for manifest entries that have none yet (all of them if `refresh`).

    Updates are appended as they arrive (crash-safe, last line wins), then the manifest is
    compacted. Returns how many entries were updated.
    """
    todo = [e for e in read_manifest(out_dir) if refresh or e.beatleader_at is None]
    updated = 0
    try:
        for i, entry in enumerate(todo, 1):
            new = fetch_ratings(bl, entry)
            if new.beatleader_at is None:
                continue
            append_manifest(out_dir, new)
            updated += 1
            if i % 50 == 0:
                log.info("BeatLeader ratings: %d/%d", i, len(todo))
    finally:
        if updated:
            write_manifest(out_dir, read_manifest(out_dir))
    return updated


def scrape(
    client: BeatSaverClient, cfg: ScrapeConfig, ratings: BeatLeaderClient | None = None
) -> ScrapeResult:
    """Run (or resume) a scrape. With `ratings`, also attach BeatLeader ratings, first to
    already-downloaded maps that lack them and then to each new map."""
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if ratings is not None:
        n = backfill_ratings(ratings, out)
        if n:
            log.info("Added BeatLeader ratings to %d existing maps.", n)
    entries = read_manifest(out)
    done = {e.record.key for e in entries}
    total_bytes = sum(e.bytes for e in entries)
    result = ScrapeResult(total=len(done), total_bytes=total_bytes)
    result.gold = sum(e.gold for e in entries)
    result.matching = sum(cfg.filters.wants(e.record) for e in entries)

    params = cfg.filters.search_params()
    state = ScrapeState.load(out, params)
    if not state.updated_at:
        log.info("New search cursor for these filters (state/%s.json).", query_key(params))
    page = state.next_page
    pages_this_run = 0
    _cleanup_partials(out)

    try:
        while True:
            if result.matching >= cfg.limit:
                result.stop_reason = "limit"
                break
            if state.exhausted:
                result.stop_reason = "exhausted"
                break
            if cfg.max_pages is not None and pages_this_run >= cfg.max_pages:
                result.stop_reason = "max_pages"
                break
            data = client.search(page, params)
            docs = data.get("docs") or []
            pages_this_run += 1
            log.info(
                "Page %d: %d maps (%d/%d toward the limit)",
                page,
                len(docs),
                result.matching,
                cfg.limit,
            )
            if not docs:
                state.exhausted = True
                continue
            for doc in docs:
                if result.matching >= cfg.limit:
                    break
                budget_stop = _budget_stop(out, cfg, result.total_bytes)
                if budget_stop:
                    result.stop_reason = budget_stop
                    return result
                rec = MapRecord.from_api(doc)
                if rec is None:
                    state.rejected[str(doc.get("id"))] = "unpublished"
                    continue
                if rec.key in done or rec.key in result.failed:
                    continue
                reason = check(rec, cfg.filters)
                if reason:
                    state.rejected[rec.key] = reason
                    continue
                state.rejected.pop(rec.key, None)
                entry = _download(client, out, rec, cfg.curated_keys, result)
                if entry is None:
                    continue
                if ratings is not None:
                    entry = fetch_ratings(ratings, entry)
                append_manifest(out, entry)
                done.add(rec.key)
                result.downloaded += 1
                result.total += 1
                result.matching += 1
                result.total_bytes += entry.bytes
                result.gold += entry.gold
                log.info(
                    "[%d/%d] %s %s - %s (%.1f MB%s)",
                    result.matching,
                    cfg.limit,
                    rec.key,
                    rec.song_author,
                    rec.song_name,
                    entry.bytes / 1e6,
                    ", gold" if entry.gold else "",
                )
            # Only advance once the whole page is handled, so an interrupted page is redone
            # (cheaply: it is cached and finished maps are skipped).
            if result.matching < cfg.limit:
                page += 1
                state.next_page = page
            state.save(out)
    finally:
        state.save(out)
        result.rejected = Counter(state.rejected.values())
    return result


def _download(
    client: BeatSaverClient,
    out: Path,
    rec: MapRecord,
    curated_keys: frozenset[str],
    result: ScrapeResult,
) -> ManifestEntry | None:
    rel = f"{ZIP_DIR}/{rec.key}.zip"
    dest = out / rel
    try:
        # A valid zip without a manifest line means we crashed right after the download.
        if dest.exists() and _validate_zip(dest) is None:
            size = dest.stat().st_size
        else:
            size = client.download(rec.download_url, dest)
    except BeatSaverError as e:
        log.warning("Download failed for %s: %s", rec.key, e)
        result.failed[rec.key] = str(e)
        return None
    problem = _validate_zip(dest)
    if problem:
        log.warning("Bad zip for %s: %s", rec.key, problem)
        dest.unlink(missing_ok=True)
        result.failed[rec.key] = problem
        return None
    reasons = gold_reasons(rec, curated_keys)
    return ManifestEntry(
        record=rec,
        zip_path=rel,
        bytes=size,
        gold=bool(reasons),
        gold_reasons=reasons,
        downloaded_at=now_iso(),
    )


def _validate_zip(path: Path) -> str | None:
    if not zipfile.is_zipfile(path):
        return "not a zip file"
    try:
        with zipfile.ZipFile(path) as zf:
            names = {Path(n).name.lower() for n in zf.namelist()}
    except zipfile.BadZipFile as e:
        return f"corrupt zip: {e}"
    if "info.dat" not in names:
        return "no Info.dat in zip"
    return None


def _budget_stop(out: Path, cfg: ScrapeConfig, total_bytes: int) -> str | None:
    if total_bytes >= cfg.max_bytes:
        return "max_bytes"
    if shutil.disk_usage(out).free < cfg.min_free_bytes:
        return "min_free_disk"
    return None


def _cleanup_partials(out: Path) -> None:
    zips = out / ZIP_DIR
    if zips.exists():
        for part in zips.glob("*.part"):
            part.unlink(missing_ok=True)
