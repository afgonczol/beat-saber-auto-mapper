"""Resumable BeatSaver scrape: search by rating, filter, download zips, append to manifest.

Output layout (under `out_dir`, default `data/raw/`):

    zips/<key>.zip      the map as served by BeatSaver
    manifest.jsonl      one line per downloaded map (metadata, gold tag, zip path, size)
    state.json          search cursor + rejection reasons, so a rerun picks up where it stopped
    cache/search/       cached search pages (see BeatSaverClient)

Rerunning with the same filters continues from the saved page and skips maps already in the
manifest. Changing the filters restarts the cursor but still keeps every downloaded map.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from bsmapper.data.beatsaver import BeatSaverClient, BeatSaverError
from bsmapper.data.filters import FilterConfig, MapRecord, check, gold_reasons
from bsmapper.data.manifest import (
    ZIP_DIR,
    ManifestEntry,
    ScrapeState,
    append_manifest,
    now_iso,
    read_manifest,
)

log = logging.getLogger(__name__)

GB = 1024**3


@dataclass(slots=True)
class ScrapeConfig:
    out_dir: Path = Path("data/raw")
    limit: int = 1000  # total maps wanted in the manifest
    max_bytes: int = 50 * GB  # stop once downloaded zips reach this size
    min_free_bytes: int = 20 * GB  # stop if the disk has less free space than this
    max_pages: int | None = None  # safety stop for a single run
    filters: FilterConfig = field(default_factory=FilterConfig)
    curated_keys: frozenset[str] = frozenset()


@dataclass(slots=True)
class ScrapeResult:
    downloaded: int = 0  # new this run
    total: int = 0  # in the manifest after this run
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


def scrape(client: BeatSaverClient, cfg: ScrapeConfig) -> ScrapeResult:
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    entries = read_manifest(out)
    done = {e.record.key for e in entries}
    total_bytes = sum(e.bytes for e in entries)
    result = ScrapeResult(total=len(done), total_bytes=total_bytes)
    result.gold = sum(e.gold for e in entries)

    params = cfg.filters.search_params()
    state = ScrapeState.load(out)
    if state.query != params:
        if state.query:
            log.info("Filters changed since the last run; restarting the search cursor.")
        state = ScrapeState(query=params, rejected=state.rejected)
    page = state.next_page
    pages_this_run = 0
    _cleanup_partials(out)

    try:
        while True:
            if result.total >= cfg.limit:
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
            log.info("Page %d: %d maps (manifest has %d)", page, len(docs), result.total)
            if not docs:
                state.exhausted = True
                continue
            for doc in docs:
                if result.total >= cfg.limit:
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
                append_manifest(out, entry)
                done.add(rec.key)
                result.downloaded += 1
                result.total += 1
                result.total_bytes += entry.bytes
                result.gold += entry.gold
                log.info(
                    "[%d/%d] %s %s - %s (%.1f MB%s)",
                    result.total,
                    cfg.limit,
                    rec.key,
                    rec.song_author,
                    rec.song_name,
                    entry.bytes / 1e6,
                    ", gold" if entry.gold else "",
                )
            # Only advance once the whole page is handled, so an interrupted page is redone
            # (cheaply: it is cached and finished maps are skipped).
            if result.total < cfg.limit:
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
