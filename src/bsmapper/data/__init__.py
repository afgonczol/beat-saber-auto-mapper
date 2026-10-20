"""Dataset tools (WS3): BeatSaver scraper, curation filters, manifest."""

from bsmapper.data.beatleader import BeatLeaderClient
from bsmapper.data.beatsaver import BeatSaverClient, BeatSaverError
from bsmapper.data.filters import (
    BeatLeaderRating,
    DiffInfo,
    FilterConfig,
    MapRecord,
    check,
    gold_reasons,
)
from bsmapper.data.manifest import ManifestEntry, ScrapeState, read_manifest
from bsmapper.data.scraper import (
    ScrapeConfig,
    ScrapeResult,
    backfill_ratings,
    load_curated_keys,
    ratings_coverage,
    scrape,
)

__all__ = [
    "BeatLeaderClient",
    "BeatLeaderRating",
    "BeatSaverClient",
    "BeatSaverError",
    "DiffInfo",
    "FilterConfig",
    "ManifestEntry",
    "MapRecord",
    "ScrapeConfig",
    "ScrapeResult",
    "ScrapeState",
    "backfill_ratings",
    "check",
    "gold_reasons",
    "load_curated_keys",
    "ratings_coverage",
    "read_manifest",
    "scrape",
]
