"""Dataset tools (WS3): BeatSaver scraper, curation filters, manifest, dataset builder."""

from bsmapper.data.beatleader import BeatLeaderClient
from bsmapper.data.beatsaver import BeatSaverClient, BeatSaverError
from bsmapper.data.build import BuildConfig, BuildResult, run_build
from bsmapper.data.dataset import ProcessedDataset
from bsmapper.data.filters import (
    BeatLeaderRating,
    DiffInfo,
    FilterConfig,
    MapRecord,
    check,
    gold_reasons,
)
from bsmapper.data.manifest import ManifestEntry, ScrapeState, read_manifest
from bsmapper.data.pack import PackConfig, pack
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
    "BuildConfig",
    "BuildResult",
    "DiffInfo",
    "FilterConfig",
    "ManifestEntry",
    "MapRecord",
    "PackConfig",
    "ProcessedDataset",
    "ScrapeConfig",
    "ScrapeResult",
    "ScrapeState",
    "backfill_ratings",
    "check",
    "gold_reasons",
    "load_curated_keys",
    "pack",
    "ratings_coverage",
    "read_manifest",
    "run_build",
    "scrape",
]
