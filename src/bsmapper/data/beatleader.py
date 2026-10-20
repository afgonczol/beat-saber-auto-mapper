"""BeatLeader per-difficulty ratings (tech / pass / acc / stars) for the style labels.

BeatLeader rates most maps it knows, including unranked ones, with its AI model. These
ratings are the calibration target for WS3b's heuristic tech score (the flowy <-> technical
slider). A map or difficulty that BeatLeader does not know simply gets no rating.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from bsmapper.data.filters import BeatLeaderRating, MapRecord
from bsmapper.data.http import HttpError, PoliteClient

log = logging.getLogger(__name__)

API_URL = "https://api.beatleader.com"

# BeatLeader's DifficultyStatus enum.
STATUS_NAMES = {
    0: "unranked",
    1: "nominated",
    2: "qualified",
    3: "ranked",
    4: "unrankable",
    5: "outdated",
    6: "event",
}


class BeatLeaderClient(PoliteClient):
    """Looks maps up by hash. Parsed ratings are cached for a week (misses too)."""

    base_url = API_URL
    cache_ttl_default = 7 * 24 * 3600.0

    def ratings(self, map_hash: str) -> list[BeatLeaderRating] | None:
        """Ratings for every difficulty of the map with this hash, or None if BL lacks it."""
        h = map_hash.lower()
        path = self._cache_path("beatleader", {"hash": h}, prefix=f"{h[:8]}-")
        cached = self._cache_get(path)
        if cached is not None:
            if not cached.get("found"):
                return None
            return [BeatLeaderRating(**r) for r in cached["ratings"]]
        try:
            data = self._request("GET", f"/map/hash/{h}").json()
        except HttpError as e:
            if e.status != 404:
                raise
            self._cache_put(path, {"found": False})
            return None
        ratings = parse_ratings(data, h)
        self._cache_put(path, {"found": True, "ratings": [r.to_dict() for r in ratings]})
        return ratings


def parse_ratings(bl_map: dict[str, Any], map_hash: str) -> list[BeatLeaderRating]:
    out = []
    for d in bl_map.get("difficulties") or ():
        if str(d.get("hash") or map_hash).lower() != map_hash.lower():
            continue
        status = d.get("status")
        out.append(
            BeatLeaderRating(
                characteristic=str(d.get("modeName", "")),
                difficulty=str(d.get("difficultyName", "")),
                status=STATUS_NAMES.get(status, str(status)) if status is not None else None,
                stars=_float(d.get("stars")),
                tech_rating=_float(d.get("techRating")),
                pass_rating=_float(d.get("passRating")),
                acc_rating=_float(d.get("accRating")),
                predicted_acc=_float(d.get("predictedAcc")),
            )
        )
    return out


def attach_ratings(rec: MapRecord, ratings: list[BeatLeaderRating] | None) -> MapRecord:
    """Set `DiffInfo.beatleader` on each matching (characteristic, difficulty); others None."""
    by_key = {(r.characteristic.lower(), r.difficulty.lower()): r for r in ratings or ()}
    diffs = tuple(
        replace(d, beatleader=by_key.get((d.characteristic.lower(), d.difficulty.lower())))
        for d in rec.diffs
    )
    return replace(rec, diffs=diffs)


def _float(v: Any) -> float | None:
    return None if v is None else float(v)
