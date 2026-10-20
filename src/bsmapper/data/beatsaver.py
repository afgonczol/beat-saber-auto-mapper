"""Client for api.beatsaver.com: paginated map search and zip download."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bsmapper.data.http import USER_AGENT, HttpError, PoliteClient, RateLimiter, query_value

API_URL = "https://api.beatsaver.com"
PAGE_SIZE = 100  # BeatSaver's maximum

BeatSaverError = HttpError

__all__ = ["API_URL", "USER_AGENT", "BeatSaverClient", "BeatSaverError", "RateLimiter"]


class BeatSaverClient(PoliteClient):
    """Thin wrapper over the endpoints the scraper needs. Search pages are cached for 6 h."""

    base_url = API_URL

    def search(self, page: int, params: dict[str, Any]) -> dict[str, Any]:
        """Return one page of `/search/text/{page}` as JSON (`{"docs": [...], "info": {...}}`)."""
        query = {k: query_value(v) for k, v in params.items() if v is not None}
        query.setdefault("pageSize", PAGE_SIZE)
        path = self._cache_path("search", {"page": page, **query}, prefix=f"{page:05d}-")
        cached = self._cache_get(path)
        if cached is not None:
            return cached
        data = self._request("GET", f"/search/text/{page}", params=query).json()
        self._cache_put(path, data)
        return data

    def download(self, url: str, dest: Path) -> int:
        """Stream a map zip to `dest` atomically. Returns the byte count."""
        return self._download(url, dest)
