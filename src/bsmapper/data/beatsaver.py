"""Polite HTTP client for api.beatsaver.com.

One `httpx.Client`, a minimum interval between requests, retries with backoff on 429/5xx
and transport errors, and an on-disk cache for search pages so a resumed scrape does not
hit the API again for pages it already saw.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from bsmapper import __version__

API_URL = "https://api.beatsaver.com"
PROJECT_URL = "https://github.com/afgonczol/beat-saber-auto-mapper"
USER_AGENT = f"bsmapper/{__version__} (+{PROJECT_URL}; dataset scraper)"
PAGE_SIZE = 100  # BeatSaver's maximum


class BeatSaverError(RuntimeError):
    """A request failed for good (after retries, or with a non-retryable status)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class RateLimiter:
    """Enforces a minimum interval between calls to `wait()`."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            delay = self._last + self.min_interval - now
            if delay > 0:
                self._sleep(delay)
                now += delay
        self._last = now


class BeatSaverClient:
    """Thin wrapper over the endpoints the scraper needs: map search and zip download."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl: float = 6 * 3600,
        requests_per_second: float = 2.0,
        max_retries: int = 5,
        backoff: float = 2.0,
        timeout: float = 60.0,
        user_agent: str = USER_AGENT,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self._wall_clock = time.time
        self._limiter = RateLimiter(1.0 / requests_per_second, clock=clock, sleep=sleep)
        self._http = httpx.Client(
            base_url=API_URL,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> BeatSaverClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- search --------------------------------------------------------------------------

    def search(self, page: int, params: dict[str, Any]) -> dict[str, Any]:
        """Return one page of `/search/text/{page}` as JSON (`{"docs": [...], "info": {...}}`)."""
        query = {k: _query_value(v) for k, v in params.items() if v is not None}
        query.setdefault("pageSize", PAGE_SIZE)
        cached = self._cache_get(page, query)
        if cached is not None:
            return cached
        resp = self._request("GET", f"/search/text/{page}", params=query)
        data = resp.json()
        self._cache_put(page, query, data)
        return data

    # -- download ------------------------------------------------------------------------

    def download(self, url: str, dest: Path) -> int:
        """Stream `url` to `dest` atomically (via a `.part` file). Returns the byte count."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            try:
                with self._http.stream("GET", url) as resp:
                    if self._should_retry(resp.status_code) and attempt < self.max_retries:
                        self._sleep(self._retry_delay(resp, attempt))
                        continue
                    if resp.status_code != 200:
                        raise BeatSaverError(
                            f"GET {url} -> HTTP {resp.status_code}", resp.status_code
                        )
                    size = 0
                    with part.open("wb") as f:
                        for chunk in resp.iter_bytes(1 << 16):
                            f.write(chunk)
                            size += len(chunk)
                part.replace(dest)
                return size
            except httpx.TransportError as e:
                part.unlink(missing_ok=True)
                if attempt >= self.max_retries:
                    raise BeatSaverError(f"GET {url} failed: {e}") from e
                self._sleep(self.backoff * 2**attempt)
        raise BeatSaverError(f"GET {url} failed after {self.max_retries} retries")

    # -- internals -----------------------------------------------------------------------

    @staticmethod
    def _should_retry(status: int) -> bool:
        return status == 429 or status >= 500

    def _retry_delay(self, resp: httpx.Response, attempt: int) -> float:
        retry_after = resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
        return self.backoff * 2**attempt

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            self._limiter.wait()
            try:
                resp = self._http.request(method, url, **kwargs)
            except httpx.TransportError as e:
                if attempt >= self.max_retries:
                    raise BeatSaverError(f"{method} {url} failed: {e}") from e
                self._sleep(self.backoff * 2**attempt)
                continue
            if self._should_retry(resp.status_code) and attempt < self.max_retries:
                self._sleep(self._retry_delay(resp, attempt))
                continue
            if resp.status_code != 200:
                raise BeatSaverError(f"{method} {url} -> HTTP {resp.status_code}", resp.status_code)
            return resp
        raise BeatSaverError(f"{method} {url} failed after {self.max_retries} retries")

    def _cache_path(self, page: int, query: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps({"page": page, **query}, sort_keys=True)
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / "search" / f"{page:05d}-{digest}.json"

    def _cache_get(self, page: int, query: dict[str, Any]) -> dict[str, Any] | None:
        path = self._cache_path(page, query)
        if path is None or not path.exists():
            return None
        if self._wall_clock() - path.stat().st_mtime > self.cache_ttl:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_put(self, page: int, query: dict[str, Any], data: dict[str, Any]) -> None:
        path = self._cache_path(page, query)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)


def _query_value(v: Any) -> Any:
    if isinstance(v, bool):
        return "true" if v else "false"
    return v
