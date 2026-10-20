"""Shared polite-HTTP plumbing for the BeatSaver and BeatLeader clients.

One `httpx.Client`, a minimum interval between requests, retries with backoff on 429/5xx
and transport errors, and an on-disk JSON cache so a resumed run does not repeat requests.
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

PROJECT_URL = "https://github.com/afgonczol/beat-saber-auto-mapper"
USER_AGENT = f"bsmapper/{__version__} (+{PROJECT_URL}; dataset scraper)"


class HttpError(RuntimeError):
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


class PoliteClient:
    """Base for API clients: rate limiting, retries and a JSON cache under `cache_dir`."""

    base_url = ""
    cache_ttl_default = 6 * 3600.0

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl: float | None = None,
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
        self.cache_ttl = self.cache_ttl_default if cache_ttl is None else cache_ttl
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self._wall_clock = time.time
        self._limiter = RateLimiter(1.0 / requests_per_second, clock=clock, sleep=sleep)
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- requests ------------------------------------------------------------------------

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
                    raise HttpError(f"{method} {url} failed: {e}") from e
                self._sleep(self.backoff * 2**attempt)
                continue
            if self._should_retry(resp.status_code) and attempt < self.max_retries:
                self._sleep(self._retry_delay(resp, attempt))
                continue
            if resp.status_code != 200:
                raise HttpError(f"{method} {url} -> HTTP {resp.status_code}", resp.status_code)
            return resp
        raise HttpError(f"{method} {url} failed after {self.max_retries} retries")

    def _download(self, url: str, dest: Path) -> int:
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
                        raise HttpError(f"GET {url} -> HTTP {resp.status_code}", resp.status_code)
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
                    raise HttpError(f"GET {url} failed: {e}") from e
                self._sleep(self.backoff * 2**attempt)
        raise HttpError(f"GET {url} failed after {self.max_retries} retries")

    # -- cache ---------------------------------------------------------------------------

    def _cache_path(self, namespace: str, key: dict[str, Any], prefix: str = "") -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        return self.cache_dir / namespace / f"{prefix}{digest}.json"

    def _cache_get(self, path: Path | None) -> Any | None:
        if path is None or not path.exists():
            return None
        if self._wall_clock() - path.stat().st_mtime > self.cache_ttl:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _cache_put(path: Path | None, data: Any) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)


def query_value(v: Any) -> Any:
    if isinstance(v, bool):
        return "true" if v else "false"
    return v
