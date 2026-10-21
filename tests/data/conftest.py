"""A fake BeatSaver (search + CDN) on top of httpx.MockTransport."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from bsmapper.data.beatleader import BeatLeaderClient
from bsmapper.data.beatsaver import BeatSaverClient


def make_doc(key: str, **over: Any) -> dict[str, Any]:
    """A search result shaped like api.beatsaver.com's, passing the default filters."""
    h = (key * 40)[:40]
    diff = {
        "characteristic": "Standard",
        "difficulty": "Expert",
        "notes": 700,
        "bombs": 10,
        "obstacles": 5,
        "nps": 5.0,
        "njs": 16.0,
        "offset": 0.0,
        "seconds": 140.0,
        "events": 1500,
        "chroma": False,
        "me": False,
        "ne": False,
        "cinema": False,
        "environment": "WeaveEnvironment",
        "paritySummary": {"errors": 0, "warns": 1, "resets": 2},
    }
    diff.update(over.pop("diff", {}))
    doc = {
        "id": key,
        "name": f"Map {key}",
        "uploader": {"name": "mapper"},
        "metadata": {
            "bpm": 128.0,
            "duration": 150,
            "songName": f"Song {key}",
            "songSubName": "",
            "songAuthorName": "Artist",
            "levelAuthorName": "Mapper",
        },
        "stats": {"upvotes": 200, "downvotes": 10, "score": 0.9},
        "uploaded": "2023-05-01T12:00:00.000Z",
        "automapper": False,
        "declaredAi": "None",
        "ranked": False,
        "blRanked": False,
        "tags": ["dance"],
        "versions": [
            {
                "hash": h,
                "key": key,
                "state": "Published",
                "createdAt": "2023-05-01T12:00:00.000Z",
                "diffs": [diff],
                "downloadURL": f"https://cdn.beatsaver.com/{h}.zip",
            }
        ],
    }
    for k, v in over.items():
        if k in ("score", "upvotes", "downvotes"):
            doc["stats"][k] = v
        elif k == "duration":
            doc["metadata"]["duration"] = v
        else:
            doc[k] = v
    return doc


def make_zip(info: bool = True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if info:
            zf.writestr("Info.dat", '{"_version": "2.1.0"}')
        zf.writestr("ExpertStandard.dat", '{"version": "3.3.0"}')
    return buf.getvalue()


class FakeBeatSaver:
    """Serves `docs` in pages of `page_size`; counts requests; lets tests inject failures."""

    def __init__(self, docs: list[dict[str, Any]], page_size: int = 3) -> None:
        self.docs = docs
        self.page_size = page_size
        self.search_calls: list[int] = []
        self.download_calls: list[str] = []
        self.bad_zips: set[str] = set()  # hashes served as junk
        self.missing: set[str] = set()  # hashes that 404
        self.on_download: Callable[[str], None] | None = None
        self.last_params: dict[str, str] = {}
        self.user_agents: set[str] = set()
        # BeatLeader: hash -> difficulties. Hashes not listed 404; `bl_down` makes it 500.
        self.bl_maps: dict[str, list[dict[str, Any]]] = {}
        self.bl_calls: list[str] = []
        self.bl_down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.user_agents.add(request.headers.get("user-agent", ""))
        if request.url.host == "api.beatleader.com":
            h = request.url.path.rsplit("/", 1)[1]
            self.bl_calls.append(h)
            if self.bl_down:
                return httpx.Response(500)
            if h not in self.bl_maps:
                return httpx.Response(404, json={"title": "Not Found"})
            return httpx.Response(200, json={"hash": h, "difficulties": self.bl_maps[h]})
        if request.url.host == "api.beatsaver.com":
            page = int(request.url.path.rsplit("/", 1)[1])
            self.search_calls.append(page)
            self.last_params = dict(request.url.params)
            docs = self.docs
            if "characteristics" in self.last_params:  # OR, like the real API
                chars = set(self.last_params["characteristics"].split(","))
                docs = [d for d in docs if _characteristics(d) & chars]
            chunk = docs[page * self.page_size : (page + 1) * self.page_size]
            return httpx.Response(200, json={"docs": chunk, "info": {"total": len(docs)}})
        h = request.url.path.strip("/").removesuffix(".zip")
        self.download_calls.append(h)
        if self.on_download:
            self.on_download(h)
        if h in self.missing:
            return httpx.Response(404)
        if h in self.bad_zips:
            return httpx.Response(200, content=b"<html>nope</html>")
        return httpx.Response(200, content=make_zip())

    def client(self, cache_dir=None, **kw: Any) -> BeatSaverClient:
        kw.setdefault("sleep", lambda s: None)
        kw.setdefault("requests_per_second", 1000.0)
        return BeatSaverClient(
            cache_dir=cache_dir, transport=httpx.MockTransport(self.handler), **kw
        )

    def bl_client(self, cache_dir=None, **kw: Any) -> BeatLeaderClient:
        kw.setdefault("sleep", lambda s: None)
        kw.setdefault("requests_per_second", 1000.0)
        kw.setdefault("max_retries", 1)
        return BeatLeaderClient(
            cache_dir=cache_dir, transport=httpx.MockTransport(self.handler), **kw
        )


def _characteristics(doc: dict[str, Any]) -> set[str]:
    return {d["characteristic"] for v in doc["versions"] for d in v["diffs"]}


def bl_diff(mode: str = "Standard", diff: str = "Expert", **over: Any) -> dict[str, Any]:
    """One entry of BeatLeader's `difficulties` list."""
    d = {
        "modeName": mode,
        "difficultyName": diff,
        "status": 3,
        "stars": 5.1,
        "passRating": 2.7,
        "accRating": 7.1,
        "techRating": 3.6,
        "predictedAcc": 0.98,
    }
    d.update(over)
    return d


@pytest.fixture
def fake() -> Callable[..., FakeBeatSaver]:
    return FakeBeatSaver


@pytest.fixture
def doc() -> Callable[..., dict[str, Any]]:
    return make_doc


@pytest.fixture
def bl() -> Callable[..., dict[str, Any]]:
    return bl_diff
