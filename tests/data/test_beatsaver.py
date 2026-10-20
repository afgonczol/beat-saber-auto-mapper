import httpx
import pytest

from bsmapper.data.beatsaver import USER_AGENT, BeatSaverClient, BeatSaverError, RateLimiter


def client_for(handler, sleeps, **kw) -> BeatSaverClient:
    """`sleeps` records retry sleeps; the clock jumps so the rate limiter never sleeps."""
    ticks = iter(range(0, 10**9, 1000))
    return BeatSaverClient(
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        clock=lambda: float(next(ticks)),
        **kw,
    )


def test_rate_limiter_spaces_calls():
    t = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t[0] += s

    rl = RateLimiter(0.5, clock=lambda: t[0], sleep=sleep)
    rl.wait()
    rl.wait()
    t[0] += 0.2
    rl.wait()
    t[0] += 1.0
    rl.wait()
    assert sleeps == pytest.approx([0.5, 0.3])


def test_user_agent_and_bool_params():
    seen = {}

    def handler(req):
        seen["ua"] = req.headers["user-agent"]
        seen["params"] = dict(req.url.params)
        return httpx.Response(200, json={"docs": []})

    client_for(handler, []).search(0, {"order": "Rating", "noodle": False, "skip": None})
    assert seen["ua"] == USER_AGENT and "github.com" in USER_AGENT
    assert seen["params"] == {"order": "Rating", "noodle": "false", "pageSize": "100"}


def test_retries_429_with_retry_after_then_succeeds():
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(503),
            httpx.Response(200, json={"docs": [1]}),
        ]
    )
    sleeps = []
    c = client_for(lambda req: next(responses), sleeps, backoff=1.0)
    assert c.search(0, {}) == {"docs": [1]}
    assert sleeps == [7.0, 2.0]  # Retry-After, then backoff * 2**1


def test_retries_transport_errors():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"docs": []})

    assert client_for(handler, [])._request("GET", "/x").status_code == 200
    assert len(calls) == 3


def test_gives_up_after_max_retries():
    sleeps = []
    c = client_for(lambda req: httpx.Response(500), sleeps, max_retries=2)
    with pytest.raises(BeatSaverError):
        c.search(0, {})
    assert len(sleeps) == 2


def test_404_is_not_retried():
    sleeps = []
    c = client_for(lambda req: httpx.Response(404), sleeps)
    with pytest.raises(BeatSaverError) as e:
        c.search(0, {})
    assert e.value.status == 404 and sleeps == []


def test_search_cache(tmp_path):
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json={"docs": [len(calls)]})

    c = client_for(handler, [], cache_dir=tmp_path)
    assert c.search(0, {"q": "a"}) == {"docs": [1]}
    assert c.search(0, {"q": "a"}) == {"docs": [1]}  # cached
    assert c.search(0, {"q": "b"}) == {"docs": [2]}  # different query
    assert c.search(1, {"q": "a"}) == {"docs": [3]}  # different page
    assert len(calls) == 3

    expired = client_for(handler, [], cache_dir=tmp_path, cache_ttl=-1)
    assert expired.search(0, {"q": "a"}) == {"docs": [4]}


def test_download_is_atomic(tmp_path):
    dest = tmp_path / "z" / "a.zip"
    c = client_for(lambda req: httpx.Response(200, content=b"x" * 100_000), [])
    assert c.download("https://cdn.beatsaver.com/a.zip", dest) == 100_000
    assert dest.read_bytes() == b"x" * 100_000
    assert not dest.with_name("a.zip.part").exists()

    bad = client_for(lambda req: httpx.Response(404), [])
    with pytest.raises(BeatSaverError):
        bad.download("https://cdn.beatsaver.com/b.zip", tmp_path / "z" / "b.zip")
    assert not (tmp_path / "z" / "b.zip").exists()


def test_download_retries_on_503(tmp_path):
    responses = iter([httpx.Response(503), httpx.Response(200, content=b"ok")])
    sleeps = []
    c = client_for(lambda req: next(responses), sleeps)
    assert c.download("https://cdn.beatsaver.com/a.zip", tmp_path / "a.zip") == 2
    assert len(sleeps) == 1
