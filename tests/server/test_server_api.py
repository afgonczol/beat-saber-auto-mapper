"""Jobs API against the mock runner. Uploads are fake bytes, so packaging skips ffmpeg."""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bsmapper.server.app import create_app  # noqa: E402
from bsmapper.server.mock import MockJobRunner  # noqa: E402
from bsmapper.server.openapi import SPEC_PATH, dumps  # noqa: E402


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(MockJobRunner(step_delay=0), tmp_path / "work", tmp_path / "no-ui")
    with TestClient(app) as c:
        yield c


def upload(client: TestClient, name: str = "Artist - Song.mp3", **form) -> dict:
    data = {"modes": ["Standard"], "difficulties": ["Expert", "ExpertPlus"], **form}
    r = client.post("/api/jobs", files={"audio": (name, io.BytesIO(b"not audio"))}, data=data)
    assert r.status_code == 201, r.text
    return r.json()


def wait_done(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = client.get(f"/api/jobs/{job_id}").json()
        if info["status"] in ("done", "failed"):
            return info
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {info}")


def read_sse(client: TestClient, url: str) -> list[tuple[str, dict]]:
    events = []
    with client.stream("GET", url) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        kind = None
        for line in r.iter_lines():
            if line.startswith("event: "):
                kind = line[len("event: ") :]
            elif line.startswith("data: "):
                events.append((kind, json.loads(line[len("data: ") :])))
    return events


def test_create_job_returns_options(client: TestClient) -> None:
    info = upload(client, lighting="chroma", style="0.25", seed="7")
    assert info["status"] in ("queued", "running", "done")
    assert info["filename"] == "Artist - Song.mp3"
    assert info["options"] == {
        "modes": ["Standard"],
        "difficulties": ["Expert", "ExpertPlus"],
        "lighting": "chroma",
        "style": 0.25,
        "seed": 7,
        "intensity": None,
        "repetition": None,
        "light_style": None,
        "arcs": None,
        "chains": None,
        "njs": None,
        "reaction_time": None,
        "environment": None,
        "title": None,
        "artist": None,
    }
    info = upload(client, arcs="false", chains="true")
    assert (info["options"]["arcs"], info["options"]["chains"]) == (False, True)
    assert upload(client, light_style="calm")["options"]["light_style"] == "calm"
    r = client.post("/api/jobs", files={"audio": ("a.mp3", b"x", "audio/mpeg")},
                    data={"light_style": "disco"})  # fmt: skip
    assert r.status_code == 422


def test_environment_option(client: TestClient) -> None:
    info = upload(client, environment="LizzoEnvironment")
    assert info["options"]["environment"] == "LizzoEnvironment"
    r = client.post("/api/jobs", files={"audio": ("a.mp3", b"x", "audio/mpeg")},
                    data={"environment": "../evil"})  # fmt: skip
    assert r.status_code == 422


def test_environment_list(client: TestClient, tmp_path: Path) -> None:
    from bsmapper.server.app import environment_label, environment_options

    assert environment_label("TheWeekndEnvironment") == "The Weeknd"
    assert environment_label("Dragons2Environment") == "Dragons 2"
    assert environment_label("EDMEnvironment") == "EDM"
    root = tmp_path / "fx"
    for env, n in (("LizzoEnvironment", 40), ("PyroEnvironment", 31), ("QueenEnvironment", 5)):
        (root / env).mkdir(parents=True)
        index = {"maps": [{"key": str(i)} for i in range(n)]}
        (root / env / "index.json").write_text(json.dumps(index), encoding="utf-8")
    envs = environment_options(root)
    assert [(e.name, e.maps, e.auto) for e in envs[:3]] == [
        ("LizzoEnvironment", 40, True),
        ("PyroEnvironment", 31, True),
        ("QueenEnvironment", 5, False),
    ]
    assert {e.name for e in envs[3:]} >= {"WeaveEnvironment"}  # template environments
    assert all(e.maps == 0 and not e.auto for e in envs[3:])
    without = environment_options(None)
    assert [e.name for e in without if e.auto] == ["WeaveEnvironment"]
    r = client.get("/api/environments")
    assert r.status_code == 200 and all("label" in e for e in r.json())


def test_defaults_and_validation(client: TestClient) -> None:
    files = {"audio": ("a.ogg", io.BytesIO(b"x"))}
    r = client.post("/api/jobs", files=files)
    assert r.status_code == 201
    assert r.json()["options"]["style"] is None
    assert r.json()["options"]["modes"] == ["Standard"]

    bad = [{"style": "1.5"}, {"lighting": "disco"}, {"difficulties": ["Impossible"]}]
    for data in bad:
        r = client.post("/api/jobs", files={"audio": ("a.ogg", io.BytesIO(b"x"))}, data=data)
        assert r.status_code == 422, data
    r = client.post("/api/jobs", files={"audio": ("notes.txt", io.BytesIO(b"x"))})
    assert r.status_code == 415
    assert client.post("/api/jobs").status_code == 422


def test_full_flow_events_map_download(client: TestClient) -> None:
    info = upload(client, modes=["Standard", "360Degree"], lighting="v3")
    job_id = info["id"]
    events = read_sse(client, f"/api/jobs/{job_id}/events")
    kinds = [(k, e["status"]) for k, e in events]
    assert kinds[0] == ("status", "queued")
    assert kinds[-1] == ("status", "done")
    stages = {e["stage"] for k, e in events if k == "progress"}
    assert {"analyze", "generate", "lighting", "rules", "package"} <= stages
    seqs = [e["seq"] for _, e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "done" and status["progress"] == 1.0 and status["revision"] == 1

    preview = client.get(f"/api/jobs/{job_id}/map").json()
    assert preview["revision"] == 1
    assert preview["duration_s"] == 60.0  # fake audio: fallback length, with a note about it
    assert any("song length" in n for n in preview["notes"])
    bm = preview["beatmap"]
    assert bm["song_name"] == "Song" and bm["song_author"] == "Artist"
    got = {(d["characteristic"], d["difficulty"]) for d in bm["difficulties"]}
    assert got == {(m, d) for m in ("Standard", "360Degree") for d in ("Expert", "ExpertPlus")}
    for d in bm["difficulties"]:
        assert d["notes"], d["difficulty"]
        beats = [n["beat"] for n in d["notes"]]
        assert beats == sorted(beats)
        assert all(n["color"] in (0, 1) and 0 <= n["x"] <= 3 for n in d["notes"])
        assert bool(d["rotations"]) == (d["characteristic"] == "360Degree")
    assert bm["lightshow"]["basic_events"]
    assert preview["warnings"] and {"characteristic", "difficulty", "issue"} <= set(
        preview["warnings"][0]
    )

    r = client.get(f"/api/jobs/{job_id}/download")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "Song.zip" in r.headers["content-disposition"]
    names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert {"Info.dat", "cover.png", "ExpertPlusStandard.dat", "Expert360Degree.dat"} <= names

    audio = client.get(f"/api/jobs/{job_id}/audio")
    assert audio.status_code == 200 and audio.content == b"not audio"
    assert audio.headers["content-type"].startswith("audio/")


def test_events_replay_after_seq(client: TestClient) -> None:
    job_id = upload(client)["id"]
    wait_done(client, job_id)
    all_events = read_sse(client, f"/api/jobs/{job_id}/events")
    tail = read_sse(client, f"/api/jobs/{job_id}/events?after={all_events[-3][1]['seq']}")
    assert [e for _, e in tail] == [e for _, e in all_events[-2:]]
    assert read_sse(client, f"/api/jobs/{job_id}/events?after=100000") == []


def test_regenerate_one_difficulty(client: TestClient) -> None:
    job_id = upload(client, seed="1")["id"]
    wait_done(client, job_id)
    before = client.get(f"/api/jobs/{job_id}/map").json()
    last_seq = read_sse(client, f"/api/jobs/{job_id}/events")[-1][1]["seq"]

    body = {"characteristic": "Standard", "difficulty": "Expert", "seed": 99, "temperature": 2.0}
    r = client.post(f"/api/jobs/{job_id}/regenerate", json=body)
    assert r.status_code == 202, r.text
    events = read_sse(client, f"/api/jobs/{job_id}/events?after={last_seq}")
    assert events[0][1]["status"] == "queued" and events[-1][1]["status"] == "done"

    after = client.get(f"/api/jobs/{job_id}/map").json()
    assert after["revision"] == before["revision"] + 1

    def diff(preview, name):
        return next(d for d in preview["beatmap"]["difficulties"] if d["difficulty"] == name)

    assert diff(after, "ExpertPlus") == diff(before, "ExpertPlus")
    assert diff(after, "Expert") != diff(before, "Expert")


def test_regenerate_validation_and_busy(tmp_path: Path) -> None:
    app = create_app(MockJobRunner(step_delay=0.2), tmp_path / "work", tmp_path / "no-ui")
    with TestClient(app) as client:
        job_id = upload(client)["id"]
        body = {"characteristic": "Standard", "difficulty": "Expert", "seed": 1}
        assert client.post(f"/api/jobs/{job_id}/regenerate", json=body).status_code == 409
        assert client.get(f"/api/jobs/{job_id}/map").status_code == 409
        assert client.get(f"/api/jobs/{job_id}/download").status_code == 409
        bad = {**body, "temperature": 0}
        assert client.post(f"/api/jobs/{job_id}/regenerate", json=bad).status_code == 422


def test_unknown_job_is_404(client: TestClient) -> None:
    for path in ("", "/events", "/map", "/download", "/audio"):
        assert client.get(f"/api/jobs/nope{path}").status_code == 404
    body = {"characteristic": "Standard", "difficulty": "Expert", "seed": 1}
    assert client.post("/api/jobs/nope/regenerate", json=body).status_code == 404


class Boom:
    def run(self, job, progress):
        progress("analyze", 0.1, "about to fail")
        raise RuntimeError("model exploded")

    def regenerate(self, job, previous, request, progress):
        raise NotImplementedError


def test_runner_failure_is_reported(tmp_path: Path) -> None:
    app = create_app(Boom(), tmp_path / "work", tmp_path / "no-ui")
    with TestClient(app) as client:
        job_id = upload(client)["id"]
        info = wait_done(client, job_id)
        assert info["status"] == "failed" and info["error"] == "model exploded"
        events = read_sse(client, f"/api/jobs/{job_id}/events")
        assert events[-1][1]["status"] == "failed"
        # The worker keeps serving after a failure.
        assert wait_done(client, upload(client)["id"])["status"] == "failed"


def test_serves_built_ui(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>ui</html>")
    with TestClient(create_app(MockJobRunner(step_delay=0), tmp_path / "w", dist)) as client:
        assert client.get("/").text == "<html>ui</html>"
        assert client.get("/api/jobs/nope").status_code == 404
    with TestClient(create_app(MockJobRunner(step_delay=0), tmp_path / "w", tmp_path)) as client:
        assert "npm run build" in client.get("/").text


def test_openapi_spec_is_current() -> None:
    assert SPEC_PATH.read_text(encoding="utf-8") == dumps(), (
        "web/openapi.json is stale: run python -m bsmapper.server.openapi"
    )
