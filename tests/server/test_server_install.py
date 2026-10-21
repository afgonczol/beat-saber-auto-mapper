"""POST /api/jobs/{id}/install into a tmp CustomLevels set through BSMAPPER_CONFIG."""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bsmapper.generation import safe_folder_name  # noqa: E402
from bsmapper.server.app import create_app  # noqa: E402
from bsmapper.server.mock import MockJobRunner  # noqa: E402


@pytest.fixture
def game(tmp_path: Path, monkeypatch) -> Path:
    """A fake Beat Saber install; returns its CustomLevels folder."""
    levels = tmp_path / "Beat Saber" / "Beat Saber_Data" / "CustomLevels"
    levels.mkdir(parents=True)
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"[game]\ncustom_levels = '{levels.as_posix()}'\n", encoding="utf-8")
    monkeypatch.setenv("BSMAPPER_CONFIG", str(cfg))
    return levels


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(MockJobRunner(step_delay=0), tmp_path / "work", tmp_path / "no-ui")
    with TestClient(app) as c:
        yield c


def finished_job(client: TestClient, lighting: str = "v3") -> str:
    data = {"modes": ["Standard"], "difficulties": ["Expert"], "lighting": lighting}
    files = {"audio": ("Artist - Song.mp3", io.BytesIO(b"not audio"))}
    job_id = client.post("/api/jobs", files=files, data=data).json()["id"]
    deadline = time.monotonic() + 10
    while client.get(f"/api/jobs/{job_id}").json()["status"] != "done":
        assert time.monotonic() < deadline, "job did not finish"
        time.sleep(0.02)
    return job_id


def test_install_copies_and_replaces(client: TestClient, game: Path, tmp_path: Path) -> None:
    job_id = finished_job(client)
    beatmap = client.get(f"/api/jobs/{job_id}/map").json()["beatmap"]
    expected = game / safe_folder_name(beatmap["song_name"], beatmap["song_author"])

    r = client.post(f"/api/jobs/{job_id}/install")
    assert r.status_code == 200, r.text
    assert r.json() == {"path": str(expected), "warnings": []}
    assert (expected / "Info.dat").is_file()
    assert (expected / "ExpertStandard.dat").is_file()
    assert not (tmp_path / "work" / job_id / "install").exists()  # staging cleaned up

    # Installing again replaces our earlier copy instead of adding "(2)".
    (expected / "stale.txt").write_text("old")
    assert client.post(f"/api/jobs/{job_id}/install").json()["path"] == str(expected)
    assert not (expected / "stale.txt").exists()
    assert [p.name for p in game.iterdir()] == [expected.name]


def test_install_warns_when_chroma_is_missing(client: TestClient, game: Path) -> None:
    job_id = finished_job(client, lighting="chroma")
    warnings = client.post(f"/api/jobs/{job_id}/install").json()["warnings"]
    assert len(warnings) == 1 and "Chroma.dll" in warnings[0]

    plugins = game.parents[1] / "Plugins"
    plugins.mkdir()
    (plugins / "Chroma.dll").write_bytes(b"")
    assert client.post(f"/api/jobs/{job_id}/install").json()["warnings"] == []


def test_install_errors(client: TestClient, game: Path, tmp_path: Path, monkeypatch) -> None:
    assert client.post("/api/jobs/nope/install").status_code == 404

    job_id = finished_job(client)
    game.rmdir()
    r = client.post(f"/api/jobs/{job_id}/install")
    assert r.status_code == 503 and "CustomLevels" in r.json()["detail"]

    bad = tmp_path / "bad.toml"
    bad.write_text("[game\n", encoding="utf-8")
    monkeypatch.setenv("BSMAPPER_CONFIG", str(bad))
    assert client.post(f"/api/jobs/{job_id}/install").status_code == 503


def test_install_before_the_map_is_ready(tmp_path: Path, game: Path) -> None:
    app = create_app(MockJobRunner(step_delay=0.2), tmp_path / "work", tmp_path / "no-ui")
    with TestClient(app) as client:
        files = {"audio": ("song.mp3", io.BytesIO(b"not audio"))}
        job_id = client.post("/api/jobs", files=files, data={"modes": ["Standard"]}).json()["id"]
        assert client.post(f"/api/jobs/{job_id}/install").status_code == 409
    assert list(game.iterdir()) == []
