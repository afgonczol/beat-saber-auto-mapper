"""Jobs that outlive the server, remaking a song, renaming it, and GET /api/settings."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bsmapper.formats import read_map  # noqa: E402
from bsmapper.schema import Beatmap  # noqa: E402
from bsmapper.server import store  # noqa: E402
from bsmapper.server.app import create_app  # noqa: E402
from bsmapper.server.jobs import JobManager  # noqa: E402
from bsmapper.server.mock import MockJobRunner  # noqa: E402
from bsmapper.server.models import JobInfo, JobOptions, JobStatus  # noqa: E402
from bsmapper.server.runner import JobResult  # noqa: E402


@pytest.fixture
def game(tmp_path: Path, monkeypatch) -> Path:
    levels = tmp_path / "Beat Saber" / "Beat Saber_Data" / "CustomLevels"
    levels.mkdir(parents=True)
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"[game]\ncustom_levels = '{levels.as_posix()}'\n\n[generate]\nintensity = 0.8\n"
        "chains = false\n\n[jump.ExpertPlus]\nreaction_time = 0.75\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BSMAPPER_CONFIG", str(cfg))
    return levels


def make_client(work: Path) -> TestClient:
    return TestClient(create_app(MockJobRunner(step_delay=0), work, work.parent / "no-ui"))


def upload(client: TestClient, name: str = "Artist - Song.mp3", **form) -> dict:
    data = {"modes": ["Standard"], "difficulties": ["Expert"], **form}
    r = client.post("/api/jobs", files={"audio": (name, io.BytesIO(b"not audio"))}, data=data)
    assert r.status_code == 201, r.text
    return r.json()


def wait_status(client: TestClient, job_id: str, *wanted: str) -> dict:
    deadline = time.monotonic() + 10
    while True:
        info = client.get(f"/api/jobs/{job_id}").json()
        if info["status"] in wanted:
            return info
        assert time.monotonic() < deadline, f"job {job_id} stuck: {info}"
        time.sleep(0.02)


def test_finished_jobs_come_back_after_a_restart(tmp_path: Path) -> None:
    work = tmp_path / "work"
    with make_client(work) as client:
        job = upload(client, seed="7")
        done = wait_status(client, job["id"], "done")
        assert done["title"] == "Song" and done["artist"] == "Artist"
        assert done["started_at"] and done["finished_at"]
        before = client.get(f"/api/jobs/{job['id']}/map").json()
    (work / "old-version-folder").mkdir()  # no job.json: ignored

    with make_client(work) as client:
        listed = client.get("/api/jobs").json()
        assert [j["id"] for j in listed] == [job["id"]]
        assert listed[0]["status"] == "done" and listed[0]["options"]["seed"] == 7
        after = client.get(f"/api/jobs/{job['id']}/map").json()
        assert after["beatmap"] == before["beatmap"]
        assert after["warnings"] == before["warnings"]
        r = client.get(f"/api/jobs/{job['id']}/download")
        assert r.status_code == 200 and zipfile.is_zipfile(io.BytesIO(r.content))
        # It can still be regenerated: the upload is kept.
        body = {"characteristic": "Standard", "difficulty": "Expert", "seed": 3}
        assert client.post(f"/api/jobs/{job['id']}/regenerate", json=body).status_code == 202
        assert wait_status(client, job["id"], "done")["revision"] == 2


class _Never:
    """A runner that never gets to run anything (the server stops first)."""

    def run(self, job, progress):
        raise AssertionError("not expected to run")

    regenerate = run


def saved_job(work: Path, job_id: str, status: JobStatus, minute: int, **info) -> Path:
    """A job folder as an earlier server left it; returns the folder."""
    job_dir = work / job_id
    (job_dir / "upload").mkdir(parents=True)
    audio = job_dir / "upload" / f"{job_id}.mp3"
    audio.write_bytes(b"x")
    saved = JobInfo(
        id=job_id,
        status=status,
        created_at=f"2026-10-28T12:{minute:02d}:00Z",
        filename=f"{job_id}.mp3",
        options=JobOptions(),
        **info,
    )
    store.save_info(job_dir, saved, audio)
    return job_dir


def test_interrupted_jobs_are_queued_again(tmp_path: Path) -> None:
    work = tmp_path / "work"
    # Saved in another order than created: restore sorts them by creation time.
    regen = saved_job(work, "three", JobStatus.RUNNING, 3)
    saved_job(work, "one", JobStatus.QUEUED, 1, install_when_done=True)
    gone = saved_job(work, "two", JobStatus.RUNNING, 2)
    (gone / "upload" / "two.mp3").unlink()
    # An interrupted regenerate: the job already has a map.
    zip_path = regen / "map.zip"
    zip_path.write_bytes(b"zip")
    store.save_result(regen, JobResult(Beatmap("three", bpm=120.0), 30.0, zip_path))

    restored = JobManager(MockJobRunner(step_delay=0), work)
    assert restored.restore() == 3
    jobs = restored.all()
    assert [j.info.id for j in jobs] == ["one", "two", "three"]
    one, two, three = jobs
    assert two.info.status == JobStatus.FAILED and "missing" in (two.info.error or "")
    assert three.info.status == JobStatus.DONE and three.result is not None
    assert three.info.message.startswith("Interrupted")
    # Queued again (restore starts the worker, so it may already be running).
    assert one.info.status != JobStatus.FAILED and one.info.install_when_done
    deadline = time.monotonic() + 10
    while one.info.status != JobStatus.DONE:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    restored.stop()
    assert one.result is not None


def test_done_job_without_its_map_is_failed(tmp_path: Path) -> None:
    saved_job(tmp_path / "work", "abc", JobStatus.DONE, 0)
    manager = JobManager(_Never(), tmp_path / "work")
    manager.restore()
    assert manager.get("abc").info.status == JobStatus.FAILED


def test_rerun_makes_a_new_job_from_the_upload(tmp_path: Path) -> None:
    with make_client(tmp_path / "work") as client:
        source = upload(client)
        wait_status(client, source["id"], "done")
        same = client.post(f"/api/jobs/{source['id']}/rerun", json={})
        assert same.status_code == 201, same.text
        same = same.json()
        assert same["id"] != source["id"] and same["source_id"] == source["id"]
        assert same["filename"] == source["filename"] and same["options"] == source["options"]

        options = {**source["options"], "seed": 42, "difficulties": ["Hard", "Expert"]}
        other = client.post(
            f"/api/jobs/{source['id']}/rerun", json={"options": options, "install": True}
        ).json()
        assert other["options"]["seed"] == 42 and other["install_when_done"] is True
        done = wait_status(client, other["id"], "done")
        diffs = client.get(f"/api/jobs/{done['id']}/map").json()["beatmap"]["difficulties"]
        assert sorted(d["difficulty"] for d in diffs) == ["Expert", "Hard"]

        assert client.post("/api/jobs/nope/rerun", json={}).status_code == 404
        bad = {"options": {**options, "intensity": 3}}
        assert client.post(f"/api/jobs/{source['id']}/rerun", json=bad).status_code == 422
        wait_status(client, same["id"], "done")
        client.delete(f"/api/jobs/{source['id']}")
        # The remake has its own copy of the song.
        assert client.post(f"/api/jobs/{same['id']}/rerun", json={}).status_code == 201


def test_song_info_renames_map_zip_and_installed_copy(tmp_path: Path, game: Path) -> None:
    with make_client(tmp_path / "work") as client:
        job = upload(client, name="Some Upload (Official Video).mp3")
        wait_status(client, job["id"], "done")
        installed = client.post(f"/api/jobs/{job['id']}/install").json()["path"]
        assert Path(installed).is_dir()

        r = client.put(f"/api/jobs/{job['id']}/song", json={"title": " Nice ", "artist": "Band"})
        assert r.status_code == 200, r.text
        info = r.json()
        assert (info["title"], info["artist"], info["revision"]) == ("Nice", "Band", 2)
        assert info["installed"]["path"] != installed and not Path(installed).exists()
        in_game = read_map(Path(info["installed"]["path"]))
        assert (in_game.song_name, in_game.song_author) == ("Nice", "Band")

        preview = client.get(f"/api/jobs/{job['id']}/map").json()["beatmap"]
        assert (preview["song_name"], preview["song_author"]) == ("Nice", "Band")
        r = client.get(f"/api/jobs/{job['id']}/download")
        assert "Nice%20-%20Band.zip" in r.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            assert b'"Nice"' in zf.read("Info.dat")
        assert len(list((tmp_path / "work" / job["id"]).glob("*.zip"))) == 1

        assert client.put(f"/api/jobs/{job['id']}/song", json={"title": ""}).status_code == 422
        assert client.put("/api/jobs/nope/song", json={"title": "x"}).status_code == 404

    # The new name is what a restarted server shows.
    with make_client(tmp_path / "work") as client:
        assert client.get(f"/api/jobs/{job['id']}").json()["title"] == "Nice"


def test_song_info_needs_a_finished_map(tmp_path: Path) -> None:
    with TestClient(create_app(_Never(), tmp_path / "work", tmp_path / "no-ui")) as client:
        job_id = upload(client)["id"]
        r = client.put(f"/api/jobs/{job_id}/song", json={"title": "x"})
        assert r.status_code == 409


def test_settings_show_config_defaults(tmp_path: Path, game: Path) -> None:
    with make_client(tmp_path / "work") as client:
        s = client.get("/api/settings").json()
    assert s["generator"] == "mock"
    assert s["intensity"] == 0.8 and s["repetition"] is None
    assert s["arcs"] is None and s["chains"] is False
    assert s["light_style"] == "auto"
    assert s["jump"]["ExpertPlus"] == {"njs": 17.0, "reaction_time": 0.75}
    assert s["jump"]["Expert"]["reaction_time"] == 0.8
    assert s["install"]["found"] is True and s["install"]["path"] == str(game)
    assert s["install"]["chroma"] is False


def test_settings_without_config_or_game(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f"[game]\ncustom_levels = '{(tmp_path / 'nope').as_posix()}'\n", "utf-8")
    monkeypatch.setenv("BSMAPPER_CONFIG", str(cfg))
    with make_client(tmp_path / "work") as client:
        s = client.get("/api/settings").json()
    assert s["intensity"] == 0.5 and s["install"]["found"] is False
    cfg.write_text("[game\n", "utf-8")
    with make_client(tmp_path / "work") as client:
        s = client.get("/api/settings").json()
    assert s["install"]["error"] and s["intensity"] == 0.5
