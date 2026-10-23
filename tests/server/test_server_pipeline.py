"""Jobs API on the real pipeline runner, with a fake analyzer and ffmpeg-free packaging."""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bsmapper.formats import write_map, zip_map  # noqa: E402
from bsmapper.generation import Pipeline  # noqa: E402
from bsmapper.server.app import create_app  # noqa: E402
from bsmapper.server.pipeline_runner import STUB_NOTE, PipelineJobRunner  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "generation"))
from gen_synthetic import FakeAnalyzer  # noqa: E402


class NoFfmpegPipeline(Pipeline):
    def package(self, song, beatmap, map_dir, zip_path=None, version="v3"):
        write_map(beatmap, map_dir, version)
        (map_dir / "song.egg").write_bytes(b"padded egg")
        return zip_map(map_dir, zip_path) if zip_path else map_dir


@pytest.fixture
def setup(tmp_path: Path):
    analyzer = FakeAnalyzer(duration=20.0, offset_s=0.25)
    runner = PipelineJobRunner(NoFfmpegPipeline(analyzer=analyzer))
    app = create_app(runner, tmp_path / "work", tmp_path / "no-ui")
    with TestClient(app) as client:
        yield client, analyzer


def wait_done(client: TestClient, job_id: str, revision: int = 0) -> dict:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        info = client.get(f"/api/jobs/{job_id}").json()
        if info["status"] in ("done", "failed") and info["revision"] > revision:
            return info
        if info["status"] == "failed":
            return info
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {info}")


def test_real_runner_end_to_end(setup) -> None:
    client, analyzer = setup
    data = {"modes": ["Standard"], "difficulties": ["Hard", "Expert"], "lighting": "v3"}
    files = {"audio": ("Artist - Song.mp3", io.BytesIO(b"fake mp3"))}
    job = client.post("/api/jobs", files=files, data=data).json()
    info = wait_done(client, job["id"])
    assert info["status"] == "done", info

    preview = client.get(f"/api/jobs/{job['id']}/map").json()
    beatmap = preview["beatmap"]
    assert beatmap["song_name"] == "Song" and beatmap["mapper"] == "bsmapper (stub)"
    assert [d["difficulty"] for d in beatmap["difficulties"]] == ["Hard", "Expert"]
    assert [d["label"] for d in beatmap["difficulties"]] == ["Stub Hard", "Stub Expert"]
    assert all(d["notes"] for d in beatmap["difficulties"])
    assert STUB_NOTE in preview["notes"]
    assert preview["duration_s"] == pytest.approx(20.0 + 1.75)  # padded by the lead-in
    assert client.get(f"/api/jobs/{job['id']}/audio").content == b"padded egg"
    assert client.get(f"/api/jobs/{job['id']}/download").headers["content-type"] == (
        "application/zip"
    )

    before = {d["difficulty"]: d["notes"] for d in beatmap["difficulties"]}
    body = {"characteristic": "Standard", "difficulty": "Hard", "seed": 7, "temperature": 1.2}
    assert client.post(f"/api/jobs/{job['id']}/regenerate", json=body).status_code == 202
    info = wait_done(client, job["id"], revision=info["revision"])
    assert info["status"] == "done", info
    after = client.get(f"/api/jobs/{job['id']}/map").json()["beatmap"]["difficulties"]
    assert after[0]["notes"] != before["Hard"]
    assert after[1]["notes"] == before["Expert"]
    assert len(after[0]["notes"]) < len(after[1]["notes"])
    assert analyzer.calls == 1  # the regenerate reused the cached analysis


def test_job_options_set_note_timing(monkeypatch, tmp_path: Path) -> None:
    from bsmapper.generation import JumpOverride
    from bsmapper.schema import Difficulty
    from bsmapper.server.models import JobOptions
    from bsmapper.server.pipeline_runner import _options

    monkeypatch.setenv("BSMAPPER_CONFIG", str(tmp_path / "none.toml"))
    opts = _options(JobOptions(njs=15.0, reaction_time=0.9))
    assert opts.jump_for(Difficulty.EXPERT_PLUS) == JumpOverride(15.0, 0.9)
    assert opts.jump_for(Difficulty.HARD) == JumpOverride(15.0, None)
    with pytest.raises(ValueError):
        JobOptions(reaction_time=5.0)


def test_bad_upload_fails_the_job(tmp_path: Path) -> None:
    runner = PipelineJobRunner(Pipeline(tracker="librosa"))
    app = create_app(runner, tmp_path / "work", tmp_path / "no-ui")
    with TestClient(app) as client:
        files = {"audio": ("song.mp3", io.BytesIO(b"definitely not audio"))}
        job = client.post("/api/jobs", files=files, data={"modes": ["Standard"]}).json()
        info = wait_done(client, job["id"])
        assert info["status"] == "failed" and info["error"]


def test_job_intensity_falls_back_to_the_config(monkeypatch, tmp_path: Path) -> None:
    from bsmapper.server.models import JobOptions
    from bsmapper.server.pipeline_runner import _options

    cfg = tmp_path / "config.toml"
    cfg.write_text("[generate]\nintensity = 0.8\n", encoding="utf-8")
    monkeypatch.setenv("BSMAPPER_CONFIG", str(cfg))
    assert _options(JobOptions()).intensity == 0.8
    assert _options(JobOptions(intensity=0.3)).intensity == 0.3
