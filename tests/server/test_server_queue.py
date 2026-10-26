"""The job queue with many songs: order, cancel, remove, failures and install when done."""

from __future__ import annotations

import contextlib
import io
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bsmapper.schema import Beatmap  # noqa: E402
from bsmapper.server.app import create_app  # noqa: E402
from bsmapper.server.jobs import (  # noqa: E402
    Job,
    JobBusyError,
    JobManager,
    JobNotActiveError,
)
from bsmapper.server.models import (  # noqa: E402
    InstallResult,
    JobOptions,
    JobStatus,
    RegenerateRequest,
)
from bsmapper.server.runner import JobInput, JobResult  # noqa: E402


class GatedRunner:
    """Each run reports progress, then waits until its song is released (or fails on cue)."""

    def __init__(self, gated: bool = True) -> None:
        self.gated = gated
        self.gates: dict[str, threading.Event] = {}
        self.started: list[str] = []
        self.running = 0
        self.max_running = 0
        self._lock = threading.Lock()

    def gate(self, name: str) -> threading.Event:
        with self._lock:
            return self.gates.setdefault(name, threading.Event())

    def release(self, name: str) -> None:
        self.gate(name).set()

    def run(self, job: JobInput, progress) -> JobResult:
        name = job.audio_path.stem
        with self._lock:
            self.started.append(name)
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            progress("analyze", 0.1, "Analysing")
            if self.gated:
                assert self.gate(name).wait(10), f"{name} was never released"
            progress("generate", 0.5, "Generating")
            if name.startswith("bad"):
                raise RuntimeError(f"cannot read {name}")
            zip_path = job.work_dir / "map.zip"
            zip_path.write_bytes(b"zip")
            return JobResult(Beatmap(song_name=name, bpm=120.0), 60.0, zip_path)
        finally:
            with self._lock:
                self.running -= 1

    def regenerate(self, job, previous, request, progress) -> JobResult:
        return self.run(job, progress)


def wait_for(cond, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def status(job: Job) -> JobStatus:
    with job.lock:
        return job.info.status


@pytest.fixture
def runner() -> GatedRunner:
    return GatedRunner()


@pytest.fixture
def installs() -> list[str]:
    return []


@pytest.fixture
def manager(runner: GatedRunner, installs: list[str], tmp_path: Path):
    def installer(job: JobInput, result: JobResult) -> InstallResult:
        if result.beatmap.song_name.startswith("noinstall"):
            raise FileNotFoundError("CustomLevels not found")
        installs.append(result.beatmap.song_name)
        return InstallResult(path=f"CustomLevels/{result.beatmap.song_name}")

    m = JobManager(runner, tmp_path / "work", installer=installer)
    m.start()
    yield m
    for gate in list(runner.gates.values()):
        gate.set()
    m.stop()


def submit(manager: JobManager, name: str, **kw) -> Job:
    return manager.submit(f"{name}.mp3", io.BytesIO(b"x"), JobOptions(), **kw)


def test_many_jobs_run_in_order_one_at_a_time(manager: JobManager, runner: GatedRunner) -> None:
    runner.gated = False
    jobs = [submit(manager, f"song{i:02d}") for i in range(30)]
    wait_for(lambda: all(status(j) == JobStatus.DONE for j in jobs))
    assert runner.started == [f"song{i:02d}" for i in range(30)]
    assert runner.max_running == 1
    assert [j.info.id for j in manager.all()] == [j.info.id for j in jobs]


def test_cancel_queued_job_skips_it(manager: JobManager, runner: GatedRunner) -> None:
    first, second, third = (submit(manager, n) for n in ("first", "second", "third"))
    wait_for(lambda: status(first) == JobStatus.RUNNING)
    manager.cancel(second)
    assert status(second) == JobStatus.CANCELLED
    runner.release("first")
    runner.release("third")
    wait_for(lambda: status(third) == JobStatus.DONE)
    assert runner.started == ["first", "third"]
    assert status(first) == JobStatus.DONE and second.result is None
    with pytest.raises(JobNotActiveError):
        manager.cancel(second)
    with pytest.raises(JobNotActiveError):
        manager.cancel(first)


def test_cancel_running_job_stops_at_next_progress(
    manager: JobManager, runner: GatedRunner
) -> None:
    first, second = submit(manager, "first"), submit(manager, "second")
    wait_for(lambda: status(first) == JobStatus.RUNNING and runner.started == ["first"])
    manager.cancel(first)
    assert first.info.message == "Cancelling…"
    runner.release("first")  # it reports progress next, and stops there
    wait_for(lambda: status(first) == JobStatus.CANCELLED)
    assert first.result is None
    runner.release("second")
    wait_for(lambda: status(second) == JobStatus.DONE)


def test_cancel_is_not_swallowed_by_runner_fallbacks(tmp_path: Path) -> None:
    class FallbackRunner(GatedRunner):
        def run(self, job, progress):
            self.gate("song").wait(10)
            # Like the pipeline's "any model failure: the rules light the map" fallbacks.
            with contextlib.suppress(Exception):
                progress("lighting", 0.6, "Model lights")
            return super().run(job, progress)

    runner = FallbackRunner(gated=False)
    manager = JobManager(runner, tmp_path / "work")
    job = submit(manager, "song")
    wait_for(lambda: status(job) == JobStatus.RUNNING)
    manager.cancel(job)
    runner.release("song")
    wait_for(lambda: status(job) == JobStatus.CANCELLED)
    assert runner.started == []
    manager.stop()


def test_one_failure_does_not_stop_the_queue(manager: JobManager, runner: GatedRunner) -> None:
    runner.gated = False
    jobs = [submit(manager, n) for n in ("one", "bad two", "three")]
    wait_for(lambda: status(jobs[2]) == JobStatus.DONE)
    assert [status(j) for j in jobs] == [JobStatus.DONE, JobStatus.FAILED, JobStatus.DONE]
    assert jobs[1].info.error == "cannot read bad two"


def test_remove_queued_and_finished_jobs(manager: JobManager, runner: GatedRunner) -> None:
    first, second, third = (submit(manager, n) for n in ("first", "second", "third"))
    wait_for(lambda: status(first) == JobStatus.RUNNING)
    with pytest.raises(JobBusyError):
        manager.remove(first)
    second_dir = second.input.work_dir
    assert second_dir.is_dir()
    manager.remove(second)
    assert manager.get(second.info.id) is None and not second_dir.exists()
    runner.release("first")
    runner.release("third")
    wait_for(lambda: status(third) == JobStatus.DONE)
    assert runner.started == ["first", "third"]
    manager.remove(first)
    assert [j.info.id for j in manager.all()] == [third.info.id]


def test_install_when_done(manager: JobManager, runner: GatedRunner, installs: list[str]) -> None:
    runner.gated = False
    auto = submit(manager, "auto", install_when_done=True)
    manual = submit(manager, "manual")
    broken = submit(manager, "noinstall", install_when_done=True)
    wait_for(lambda: all(status(j) == JobStatus.DONE for j in (auto, manual, broken)))
    assert installs == ["auto"]
    assert auto.info.installed == InstallResult(path="CustomLevels/auto")
    assert auto.info.message == "Done, installed to the game"
    assert manual.info.installed is None and manual.info.install_error is None
    # A failed install does not fail the job: the map is there to download or retry.
    assert broken.result is not None and broken.info.installed is None
    assert broken.info.install_error == "CustomLevels not found"
    # A regenerate does not reinstall by itself.
    request = RegenerateRequest(characteristic="Standard", difficulty="Expert", seed=1)
    manager.regenerate(auto, request)
    wait_for(lambda: status(auto) == JobStatus.DONE and auto.info.revision == 2)
    assert installs == ["auto"]


def test_cancelled_regenerate_keeps_the_map(manager: JobManager, runner: GatedRunner) -> None:
    first = submit(manager, "first")
    runner.release("first")
    wait_for(lambda: status(first) == JobStatus.DONE)
    blocker = submit(manager, "blocker")
    wait_for(lambda: status(blocker) == JobStatus.RUNNING)
    request = RegenerateRequest(characteristic="Standard", difficulty="Expert", seed=1)
    manager.regenerate(first, request)
    assert status(first) == JobStatus.QUEUED
    manager.cancel(first)
    assert status(first) == JobStatus.DONE and first.info.message == "Cancelled"
    assert first.info.revision == 1
    runner.release("blocker")
    wait_for(lambda: status(blocker) == JobStatus.DONE)
    assert runner.started == ["first", "blocker"]


def test_stop_finishes_the_running_job_only(manager: JobManager, runner: GatedRunner) -> None:
    first, second = submit(manager, "first"), submit(manager, "second")
    wait_for(lambda: status(first) == JobStatus.RUNNING)
    runner.release("first")
    runner.release("second")
    start = time.monotonic()
    manager.stop()
    assert time.monotonic() - start < 4
    assert status(first) == JobStatus.DONE
    assert status(second) == JobStatus.QUEUED and runner.started == ["first"]


# --- the HTTP API --------------------------------------------------------------------------


@pytest.fixture
def client(runner: GatedRunner, tmp_path: Path):
    app = create_app(runner, tmp_path / "work", tmp_path / "no-ui")
    with TestClient(app) as c:
        yield c
        for gate in list(runner.gates.values()):
            gate.set()


def post(client: TestClient, name: str, **form) -> dict:
    files = {"audio": (f"{name}.mp3", io.BytesIO(b"x"))}
    r = client.post("/api/jobs", files=files, data={"difficulties": ["Expert"], **form})
    assert r.status_code == 201, r.text
    return r.json()


def test_api_queue_list_cancel_and_remove(client: TestClient, runner: GatedRunner) -> None:
    ids = [post(client, n)["id"] for n in ("first", "second", "third")]
    assert post(client, "fourth", install="true")["install_when_done"] is True
    wait_for(lambda: client.get(f"/api/jobs/{ids[0]}").json()["status"] == "running")
    listed = client.get("/api/jobs").json()
    assert [j["filename"] for j in listed] == ["first.mp3", "second.mp3", "third.mp3", "fourth.mp3"]
    assert [j["status"] for j in listed] == ["running", "queued", "queued", "queued"]
    assert listed[1]["install_when_done"] is False

    r = client.post(f"/api/jobs/{ids[1]}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert client.post(f"/api/jobs/{ids[1]}/cancel").status_code == 409
    assert client.delete(f"/api/jobs/{ids[0]}").status_code == 409  # running
    assert client.delete(f"/api/jobs/{ids[2]}").status_code == 204  # queued
    assert client.get(f"/api/jobs/{ids[2]}").status_code == 404
    assert client.delete(f"/api/jobs/{ids[2]}").status_code == 404
    assert [j["filename"] for j in client.get("/api/jobs").json()] == [
        "first.mp3",
        "second.mp3",
        "fourth.mp3",
    ]
    assert client.post("/api/jobs/nope/cancel").status_code == 404

    r = client.post(f"/api/jobs/{ids[0]}/cancel")
    assert r.status_code == 200 and r.json()["message"] == "Cancelling…"
    runner.release("first")
    wait_for(lambda: client.get(f"/api/jobs/{ids[0]}").json()["status"] == "cancelled")
    # The SSE stream ends on the cancelled status, like on done or failed.
    with client.stream("GET", f"/api/jobs/{ids[0]}/events") as stream:
        lines = list(stream.iter_lines())
    assert any('"status":"cancelled"' in line for line in lines)
    wait_for(lambda: runner.started == ["first", "fourth"])
