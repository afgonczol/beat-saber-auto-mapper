"""FastAPI app: the jobs API under ``/api`` and the built web UI (``web/dist``) at ``/``."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from bsmapper import __version__
from bsmapper.interfaces import LightingStyle
from bsmapper.schema import Characteristic, Difficulty
from bsmapper.server.install import InstallUnavailable, MapNotReady
from bsmapper.server.jobs import TERMINAL, Job, JobBusyError, JobManager, JobNotActiveError
from bsmapper.server.models import (
    EnvironmentInfo,
    ErrorResponse,
    Feedback,
    InstallResult,
    JobEvent,
    JobInfo,
    JobOptions,
    LightStyle,
    MapPreview,
    RegenerateRequest,
    RerunRequest,
    ServerSettings,
    SongInfo,
)
from bsmapper.server.runner import JobRunner
from bsmapper.server.settings import server_settings

REPO_WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
DEFAULT_WORK_DIR = Path("out") / "server"
SSE_POLL_S = 0.1
SSE_KEEPALIVE_S = 15.0

_DEFAULTS = JobOptions()
NOT_FOUND = {404: {"model": ErrorResponse, "description": "No such job."}}
NOT_READY = {409: {"model": ErrorResponse, "description": "The job has no finished map yet."}}


def create_app(
    runner: JobRunner | None = None,
    work_dir: Path | None = None,
    web_dist: Path | None = None,
) -> FastAPI:
    """Build the app. Defaults: the real pipeline, ``out/server`` and the repo's ``web/dist``."""
    if runner is None:
        from bsmapper.server.pipeline_runner import PipelineJobRunner

        runner = PipelineJobRunner()
    manager = JobManager(runner, work_dir or DEFAULT_WORK_DIR)
    if web_dist is None:
        env = os.environ.get("BSMAPPER_WEB_DIST")
        web_dist = Path(env) if env else REPO_WEB_DIST

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        manager.restore()  # finished maps and interrupted songs from the last run
        manager.start()
        yield
        manager.stop()

    app = FastAPI(
        title="bsmapper",
        version=__version__,
        description="Local jobs API behind the bsmapper web UI. Map times are in beats.",
        lifespan=lifespan,
    )
    app.state.jobs = manager
    app.include_router(_api(manager), prefix="/api")
    _add_event_schema(app)

    if (web_dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    else:

        @app.get("/", include_in_schema=False)
        def no_ui() -> HTMLResponse:
            return HTMLResponse(
                "<h1>bsmapper</h1><p>The web UI is not built. Run <code>npm install && "
                "npm run build</code> in <code>web/</code>, or see <a href='/docs'>/docs</a> "
                "for the API.</p>"
            )

    return app


def environment_label(name: str) -> str:
    """``TheWeekndEnvironment`` -> ``The Weeknd``, ``Dragons2Environment`` -> ``Dragons 2``."""
    base = re.sub(r"Environment$", "", name) or name
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])", " ", base)


def environment_options(effects_root: Path | None) -> list[EnvironmentInfo]:
    """``GET /api/environments`` for the effect libraries under ``effects_root`` (if any)."""
    from bsmapper.lighting import DEFAULT_V3_ENVIRONMENT, EffectRenderer
    from bsmapper.lighting.profiles import available_environments

    out: list[EnvironmentInfo] = []
    if effects_root is not None:
        effects = EffectRenderer(effects_root)
        auto = set(effects.auto_environments())
        out = [
            EnvironmentInfo(name=env, label=environment_label(env), maps=n, auto=env in auto)
            for env, n in effects.map_counts().items()
        ]
        out.sort(key=lambda e: (-e.maps, e.label))
    known = {e.name for e in out}
    for env in available_environments():
        if env not in known:
            auto = not known and env == DEFAULT_V3_ENVIRONMENT
            out.append(EnvironmentInfo(name=env, label=environment_label(env), maps=0, auto=auto))
    return out


def _api(manager: JobManager) -> APIRouter:
    router = APIRouter(tags=["jobs"])

    def get_job(job_id: str) -> Job:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        return job

    def snapshot(job: Job) -> JobInfo:
        with job.lock:
            return job.info.model_copy(deep=True)

    @router.get("/environments", tags=["options"])
    def list_environments() -> list[EnvironmentInfo]:
        """Environments v3 lighting can use: those with human light effects (most source
        maps first), then those with the built-in template."""
        from bsmapper.generation.runtime import find_light_effects

        return environment_options(find_light_effects())

    @router.get("/settings", tags=["options"])
    def get_settings() -> ServerSettings:
        """Defaults for the options a job leaves out, where notes come from, and where maps
        install (``~/.bsmapper/config.toml``)."""
        describe = getattr(manager.runner, "describe", None)
        generator, detail = describe() if describe else ("unknown", "")
        return server_settings(generator, detail)

    @router.get("/jobs")
    def list_jobs() -> list[JobInfo]:
        """Every job the server knows, oldest first: the queue order. Jobs last until restart."""
        return [snapshot(job) for job in manager.all()]

    @router.post(
        "/jobs",
        status_code=201,
        responses={415: {"model": ErrorResponse, "description": "Not an audio file."}},
    )
    def create_job(
        audio: Annotated[UploadFile, File(description="The song (mp3, ogg/egg, wav, flac, ...).")],
        modes: Annotated[
            list[Characteristic], Form(description="Repeat the field for several modes.")
        ] = _DEFAULTS.modes,
        difficulties: Annotated[
            list[Difficulty], Form(description="Repeat the field for several difficulties.")
        ] = _DEFAULTS.difficulties,
        lighting: Annotated[LightingStyle, Form()] = _DEFAULTS.lighting,
        style: Annotated[
            float | None,
            Form(ge=0.0, le=1.0, description="0 = flowy, 1 = technical; omit for no preference."),
        ] = None,
        seed: Annotated[int, Form()] = 0,
        intensity: Annotated[
            float | None,
            Form(
                ge=0.0,
                le=1.0,
                description="0.5 = typical human map, 1 = about the hardest 10%; omit for the "
                "configured default.",
            ),
        ] = None,
        repetition: Annotated[
            float | None,
            Form(
                ge=0.0,
                le=1.0,
                description="Pattern reuse on repeated music: 0 = never, 0.5 = learned, 1 = "
                "twice as strongly; omit for the configured default.",
            ),
        ] = None,
        light_style: Annotated[
            LightStyle | None,
            Form(
                description="Look of model lights: auto, none, pulse, calm, scenes, flash or "
                "accent; omit for the configured default."
            ),
        ] = None,
        # Text, not bool: FastAPI reads a missing optional bool form field as false.
        arcs: Annotated[
            Literal["true", "false"] | None,
            Form(description="Arcs on or off; omit for the configured default (else Hard up)."),
        ] = None,
        chains: Annotated[
            Literal["true", "false"] | None,
            Form(description="Chains on or off; omit for the configured default (else Expert up)."),
        ] = None,
        njs: Annotated[
            float | None,
            Form(ge=1.0, le=40.0, description="NJS for every difficulty; omit for defaults."),
        ] = None,
        reaction_time: Annotated[
            float | None,
            Form(
                ge=0.3,
                le=2.0,
                description="Seconds from spawn to strike for Expert and Expert+; omit for "
                "defaults.",
            ),
        ] = None,
        environment: Annotated[
            str | None,
            Form(
                pattern=r"^[A-Za-z0-9]{1,64}$",
                description="Info.dat environment for v3 lighting (GET /api/environments); "
                "omit to pick one for each song.",
            ),
        ] = None,
        title: Annotated[
            str | None,
            Form(max_length=200, description="Song title; omit to read it from the file."),
        ] = None,
        artist: Annotated[
            str | None,
            Form(max_length=200, description="Song artist; omit to read it from the file."),
        ] = None,
        install: Annotated[
            Literal["true", "false"],
            Form(description="Copy the map into the game as soon as it is generated."),
        ] = "false",
    ) -> JobInfo:
        """Upload a song and queue a generation job. Options are multipart form fields.

        Jobs run one at a time in the order they were queued; post one per song to queue a
        batch.
        """
        options = JobOptions(
            modes=modes,
            difficulties=difficulties,
            lighting=lighting,
            style=style,
            seed=seed,
            intensity=intensity,
            repetition=repetition,
            light_style=light_style,
            arcs=None if arcs is None else arcs == "true",
            chains=None if chains is None else chains == "true",
            njs=njs,
            reaction_time=reaction_time,
            environment=environment,
            title=(title.strip() or None) if title is not None else None,
            artist=artist.strip() if artist is not None else None,
        )
        try:
            job = manager.submit(
                audio.filename or "song", audio.file, options, install_when_done=install == "true"
            )
        except ValueError as exc:
            raise HTTPException(415, str(exc)) from exc
        return snapshot(job)

    @router.get("/jobs/{job_id}", responses=NOT_FOUND)
    def job_status(job_id: str) -> JobInfo:
        """Status and progress of a job."""
        return snapshot(get_job(job_id))

    @router.delete(
        "/jobs/{job_id}",
        status_code=204,
        response_class=Response,
        responses={**NOT_FOUND, 409: {"model": ErrorResponse, "description": "Job is running."}},
    )
    def remove_job(job_id: str) -> Response:
        """Take a job off the list (and the queue, if queued) and delete its files.

        Maps already installed in the game stay there. Cancel a running job first.
        """
        try:
            manager.remove(get_job(job_id))
        except JobBusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        return Response(status_code=204)

    @router.post(
        "/jobs/{job_id}/cancel",
        responses={
            **NOT_FOUND,
            409: {"model": ErrorResponse, "description": "Job is not queued or running."},
        },
    )
    def cancel_job(job_id: str) -> JobInfo:
        """Cancel a queued job, or stop a running one at its next pipeline step.

        A cancelled regenerate keeps the earlier map (the job stays done).
        """
        job = get_job(job_id)
        try:
            manager.cancel(job)
        except JobNotActiveError as exc:
            raise HTTPException(409, str(exc)) from exc
        return snapshot(job)

    @router.get(
        "/jobs/{job_id}/events",
        response_class=StreamingResponse,
        responses={
            **NOT_FOUND,
            200: {
                "description": "Server-sent events. Each `data:` line is one JobEvent as JSON, "
                "`event:` is its type and `id:` its seq. The stream ends after a done/failed "
                "status event.",
                "content": {
                    "text/event-stream": {"schema": {"$ref": "#/components/schemas/JobEvent"}}
                },
            },
        },
    )
    async def job_events(
        job_id: str,
        request: Request,
        after: Annotated[int | None, Query(description="Only events with seq > after.")] = None,
        last_event_id: Annotated[int | None, Header(include_in_schema=False)] = None,
    ) -> StreamingResponse:
        """Stream progress as server-sent events (replays missed ones on reconnect)."""
        job = get_job(job_id)
        # A reconnecting EventSource repeats the URL and adds Last-Event-ID; take the later one.
        start = max(after or 0, last_event_id or 0)

        async def stream() -> AsyncIterator[str]:
            seq, idle = start, 0.0
            yield "retry: 2000\n\n"
            while not await request.is_disconnected():
                new = job.events_after(seq)
                for event in new:
                    seq = event.seq
                    yield _sse(event)
                if not new and job.info.status in TERMINAL:
                    return
                idle = 0.0 if new else idle + SSE_POLL_S
                if idle >= SSE_KEEPALIVE_S:
                    idle = 0.0
                    yield ": keepalive\n\n"
                await asyncio.sleep(SSE_POLL_S)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/jobs/{job_id}/map", responses={**NOT_FOUND, **NOT_READY})
    def job_map(job_id: str) -> MapPreview:
        """The generated map for the preview, plus playability warnings."""
        preview = get_job(job_id).preview()
        if preview is None:
            raise HTTPException(409, "map not ready")
        return preview

    @router.get(
        "/jobs/{job_id}/audio",
        response_class=FileResponse,
        responses={**NOT_FOUND, 200: {"content": {"audio/*": {}}, "description": "The song."}},
    )
    def job_audio(job_id: str) -> FileResponse:
        """The song for the preview, aligned with the map (supports range requests)."""
        # The packaged song.egg once the map is ready (it may be padded), else the upload.
        job = get_job(job_id)
        path = job.input.audio_path
        if job.result is not None and job.result.audio_path and job.result.audio_path.is_file():
            path = job.result.audio_path
        media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".egg":
            media = "audio/ogg"
        return FileResponse(path, media_type=media)

    @router.get(
        "/jobs/{job_id}/download",
        response_class=FileResponse,
        responses={
            **NOT_FOUND,
            **NOT_READY,
            200: {"content": {"application/zip": {}}, "description": "Game-ready map zip."},
        },
    )
    def job_download(job_id: str) -> FileResponse:
        """The packaged map as a zip (Info.dat, difficulties, song.egg, cover)."""
        job = get_job(job_id)
        if job.result is None:
            raise HTTPException(409, "map not ready")
        zip_path = job.result.zip_path
        return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)

    @router.post(
        "/jobs/{job_id}/regenerate",
        status_code=202,
        responses={**NOT_FOUND, 409: {"model": ErrorResponse, "description": "Job is busy."}},
    )
    def job_regenerate(job_id: str, request: RegenerateRequest) -> JobInfo:
        """Redo one difficulty with a new seed/temperature. Follow it on /events."""
        job = get_job(job_id)
        try:
            manager.regenerate(job, request)
        except JobBusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        return snapshot(job)

    @router.post(
        "/jobs/{job_id}/rerun",
        status_code=201,
        responses={
            **NOT_FOUND,
            410: {"model": ErrorResponse, "description": "The uploaded song is gone."},
        },
    )
    def job_rerun(job_id: str, request: RerunRequest) -> JobInfo:
        """Queue the song again as a new job, from the upload the server already has.

        Use it to retry a failed song, or to make the whole map again with other options or
        a new seed. The source job stays as it is.
        """
        try:
            job = manager.rerun(get_job(job_id), request.options, request.install)
        except FileNotFoundError as exc:
            raise HTTPException(410, str(exc)) from exc
        return snapshot(job)

    @router.put(
        "/jobs/{job_id}/song",
        responses={**NOT_FOUND, **NOT_READY},
    )
    def job_song_info(job_id: str, request: SongInfo) -> JobInfo:
        """Change the song title and artist of a finished map (what the game lists).

        The map is not generated again. If it was installed, the game copy is replaced.
        """
        job = get_job(job_id)
        try:
            manager.set_song_info(job, request.title, request.artist)
        except (JobBusyError, MapNotReady) as exc:
            raise HTTPException(409, str(exc)) from exc
        return snapshot(job)

    @router.put("/jobs/{job_id}/feedback", responses=NOT_FOUND)
    def job_feedback(job_id: str, request: Feedback) -> JobInfo:
        """Save a playtest rating and notes for the map (``rating`` 0 and no note clears it).

        They are kept in the job's ``job.json``, so they outlive the browser and the server.
        """
        job = get_job(job_id)
        manager.set_feedback(job, request)
        return snapshot(job)

    @router.post(
        "/jobs/{job_id}/install",
        responses={
            **NOT_FOUND,
            409: {"model": ErrorResponse, "description": "No finished map, or the job is busy."},
            503: {
                "model": ErrorResponse,
                "description": "CustomLevels not found or the config file is invalid.",
            },
        },
    )
    def job_install(job_id: str) -> InstallResult:
        """Copy the map into Beat Saber's CustomLevels (set in ``~/.bsmapper/config.toml``).

        Installing again replaces the earlier copy of this song.
        """
        try:
            return manager.install(get_job(job_id))
        except MapNotReady as exc:
            raise HTTPException(409, str(exc)) from exc
        except InstallUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc

    return router


def _add_event_schema(app: FastAPI) -> None:
    """JobEvent only appears inside the SSE stream, so FastAPI would leave it out of the spec."""
    base = app.openapi

    def openapi() -> dict:
        spec = base()
        schemas = spec.setdefault("components", {}).setdefault("schemas", {})
        if "JobEvent" not in schemas:
            event = JobEvent.model_json_schema(ref_template="#/components/schemas/{model}")
            schemas.update(event.pop("$defs", {}))
            schemas["JobEvent"] = event
        return spec

    app.openapi = openapi  # type: ignore[method-assign]


def _sse(event: JobEvent) -> str:
    data = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
    return f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"
