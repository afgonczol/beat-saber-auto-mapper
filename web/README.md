# Web UI (WS9)

Vite + React + TypeScript + three.js. It talks to the FastAPI server in `src/bsmapper/server/`.
See the WS9 section of [docs/PLAN.md](../docs/PLAN.md).

## Run it

```bash
pip install -e ".[server]"
cd web && npm install && npm run build   # builds web/dist
bsmapper serve                           # http://127.0.0.1:8765, API docs at /docs
```

`bsmapper serve` serves `web/dist` at `/` and the jobs API under `/api`. For UI work, run
`bsmapper serve` and `npm run dev` side by side: Vite (port 5173) proxies `/api` to 8765.

The server runs the real WS7 pipeline (`PipelineJobRunner`; notes come from the stub
generator until the WS6 model lands). For UI work without audio analysis, `bsmapper serve
--mock` uses `MockJobRunner` instead: a canned 16-beat v3 map
(`src/bsmapper/server/mock_map/`) tiled over the song, with fake warnings. `--mock-delay`
sets its seconds per pipeline step.

## API contract

`openapi.json` is generated from the server's pydantic models:

```bash
python -m bsmapper.server.openapi   # rewrite web/openapi.json (a test fails if it is stale)
npm run gen:api                      # rewrite src/api/schema.d.ts from it
```

| Endpoint | What |
| --- | --- |
| `GET /api/jobs` | every job, oldest first (the queue order); the queue panel polls this. Jobs last until the server restarts |
| `POST /api/jobs` | multipart: `audio` file + `modes`, `difficulties` (repeat the field), `lighting` (`vanilla`/`v3`/`chroma`), `style` (0 flowy .. 1 technical, omit for none), `seed`, ..., `install` (`true`: copy into the game when done). One song per request; post several to queue a batch. Jobs run one at a time |
| `GET /api/jobs/{id}` | status (`queued`/`running`/`done`/`failed`/`cancelled`), stage, progress, `last_seq`, `installed` / `install_error` |
| `DELETE /api/jobs/{id}` | take a job off the list (and the queue) and delete its files; 409 while running |
| `POST /api/jobs/{id}/cancel` | cancel a queued job, or stop a running one at its next progress step |
| `GET /api/jobs/{id}/events?after=` | SSE progress; ends after a `done`/`failed`/`cancelled` status event |
| `GET /api/jobs/{id}/map` | preview JSON: the `schema.Beatmap` (times in beats) + warnings |
| `GET /api/jobs/{id}/audio` | the song for playback: the packaged `song.egg` (aligned with the map's beats) once the map is ready, else the upload |
| `GET /api/jobs/{id}/download` | the game-ready zip |
| `POST /api/jobs/{id}/regenerate` | one difficulty with a new seed/temperature; follow it on `/events` |
| `POST /api/jobs/{id}/install` | copies the map into `CustomLevels` ("Install to game"); returns the installed `path` and `warnings` (e.g. Chroma lighting without `Chroma.dll`). 503 if `CustomLevels` is not found: see the config in [docs/GENERATION.md](../docs/GENERATION.md#install) |

## Layout

- `src/api/`: generated types and the fetch/SSE client
- `src/useJob.ts`: job state (status, events, map) and regenerate
- `src/components/`: upload + options, progress, preview page, timeline, warnings
- `src/preview/TrackScene.ts`: three.js track, driven by the `<audio>` element's clock
- `src/preview/LightRig.ts`: basic-event lights (back/ring/left/right/center, ring spin,
  laser speed, boost, Chroma colors). v3 event box groups are not drawn yet.
