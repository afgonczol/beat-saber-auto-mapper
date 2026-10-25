# bsmapper: Implementation Plan

An ML-first automatic Beat Saber mapper. You give it any song and it outputs a ready-to-play
map folder with Easy through Expert+ for Standard, One Saber, 90° and 360°, plus a
synchronized lightshow (vanilla, v3 group lighting or Chroma). It runs as a CLI and
as a local web UI with an in-browser preview.

## 1. Decisions so far

| Topic | Decision |
|---|---|
| Generation approach | **ML-first.** A transformer trained on curated community maps. Hand-written rules are used only as constraints during decoding and as evaluation metrics, never as the note generator. |
| Interface | Python CLI core, plus a FastAPI server and a browser UI with a 3D preview |
| Modes (v1) | Standard, One Saber, 90°, 360°. All five difficulties. |
| Lighting (v1) | Vanilla basic events, v3 group lighting and Chroma colours. **The default is v3 group lighting**, which works in the unmodded game. Chroma is opt-in, and is only listed as a *suggestion* (never a requirement) so maps still load without the mod. |
| Target setup | PC Beat Saber **1.45.3** via Air Link, currently unmodded (the old 1.37.5 plugins were disabled by the update). Install path: `C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\CustomLevels`. |
| Difficulty priority | Hard, Expert and Expert+ quality matter most. **Easy is low priority**: it may be weaker (dataset v0 has only 440 Standard Easy difficulties), so don't spend oversampling or tuning effort on it. |
| Mapping style | A **flowy ↔ technical slider** (0..1), passed to the model as a style conditioning token. See WS3/WS4/WS7. |
| Language / stack | Python 3.11+, PyTorch, librosa; web UI in Vite + React + TypeScript + three.js |
| Hardware | One RTX 3090 (24 GB). This caps the model at roughly 50–150M parameters in bf16. Only one training job runs at a time. |
| Output format | Write **v3** difficulty files and a v2 Info.dat by default (the widest support across game versions and editors). v4 is optional. Read v2, v3 and v4. |
| Repo | Public, MIT licensed. **Never commit datasets, map zips, audio or model weights.** |

## 2. Prior art and lessons

- **Dance Dance Convolution** (Donahue et al., 2017) introduced the two-stage split: *where*
  notes go (onset placement) and *which* notes (step selection). Its main lesson is to
  condition on difficulty explicitly.
- **Mapperatorinator / osuT5** (osu!) use a Whisper-style encoder-decoder. The audio
  encoder feeds a decoder that emits event tokens with time tokens, and previous windows
  are carried as context. This is the closest blueprint and the one we follow.
- **Beat Sage, DeepSaber, InfernoSaber** are earlier Beat Saber automappers. Their common
  failures are bad parity (awkward swings), vision blocks, density that ignores the music,
  and generic lighting. Our rules engine targets exactly these.
- BeatSaver flags automapped maps (`automapper: true`). They **must** be excluded from
  training data, or we would learn from other bots.

## 3. Architecture

```
               ┌──────────────┐
 song.mp3 ───► │ audio (WS2)  │  BPM/beat grid, downbeats, sections, energy,
               └──────┬───────┘  log-mel + beat-synchronous features (24 ticks/beat)
                      │ AudioAnalysis
        ┌─────────────┼──────────────────────────┐
        ▼             ▼                          ▼
 ┌─────────────┐ ┌──────────────────┐   ┌───────────────────┐
 │ note model  │ │ constrained      │   │ lighting (WS8)    │
 │ (WS6)       │◄┤ decoding + rules │   │ ML events model + │
 │ enc-dec     │ │ (WS5, WS7)       │   │ style renderers   │
 └──────┬──────┘ └──────────────────┘   └─────────┬─────────┘
        │ DifficultyMap × (modes × difficulties)  │ Lightshow
        └──────────────────┬──────────────────────┘
                           ▼
                 ┌──────────────────┐     ┌──────────────────┐
                 │ formats (WS1)    │ ──► │ map folder / zip │
                 │ v2/v3/v4 writers │     │ Info.dat, *.dat, │
                 └──────────────────┘     │ song.egg, cover  │
                                          └──────────────────┘
 CLI (WS7) and FastAPI + web UI (WS9) both drive the same pipeline.
```

**Shared contracts** (already in the repo, the first thing every agent reads):

- `src/bsmapper/schema.py`: the version-agnostic map model (beats, notes, walls, arcs,
  chains, rotations, lightshow).
- `src/bsmapper/interfaces.py`: stage protocols (`AudioAnalyzer`, `NoteGenerator`,
  `LightingGenerator`, `MapReader/Writer`, `PlayabilityChecker`) and their data types.

## 4. Repository layout

```
src/bsmapper/
  schema.py, interfaces.py   # contracts (change only via contract PR)
  cli.py                     # typer entry point
  formats/     WS1  v2/v3/v4 readers+writers, Info.dat, packaging (egg/cover/zip), validation
  audio/       WS2  decode, beat/BPM, downbeats, sections, features, beat-sync resampling
  data/        WS3  BeatSaver client, filters, downloader, dataset builder, manifest, splits
  tokenizer/   WS4  DifficultyMap <-> tokens, vocab, windowing
  rules/       WS5  parity, vision blocks, NPS, crossovers, wall/note conflicts; decode masks
  models/      WS6  audio encoder, note decoder, lighting decoder, configs
  training/    WS6  dataloaders, train loop, checkpoints, logging
  generation/  WS7  end-to-end pipeline, sampling, post-processing, stub generator
  lighting/    WS8  rule baseline, ML wrapper, renderers (vanilla / v3 groups / Chroma), env profiles
  eval/        WS10 metrics, golden-song regression, report
  server/      WS9  FastAPI app, job queue, progress events
web/           WS9  Vite + React + TS + three.js preview
configs/       model/training YAMLs
tests/         unit tests per package; fixtures/ holds tiny hand-made maps
docs/          this plan, dataset card, format notes, model card
data/ checkpoints/ out/   (gitignored)
```

## 5. Components

### WS1: Map formats (`formats/`)
- Readers for v2, v3 and v4 difficulty files and Info.dat (v2 and v4), parsing into `schema`.
  They must handle the real-world mess: missing keys, `_customData`, BPM change events
  in all three styles, Noodle/Mapping Extensions (detect and flag, then drop), and
  fractional precision noise.
- Writers for v3 (default), v2 and v4. A v4 writer emits a separate lightshow file shared
  across difficulties, which matches `Beatmap.lightshow`.
- Packaging: convert audio to `song.egg` (OGG Vorbis via ffmpeg), extract or generate a cover
  image, read tags with mutagen for song name and artist, then write Info.dat and the zip.
- Validation against the official schemas and a round-trip test suite (read → write → read must be equal).

### WS2: Audio analysis (`audio/`)
- Decode with ffmpeg/soundfile, mono, 22.05 kHz.
- Beat and downbeat tracking: evaluate **beat_this** (CPJKU, 2024, modern and maintained)
  against librosa's tracker. Output a stable BPM, an offset (first beat) and an optional
  BPM change list for songs with tempo drift.
- Features: log-mel (80–128 bins, ~100 fps), onset strength, RMS energy, spectral flux, and
  an optional percussive/harmonic split (HPSS).
- **Beat-synchronous resampling** at 24 ticks/beat (LCM of 1/8 and 1/6 beat, so it covers
  1/32 streams and triplets). Model outputs land on the grid by construction.
- Section segmentation (novelty/self-similarity) with energy per section, used for
  density targets, lighting palette changes and the 360 rotation flow.

### WS3: Dataset (`data/`)
- **BeatSaver client** (`api.beatsaver.com`): paginated search sorted by rating, with
  polite rate limiting, a descriptive User-Agent, resumable state and a local cache.
- **Filters**: exclude `automapper`/AI-declared maps; require score ≥ ~0.75 and a minimum
  vote count; prefer maps uploaded since 2021 (modern mapping style). Ranked maps from
  ScoreSaber and BeatLeader, plus curated maps, form a "gold" subset weighted higher.
  Drop maps that need Noodle or Mapping Extensions.
- **Builder**: parse with WS1, analyze audio with WS2, snap to the 24-tick grid (reject
  maps with too many off-grid notes), compute stats (NPS, parity errors), tokenize with
  WS4, then write sharded Parquet or NumPy files plus a `manifest.parquet`.
- **Style labels** (for the flowy ↔ technical slider): record BeatLeader's per-difficulty
  `techRating` (also pass/acc ratings) where a map has been rated. Compute a heuristic
  **tech score** for every map: mean angle change between consecutive swings, reset and
  crossover rate, dot ratio and rhythm irregularity. Calibrate it against the BeatLeader
  ratings, then normalize it to 0..1 within each difficulty.
- **Splits are by song, not by map**, so different mapper versions of the same song stay
  in one split. Dedupe by audio fingerprint, falling back to normalized title+artist.
- Targets: dataset v0 is about 2k maps (enough to debug training) and v1 is 10–25k maps.
  At about 5 MB of audio per map, v1 is 50–125 GB raw. The G: drive has about 338 GB
  free, and processed features are much smaller, so raw zips can be pruned after build.
- A dataset card (`docs/DATASET.md`) records the filters and counts. The data stays local
  and is never redistributed.

### WS4: Tokenizer (`tokenizer/`)
- Windows of 16 beats (384 ticks), each conditioned on the previous window's tail tokens.
- Prefix tokens: `<CHAR_*> <DIFF_*> <NPS_bucket> <STYLE_bucket> <MODE_FLAGS>`. `STYLE_0..4`
  buckets the 0..1 tech score from WS3. It is randomly dropped during training so the model
  also works with "no preference" and supports classifier-free guidance on style.
- Event tokens sorted by time: `TIME_t` (0..383) followed by an object:
  - note: `RED|BLUE  POS_x_y(12)  DIR_d(9)` (an angle-offset token only when it is nonzero)
  - bomb: `BOMB POS_x_y`; wall: `WALL X_x W_w H_y_h DUR_bucket`
  - arc/chain: head object + `ARC|CHAIN` + tail time/pos/dir (+ slice count)
  - rotation: `ROT_±15..±60`
- Exact round-trip tests (`DifficultyMap → tokens → DifficultyMap`) on the fixture maps.
- The lighting vocabulary is separate: `TIME_t EVT_type VAL_value [FLOAT_bucket] [COLOR_idx]`.

### WS5: Playability rules (`rules/`)
Pure functions over a `DifficultyMap`, usable in two ways:
1. **Checker**: returns `list[Issue]`, used for eval and the UI's warnings panel.
2. **Incremental decode mask**: given the partial state, it returns the tokens that would
   violate a hard rule, so WS7 can mask them during sampling.

Rules: per-hand swing **parity** (forehand/backhand alternation, with resets via bombs or
time gaps), **vision blocks** (centre notes hiding following notes), same-cell or overlap
conflicts, notes inside walls, **hand crossovers**, per-difficulty **NPS and swing-speed
ceilings**, wall-dodge timing minimums, and 360 rotation limits (no ±90° whiplash within
short gaps). Thresholds live in a per-difficulty YAML file so they are tunable without
code changes.

### WS6: Models and training (`models/`, `training/`)
- **Note model**: an encoder-decoder transformer.
  - The encoder takes beat-synchronous log-mel plus onset/RMS for a 16-beat window
    (384 frames). It uses a conv stem, then about 6 layers at d=512. An experiment
    compares this against frozen pretrained audio features (MERT or a Whisper encoder).
  - The decoder has about 8 layers at d=512, with a causal event-token vocabulary
    (~600 tokens). It cross-attends to the encoder and is prefixed with conditioning
    tokens and the previous window's tail.
  - It is about 60–100M parameters and trains in bf16 on the 3090, with gradient
    checkpointing if needed.
- **One model for all difficulties and characteristics**, via conditioning tokens. This is
  data-efficient: rare modes like 360 borrow skill from Standard. One Saber is the same
  model with `CHAR_OneSaber`.
- **Lighting model**: a second decoder over the lighting vocabulary. It cross-attends to the
  audio encoder **and** to an embedding of the generated note track, so lights follow
  the notes and the music. It trains only on the lightshows of the gold subset.
- Training: AdamW, cosine schedule, label smoothing, and per-difficulty and per-mode
  sampling weights. Rare modes are oversampled; Easy is not, since it is low priority (§1). The density token is computed from ground truth during training and
  set by the user or a heuristic at inference. Logging goes to TensorBoard. Configs live
  in `configs/*.yaml`.
- Experiments to run: token time-resolution ablation, context length, pretrained encoder
  vs. scratch, classifier-free guidance on the difficulty token.

### WS7: Generation pipeline and CLI (`generation/`, `cli.py`)
- `generate(audio, modes, difficulties, lighting_style)` runs: analyze, then for each
  (mode, difficulty) a window-by-window autoregressive decode with **rule masks**,
  top-p/temperature sampling, a seed, and density control. Then post-process (snap,
  dedupe, fix residual issues, NJS and offset per difficulty from tables), generate the
  lightshow, package and zip.
- **Cross-difficulty consistency**: generate Expert+ first, then condition lower
  difficulties on it (or verify that NPS is strictly increasing across difficulties).
- **StubGenerator**: places notes on beats with naive parity. It exists only so WS1, WS8,
  WS9 and WS10 can integrate before a trained model exists. It is clearly labelled and is
  never shipped as a "mode".
- CLI: `bsmapper generate song.mp3 --modes Standard,360Degree --lighting v3 --style 0.3 --zip`,
  plus `data scrape|build`, `train`, `evaluate`, `serve`. `--style` runs from 0 (flowy) to
  1 (technical); leave it out for no preference.
- `--install` copies the result into the game. By default it targets the Steam path in §1, and
  the path can be overridden in `~/.bsmapper/config.toml`.
- If `--lighting chroma` is chosen and `Plugins/Chroma.dll` is missing, print a warning.

### WS8: Lighting (`lighting/`)
- **L0 rule baseline** (ships first, and is the fallback when there is no checkpoint): a
  section-aware lightshow. Palettes change per section, back and ring lights follow
  downbeats, lasers follow onsets, ring spins and zooms hit section changes, and colour
  boost covers drops.
- **L1 ML lightshow**: the WS6 lighting decoder outputs vanilla basic events.
- **Renderers** turn a lightshow into each style:
  - `vanilla`: basic events as-is, valid in every environment.
  - `v3`: maps semantic channels onto per-environment **light groups** (event box
    groups). Environment profiles (group ids and counts) live as data files, starting with
    2–3 popular v3 environments, e.g. Weave and The Second.
  - `chroma`: adds RGB colours and gradients to basic events from a section palette, and
    sets `_suggestions: ["Chroma"]` in Info.dat.
- Lighting is generated once per song (v4 style) and copied into each difficulty for v2/v3.

### WS9: Server and web UI (`server/`, `web/`)
- FastAPI: `POST /jobs` (upload + options), `GET /jobs/{id}` (status), `GET
  /jobs/{id}/events` (SSE progress), `GET /jobs/{id}/map` (JSON for the preview), `GET
  /jobs/{id}/download` (zip), `POST /jobs/{id}/regenerate` (one difficulty, new
  seed/temperature). Jobs run in a single GPU worker queue.
- Web: drag-and-drop a song, then pick modes, difficulties, lighting style and a
  **flowy ↔ technical slider**, then follow progress. The preview is a three.js track synced to the audio, with difficulty tabs and
  simple light rendering. A side panel shows rule warnings, plus "Regenerate", "Download"
  and "Install to game".
- It is built against the OpenAPI spec and a mock server first, so it does not wait on the model.

### WS10: Evaluation (`eval/`)
- Offline metrics against held-out human maps: onset F1 (±1 tick and ±2 ticks), NPS error
  per difficulty, rule-violation rate per 100 notes, direction/position/pattern n-gram
  distribution distance, lighting event density and onset sync.
- **Golden songs**: a handful of CC-licensed tracks with expected property ranges, plus a
  synthetic click-track test that runs in CI.
- An HTML report per checkpoint, and a **playtest checklist** in
  `docs/PLAYTEST.md`. Human playtesting is the real quality bar, and you are the
  playtester.

## 6. Milestones

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Foundation | Repo, contracts, CI, plan *(done in this session)* |
| M1 | Building blocks | WS1 round-trips the fixture maps; WS2 beat-tracks 10 test songs within ±1 tick; WS3 scraper downloads 500 filtered maps; WS4 round-trips exactly; WS5 flags known-bad fixtures |
| M2 | Dataset v0/v1 | v0 is about 2k maps processed; v1 is 10k+ maps; dataset card written |
| M3 | Note model v1 (Standard) | It beats the stub on all offline metrics; Expert maps pass the rules with < 1 error/100 notes |
| M4 | **End-to-end playable** | `bsmapper generate` turns any mp3 into a zip with 5 Standard difficulties plus an L0 lightshow, loads in-game, and passes the playtest checklist on 3 songs |
| M5 | Lighting | ML lightshow plus the v3 and Chroma renderers; the lightshow is visibly synced in-game |
| M6 | All modes | One Saber, 90° and 360° generated and playable; arcs and chains enabled |
| M7 | Web UI | Upload → preview → download/install works locally |
| M8 | Release | Weights hosted on Hugging Face (`bsmapper download-model`), a Windows install guide, a model card, and a tagged v0.1.0 |

### Status (2026-10-22)

| # | State | Notes |
|---|---|---|
| M0 | ✅ done | CI workflow written but not pushed (the gh token lacks the `workflow` scope) |
| M1 | ✅ done | PRs #14–#16, #19, #22 |
| M2 | 🟡 v0 done, v1 downloaded | v0: 2,448 maps / 7,526 difficulties / 25.9M note tokens (#24). v1: 10,001 maps / 66 GB raw (#30), not processed yet |
| M3 | 🟡 trained, not evaluated | `note_v0` (88.6M params, 60k steps): val note loss 1.563, acc 0.773, mild overfit. WS10 eval (#11) next |
| M4 | 🟡 model playable | `note_v0` drives `generate` and the server (#29). First model playtest passed on timing and flow; density and reaction-time fixes are in progress (see below) |
| M5 | 🟡 L0 done | Rule lightshow + vanilla/v3/Chroma renderers, calibrated (#20, #23); laser-blocking fix (#28). note_v1 lighting decoder is the default (#35); WS8b readability limits (#36). WS8b phase 2: style token + section scene plan, the `light_v2` lights-only fine-tune, trained 2026-10-24 and awaiting playtest (LIGHTING_V2.md §5). WS8c: human group-lighting effects per environment (LIGHTING.md) after the playtest showed every light variant looked alike |
| M6 | 🟡 awaiting playtest | WS6b (#12): arcs on from Hard up, arc/chain repair and rules, `rotation_step`. One Saber, 90° and 360° from note_v1 pass every rule check; no fine-tuning was needed so far |
| M7 | 🟡 mostly done | UI shell (#21), real pipeline + install button (#25, #26) |
| M8 | ⬜ not started | |

### Playtest log

| Date | Map | Findings | Follow-up |
|---|---|---|---|
| 2026-10-21 | Stronger (stub notes, L0 v3 lights) | Loads in-game; inner lasers stuck at 60° and blocked the notes in the drop | Fixed in #28 (rest pose, 30° cap) |
| 2026-10-22 | Stronger (`note_v0`, model lights) | Notes on the beat, swings flow well. **Expert too easy, long breaks**: 2.82 NPS vs a human median of 3.68. **Expert+ spawn-to-strike feels too fast**: NJS 18 / 0.6 s, which is the human median, but the user prefers more time | Density calibration + gap filling; `--reaction-time` option and a comfier default table (Expert+ NJS 17 / 0.70 s) |
| 2026-10-23 | Stronger, Toxic (`note_v1`, model lights) | Notes "phenomenal". **Lights jitter**: the middle of Stronger is "very spastic". Diagnosis: v3 spin whips and laser sweep restarts (renderer), colour flips with no section plan (decoder), dark starts. Each human map has its own style, and one model mixes them | WS8b phase 1: renderer fixes, readability post-filter, dark-window fill; design options in [LIGHTING_V2.md](LIGHTING_V2.md), awaiting a choice |
| 2026-10-24 | Stronger, Toxic (`note_v1`, PR #36 build) | "Arcs are great and lighting is way better." | WS8b phase 2: (a) style token + (b2) scene plan, approved |
| 2026-10-25 | Stronger, Toxic (`light_v2`, 4 light styles each) | Notes "great"; every light version is an empty stage with lots of spinning, nothing like the human Espresso / How It's Done | WS8c: human group-lighting effects (the model only drives the 5 basic channels; the v3 template made every show alike) |

## 7. Splitting work between agents

### Principles
1. **Contracts first.** `schema.py` and `interfaces.py` already exist. Every workstream codes
   against them. Changing them needs its own small PR, announced to the other open sessions.
2. **One owner per directory.** Each agent works only inside its package (plus its tests
   and docs). Cross-package needs become issues or messages, not drive-by edits.
3. **Separate worktree and branch per workstream** (`ws/<n>-<name>`), with a PR into
   `main` per work package. CI (ruff + pytest) must pass. Keep PRs small, and merge often.
4. **Fixtures over dependencies.** If WS6 needs data before WS3 is done, it uses
   synthetic/fixture data. WS9 builds against a mock API, and WS8 against the stub generator.
5. **One GPU.** Only one agent at a time runs training or heavy dataset builds. Other
   agents use CPU and tiny configs.
6. Each work package is a GitHub issue with scope, deliverables and acceptance criteria.
   An agent session is started from its issue.

### Dependency graph

```
 schema/interfaces (M0)
   ├─► WS1 formats ──────┬─► WS3b dataset build ─► WS6 train ─► WS7 integrate ─► M4
   ├─► WS2 audio ────────┤                                 ▲        │
   ├─► WS3a scraper ─────┘                                 │        ├─► WS9 web UI wiring
   ├─► WS4 tokenizer ──────────────────────────────────────┤        └─► WS10 eval reports
   ├─► WS5 rules ─────────────────────────────► WS7 masks ─┘
   ├─► WS8 lighting L0 (uses stub) ─────────────────► WS8 L1 (needs WS6)
   └─► WS9 web shell (mock API)
```

### Phases (how many sessions to run at once)

| Phase | Parallel sessions | Workstreams |
|---|---|---|
| **1: building blocks** | 4–6 | WS1 formats, WS2 audio, WS3a scraper (long-running, start first), WS4 tokenizer, WS5 rules, WS9 web shell |
| **2: data + model** | 3–4 | WS3b dataset build (GPU/CPU heavy), WS6 model code on synthetic data, then training, WS7 pipeline + stub + CLI, WS8 L0 lighting, WS10 metrics |
| **3: integrate** | 2–3 | WS7 swaps in the real model, WS8 L1 + renderers, WS9 wires to the real server, WS6 adds 360/90 and One Saber fine-tuning |
| **4: polish** | 1–2 | Playtest loop: you play, report issues, then rules/model fixes; release packaging |

An "integrator" role (this session, or a fresh one per phase) reviews and merges PRs,
keeps `main` green and updates this plan.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Copyright: training maps contain copyrighted audio | Data stays local and is never committed or redistributed. Only derived weights may be published, and that decision is made at M8. Users map songs they own. |
| Bad parity / unplayable output | Hard rule masks during decoding (WS5), plus post-fixing and metrics |
| Model learns "average" bland maps | Gold-subset weighting, temperature/top-p, an optional style token, and classifier-free guidance |
| Beat tracking fails on tempo-drifting or live songs | BPM-change support, a manual BPM/offset override in CLI and UI, and a confidence warning |
| v3 group lighting differs per environment | Data-driven environment profiles; ship 2–3 environments and fall back to vanilla |
| Scarce 360/90 data | A shared model with a characteristic token, oversampling, and fine-tuning from Standard |
| One GPU bottleneck | Small debug configs on CPU; a single training queue |
| Format churn (game updates) | Isolated in WS1, with writers per version and v3 as the default |

## 9. Open questions (answer when relevant)
1. Should trained weights be published publicly at M8?
2. Before testing Chroma lighting in-game (M5), reinstall mods for 1.45.3 (e.g. with BSManager or ModAssistant).
3. Should v4 be the default writer? Your game (1.45.3) reads v4 natively; v3 is kept as the default for editor compatibility.

Answered: PC + Air Link on 1.45.3, CustomLevels path, and a flowy ↔ technical slider (§1).
