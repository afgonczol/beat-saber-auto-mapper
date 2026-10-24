# Generation pipeline (WS7)

`bsmapper generate` and the web server (`bsmapper serve`) both run
`bsmapper.generation.Pipeline`, built by `make_pipeline` (see [Choosing the
generator](#choosing-the-generator)):

| Stage | What happens |
| --- | --- |
| analyze | WS2 analysis. The song is padded with `BeatGrid.lead_in_s()` of silence so the bar before the first downbeat starts at 0 s (Beat Saber's beat 0). Map beat = grid beat + beats per bar. `timing.song_time_analysis` shifts the analysis the same way, so generators see `beat_times[i]` = map beat `i`. |
| generate | Per mode, every difficulty in one batch (`ModelGenerator.generate_many`), or one `NoteGenerator.generate` call each. Then from the hardest down: a difficulty at 95% or more of the next harder one's NPS is generated again at 80% of it, so NPS strictly rises. |
| rules | `postprocess`: snap to the 24-tick grid, dedupe, replay through `bsmapper.rules` and repair (turn or flip a note, move it to a nearby cell, soften a rotation) or drop anything that breaks a hard rule. Then `fix_spans` re-anchors arcs and chains on the notes that are left (see [Arcs and chains](#arcs-and-chains)), and NJS and offset are set (`njs.JUMP_TABLE`). Warnings are kept for the UI. |
| lighting | Model lighting (L1) when there is a model, else the WS8 L0 rule lightshow. Both render through WS8 (`lighting.render`), v3 group lighting by default (WeaveEnvironment). Any model failure falls back to L0. |
| package | WS1 `package_map`: padded `song.egg`, cover (embedded art or a placeholder), v3 difficulty files with a v2 Info.dat, zip. Title and artist come from the tags, else from an `Artist - Title` filename. |

```bash
bsmapper generate song.mp3                                  # 5 Standard difficulties, v3 lights, zip
bsmapper generate song.mp3 --modes Standard,OneSaber,90,360 --difficulties Hard,Expert+ --style 0.7
bsmapper generate song.mp3 --lighting chroma --install      # warns if Plugins/Chroma.dll is missing
bsmapper generate song.mp3 --generator stub --lighting-source rules   # no model at all
bsmapper generate song.mp3 --intensity 0.8 --install          # harder than a typical map
bsmapper serve            # web UI on the same pipeline; --mock for canned maps during UI work
```

The CLI prints, per difficulty: notes, NPS, NJS/offset, rule errors and warnings per 100
notes after post-processing, and generation time.

## Choosing the generator

`--generator auto|model|stub` (default `auto`) and `--lighting-source auto|model|rules`.
`auto` uses the model when a checkpoint is found and torch is installed, and lights from
the model whenever the notes come from it. The checkpoint is, in order:

1. `--model PATH`
2. `[model] checkpoint` in `~/.bsmapper/config.toml`
3. `checkpoints/note_v1/best.pt`, then `checkpoints/note_v0/best.pt`, under the current folder
   or the repository

```toml
[model]
checkpoint = 'G:\ClaudeProjects\beat-saber-auto-mapper\checkpoints\note_v0\best.pt'
```

The model is loaded once per process (`runtime.load_cached`), on CUDA when available
(`--device cpu` to force the CPU; a failed CUDA load falls back to the CPU). The server
loads it on its first job and keeps it for every later one.

## Generators

- **`ModelGenerator`** runs WS6's `NoteSampler` window by window (prefix, the previous
  window's tail, the body), all difficulties of a mode in one batch. On top of WS6's
  grammar masks it adds:
  - **Conditioning**: `Conditioning.from_request`. `target_nps` is the NPS wanted in the
    map. Without it, the NPS comes from `density.target_nps`, a per-difficulty fit of
    NPS against BPM on the v0 train split. The model is conditioned on the wanted NPS
    times `density.NPS_GAIN` (1.12 to 1.33 per difficulty): v0 reaches only part of the
    NPS it is conditioned on (the median achieved/requested ratio on three songs).
    `--style` goes in the STYLE token, with classifier-free guidance
    (`ModelSettings.style_guidance`, 1.5).
  - **Density feedback** (`DensityFeedback`): the achieved/requested ratio varies a lot
    between songs (0.4 to 1.2), and v0 leaves long holes in loud music. So every
    difficulty decodes in the same batch as a copy conditioned at 1.6× its NPS
    (`ModelSettings.density_boosts`). After each 16-beat window the better candidate is
    kept. "Better" means two things:
    - closest to the expected note count, which is the wanted NPS times the window's
      length times its loudness, so quiet windows expect fewer notes;
    - fewest gaps over loud music longer than `density.GAP_LIMIT_BEATS` (Easy 8,
      Normal 6, Hard 4, Expert 2, Expert+ 2 beats).

    Both rows then continue from the chosen window, with its tokens as context and its
    rules state copied, so a resampled window joins seamlessly. This costs about
    30–50% more decoding time. `density_boosts=()` turns it off.
  - **Object flags**: `density.object_flags` turns bombs on only where most human maps of
    that mode and difficulty use them. The flags tell the model "this map uses X", so
    turning everything on makes every map use everything. Arcs are on from Hard up and
    chains from Expert up in every mode with arrows (see [Arcs and
    chains](#arcs-and-chains)). `--bombs/--walls/--arcs/--chains` (and their `--no-`
    forms) override this, and so do `bombs`, `walls`, `arcs` and `chains` in the
    `[generate]` table of the config (true or false).
  - **Request mask**: forbidden object types, red notes in One Saber, arcs and chains in
    No Arrows, rotations outside 90/360.
  - **Rules mask** (`RuleMasker`, WS6's `mask_fn`): one `MapState` per sequence, fed each
    object as soon as the parser completes it. Before each token, note and bomb heads,
    cells and directions that would break a hard rule at the current tick are masked out.
  - **Features**: `song_features` resamples the song-time analysis onto the map's tempo
    map with WS3b's `map_grid_features`, the exact training layout (80 mel, onset, flux,
    RMS; silence outside the audio).

  A batch shares one RNG, so a seed reproduces a whole mode, not one difficulty alone.
- **`ModelLightingGenerator`** (L1) runs WS6's `LightSampler`. The lights follow the
  hardest Standard difficulty, re-encoded after post-processing. It decodes vanilla basic
  events and renders them with WS8's `render`, using the section palettes from the L0
  planner. So v3 keeps PR #28's rest pose and capped laser sweeps, and Chroma keeps the
  section colours. A lightshow with fewer than 0.5 light events per bar counts as a
  failure, and the map gets the L0 rules (with a note).
- **`StubGenerator`** is a placeholder, not the ML model, used when there is no
  checkpoint. It picks note times from a per-difficulty grid by onset strength and section
  energy, up to a target NPS (Easy 1.0 to Expert+ 4.8, capped per beat so slow songs stay
  calm; One Saber gets 60%). Hands alternate, and each hand alternates down and up swings.
  Every candidate note is checked with `MapState.candidates_mask` before it is placed.
  Every few bars both hands may drift a column left or right (lean phrases). From Normal
  up, a full-height wall over the two columns on the other side often makes the player
  lean around it. For 90/360 it adds a rotation every few bars (90 stays within ±45°). Its
  maps say so: the mapper is `bsmapper (stub)` and the difficulty labels are
  `Stub <difficulty>`.

## Model vs stub (Stronger, 104 BPM, RTX 3090)

`note_v0/best.pt`, seed 0, beat_this grid, all five Standard difficulties.

| | Easy | Normal | Hard | Expert | Expert+ |
| --- | --- | --- | --- | --- | --- |
| human median NPS at 104 BPM (v0 fit) | 1.45 | 2.00 | 2.80 | 3.68 | 4.50 |
| model NPS | 1.22 | 2.27 | 2.66 | 2.82 | 4.08 |
| stub NPS | 0.88 | 1.56 | 2.17 | 3.05 | 4.34 |
| model rule errors / warnings per 100 notes | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| stub rule errors / warnings per 100 notes | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

- Time: the model takes 28 s of note decoding for all five difficulties (one batch) and
  about 25 s for the lights; the whole `generate` run with packaging takes 94 s. The stub
  takes under 1 s to generate and 12 s end to end.
- The model's notes needed no repairs: the rules masks kept every hard rule during
  decoding.
- 38–49% of its notes are in the outer lanes.
- Most of its walls are one-lane side walls and short "flick" accents. On Expert+, 24
  walls reach head height in the centre lanes and make the player move.

## Density before and after the feedback loop

Seed 0, `note_v0/best.pt`, all Standard difficulties, NPS from first to last note. A loud
gap is the longest gap between notes over music at least half as loud as the song's
typical level.

| song | | Easy | Normal | Hard | Expert | Expert+ |
| --- | --- | --- | --- | --- | --- | --- |
| Stronger (104 BPM) | wanted NPS | 1.45 | 2.00 | 2.80 | 3.68 | 4.50 |
| | NPS before → after | 1.22 → 1.47 | 2.27 → 1.83 | 2.66 → 2.69 | 2.82 → 3.52 | 4.08 → 4.24 |
| | loud gap before → after | 8 → 5 | 2.5 → 3 | 3 → 4 | 13.25 → 2.33 | 2.25 → 4 |
| Toxic (143 BPM) | wanted NPS | 1.59 | 2.26 | 3.14 | 4.13 | 5.15 |
| | NPS before → after | 1.64 → 1.65 | 2.23 → 2.22 | 3.31 → 3.38 | 4.08 → 4.60 | 4.72 → 5.13 |
| | loud gap before → after | 14.5 → 4 | 23.4 → 7 | 8 → 6.5 | 5.8 → 2 | 10.25 → 2.96 |
| trap song (82.5 BPM) | wanted NPS | 1.38 | 1.85 | 2.60 | 3.43 | 4.14 |
| | NPS before → after | 0.39 → 0.92 | 0.38 → 1.20 | 1.43 → 1.61 | 2.13 → 2.69 | 2.54 → 2.90 |
| | loud gap before → after | 8 → 4 | 16 → 2.5 | 12.7 → 2 | 1.5 → 0.5 | 2 → 2 |

- The rules masks and post-processing remove nothing. Raw model output and the final map
  differ by 0–2 notes per difficulty. The shortfall was the model's own output.
- The trap song is still sparse (62–78% of the wanted NPS). Its grid is uncertain:
  librosa gives 110 BPM, beat_this gives 82.5.
- A second boosted copy (1.6× and 2.4×) closed Stronger's Expert+ gap. But Toxic
  overshot by about 25%, and decoding took 60–70 s instead of about 40 s, so the default
  keeps one.

## note_v1

note_v1 is the default checkpoint; note_v0 still works with `--model
checkpoints/note_v0/best.pt`. The tokenizer version comes from the checkpoint's
vocabulary (`ModelGenerator.version`):

- version 2 (note_v1) has the INTENSITY token and the structure memory;
- version 1 (note_v0) has neither.

What note_v1 adds:

- **Intensity** goes into the INTENSITY token (`Conditioning.from_request`). The #32
  DIFF up-shift above 0.75 stays on (`ModelSettings.intensity_shift`; see the table
  below):
  - at 0.8, the token alone put Expert at 0.82–1.33× the human median on three songs;
  - the token plus the shift put it at 0.95–1.33×;
  - neither had a rule error.
- **Conditioning gain**: note_v1 follows its NPS token closely, so its gains are 1.0
  (`density.gains_for`). note_v0 keeps 1.12–1.33. With v0's gains, v1 overshot by
  14–44%.
- **Structure memory**:
  - `Pipeline.analyze` adds a `chroma` frame feature, computed from the audio the way
    the dataset build does, when the generator needs it.
  - `ModelGenerator` finds each window's most similar earlier span
    (`data.structure.find_references`, threshold 0.85) and passes `WindowRefs` to the
    sampler. The reference notes are the ones already generated: the copy the density
    feedback kept.
- **`--repetition 0..1`**: also `[generate] repetition` in the config, `repetition` in the
  jobs API, and the web UI's Repetition slider. It scales the reference guidance:
  0 = ignore the reference, 0.5 = the learned amount, 1 = twice as strong. Unset
  (`None`, the default) is the learned behaviour, with no extra batch rows.
- **Lighting**: from note_v1's lighting decoder (L1), through WS8's renderers with the
  rest pose and capped sweeps, and the L0 fallback. The note track it follows now gets
  the checkpoint's prefix layout. Before, a note_v0 checkpoint would have been handed
  INTENSITY ids outside its vocabulary.

### note_v1 vs note_v0 at intensity 0.8

Hard / Expert / Expert+, seed 0, learned repetition. Each cell is NPS / longest gap over
loud music (beats) / rule errors and warnings per 100 notes / exact and mirrored shares
of repeated 8-beat segments (`bsmapper.eval.repetition`; human: 29% and 14%). Build time
covers the three difficulties (RTX 3090).

| song | model | build | Hard | Expert | Expert+ |
| --- | --- | --- | --- | --- | --- |
| Stronger, 104 BPM | note_v0 (#32: shift) | 68 s | 3.59 / 2 / 0.00, 0.00 / 0%, 0% | 3.95 / 1.75 / 0.00, 0.00 / 0%, 0% | 4.43 / 2 / 0.00, 0.00 / 0%, 0% |
|  | note_v1, token only | 58 s | 3.52 / 4 / 0.00, 0.00 / 39%, 0% | 4.88 / 4 / 0.00, 0.00 / 17%, 17% | 5.70 / 2.25 / 0.00, 0.28 / 17%, 0% |
|  | **note_v1, token + shift (default)** | 66 s | 4.44 / 2 / 0.00, 0.00 / 33%, 17% | 4.89 / 2.75 / 0.00, 0.00 / 0%, 0% | 5.25 / 1.5 / 0.00, 0.00 / 11%, 6% |
| Toxic, 143 BPM | note_v0 (#32: shift) | 55 s | 4.51 / 3 / 0.00, 0.00 / 0%, 0% | 5.00 / 2.5 / 0.00, 0.00 / 0%, 0% | 5.94 / 2 / 0.00, 0.00 / 0%, 0% |
|  | note_v1, token only | 57 s | 3.63 / 6 / 0.00, 0.00 / 21%, 19% | 4.21 / 2.5 / 0.00, 0.00 / 6%, 6% | 4.93 / 1.5 / 0.00, 0.00 / 0%, 0% |
|  | **note_v1, token + shift (default)** | 62 s | 4.14 / 2 / 0.00, 0.00 / 8%, 8% | 4.62 / 2 / 0.00, 0.00 / 2%, 8% | 5.01 / 2 / 0.00, 0.00 / 19%, 2% |
| trap, 82.5 BPM | note_v0 (#32: shift) | 74 s | 1.01 / 2 / 0.00, 0.00 / 0%, 0% | 2.21 / 1.5 / 0.00, 0.00 / 0%, 0% | 4.18 / 1 / 0.00, 0.00 / 0%, 0% |
|  | note_v1, token only | 63 s | 1.97 / 1 / 0.00, 0.00 / 0%, 0% | 2.81 / 1.25 / 0.00, 0.00 / 27%, 33% | 3.58 / 1.25 / 0.00, 0.00 / 0%, 20% |
|  | **note_v1, token + shift (default)** | 53 s | 2.81 / 1.5 / 0.00, 0.00 / 13%, 7% | 3.25 / 2 / 0.00, 0.00 / 0%, 0% | 3.58 / 1.25 / 0.00, 0.00 / 0%, 20% |

Wanted NPS at 0.8 (Hard / Expert / Expert+): Stronger 3.22 / 4.13 / 5.04, Toxic 3.62 /
4.63 / 5.77, trap 2.99 / 3.85 / 4.64. The human medians at 0.5 are Stronger 2.80 / 3.68
/ 4.50, Toxic 3.14 / 4.13 / 5.15 and trap 2.60 / 3.43 / 4.14.

- **Density**: with the default, Expert lands at 1.33 / 1.12 / 0.95× the human median on
  the three songs (note_v0: 1.07 / 1.21 / 0.64). Expert+ lands at 1.17 / 0.97 / 0.86.
  The trap song, with its uncertain grid, stays the sparsest.
- **Structure**: note_v0 never copies or mirrors. note_v1 does, at 0–33% exact and
  0–20% mirrored per difficulty, below the human 29% / 14% on average.
- **Time**: about the same for both models, 53–74 s for three difficulties. note_v1's
  decoder is larger, but it needed fewer retries.

## Intensity

`--intensity` (0 to 1) sets how hard a map is *within* its label (`generation.intensity`).
0.5 is a typical human map for the difficulty and BPM, and 1.0 is about the hardest 10%.
Without the option, `[generate] intensity` in `~/.bsmapper/config.toml` is used, else 0.5.
The server's `intensity` job option and the web UI's Intensity slider work the same way.

- **Target density**: intensity maps linearly to a percentile of the human NPS spread
  for that difficulty, from p10 at 0 to p90 at 1. The spread is NPS over the fitted
  median at the map's BPM, from the v0 Standard train split.
- **Above 0.75**: the v0 model has no intensity token, so a label is decoded with the
  next difficulty's DIFF token (Hard as Expert, Expert as Expert+) at its own percentile
  target. Expert+ cannot shift, so it gets the top of its NPS range, plus the per-window
  density feedback. The gap limit is the harder difficulty's.
- **NJS and reaction time** always follow the label's table, so the note speed doesn't
  change.
- **Rule ceilings** follow the shift, but only up to what human maps of the harder
  difficulty do:
  - the NPS ceiling (densest 2 s) goes up to that difficulty's human p95: Hard 9.5,
    Expert 12, Expert+ 14;
  - the minimum same-hand swing interval goes down to its human p5: 0.135 / 0.117 /
    0.098 s.

  For example, Expert at 0.8 gets 14 NPS and 0.098 s; Expert+ keeps its own 20 NPS and
  0.08 s.
- **note_v1**: once the model has an intensity token, `intensity.plan(...,
  model_intensity=True)` keeps the label and drops the shift. The percentile targets stay
  the same.

Wanted NPS at 104 BPM (Standard):

| intensity | Easy | Normal | Hard | Expert | Expert+ |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 1.45 | 2.00 | 2.80 | 3.68 | 4.50 |
| 0.8 (shifted) | 1.81 | 2.40 | 3.22 | 4.13 | 5.04 |
| 1.0 (shifted) | 2.24 | 2.85 | 3.70 | 4.61 | 5.64 |

Two guards apply above 0.75:

- **Conditioning cap**: every candidate's conditioning NPS is capped at the human p95 of
  the difficulty it is decoded as (× its gain). Without the cap, intensity 1.0 pushed the
  boosted copy into the top buckets (12+ NPS), where v0 saw few maps. Its output then got
  erratic, often *sparser* than at 0.8; for example the trap song's Expert fell to 1.37.
- **Not easier than typical**: a shifted label under 85% of its wanted NPS is decoded
  once more with the next seed, and the closer result is kept. If it is still under 80%
  of the label's typical (intensity 0.5) NPS, it is decoded unshifted at 0.75.

Measured with `note_v0/best.pt`, seed 0, Standard. Each cell is NPS / longest gap over
loud music in beats. Rule errors are 0 per 100 notes everywhere, and warnings are at most
0.91 per 100 (Toxic Easy at 0.8). 0.5 is the PR #31 behaviour. 0.8 and 1.0 include the
conditioning cap, but not the next-seed retry, which was added after this run.

| song | intensity | Easy | Normal | Hard | Expert | Expert+ |
| --- | --- | --- | --- | --- | --- | --- |
| Stronger, 104 BPM | 0.5 | 1.47 / 5 | 1.83 / 3 | 2.69 / 4 | 3.52 / 2.33 | 4.24 / 4 |
| | 0.8 | 1.84 / 4 | 2.76 / 4 | 2.98 / 3 | 3.51 / 1.62 | 4.40 / 1.5 |
| | 1.0 | 2.24 / 4 | 2.99 / 5 | 3.47 / 2 | 4.84 / 4 | 5.34 / 4 |
| Toxic, 143 BPM | 0.5 | 1.65 / 4 | 2.22 / 7 | 3.38 / 6.5 | 4.60 / 2 | 5.13 / 2.96 |
| | 0.8 | 2.22 / 10 | 3.46 / 5 | 4.38 / 2 | 5.67 / 2 | 6.12 / 3 |
| trap, 82.5 BPM | 0.5 | 0.92 / 4 | 1.20 / 2.5 | 1.61 / 2 | 2.69 / 0.5 | 2.90 / 2 |
| | 0.8 | 0.90 / 4 | 2.07 / 1.5 | 2.32 / 1.5 | 2.76 / 0.5 | 3.51 / 1 |
| | 1.0 | 1.47 / 2 | 1.87 / 3.5 | 2.32 / 1.5 | 2.77 / 0.5 | 3.53 / 1 |

v0 varies a lot between seeds. In an earlier run without the cap, Stronger Expert at 0.8
came out at 4.18 NPS; in the run above it came out at 3.51, the same as at 0.5. That
case is what the next-seed retry is for. Note timing is unchanged at every intensity:
Expert 0.80 s, Expert+ 0.69 s.

## NJS and offset

The offset brings the reaction time (half jump duration in seconds, as the game computes
it) to a per-difficulty target, so a slow song does not give Expert+ two seconds of
warning:

| | Easy | Normal | Hard | Expert | Expert+ |
| --- | --- | --- | --- | --- | --- |
| NJS | 10 | 12 | 13 | 16 | 17 |
| reaction time | 1.20 s | 1.00 s | 0.88 s | 0.80 s | 0.70 s |
| human v0 median | 10 / 1.20 s | 12 / 1.00 s | 14 / 0.83 s | 16 / 0.69 s | 18 / 0.59 s |

- Easy to Hard follow the human medians; Hard uses the 90–120 BPM ones.
- Expert and Expert+ give about the 75th percentile of human reaction time (Expert+
  ≈ 23.8 m jump distance). The playtest found 0.6 s on Expert+ rushed.

To override them:

- **CLI**: `--njs 16` applies to every difficulty, `--reaction-time 0.85` to Expert and
  Expert+. Both also take per-difficulty values, such as
  `--reaction-time Hard=0.95,Expert+=0.75` or `--njs Expert=16,Expert+=18`.
- **Server and web UI**: `njs` (every difficulty) and `reaction_time` (Expert and Expert+)
  in the job options, under "Note timing" in the upload panel.
- **Defaults** in `~/.bsmapper/config.toml`. They are used by both the CLI and the server,
  and explicit options win over them:

```toml
[jump.ExpertPlus]   # or [jump."Expert+"]
njs = 17
reaction_time = 0.75
```

## Install

`--install` copies the map folder into `CustomLevels`. The web UI's "Install to game"
(`POST /api/jobs/{id}/install`) does the same, with the folder named `Title - Artist`.
It replaces an earlier bsmapper install of the same name, and it never touches a folder
another mapper made: it installs next to it as `Name (2)`. Paths come from
`~/.bsmapper/config.toml` (or `$BSMAPPER_CONFIG`):

```toml
[game]
beat_saber_dir = 'C:\Program Files (x86)\Steam\steamapps\common\Beat Saber'
# custom_levels = 'D:\Beat Saber\Beat Saber_Data\CustomLevels'  # optional override
```
