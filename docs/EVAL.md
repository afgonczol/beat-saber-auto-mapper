# Evaluation (WS10)

`bsmapper evaluate` scores generated maps against held-out human maps and writes a
self-contained HTML report, a JSON summary and every generated map. Two runs can then be
diffed with `--compare`. Human playtesting is still the real bar; see
[PLAYTEST.md](PLAYTEST.md).

```bash
# note_v0 and the stub against the human maps of the frozen v0 test split
bsmapper evaluate -m checkpoints/note_v0/best.pt
# tomorrow: note_v1, then compare it with the v0 run
bsmapper evaluate -m checkpoints/note_v1/best.pt
bsmapper evaluate --compare out/eval/note_v0-.../summary.json out/eval/note_v1-.../summary.json
# a quick subset: 30 songs, Standard Hard/Expert/Expert+ only
bsmapper evaluate -m checkpoints/note_v0/best.pt --limit 30 --modes Standard --difficulties Hard,Expert,Expert+
```

Outputs go to `out/eval/<name>/` (`--out`, `--name`; the default name is the checkpoint
folders and the date):

| File | What |
|---|---|
| `report.html` | the report: headline table, per mode and difficulty, rules by rule, repetition, style sweep, lighting, run times, metric glossary |
| `summary.json` | everything in the report, machine-readable (input of `--compare`) |
| `records.jsonl` | one line per system x song x difficulty (metrics, distribution counts, repetition counts), per lightshow and per style sample |
| `maps/<system>/<key>.json.gz` | every generated difficulty and lightshow (`<key>@style<s>` for the style sweep) |

`--resume` continues an interrupted run in the same folder (failed songs are retried).
`--rescore RUN_DIR` scores the saved maps of an earlier run again without decoding, so a
metric change costs seconds instead of hours.

## Test set

The human maps are the dataset's `test` split (song-level, never trained on). Dataset
rebuilds can change splits, so `--snapshot` freezes one:

```bash
bsmapper evaluate --data data/processed --snapshot data/eval_v0_test
```

`data/eval_v0_test` holds the v0 test split (108 maps, 314 difficulties, 194 MB: the test
shards, all manifest rows and a `timing.json` with each map's exact tempo map). It is the
default `--data`, so v0 and v1 are scored on the same songs whatever the v1 rebuild does.
The other splits' manifest rows stay in it as the human reference for NPS and tech score.

## How a system is scored

Every song is evaluated **on the human map's own beat grid**. `eval.data.song_analysis`
turns the dataset's feature block (80 mel, onset, flux, RMS at 24 rows per beat, aligned
to the map's BPM, BPM changes, offset and phase) into a song-time `AudioAnalysis` whose
beat `i` is map beat `i`. Constant-tempo maps get one frame per tick, so the model sees
exactly its training features. Beat tracking is left out on purpose: a v0-vs-v1 diff
then measures the note model, and a generated note at beat `b` sits at the same point in
the audio as a human note at `b`. Sections come from WS2's `segment_sections` on the same
features. No audio is decoded.

Systems:

- **human**: the test maps themselves (the reference column).
- **stub**: `StubGenerator` with the L0 rule lightshow.
- **one per checkpoint** (`-m`, repeatable; named after the checkpoint folder, e.g.
  `note_v0`) and per intensity (`--intensity 0.5,0.8`, named `note_v0@i0.8`; it needs a
  generator with the intensity option, otherwise only the default runs).

Generated systems run the WS7 pipeline's stages: `Pipeline.generate_mode` (hardest first
with strictly rising NPS), `postprocess` (snap, dedupe, rule repairs, NJS), then
`Pipeline.light`, rendered `vanilla` so its events compare with the human basic events.
Only the difficulties the human map has are generated, with seed 0, no style and the
default NPS heuristic: what `bsmapper generate` would hand the player.

GPU: `--batch-difficulties 2` (default) decodes two difficulties at a time, four rows with
the density-boost copies, about 2 GB of VRAM. Without `--device`, CUDA is used when
`nvidia-smi` reports at least 4 GB free, else the CPU.

## Metrics

Per difficulty, against the same song's human difficulty where one is needed:

| Metric | Meaning | Better |
|---|---|---|
| `onset_f1_{0,1,2}` | note onset ticks (either colour) matched one-to-one with the human map's within 0 / 1 / 2 ticks; also precision `onset_p_*` and recall `onset_r_*` | higher |
| `nps`, `nps_ratio`, `nps_error` | colour notes per second (first to last note); over the human map's; minus it | ratio near 1 |
| `nps_percentile` | percentile among human train/val maps of the mode and difficulty within ±15% BPM (Standard's when a mode has under 40 maps); 50 = typical | near 50 |
| `step_offset` | NPS placed on the ladder of human median NPS per difficulty at that BPM (0 Easy … 4 Expert+), minus the map's level. −1 = as dense as a typical map one difficulty lower; this is the "one step too easy" number | near the human column |
| `audio_sync` | share of note onsets within 2 ticks of an audio onset peak (`_lift`: over chance) | near human |
| `errors_per_100`, `warnings_per_100`, `rule_<rule>_per_100` | `bsmapper.rules` issues per 100 notes | lower |
| `jsd_*` | Jensen-Shannon divergence (base 2, 0..1) of counts pooled over a group's maps against the human maps' pooled counts: `direction`, `position` (cell), `pattern` (per swing: direction + cell, per hand), `pattern_2gram`, `pattern_3gram`, `rhythm` (gaps between onsets in ticks). The human column is a split-half floor (half the songs against the other half) | lower |
| `red_share`, `crossover_rate` | share of red notes; share of swings landing past the other hand's latest swing within one beat | near human |
| `tech_score`, `tech_*` | the dataset's per-difficulty tech-score calibration (techRating z units) and its features | near human |
| `rep_*` | repetition on repeated audio (below) | near human |

**Repetition.** The song is cut into 8-beat segments on every bar line. A segment
repeats an earlier, non-overlapping one when the cosine similarity of their descriptors
(per-beat mel z-scored over the song, plus the onset envelope at 1/8 beat) is at least
0.85; each segment keeps its best earlier match. The pairs depend only on the audio, so
every system is scored on the same pairs (about 25 per song). For each pair where either
segment has notes, the map's notes at the same offsets are compared (F1 ≥ 0.8 counts):
`exact` (same tick, colour, cell, direction), `mirrored` (colours swapped, x → 3 − x,
left/right flipped), `same_rhythm` (same onset ticks only) or `different`. `rep_reuse`
is exact + mirrored; `rep_distance` is the total variation distance between a system's
class shares and the human maps'. On 40 test songs the human Expert/Expert+ maps reuse
more the more similar the audio is: exact 18% / 29% / 40% at similarity 0.7 / 0.85 / 0.95.

**Style.** On the first `--style-songs` songs (default 12) with a Standard Expert map,
Expert is generated again at `style` 0, 0.5 and 1 without lights. The summary gives the
mean tech score and its percentile among human Expert maps per style, the high − low
difference and Spearman ρ between requested style and tech score.

**Lighting** (one lightshow per song, light events = types 0–4):

| Metric | Meaning |
|---|---|
| `events_per_bar` | light events per 4 beats |
| `onset_precision`, `onset_lift` | share of light-event ticks within 2 ticks of an onset peak (local maximum above the song's 90th percentile of onset strength, about 1.6 per beat); lift = that over the share of all ticks near a peak (1 = chance) |
| `onset_recall` | share of strong onset peaks (97th percentile) with a light event |
| `on_beat`, `on_downbeat` | share of event ticks on a beat / bar line |
| `section_response`, `bar_response`, `section_lift` | share of section boundaries with a ring spin/zoom, boost or colour change within a beat; the same at other bar lines; their ratio. Dense human shows change colour almost everywhere, so read the lift |
| `section_energy_r` | Spearman r of light density per section and section energy |

Human lights are the dataset's basic events; v3 event box groups are not stored, so
songs whose human map lights only with box groups have no human lighting record.

## `--compare`

`bsmapper evaluate --compare A.json B.json` prints, for the groups all, Standard Hard,
Expert and Expert+, plus lighting and style, each headline metric for A, B, B − A and the
human value, marking where B is better or worse (higher, lower, or closer to the human
maps, per metric). It picks each file's first model system (else the stub);
`--system-a` / `--system-b` choose others. It warns when the runs scored different songs.

## Results: note_v0 vs the stub (2026-10-22)

```bash
bsmapper evaluate -m checkpoints/note_v0/best.pt --limit 30 --modes Standard \
  --difficulties Hard,Expert,Expert+ --style-songs 6 --name note_v0-subset30
```

**Subset.** The v1 dataset build had the CPU at 100% and the rule masks are CPU-bound, so
note_v0 took about 4–5 minutes per song (2.5 h in all). The run therefore used a **fixed
subset**: the first 30 test songs by key that have a Standard Hard, Expert or Expert+ map,
those three difficulties only (58 difficulties: 13 Hard, 19 Expert, 26 Expert+), and the
style sweep on the first 6 of them. Seed 0, no style, default NPS heuristic, intensity
not available yet. Score note_v1 with the same command to compare.

| all 58 difficulties | human | stub | note_v0 |
|---|---|---|---|
| onset F1 ±1 / ±2 ticks | – | **0.637 / 0.639** | 0.579 / 0.583 |
| note onsets on audio peaks | 0.774 | 0.708 | 0.608 |
| NPS / human NPS | 1.00 | 0.87 | 1.19 |
| NPS percentile (median) | 48 | 26 | 80 |
| difficulty step offset (median) | −0.03 | −0.54 | **+0.77** |
| rule errors / warnings per 100 notes | 0.26 / 1.75 | 0 / 0 | 0 / 0.90 |
| JSD direction / position | 0.000 / 0.002 | 0.048 / 0.091 | **0.020 / 0.007** |
| JSD pattern 2-gram / 3-gram | 0.085 / 0.241 | 0.659 / 0.802 | **0.106 / 0.202** |
| JSD rhythm | 0.020 | 0.067 | 0.032 |
| red share / crossover rate | 0.491 / 0.050 | 0.501 / 0.005 | 0.491 / 0.033 |
| repeats: exact / mirrored / same rhythm / different | 29% / 14% / 47% / 11% | 0 / 0 / 87% / 13% | 0.1% / 0.1% / 41% / 59% |
| repetition distance to human | 0 | 0.42 | 0.48 |

The human JSD column is the split-half floor; note_v0's pattern 3-gram JSD is below it
because the floor compares two halves of 30 songs, each sparser than the pooled set.

| Step offset (mean) | human | stub | note_v0 |
|---|---|---|---|
| Hard | +0.09 | −0.61 | +0.60 |
| Expert | +0.10 | −0.64 | +0.91 |
| Expert+ | +0.17 | −0.51 | +0.78 |

**Style** (Standard Expert, 6 songs): note_v0's tech score goes from −0.79 at style 0 to
+0.53 at style 1 (human percentile 6 → 84, Spearman ρ 0.94): the style token works. The
stub stays at the 99th+ percentile whatever the style.

**Lighting** (30 songs, vanilla events):

| | human | stub (L0 rules) | note_v0 (L1 model) |
|---|---|---|---|
| light events / bar | 27.2 | 7.2 | 26.9 |
| events on onsets / chance | 1.69 | 2.58 | 1.36 |
| strong onsets lit | 0.80 | 0.89 | 0.83 |
| section changes answered / lift over bar lines | 0.85 / 1.09 | 1.00 / 1.14 | 0.64 / 1.00 |
| density ~ section energy (r) | 0.53 | 0.63 | 0.27 |

What it says:

- **Density:** on the human grid with #31's density feedback, note_v0 is *denser* than
  the human maps of the same songs (NPS ×1.19, about 0.8 of a difficulty step above),
  not sparser. The playtest's "one step too easy" therefore isn't explained by note
  count on these songs. Candidates are the lower audio sync (61% of onsets on audio peaks
  against 77% for humans), the flatter, less technical patterns (tech score −0.26 against
  −0.01) and the missing reuse on repeated sections. The playtest map (Stronger, before
  #31) was also sparser than these.
- **Patterns:** note_v0 is far closer to human pattern statistics than the stub, but
  places onsets less like the human mapper (F1 0.58 against the stub's 0.64, which snaps
  to onset peaks).
- **Repetition:** note_v0 almost never reuses a pattern on repeated music (0.2% exact
  or mirrored against 42% for humans). This is the gap note_v1's structure memory targets.
- **Lighting:** L1 matches human event density but follows onsets and sections less than
  the human shows and the L0 rules.

## Limitations

- The oracle grid hides beat-tracking errors; `bsmapper generate` on raw audio can
  still drift where WS2's grid is wrong.
- Onset F1 compares with one human mapper's choices; another human map of the same song
  would not score 1 either. Read it next to `audio_sync` and the human column.
- NPS percentiles use the v0 train/val maps; the v1 dataset may shift them slightly.
- Golden CC-licensed songs (PLAN §5 WS10) are not in yet; the synthetic click-track test
  (`tests/eval`) covers the pipeline in CI.
