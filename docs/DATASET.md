# Dataset card

The data stays on the local machine. It is never committed or redistributed (PLAN §8).

## Raw maps: `bsmapper data scrape` (WS3a)

```bash
bsmapper data scrape --out data/raw --limit 2000 --max-gb 50
```

It searches `api.beatsaver.com` ordered by rating (100 maps per page), filters each map,
downloads the zip of its latest published version and appends one line to the manifest.
Run the same command again to resume. `--limit` is the total number of maps wanted in the
manifest, not the number to add in this run. With `--characteristics` (see
[Rare modes](#rare-modes-characteristic-targeted-scrape)), it counts only the maps that have
one of those characteristics.

### Filters (defaults)

| Filter | Default | Flag |
|---|---|---|
| AI maps (`automapper`, or `declaredAi` other than `None`) | excluded | |
| BeatSaver rating | ≥ 0.75 | `--min-score` |
| Votes (up + down) | ≥ 30 | `--min-votes` |
| Uploaded | on or after 2021-01-01 | `--since` |
| Song length | 30 s – 10 min | |
| Characteristics (at least one difficulty in one of them) | any | `--characteristics` |
| Noodle Extensions, Mapping Extensions or Vivify in any difficulty | excluded | |
| Zip must be valid and contain `Info.dat` | required | |

The search request repeats most of these filters on the server side. Every map is still
checked again on the client. Note BeatSaver's `automapper` parameter: leaving it out
means "no AI", `false` means "only AI", and `true` means "both". The scraper leaves it out.

### Gold subset

A map is tagged `gold` when any of these is true (`gold_reasons` in the manifest says which):
`scoresaber_ranked`, `beatleader_ranked`, `beatsaver_curated` (picked by a BeatSaver
curator), or `curated_list` (its key is in the file passed with `--gold-list`, one key per
line, `#` for comments).

### BeatLeader ratings (for the flowy ↔ technical style labels)

For each map, the scraper looks up `api.beatleader.com/map/hash/{hash}`. It attaches
BeatLeader's ratings to every matching (characteristic, difficulty) as
`diffs[i].beatleader`. The fields are `status` (`ranked`, `unranked`, `unrankable`, …),
`stars`, `tech_rating`, `pass_rating`, `acc_rating` and `predicted_acc`. BeatLeader's AI
model rates most maps, including unranked ones, so coverage is high.

- If a map or difficulty is unknown to BeatLeader, `beatleader` is `null`. Single values
  can also be `null`.
- `beatleader_at` on the manifest line records when the lookup ran. `null` means it has
  not run yet, for example because BeatLeader was unreachable. The next `scrape` or
  `bsmapper data ratings` run retries those maps. A BeatLeader failure never stops the
  download.
- `bsmapper data ratings --out data/raw` backfills an existing manifest.
  `--refresh` fetches the ratings again for every map.
- `--no-ratings` on `scrape` skips the lookups entirely.
- Parsed ratings are cached for 7 days in `cache/beatleader/` (about 3 KB per map).

The heuristic tech score is **not** computed here. The dataset builder (WS3b) computes it
and calibrates it against `tech_rating`.

### Rare modes: characteristic-targeted scrape

The main query is ordered by rating and is almost all Standard, so OneSaber, 360Degree and
90Degree get their own queries:

```bash
bsmapper data scrape --out data/raw --characteristics 360Degree,90Degree     --min-score 0.7 --min-votes 10 --since 2019-12-01 --limit 100000
bsmapper data scrape --out data/raw --characteristics OneSaber     --min-score 0.7 --min-votes 10 --limit 100000
```

- `--characteristics` is sent as BeatSaver's `characteristics` search filter. It is a
  comma-separated **OR** list, so a map qualifies if any of its difficulties uses any of
  the names. BeatSaver silently ignores names it does not know and then returns every map,
  so the CLI rejects unknown names up front (`Standard`, `OneSaber`, `NoArrows`, `90Degree`,
  `360Degree`, `Lightshow`, `Lawless`, `Legacy`). The client re-checks each map
  (rejection reason `characteristic`).
- These runs write to the same `data/raw` and manifest. Each has its own cursor (see
  [Resume](#resume-and-crash-safety)), so the main `--limit 2000` cursor does not move.
  Maps the main run already has are skipped, and the main run later skips the ones these
  runs fetched.
- `--limit` counts the manifest maps that have one of the characteristics, so a large
  value means "until the pool runs out". Note that the main run's `--limit` still counts
  every map, including the ones the targeted runs added.
- A rare-mode map brings its Standard difficulties with it. These maps can be older and
  lower rated than the main query allows. `uploaded`, `score` and `upvotes`/`downvotes`
  are in the manifest, so the builder can hold Standard to the stricter bar if it wants to.

Pool sizes from live probes on 2026-10-20 (maps on BeatSaver matching the query, 30 s – 10
min, before the client-side AI and mods checks):

| Rating, votes | Since | OneSaber | 360Degree | 90Degree |
|---|---|---|---|---|
| ≥ 0.75, ≥ 30 (main defaults) | 2021-01-01 | 325 | 116 | 34 |
| ≥ 0.7, ≥ 10 | 2021-01-01 | 451 | 160 | 56 |
| ≥ 0.6, ≥ 5 | 2021-01-01 | 536 | 181 | 65 |
| ≥ 0.75, ≥ 30 | 2018-05-01 | 441 | 291 | 90 |
| ≥ 0.7, ≥ 10 | 2018-05-01 | 572 | 347 | 116 |
| no rating/vote filter | 2021-01-01 | 864 | 233 | 82 |

360Degree + 90Degree at ≥ 0.7 / ≥ 10 is 400 maps whether it starts in 2019-06 or 2019-12,
because both modes shipped in late 2019. Their mapping peaked in 2020, so the 2021 cutoff
would halve their pool. OneSaber keeps the 2021 cutoff, since its pool is large enough.
Going from 0.75 / 30 to 0.7 / 10 adds about 40–65 % to each pool. Going lower to 0.6 / 5
adds much less, and the maps it adds have barely been voted on.

### Budget

`--max-gb` (default 50) stops the run once the zips reach that size. `--min-free-gb`
(default 20) stops it when the disk gets that full. An average map zip is about 5–6 MB.

### Politeness

Requests are spaced to at most `--rps` per second (default 2). The User-Agent names the
project and links to its repo. A 429 waits for `Retry-After`. 5xx responses and network
errors are retried with exponential backoff. Search pages are cached for 6 hours in
`data/raw/cache/`, so a resume does not fetch them again.

### Output layout

```
data/raw/
  zips/<key>.zip    the map exactly as BeatSaver serves it
  manifest.jsonl    one JSON object per downloaded map (see below)
  state/<key>.json  one search query's cursor + rejection reason per rejected map key
  cache/search/     cached search pages
  cache/beatleader/ cached BeatLeader ratings by map hash
```

Each manifest line is a `MapRecord` (`bsmapper.data.filters`): key, version hash, song and
mapper names, bpm, duration, upload date, score, votes, AI flags, ranked/curated flags,
tags, and a per-difficulty summary (notes, nps, njs, offset, environment, Chroma/NE/ME
flags, BeatSaver parity summary and BeatLeader ratings). It also holds `zip_path`, `bytes`,
`gold`, `gold_reasons`, `downloaded_at` and `beatleader_at`. Read it with `bsmapper.data.read_manifest`.

### Resume and crash safety

Each zip is written to `<key>.zip.part` and then renamed. Its manifest line is appended
(and fsynced) only after that, so the manifest always matches the zips. On restart the
scraper deletes leftover `.part` files, skips torn manifest lines, reuses any valid zip
that is missing from the manifest, and continues from the last unfinished search page.

Each distinct set of filters (the exact search query) has its own cursor file,
`state/<first 12 hex of sha1(query)>.json`, which also stores the query itself. New filters
start a new cursor at page 0. Going back to earlier filters resumes their old cursor. Maps
already downloaded are always kept and skipped. The single `state.json` from before
per-query cursors is moved to `state/` the first time its own query runs again.

### Counts

| Date | Maps | Gold | Size | Notes |
|---|---|---|---|---|
| 2026-10-19 | 500 | 392 | 3.4 GB | First acceptance run with the default filters, hard-killed twice and resumed. Rejected: 65 maps for mods. Lowest rating 0.968, fewest votes 156. |
| 2026-10-20 | 2000 | 1405 | 13.4 GB | **Dataset v0.** Resumed from the 500 above with `--limit 2000 --max-gb 15`. 1500 new maps in 15 min, 0 failed. Rejected: 173 maps for mods. Lowest rating 0.955, fewest votes 91. |
| 2026-10-20 | 2712 | 1740 | 17.8 GB | **Rare-mode top-up** of v0 (see below). 712 new maps in 9 min, 1 failed. Both pools ran out. |
| 2026-10-22 | 10001 | 4949 | 64.9 GB | **Dataset v1** (see below). Same filters as v0, no relaxation. 7289 new maps in 70 min, 1 failed (the same broken zip as before). Rejected: 682 maps for mods, 1 for duration. |

Note for the dataset builder (WS3b): the top of the rating list is mostly BeatSaver-curated
maps (391 of 500), so at this size `beatsaver_curated` alone tags most maps as gold. Ranked
maps (38) are a much smaller, stricter set, and the builder may want to weight them on
their own. The difficulties include `Lightshow` and `Lawless` characteristics, which the
builder should drop.

BeatLeader ratings for the same 500 maps (backfilled with `bsmapper data ratings`):
- All 500 maps were found.
- 1683 of the 1708 difficulties have a `tech_rating`. The ones without are 15
  `Lightshow`, 2 `NoArrows` and 8 `Standard` difficulties.
- Status: 1602 unranked, 91 ranked, 15 unrankable.
- Standard `tech_rating` quartiles: 1.80 / 2.84 / 4.15, ranging from 0 to 10.7.

#### Dataset v0 (2026-10-20, 2000 maps)

Command (main checkout, BeatLeader ratings on):
`bsmapper data scrape --out data/raw --limit 2000 --max-gb 15`

- **Disk:** 13.38 GB of zips (2000 files) plus 8.8 MB of cache. 318 GB of the G: drive was
  still free afterwards.
- **Gold:** 1405 maps, mostly picked by BeatSaver curators.

  | Gold reason | Maps |
  |---|---|
  | `beatsaver_curated` | 1394 |
  | `beatleader_ranked` | 79 |
  | `scoresaber_ranked` | 54 |
  | ranked on either leaderboard | 106 |

- **Upload year:**

  | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
  |---|---|---|---|---|---|
  | 520 | 617 | 501 | 256 | 89 | 17 |

- **Difficulties:** 5933 in total. 1995 maps have at least one Standard difficulty.

  | Characteristic | Difficulties |
  |---|---|
  | Standard | 5589 |
  | Lightshow | 123 |
  | Lawless | 102 |
  | OneSaber | 74 |
  | 360Degree | 25 |
  | NoArrows | 12 |
  | Legacy | 8 |
  | 90Degree | 0 |

  | Difficulty | All characteristics | Standard only |
  |---|---|---|
  | Easy | 490 | 440 |
  | Normal | 585 | 561 |
  | Hard | 1263 | 1238 |
  | Expert | 1648 | 1585 |
  | ExpertPlus | 1947 | 1765 |

- **BeatLeader ratings:**
  - All 2000 maps were looked up and found.
  - 5840 of the 5933 difficulties have a `tech_rating`. The 93 without one are 72
    Lightshow, 16 Standard, 4 NoArrows and 1 Lawless.
  - Status: 5651 unranked (AI-rated), 252 ranked, 30 unrankable.
  - Standard `tech_rating` quartiles: 1.86 / 2.95 / 4.21, ranging from 0 to 10.7. That is
    5573 rated Standard difficulties.

The rare modes are thin. There are 74 OneSaber and 25 360Degree difficulties, and no
90Degree at all, so PLAN's mode fine-tuning (WS6) needs a targeted scrape (next section).
The top-rated slice also leans heavily toward Expert and ExpertPlus, with only 440 Standard
Easy difficulties.

#### Rare-mode top-up (2026-10-20, 2712 maps)

The two `--characteristics` commands from
[Rare modes](#rare-modes-characteristic-targeted-scrape) were run on top of v0, in the main
checkout with BeatLeader ratings on. Both ran until their pool was exhausted.

| Run | New maps | Maps with the mode(s) afterwards | Rejected | Failed |
|---|---|---|---|---|
| `360Degree,90Degree`, ≥ 0.7, ≥ 10 votes, since 2019-12-01 | 362 | 379 | 21 mods | 1 (not a zip) |
| `OneSaber`, ≥ 0.7, ≥ 10 votes, since 2021-01-01 | 350 | 428 | 33 mods | 0 |

- **Disk:** 4.37 GB for the 712 new maps, 17.75 GB in total. The main cursor
  (`state.json`, page 21 of the `--limit 2000` query) did not move.
- **New maps:** 335 are gold, 39 of them ranked. 585 of the 712 also have Standard
  difficulties. Lowest rating 0.705, fewest votes 10. By upload year: 2019: 26, 2020: 183,
  2021: 118, 2022: 117, 2023: 151, 2024: 82, 2025: 22, 2026: 13. The 209 maps from
  before 2021 are all from the 360Degree/90Degree run.
- **Difficulties per characteristic** (whole manifest):

  | Characteristic | v0 diffs | Now diffs | Now maps |
  |---|---|---|---|
  | Standard | 5589 | 7387 | 2580 |
  | OneSaber | 74 | 738 | 428 |
  | 360Degree | 25 | 416 | 329 |
  | 90Degree | 0 | 141 | 110 |
  | Lawless | 102 | 241 | 137 |
  | Lightshow | 123 | 205 | 181 |
  | NoArrows | 12 | 86 | 37 |
  | Legacy | 8 | 19 | 14 |

  | Rare mode | Easy | Normal | Hard | Expert | ExpertPlus |
  |---|---|---|---|---|---|
  | OneSaber | 46 | 74 | 116 | 203 | 299 |
  | 360Degree | 19 | 20 | 37 | 136 | 204 |
  | 90Degree | 7 | 9 | 28 | 43 | 54 |

- **BeatLeader ratings:** all 2712 maps were looked up. 9095 of the 9233 difficulties
  have a `tech_rating`, including 735 of 738 OneSaber, 415 of 416 360Degree and 140 of 141
  90Degree.

90Degree is still small (141 difficulties), and the rare modes lean toward Expert and
ExpertPlus even more than Standard does. Relaxing the filters further would add few maps
(see the pool table), so WS6 should rely on oversampling and on the shared model.

#### Dataset v1 (2026-10-22, 10001 maps)

Why: note_v0, trained on v0, plateaued at val loss 1.563 while train loss kept falling, so
it was overfitting on about 2k songs. v1 grows `data/raw` from 2712 to about 10k maps.

The commands ran in the main checkout on `a33a7e3`, one after another because they share
the manifest, with BeatLeader ratings on:

```bash
bsmapper data scrape --out data/raw --limit 10000 --max-gb 78
bsmapper data scrape --out data/raw --characteristics 360Degree,90Degree --min-score 0.7 --min-votes 10 --since 2019-12-01 --limit 100000 --max-gb 78
bsmapper data scrape --out data/raw --characteristics OneSaber --min-score 0.7 --min-votes 10 --limit 100000 --max-gb 78
```

- **Filters: unchanged.** The main query keeps score ≥ 0.75, ≥ 30 votes, uploaded since
  2021-01-01 and 30 s – 10 min. Its pool on 2026-10-22 was 23,805 maps, roughly 22k after
  the client-side mods check, so 10k did not need any relaxation. For reference, relaxing
  would have added 28,621 maps at ≥ 20 votes, 24,497 at score ≥ 0.70, and 30,933 with
  uploads since 2020.
- **Main run:** it resumed at page 21, continuing the v0 cursor (the old `state.json` was
  moved into `state/`). It stopped at the 10,000 limit at page 101. There is still plenty
  of pool left for a later v2.
  - New maps have a median score of 0.937 (10th percentile 0.924, lowest 0.854).
  - BeatSaver's "Rating" order is not a plain sort by score, so a few lower-scored maps
    with few votes come early.
- **Rare-mode rescans:** both cursors had been marked `exhausted` on 2026-10-20, and
  `exhausted` sticks, so a plain rerun stops at once. Their two state files were moved to
  `state/old-v0/` so the queries could scan again from page 0. That added 1 new map; the
  1 failure was `7ab0`, a broken zip that failed on 10-20 too. The rare pools are fully
  covered.
- **Gold:** 4949 maps (4818 `beatsaver_curated`, 349 `beatleader_ranked`, 282
  `scoresaber_ranked`). 499 maps are ranked on at least one leaderboard.
- **Filters met:** 9672 maps pass the main filters. The other 329 came in through the
  rare-mode queries: 195 are older than 2021, 106 have fewer than 30 votes and 28 have a
  score below 0.75.
- **Upload year:**

  | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
  |---|---|---|---|---|---|---|---|
  | 26 | 183 | 2262 | 2905 | 2380 | 1631 | 413 | 201 |

- **Disk:**
  - 64.9 GB of zips, 7.0 MB per map on average; the run added 47.2 GB.
  - Cache: 40 MB. All of `data/raw` is 66 GB.
  - G: had 235 GB free afterwards.
- **Difficulties:** 26,247 in total, rows are characteristics:

  | Characteristic | Maps | Easy | Normal | Hard | Expert | ExpertPlus | Total |
  |---|---|---|---|---|---|---|---|
  | Standard | 9863 | 1854 | 2418 | 4719 | 6583 | 7993 | 23567 |
  | OneSaber | 429 | 46 | 74 | 117 | 203 | 300 | 740 |
  | Lawless | 446 | 34 | 33 | 55 | 128 | 385 | 635 |
  | Lightshow | 526 | 184 | 31 | 14 | 43 | 283 | 555 |
  | 360Degree | 331 | 20 | 21 | 38 | 138 | 205 | 422 |
  | 90Degree | 110 | 7 | 9 | 28 | 43 | 54 | 141 |
  | NoArrows | 52 | 30 | 21 | 27 | 28 | 22 | 128 |
  | Legacy | 38 | 1 | 1 | 7 | 17 | 33 | 59 |
  | **All** | 10001 | 2176 | 2608 | 5005 | 7183 | 9275 | 26247 |

- **BeatLeader ratings:**
  - All 10001 maps were looked up (`beatleader_at` is set on every entry).
  - 25,828 of the 26,247 difficulties have a `tech_rating`.

    | Characteristic | With `tech_rating` |
    |---|---|
    | Standard | 23,492 of 23,567 |
    | OneSaber | 737 of 740 |
    | 360Degree | 421 of 422 |
    | 90Degree | 140 of 141 |
    | Lightshow | 223 of 555 |

  - Only 3 maps (`14ac9`, `22025`, `53b1e`) have no rating at all.
  - Status: 25,079 unranked (AI-rated), 1062 ranked, 92 unrankable, 10 nominated,
    4 outdated.
  - Standard `tech_rating` quartiles: 1.97 / 3.17 / 4.47, ranging from 0 to 13.4.

Compared with v0, Standard grew about 4.2× (5589 → 23567 difficulties). The rare modes did
not grow, because their pools were already exhausted. Easy and Normal are still the
thinnest Standard difficulties (1854 and 2418). The processed dataset v1 is in
`data/processed_v1` (see [Build stats](#dataset-v1-2026-10-22-10001-maps-1)); `data/processed`
stays dataset v0 for note_v0.

## Processed dataset: `bsmapper data build` (WS3b)

```bash
bsmapper data build --raw data/raw --out data/processed --workers 10
```

It turns every map in `data/raw/manifest.jsonl` into training data for WS6. Run the same
command again to resume: a map whose work file exists is not processed again (pass
`--retry-errors` to retry maps that crashed). `--pack-only` rewrites the shards, the
manifest and the stats from the work files without touching the audio, so changing a
filter or the tech-score calibration takes about a minute. `--limit N` and
`--keys a,b,c` build a subset. `--tracker none` skips the beat tracking, which only
feeds the grid-drift stat.

### Pipeline

Stage 1 runs per map in `--workers` CPU processes (default: cores − 2):

1. **Read** the zip with `bsmapper.formats.read_map_report`. Maps that need Noodle or
   Mapping Extensions are rejected (`needs_mods`). Only Standard, OneSaber, 90Degree and
   360Degree difficulties are kept. Lightshow and NoArrows are skipped (reason
   `characteristic`). The WS1 reader already drops Lawless and Legacy.
2. **Audio**: decode the song (`bsmapper.audio.load_audio`) and compute the frame
   features: 80-band log-mel, onset strength, spectral flux and RMS, at about 100 frames/s.
3. **Resample onto the map's own grid.** The human map is the ground truth. Feature
   row `j` is map beat `j / 24` on the map's BPM, BPM changes and `_songTimeOffset`, not
   on the grid WS2 detects. Rows outside the audio are silence (mel −80 dB, the rest 0).
   **Phase correction:** some mappers put every note at a constant fraction of a beat off
   their own grid instead of setting an offset. The builder takes the most common note
   position modulo half a beat (`audio.benchmark.note_phase`). If removing it moves at
   least 10 % of the notes onto the 1/24 grid, the whole map (notes and lights) is moved
   back by that phase. The features are sampled with the same shift, so the notes stay
   exactly where they were in the audio.
4. **Snap** with `tokenizer.snap_map`. The off-grid ratio used for filtering is the
   share of notes and bombs more than 0.1 tick (1/240 beat) from the 1/24 grid, measured
   before snapping. Walls and arc tails are snapped but not counted.
5. **Stats** per difficulty: NPS (`tokenizer.notes_per_second`, first to last note),
   `bsmapper.rules` errors and warnings per 100 notes with per-rule counts, and an
   onset-alignment check (see [Build stats](#build-stats)).
6. **Tech features** (`bsmapper.data.tech`) for the style label (next section).
7. **Tokenize** every difficulty with `MapTokenizer`. The map's richest lightshow (the
   shared one or a difficulty's own) goes through `prepare_lightshow` and
   `LightingTokenizer`. All difficulties of a map get the same number of windows: the
   audio's length in 16-beat windows, or more if objects run past the end of the song.
8. **Fingerprint** the audio for the song-level split (below).
9. **Structure features** (builder v3, dataset v1): per map beat, 12 chroma bins
   (`librosa.feature.chroma_stft`, averaged over the beat), the 80 mel bins averaged into
   20 bands, the beat's onset strength in 8 slots of 3 ticks, and RMS
   (`bsmapper.data.structure`). Saved as `structure` in the work file; the pack stage
   turns them into reference windows (see [Structure](#structure-reference-windows-dataset-v1)).

Stage 2 runs at the same time, in one process on the GPU. It runs the full WS2
`Analyzer` (beat_this) on each song and saves the *detected* grid for the grid-drift
stat.

Stage 3 (pack) filters the difficulties, calibrates the tech score, groups the songs,
splits them and writes the output. Default filters (`bsmapper.data.pack.PackConfig`),
applied per difficulty:

| Filter | Default | Reason |
|---|---|---|
| Notes | ≥ 20 | `too_few_notes` |
| Notes and bombs > 0.1 tick off the 1/24 grid | ≤ 10 % | `off_grid` |
| NPS | ≤ 20 | `nps_extreme` |
| Rule errors per 100 notes | ≤ 10 | `rule_errors` |
| Objects after the end of the audio | ≤ 1 window | `past_audio_end` |

A map is kept when at least one of its difficulties is kept.

### Style label: the tech score

`tech_features` groups each hand's notes into swings. Notes of one colour within 1/8
beat of each other are one swing (sliders, stacks). It then measures:

| Feature | Meaning |
|---|---|
| `angle_change` | mean of (180° − angle between the previous and this cut) / 180, per hand: 0 for pure down-up flow, 1 for same-direction resets |
| `reset_rate` | share of consecutive swings that cut within 90° of the previous one |
| `crossover_rate` | share of swings that land past the other hand's latest swing (within 1 beat) |
| `dot_ratio` | dot notes / notes |
| `rhythm_irregularity` | mean absolute log2 of (gap / previous gap) between a hand's swings |
| `odd_snap_ratio` | share of swings off the 1/4-beat grid (triplets, 1/6, 1/12, …) |
| `travel` | mean grid distance between a hand's consecutive swings |

Each difficulty (Easy … ExpertPlus, all characteristics pooled) gets its own ridge
regression on the standardized features. The target is the BeatLeader `techRating`
standardized within that difficulty, and the fit uses every kept difficulty that has a
rating (`bsmapper.data.tech.calibrate_by_difficulty`). `tech_score` is the prediction:
how many standard deviations of techRating the map would sit above its difficulty's
mean. The pack stage reports each fit's 5-fold cross-validated correlation. One fit per
difficulty is used because techRating grows with difficulty, so a single pooled fit
mostly learns which difficulty a map is. `style` is the score's percentile rank within
its difficulty, so each of `STYLE_0..4` holds about a fifth of every difficulty. Difficulties
with fewer than 16 notes have no features. They get `style = NaN` and
`style_bucket = -1`, so use `STYLE_NONE` for them.

### Intensity label (dataset v1)

`intensity` (0..1) says how challenging a difficulty is *for its label*, for the note_v1
`INTENSITY` token (`bsmapper.data.intensity`). It is a mid-rank percentile within a group
of comparable difficulties: the same characteristic and difficulty, and the same tempo
bucket (the BPM folded into 90–180 by octaves, since mappers often double it, then cut at
110 / 130 / 150). Groups with fewer than 40 difficulties fall back to (characteristic,
difficulty), then (all characteristics, difficulty, tempo), then (difficulty). The value
ranked is BeatLeader's `passRating` (how hard the map is to pass; its AI rates almost every
map), else `stars`, else the map's NPS. All three are ranks inside the same group, so they
share one uniform scale: 0.5 is a typical map of that label and tempo, 0.9+ the hardest
10 % (`INTENSITY_4`). The calibration table in [Build stats](#build-stats) compares the
BeatLeader rank with the NPS rank on maps that have both, which is what the NPS fallback
relies on.

### Structure: reference windows (dataset v1)

Human mappers reuse patterns when the music repeats (a second chorus, a repeated drop),
often mirrored. For each 16-beat window, the pack stage finds the most similar *earlier*
16-beat span of the same song (`find_references`): candidates start a whole number of
bars (4 beats) before the window and end at or before its start, so they keep the
window's position in the bar and only use music that comes before it. The score is the
mean, over the 16 beats, of the cosine similarity of the per-beat structure features
(chroma, z-scored mel bands and onset pattern, equally weighted); silent beats score 0.
The best start and its score are stored per window (`ref_start` in ticks, −1 for none, and
`ref_score`); training keeps a reference when the score is at least 0.85
(`DataConfig.ref_threshold`). The threshold comes from the human repetition stats below:
it is where most matched windows share at least their rhythm with the reference.

`repetition_kind` classifies a window's notes against its reference: **copy** (at least
80 % of the notes agree in time, colour, position and direction, by F1), **mirror** (the
same after swapping colours, flipping x and mirroring directions), **rhythm** (80 % of the
note times agree) or **different**. `repetition_profile(dmap, refs)` gives the shares for
a whole map, human or generated.

### Song-level split

Maps are grouped by song, so all maps of a song land in the same split. Two maps are the
same song if their **audio fingerprints** match. As the fallback, they are also the same
song if their normalized title and artist are equal (case-folded, with bracketed parts
and "feat. …" removed) and their durations differ by at most 3 s. The groups are the
connected components of both relations.

The fingerprint (`bsmapper.data.songs`) follows Haitsma & Kalker. Each of its 25 words
per second has 32 bits: the sign of the change in energy difference between neighbouring
mel bands, smoothed over 80 ms. Exact words that two songs share vote for a time offset
between them. A pair with at least 8 votes at one offset is confirmed if its bit error
rate over the aligned overlap is at most 0.3 across at least 20 s. Unrelated audio scores
about 0.5. This catches re-encodes, re-trims and remasters of the same recording.
Cut-down "TV size" edits and differently mastered uploads often share too few exact
words. The title fallback catches most of those.

Each group goes to train, val or test by a hash of its id (the smallest map key in the
group), about 90 / 5 / 5. The assignment stays the same across rebuilds unless a group
gains a member.

### Output layout

```
data/processed/
  manifest.parquet        one row per kept difficulty: labels, stats, shard pointers
  difficulties.parquet    every difficulty that was read, kept or not, with the reason
  maps.parquet            every map, kept or not, with reason, song group, split, grid drift
  dataset.json            layout, vocab sizes, filters, tech calibration, all build stats
  features/<split>-NNN.npy    float16 (rows, 83): mel 0..79, onset, flux, rms; memory-mappable
  tokens/<split>-NNN.parquet  note-token window bodies, one row per kept difficulty
  lights/<split>-NNN.parquet  lighting-token window bodies, one row per kept map with lights
  work/maps/<key>.json|.npz   stage 1 output per map (the resume state)
  work/track/<key>.json       stage 2 detected grid per map
  build.log
```

A shard holds up to 64 maps. A map's features are `n_windows * 384` rows starting at
`feature_row` in `features/<feature_shard>.npy`, so window `k` is rows `k*384` to
`(k+1)*384` from there. Tokens are stored as window bodies (int16, with int32 window
offsets). The conditioning prefix and each window's context are rebuilt at load time, so
training can drop or change NPS and style per sample. Read it with
`bsmapper.data.ProcessedDataset`:

```python
from bsmapper.data import ProcessedDataset

ds = ProcessedDataset("data/processed")
row = ds.rows("train")[0]  # dict: key, characteristic, difficulty, nps, style, ...
feats = ds.features(row)  # (n_windows * 384, 83) float16 view
windows = ds.note_windows(row)  # TokenWindows: BOS prefix context SEP body EOS
no_style = ds.note_windows(row, ds.conditioning(row, style=None))  # STYLE_NONE
lights = ds.light_windows(row)  # None when the map has no lightshow
```

Useful manifest columns for WS6: `gold`, `ranked`, `curated`, `nps`, `nps_bucket`,
`style`, `style_bucket`, `tech_score`, `bl_tech_rating`, `bl_stars`, `flag_*` (the object
types the map uses), `max_window_tokens`, `rule_errors_per_100`, `note_off_grid_ratio`,
`onset_best_shift`, `phase` and `bpm_remapped`.

### Format reference (for loaders)

Everything below can be read without bsmapper (numpy + pyarrow). `ProcessedDataset`
is a thin wrapper over it.

**`dataset.json`**

| Key | Meaning |
|---|---|
| `dataset_version`, `builder_version`, `created` | 0, 2 for v0 and 1, 3 for v1; UTC build time of the pack |
| `ticks_per_beat`, `window_ticks` | 24 and 384 (16 beats) |
| `feature_dim`, `feature_columns`, `feature_dtype` | 83, `mel_0 … mel_79, onset_strength, spectral_flux, rms`, `float16` |
| `note_vocab_size`, `light_vocab_size` | 655 (v0) or 663 (v1, tokenizer v2) and 500 (`tokenizer.NOTE_VOCAB` / `LIGHT_VOCAB`; PAD 0, BOS 1, EOS 2, SEP 3). The window bodies only use v1 ids, so either loads |
| `shards` | shard count per split, e.g. `{"train": 35, "val": 3, "test": 2}` |
| `pack_config` | the filters used (`PackConfig`) |
| `tech_calibration` | per difficulty: features, mean, std, weights, intercept, n, CV r and Spearman |
| `stats` | everything in [Build stats](#build-stats), machine-readable |

**`features/<split>-NNN.npy`**: a plain `.npy`, `float16`, shape `(rows, 83)`. Open it
with `np.load(path, mmap_mode="r")`. The maps of a shard are stored one after another. A
map's block starts at `feature_row` and is `n_windows * 384` rows long. Row `j` of the
block is map beat `j / 24` (after the map's `phase` correction), so window `k` is rows
`k*384 … k*384 + 383`. The mel columns are dB relative to the song's loudest bin
(−80 … 0). The other three are scaled to the song's maximum (0 … 1). Rows past the end
of the audio are −80 / 0. The onset column peaks about 1 tick after a hit (12 ms lag).

**`tokens/<split>-NNN.parquet`** (one row per kept difficulty) and
**`lights/<split>-NNN.parquet`** (one row per kept map with a lightshow):

| Column | Type | Meaning |
|---|---|---|
| `key` | string | BeatSaver map key |
| `characteristic`, `difficulty` | string | tokens tables only |
| `tokens` | list<int16> | every window's *body* tokens, concatenated (no BOS, prefix, context, SEP or EOS) |
| `window_offsets` | list<int32> | `n_windows + 1` offsets: window `k` is `tokens[off[k]:off[k+1]]` (it may be empty) |

TIME tokens inside a body are relative to the window's start. To build a model input,
use `BOS + prefix + context + SEP + body + EOS`. The prefix is
`Conditioning(...).prefix_ids()` from the manifest row: characteristic, difficulty,
`nps`, `style`, `flag_*`. The context is `tokenizer.tail_context(previous body,
grammar, 16)`, empty for window 0. Lighting has no prefix. A manifest row points to its
token row with `token_shard` + `token_row` (the row index inside that file), and to its
lights with `light_shard` + `light_row` (`light_shard` is null and `light_row` is −1
when the map has none). All difficulties of a map share its feature block and its
lights.

**`manifest.parquet`**: one row per kept difficulty (7526 rows). Columns by group:

| Group | Columns |
|---|---|
| identity | `key`, `split` (`train`/`val`/`test`), `song_group`, `characteristic`, `difficulty`, `label` |
| pointers | `n_windows`, `feature_shard`, `feature_row`, `token_shard`, `token_row`, `light_shard`, `light_row` |
| conditioning | `nps`, `nps_bucket`, `style` (0..1, NaN = none), `style_bucket` (0..4, −1 = none), `flag_bombs`, `flag_walls`, `flag_arcs`, `flag_chains` |
| v1 labels | `intensity` (0..1), `intensity_bucket` (0..4), `bpm_bucket` (0..3), `ref_start` (list, one start tick per window, −1 = none), `ref_score` (list, NaN = none) |
| style label | `tech_score` (techRating z units within the difficulty), `tech_<feature>` × 7, `bl_tech_rating`, `bl_stars`, `bl_pass_rating`, `bl_acc_rating`, `bl_status` |
| weighting | `gold`, `ranked`, `curated`, `uploaded` |
| content | `notes`, `bombs`, `obstacles`, `arcs`, `chains`, `rotations`, `tokens` (body token count), `max_window_tokens`, `njs`, `njs_offset` |
| quality | `rule_errors_per_100`, `rule_warnings_per_100`, `rules` (JSON: count per rule), `note_off_grid_ratio`, `snap_off_grid_ratio`, `onset_ratio`, `onset_best_shift`, `onset_best_ratio` |
| song | `bpm`, `duration_s`, `song_name`, `song_author`, `mapper`, `phase`, `bpm_remapped` |

**`maps.parquet`**: one row per raw map (2712), kept or not: `status` (`kept` or
`rejected`), `reason`, the song metadata, timing (`bpm`, `bpm_changes`, `bpm_source`,
`song_time_offset`, `phase`, `n_windows`, `audio_windows`), lights (`light_source`,
`light_events`, `light_tokens`, `light_chroma`, `light_dropped_box_groups`,
`light_max_window_tokens`), grid drift (`drift_*`, `detected_bpm`,
`detected_bpm_changes`, `detected_coverage`, `tracker`) and `song_group`, `split`,
`feature_shard`, `feature_row`, `feature_rows`.

**`difficulties.parquet`**: one row per difficulty of every map that was read (8735),
with `kept`, `reason` and the same stats columns as the manifest.

### Build stats

#### Dataset v1 (2026-10-22, 10001 maps)

Built for note_v1 into a **new directory**, `data/processed_v1`; `data/processed` (v0) is
unchanged. Builder version 3 (stage 1 adds the structure features), dataset version 1
(pack adds `intensity`, `ref_start` / `ref_score` and the stats below), tokenizer v2
(`note_vocab_size` 663). The stages ran as separate detached processes from a code
snapshot in `data/processed_v1/code/` (`run_build.py process | track | pack`), so the
build survives the session that started it and resumes like any build.

- **Time:** stage 1 took 51 min for 10,001 maps (13 CPU workers, 3.3 maps/s). The pack
  took 17 min (song grouping and writing the 17 GB of feature shards dominate; finding
  the references and the repetition stats take about 2 min).
- **Beat tracking (grid drift) did not run for the new maps.** The 2,712 v0 track files
  were copied over. The tracker failed on every other map because `beat_this` is not
  installed in the project's `.venv` (`ModuleNotFoundError`), so the drift stat below
  covers only the 2,448 kept v0 maps. Training does not use it. The error files were
  deleted, so `run_build.py track` retries those maps once `beat_this` is installed.
- **Disk:** features 17 GB, note tokens 76 MB, light tokens 96 MB, manifest 7.7 MB, and
  `work/` 19 GB (deletable once the build is final).

**Maps:** 9061 of 10001 kept. Rejected: 774 because no difficulty passed the filters, 159
because they need Noodle or Mapping Extensions, 7 without a playable characteristic.

**Difficulties:** 22,004 kept of the 25,007 read. Rejected: 2049 `off_grid` (1858
Standard), 670 `characteristic` (Lightshow, NoArrows), 126 `rule_errors`, 114
`too_few_notes`, 44 `past_audio_end`.

**Splits** (by song; 8333 song groups, 581 with more than one map):

| Split | Maps | Songs | Difficulties | Map windows | Note tokens | Maps with lights | Light tokens |
|---|---|---|---|---|---|---|---|
| train | 8123 | 7483 | 19,820 | 242,164 | 73,953,913 | 8022 | 117,171,827 |
| val | 448 | 413 | 1028 | 13,664 | 3,875,921 | 440 | 6,176,050 |
| test | 490 | 437 | 1156 | 14,031 | 4,237,685 | 481 | 6,588,364 |

Compared with v0, train has 3.2× the note tokens (74.0M vs 23.4M) and 3.0× the
difficulty windows (588,379 vs 197,786).

**Kept Standard difficulties** (train / val / test): Easy 1585 (1436/73/76), Normal 2184
(1975/100/109), Hard 4249 (3827/196/226), Expert 5864 (5282/263/319), ExpertPlus 7115
(6377/351/387). The rare modes barely grew, because their pools were already exhausted in
v0: OneSaber 604, 360Degree 297, 90Degree 106 difficulties.

**Intensity calibration** (kept difficulties, all characteristics; 52 ranking groups):

| Difficulty | n | ranked by passRating | by stars | by NPS | Spearman(BL rank, NPS rank) | same bucket | within one bucket |
|---|---|---|---|---|---|---|---|
| Easy | 1638 | 1622 | 14 | 2 | 0.86 | 57 % | 95 % |
| Normal | 2268 | 2254 | 13 | 1 | 0.86 | 58 % | 94 % |
| Hard | 4391 | 4381 | 7 | 3 | 0.82 | 54 % | 92 % |
| Expert | 6162 | 6153 | 6 | 3 | 0.77 | 50 % | 90 % |
| ExpertPlus | 7545 | 7531 | 12 | 2 | 0.76 | 49 % | 88 % |

BeatLeader rates almost every difficulty, so the NPS fallback is used for only 11.
Within a difficulty and tempo, BeatLeader's passRating rank and the NPS rank agree well
(Spearman 0.76–0.86) and land within one bucket of each other about 90 % of the time. The
two are close enough to share the scale, and passRating also reflects what NPS misses
(swing angles, speed, patterns). Buckets: `INTENSITY_0..4` hold 4446 / 4389 / 4391 /
6580 / 2198 difficulties (edges 0.2 / 0.4 / 0.6 / 0.9).

**Structure (reference windows):** 92,134 of 269,859 map windows (34 %) have a
reference at the 0.85 threshold. Human repetition, from 3954 sampled difficulties, by the
reference's similarity score (all modes and difficulties pooled; a window and its
reference both empty are not counted):

| Score | Windows | Copy | Mirror | Same rhythm | Different | Mean note F1 (best of copy/mirror) |
|---|---|---|---|---|---|---|
| < 0.50 | 6447 | 0.1 % | 0.1 % | 3 % | 97 % | 0.07 |
| 0.60–0.65 | 7176 | 0.4 % | 0.5 % | 20 % | 80 % | 0.10 |
| 0.70–0.75 | 13,609 | 1 % | 1 % | 32 % | 66 % | 0.14 |
| 0.80–0.85 | 14,410 | 4 % | 5 % | 48 % | 43 % | 0.26 |
| 0.85–0.90 | 14,088 | 7 % | 11 % | 53 % | 30 % | 0.35 |
| 0.90–0.95 | 12,730 | 15 % | 19 % | 51 % | 16 % | 0.49 |
| ≥ 0.95 | 12,799 | 29 % | 34 % | 33 % | 4 % | 0.73 |

Exact and mirrored repeats only become common above about 0.85, and from there most
matched windows at least repeat the rhythm. That is the training threshold. Mirrors are
a bit more common than copies at every score.

Human repetition on matched windows (score ≥ 0.85), **Standard**, per difficulty:

| Difficulty | Windows | Copy | Mirror | Same rhythm | Different |
|---|---|---|---|---|---|
| Easy | 2598 | 21 % | 23 % | 38 % | 18 % |
| Normal | 3719 | 19 % | 19 % | 43 % | 20 % |
| Hard | 7496 | 19 % | 23 % | 41 % | 17 % |
| Expert | 10,523 | 17 % | 22 % | 46 % | 16 % |
| ExpertPlus | 13,809 | 14 % | 21 % | 49 % | 16 % |

So on repeated music, human mappers copy or mirror about 35–44 % of windows, keep the
rhythm with new notes in another 38–49 %, and do something different in under 20 %.
Higher difficulties vary the notes a little more. WS10 can compare a generated map with
these shares using `bsmapper.data.structure.repetition_profile`.

**Other stats** (the same definitions as v0): 877 maps phase-corrected, 1311 with BPM
changes, 136 remapped onto one tempo map. 8943 maps have lights (1595 with Chroma
colours). Onset alignment peaks at 0 to +2 ticks for 96 % of difficulties. Note tokens
per window: the median per-difficulty maximum is 198 and the p95 is 417.

#### Dataset v0 (2026-10-21, 2712 maps)

Command (main checkout, all 2712 maps of the raw manifest):
`bsmapper data build --raw data/raw --out data/processed --workers 10`

- **Time:** stage 1 ran at about 4.2 maps/s with 10 workers (11 min). Beat tracking ran
  at about 0.9 maps/s on the RTX 3090 (about 50 min, the bottleneck). The pack took 50 s.
- **Disk:** features 4.3 GB (float16), note tokens 24 MB, light tokens 27 MB, manifest
  1.5 MB. The resume state in `work/` is another 4.9 GB and can be deleted once the
  build is final.

**Maps:** 2448 of 2712 kept.

| Map rejected because | Maps |
|---|---|
| needs Noodle or Mapping Extensions (`needs_mods`) | 58 |
| no difficulty passed the filters (`no_kept_difficulties`) | 205 |
| no Standard/OneSaber/90Degree/360Degree difficulty (`no_playable_difficulties`) | 1 |

No map failed to read or decode. Of the kept maps, 1580 are gold and 135 are ranked.

**Difficulties:** 7526 kept of the 8735 in the maps that were read.

| Characteristic | Read | Kept | `off_grid` | `rule_errors` | `too_few_notes` | `past_audio_end` |
|---|---|---|---|---|---|---|
| Standard | 7191 | 6526 | 604 | 26 | 27 | 8 |
| OneSaber | 707 | 602 | 92 | 11 | 2 | 0 |
| 360Degree | 411 | 292 | 75 | 42 | 1 | 1 |
| 90Degree | 137 | 106 | 23 | 7 | 1 | 0 |
| Lightshow, NoArrows | 289 | 0 | skipped (`characteristic`) | | | |

`off_grid` is by far the main reason: 9.4 % of playable difficulties have more than 10 %
of their notes off the 1/24 grid (the median rejected one has 27 %). They come from every
upload year, but 2020 maps are hit hardest: 26 % of their difficulties, against about 10 %
for later years. 360Degree loses the most to `rule_errors`, mostly
`rotation_whiplash`. Among kept difficulties, the median is 0 rule errors per 100 notes,
the 90th percentile 0.45 and the 99th 4.0.

**Splits** (by song):

| Split | Maps | Songs | Difficulties | 16-beat windows | Note tokens | Maps with lights | Light tokens |
|---|---|---|---|---|---|---|---|
| train | 2207 | 2136 | 6824 | 64,547 | 23,429,897 | 2187 | 32,187,726 |
| val | 133 | 126 | 388 | 4,098 | 1,388,764 | 133 | 1,962,026 |
| test | 108 | 105 | 314 | 3,095 | 1,094,210 | 106 | 1,593,287 |
| total | 2448 | 2367 | 7526 | 71,740 | 25,912,871 | 2426 | 35,743,039 |

**Kept difficulties per characteristic and difficulty** (train / val / test, then note
tokens):

| | Easy | Normal | Hard | Expert | ExpertPlus |
|---|---|---|---|---|---|
| Standard | 586 (531/31/24), 1.05 M | 786 (718/39/29), 1.85 M | 1431 (1293/79/59), 4.25 M | 1821 (1648/92/81), 6.76 M | 1902 (1716/103/83), 8.63 M |
| OneSaber | 33 (32/1/0), 0.05 M | 63 (61/2/0), 0.15 M | 96 (89/3/4), 0.25 M | 163 (148/9/6), 0.55 M | 247 (221/17/9), 0.98 M |
| 360Degree | 13 (13/0/0), 0.03 M | 14 (14/0/0), 0.03 M | 23 (20/2/1), 0.06 M | 100 (93/3/4), 0.35 M | 142 (129/5/8), 0.59 M |
| 90Degree | 6 (6/0/0), 0.01 M | 6 (6/0/0), 0.01 M | 21 (19/1/1), 0.05 M | 34 (33/0/1), 0.10 M | 39 (34/1/4), 0.17 M |

The rare modes have almost nothing below Hard, and their val and test sets are tiny
(0–17 difficulties per cell). Evaluate them on train-split songs held out by key, or
pool them, rather than trusting those per-cell numbers.

**Tech score calibration** (one fit per difficulty, 5-fold cross-validated, all 7526 kept
difficulties have a BeatLeader techRating):

| Difficulty | n | Spearman (CV) | Pearson (CV) | Largest weights (z units) | Style vs NPS (Spearman) |
|---|---|---|---|---|---|
| Easy | 638 | 0.22 | 0.29 | crossover +0.33, angle change +0.18 | 0.20 |
| Normal | 869 | 0.30 | 0.29 | crossover +0.26, angle change +0.17 | 0.16 |
| Hard | 1571 | 0.41 | 0.42 | crossover +0.33, angle change +0.24, travel −0.20 | 0.17 |
| Expert | 2118 | 0.51 | 0.51 | angle change +0.41, travel −0.39, crossover +0.31 | 0.09 |
| ExpertPlus | 2330 | 0.58 | 0.57 | angle change +0.50, rhythm irregularity −0.31, travel −0.31 | 0.07 |

- The heuristic agrees reasonably with techRating where it matters most (Hard to
  ExpertPlus). It agrees weakly on Easy and Normal, where BeatLeader's techRating mostly
  follows density (Spearman 0.70 with NPS on Easy). Easy is low priority anyway.
- Agreement by characteristic, as the mean within-difficulty Spearman (in-sample):
  Standard 0.45, 90Degree 0.45, OneSaber 0.37, 360Degree 0.25. For 360Degree, treat
  the style label as noisy.
- **Style stays close to independent of density** within a difficulty (last column), so
  the `STYLE` and `NPS` tokens carry different information. Adding NPS as a feature
  would raise the agreement (Standard ExpertPlus to about 0.7). It is left out on purpose, because the
  slider should not just be a second density knob.
- A single fit over all difficulties reached r = 0.56, but mostly by learning the
  difficulty: within a difficulty it agreed much less (ExpertPlus 0.36, Expert 0.29). That is why there is
  one fit per difficulty.
- Each `STYLE_k` bucket holds a fifth of each difficulty (for example Expert: 423–424 per
  bucket).

**Grid drift** (WS2's detected grid against the map's own grid, 2448 kept maps, all
tracked with beat_this):

| Detected tempo vs map | Maps |
|---|---|
| same | 1902 |
| half (the map uses double the BPM) | 436 |
| double | 47 |
| 2/3 or 3/2 | 10 |
| other | 53 |

**Half and double time** are not errors. Mappers often set double the song's BPM, for
1/8 precision in their editor, and the tracker then finds every other map beat (436
"half" maps). Sometimes the tracker counts eighth notes instead (47 "double" maps).
Training is not affected: features, tokens and windows always follow the map's own BPM,
never the detected one. For the drift stat, a double-time detection is scored modulo
half a map beat, since each detected beat should still land on a half-beat of the map.
A half-time detection is scored modulo a whole map beat. The "other" maps (53, tempo
ratios such as 0.75 or 1.33) and the 2/3 and 3/2 maps (10) are left out of the error
table below. For WS7 this means a detected BPM is only right up to a factor of 2. The
generator has to pick the tempo octave itself, for example from the target NPS.

Of the 2385 maps at the same, half or double tempo, the median |error| of the detected
beats on the map's grid is:

| Median error per map | Maps |
|---|---|
| ≤ 1 tick | 2033 (85 %) |
| 1–3 ticks | 111 |
| 3–9 ticks | 70 |
| about half a beat (> 9 ticks) | 171 (7 %) |

Quantiles over those maps (5/25/50/75/95 %): median |error| 0.02 / 0.11 / 0.23 / 0.52 /
11.4 ticks; per-map p95 |error| 0.02 / 0.12 / 0.37 / 0.89 / 11.7 ticks. 1889 maps have
90 % of their detected beats within 1 tick. The half-beat cases put the detected beat on
the map's off-beat. WS7 should check the phase against onsets (or offer a half-beat
shift) before it trusts the detected grid. The fit coverage quantiles are 0.67 / 0.88 /
0.97 / 0.995 / 1.0, and 161 maps are below the 0.7 low-confidence line.

**Timing:**
- 227 maps were phase-corrected (notes at a constant offset from the map's grid).
- 372 maps have BPM changes.
- 46 maps have difficulties whose BPM events disagree; WS1 remapped them onto one
  tempo map (`bpm_remapped`). Their times in seconds are right, but one difficulty can
  end up off the musical 1/4 grid by a constant amount.
- 2 maps have a nonzero `_songTimeOffset`.
- Song length quantiles (5/25/50/75/95 %): 91 / 158 / 192 / 221 / 272 s.

**Onset alignment:** within each difficulty, the onset strength at the note ticks peaks
at a shift of 0 to +2 ticks for 96 % of difficulties (+1 tick for 75 %, matching the
onset feature's 12 ms lag). 2 % peak about half a beat away. These are mostly
syncopated maps (notes on the off-beats) and are kept. The peak onset ratio (at note
ticks over the song mean) has quantiles 2.4 / 3.3 / 4.1 / 5.0 / 6.8. Nothing is
filtered on this; the columns are in the manifest.

**Tokens per window:** note bodies have 186 tokens at the median window maximum per
difficulty, and 390 at the 95th percentile. Lightshows are much denser. 2426 maps have
one, 440 of them with Chroma colours. 216 maps had v3 event box groups that the lighting
vocabulary drops. Light tokens per map have quantiles 420 / 3485 / 6453 / 12403 / 57019.
The longest window per map is 56 / 239 / 431 / 827 / 4282 tokens. WS6 will have to cap
or subsample the densest lightshows (the gold subset is the intended lighting training
set anyway).

**Songs:** 2367 song groups. 75 groups hold more than one map. The fingerprint made 69
links and the title/artist/duration fallback 12 more.
