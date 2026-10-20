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
