"""Build the group-lighting effect library from a dataset's human maps (WS8c).

    bsmapper lights effects data/processed_v1      # writes checkpoints/light_effects/

Reads every kept map of the dataset whose lightshow has v3 event box groups (the light
tokenizer drops them, so ``maps.parquet`` counts them in ``light_dropped_box_groups``)
from the dataset's raw zips, and writes one folder per group-lighting environment:

- ``<Environment>/index.json``: per source map, its key, mapper and per-bar descriptors
  (``effects.BAR_BEATS`` beats a bar): colour activity, motion, white and red/blue
  shares and a 16-step rhythm;
- ``<Environment>/<key>.json.gz``: the map's box groups, basic events and colour boosts.

Generation (``effects.EffectLibrary``) plans on the index and only loads the source maps
it uses. Maps that require Chroma or Noodle, have too few box groups or a sequenced lightshow
(more than ``MAX_BOX_GROUPS``) are left out, and so are environments with fewer than
``MIN_MAPS`` maps. The library holds human lightshows, so like the dataset it is not
committed; the default location sits next to the checkpoints.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from bsmapper.lighting.effects import (
    BAR_BEATS,
    EFFECTS_VERSION,
    INDEX_FILE,
    LIGHT_BASIC_TYPES,
    RHYTHM_STEPS,
    SKIP_BASIC_TYPES,
)
from bsmapper.lighting.events import LightColor, decode_value

log = logging.getLogger(__name__)

DEFAULT_EFFECTS_DIR = Path("checkpoints") / "light_effects"
MIN_BOX_GROUPS = 200
MAX_BOX_GROUPS = 25_000
MIN_MAPS = 8


def _pq():
    import pyarrow.parquet as pq

    return pq


def candidate_maps(root: Path) -> list[dict[str, Any]]:
    """Kept maps with at least ``MIN_BOX_GROUPS`` box groups, from ``maps.parquet``."""
    cols = ["key", "status", "environment", "light_dropped_box_groups", "mapper", "song_name"]
    rows = _pq().read_table(root / "maps.parquet", columns=cols).to_pylist()
    return [
        r
        for r in rows
        if r["status"] == "kept" and (r["light_dropped_box_groups"] or 0) >= MIN_BOX_GROUPS
    ]


def _bar(beat: float) -> int:
    return int(math.floor(beat / BAR_BEATS + 1e-9))


def _step(beat: float) -> int:
    return int(round((beat / BAR_BEATS - _bar(beat)) * RHYTHM_STEPS)) % RHYTHM_STEPS


def describe(groups: list, basic: list, n_bars: int) -> dict[str, list]:
    """Per-bar descriptors of a lightshow (``groups``: [beat, kind, gid, boxes], ``basic``:
    [beat, type, value, float]):

    - ``act``: colour box groups plus basic light events starting in the bar;
    - ``motion``: rotation and translation box groups;
    - ``white`` / ``red`` / ``blue``: counts of colour events by colour;
    - ``rhythm``: onsets per 16th step (one per box group or basic light event).
    """
    act = [0] * n_bars
    motion = [0] * n_bars
    white, red, blue = [0] * n_bars, [0] * n_bars, [0] * n_bars
    rhythm = [[0] * RHYTHM_STEPS for _ in range(n_bars)]

    def colour(bar: int, c: int) -> None:
        if c == LightColor.WHITE:
            white[bar] += 1
        elif c == LightColor.RED:
            red[bar] += 1
        elif c == LightColor.BLUE:
            blue[bar] += 1

    for beat, kind, _gid, boxes in groups:
        bar = min(_bar(beat), n_bars - 1)
        if kind == "color":
            act[bar] += 1
            rhythm[bar][_step(beat)] += 1
            for box in boxes:
                for e in box["e"]:
                    if e["s"] > 0:
                        colour(bar, e["c"])
        elif kind in ("rotation", "translation"):
            motion[bar] += 1
    for beat, type_, value, _f in basic:
        if type_ not in LIGHT_BASIC_TYPES:
            continue
        bar = min(_bar(beat), n_bars - 1)
        act[bar] += 1
        rhythm[bar][_step(beat)] += 1
        decoded = decode_value(value)
        if decoded is not None:
            colour(bar, decoded[0])
    return {
        "act": act,
        "motion": motion,
        "white": white,
        "red": red,
        "blue": blue,
        "rhythm": rhythm,
    }


def extract(path: Path) -> tuple[dict | None, str]:
    """(source map, "") from a map zip or folder, or (None, reason) when it is left out."""
    from bsmapper.formats.reader import read_map_report

    try:
        bm, report = read_map_report(path)
    except Exception as exc:  # noqa: BLE001 - any unreadable map is just left out
        return None, f"unreadable: {exc}"
    # Required Chroma (custom environments, materials) or Noodle changes what the light
    # groups are; a Chroma suggestion only colours basic events, which are kept uncoloured.
    if report.noodle or any("chroma" in r.lower() for r in report.requirements):
        return None, "chroma or noodle"
    shows = [s for s in [bm.lightshow, *(d.lightshow for d in bm.difficulties)] if s is not None]
    if not shows:
        return None, "no lightshow"
    ls = max(shows, key=lambda s: len(s.event_box_groups))
    if not MIN_BOX_GROUPS <= len(ls.event_box_groups) <= MAX_BOX_GROUPS:
        return None, f"{len(ls.event_box_groups)} box groups"
    groups = [
        [g.beat, g.kind, g.group_id, list(g.boxes)]
        for g in sorted(ls.event_box_groups, key=lambda g: g.beat)
    ]
    basic = [
        [e.beat, e.type, e.value, e.float_value]
        for e in sorted(ls.basic_events, key=lambda e: e.beat)
        if e.type not in SKIP_BASIC_TYPES
    ]
    boost = [[e.beat, e.on] for e in sorted(ls.boost_events, key=lambda e: e.beat)]
    end = max([g[0] for g in groups] + [e[0] for e in basic])
    n_bars = _bar(end) + 1
    return {
        "bpm": bm.bpm,
        "bars": n_bars,
        "groups": groups,
        "basic": basic,
        "boost": boost,
        "desc": describe(groups, basic, n_bars),
    }, ""


def _process(args: tuple[dict, str]) -> tuple[dict, dict | None, str]:
    row, zips = args
    source, reason = extract(Path(zips) / f"{row['key']}.zip")
    return row, source, reason


def build(
    root: Path, out: Path = DEFAULT_EFFECTS_DIR, workers: int = 8, limit: int | None = None
) -> dict[str, int]:
    """Write the library for dataset ``root`` into ``out``. Returns maps per environment."""
    raw = Path(json.loads((root / "dataset.json").read_text(encoding="utf-8"))["raw_dir"])
    rows = candidate_maps(root)[:limit]
    log.info("%d candidate maps with box groups", len(rows))
    by_env: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    with Pool(workers) as pool:
        args = [(r, str(raw / "zips")) for r in rows]
        for row, source, reason in pool.imap_unordered(_process, args, chunksize=2):
            if source is None:
                skipped[reason.split(":")[0]] += 1
            else:
                by_env[row["environment"]].append((row, source))
    log.info("left out: %s", dict(skipped))
    counts: dict[str, int] = {}
    for env, items in sorted(by_env.items()):
        if len(items) < MIN_MAPS:
            log.info("%s: %d maps, fewer than %d; skipped", env, len(items), MIN_MAPS)
            continue
        folder = out / env
        folder.mkdir(parents=True, exist_ok=True)
        index = []
        for row, source in sorted(items, key=lambda it: it[0]["key"]):
            desc = source.pop("desc")
            with gzip.open(folder / f"{row['key']}.json.gz", "wt", encoding="utf-8") as f:
                json.dump(source, f, separators=(",", ":"))
            index.append(
                {
                    "key": row["key"],
                    "mapper": row["mapper"] or "",
                    "song": row["song_name"] or "",
                    "bars": source["bars"],
                    **desc,
                }
            )
        doc = {"version": EFFECTS_VERSION, "environment": env, "maps": index}
        (folder / INDEX_FILE).write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        counts[env] = len(index)
        log.info("%s: %d maps", env, len(index))
    return counts
