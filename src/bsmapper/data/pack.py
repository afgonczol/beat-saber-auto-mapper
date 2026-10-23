"""Stage 3 of the build: filter, calibrate, split and write the training shards.

Reads ``work/`` (see ``build``) and the raw manifest, and writes::

    manifest.parquet       one row per kept difficulty: labels, stats, shard pointers
    difficulties.parquet   every difficulty seen, kept or not, with the reason
    maps.parquet           every map seen, kept or not, with song group, split and grid drift
    features/<split>-NNN.npy   float16 (rows, 83) beat-synchronous features, memory-mappable
    tokens/<split>-NNN.parquet note-token windows, one row per kept difficulty
    lights/<split>-NNN.parquet lighting-token windows, one row per kept map with lights
    dataset.json           layout, vocab sizes, filters, tech calibration and build stats

Feature row ``j`` of a map is map beat ``j / 24``; a map has ``n_windows * 384`` rows,
so window ``k`` is rows ``k*384 .. (k+1)*384`` from the map's ``feature_row``.

Dataset v1 adds two labels per kept difficulty: ``intensity`` (``data.intensity``)
and, per window, the best earlier reference span ``ref_start`` / ``ref_score``
(``data.structure``; the threshold is applied at training time). ``dataset.json``
gets the intensity calibration and the human repetition stats on matched windows.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bsmapper.data.build import BUILDER_VERSION, MAPS_DIR, TRACK_DIR, WORK_DIR, read_json
from bsmapper.data.mapgrid import FEATURE_COLUMNS, FEATURE_DIM, N_MELS, MapTiming, grid_drift
from bsmapper.data.songs import SPLITS, group_songs, split_for
from bsmapper.data.structure import DEFAULT_REF_THRESHOLD
from bsmapper.data.tech import (
    FEATURES,
    calibrate_by_difficulty,
    feature_matrix,
    style_by_difficulty,
)

log = logging.getLogger(__name__)

DATASET_VERSION = 1
MANIFEST = "manifest.parquet"


@dataclass(frozen=True, slots=True)
class PackConfig:
    max_note_off_grid: float = 0.10  # share of notes/bombs > 0.1 tick off the 1/24 grid
    min_notes: int = 20
    max_nps: float = 20.0
    max_rule_errors_per_100: float = 10.0
    max_windows_past_audio: int = 1  # objects may end at most this many windows after the song
    val_fraction: float = 0.05
    test_fraction: float = 0.05
    maps_per_shard: int = 64
    tech_ridge_alpha: float = 1.0
    ref_threshold: float = DEFAULT_REF_THRESHOLD  # for the repetition stats only
    repetition_sample: int = 4000  # difficulties sampled for the repetition stats


def _pa():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("the dataset build needs pyarrow: pip install -e .[ml]") from e
    return pa, pq


def _manifest_records(raw_dir: Path) -> dict[str, Any]:
    from bsmapper.data.manifest import read_manifest

    return {e.record.key: e for e in read_manifest(raw_dir)}


def _bl(entry, characteristic: str, difficulty: str) -> dict[str, Any]:
    if entry is None:
        return {}
    for d in entry.record.diffs:
        if d.characteristic == characteristic and d.difficulty == difficulty and d.beatleader:
            return asdict(d.beatleader)
    return {}


def diff_reject_reason(d: dict, rec: dict, cfg: PackConfig) -> str | None:
    if d["notes"] < cfg.min_notes:
        return "too_few_notes"
    if d["note_off_grid_ratio"] > cfg.max_note_off_grid:
        return "off_grid"
    if d["nps"] > cfg.max_nps:
        return "nps_extreme"
    if d["rule_errors_per_100"] > cfg.max_rule_errors_per_100:
        return "rule_errors"
    if d["last_window"] >= rec["audio_windows"] + cfg.max_windows_past_audio:
        return "past_audio_end"
    return None


def _drift(rec: dict, track: dict | None) -> dict[str, Any]:
    from bsmapper.audio.grid import BeatGrid
    from bsmapper.schema import BpmChange

    if not track or track.get("status") != "ok":
        return {}
    detected = BeatGrid(
        float(track["bpm"]),
        float(track["offset_s"]),
        [BpmChange(float(b), float(v)) for b, v in track.get("bpm_changes", [])],
        int(track.get("beats_per_bar", 4)),
    )
    d = grid_drift(detected, MapTiming.from_dict(rec["timing"]), rec["duration_s"])
    out = {f"drift_{k}": v for k, v in asdict(d).items()}
    out["detected_bpm"] = detected.bpm
    out["detected_bpm_changes"] = len(detected.bpm_changes)
    out["detected_coverage"] = float(track.get("coverage", float("nan")))
    out["tracker"] = track.get("tracker")
    return out


def _clear(out_dir: Path) -> None:
    for name in ("features", "tokens", "lights"):
        p = out_dir / name
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True)


def pack(
    out_dir: Path,
    raw_dir: Path,
    keys: Sequence[str] | None = None,
    cfg: PackConfig | None = None,
) -> dict[str, Any]:
    """Write the dataset from the work files. Returns the stats also put in dataset.json."""
    pa, pq = _pa()
    cfg = cfg or PackConfig()
    work = out_dir / WORK_DIR
    if keys is None:
        keys = sorted(p.stem for p in (work / MAPS_DIR).glob("*.json"))
    manifest = _manifest_records(raw_dir)

    recs: dict[str, dict] = {}
    map_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    for key in keys:
        rec = read_json(work / MAPS_DIR / f"{key}.json")
        entry = manifest.get(key)
        base = {
            "key": key,
            "gold": bool(entry.gold) if entry else False,
            "gold_reasons": ",".join(entry.gold_reasons) if entry else "",
            "ranked": bool(entry.record.ranked_scoresaber or entry.record.ranked_beatleader)
            if entry
            else False,
            "curated": bool(entry.record.curated) if entry else False,
            "uploaded": entry.record.uploaded if entry else "",
            "score": entry.record.score if entry else float("nan"),
        }
        if rec is None or rec.get("version") != BUILDER_VERSION:
            map_rows.append({**base, "status": "rejected", "reason": "not_processed"})
            continue
        if rec["status"] != "ok":
            map_rows.append({**base, "status": "rejected", "reason": rec["reason"],
                             "error": rec.get("error")})  # fmt: skip
            continue
        drift = _drift(rec, read_json(work / TRACK_DIR / f"{key}.json"))
        kept = 0
        for i, d in enumerate(rec["diffs"]):
            reason = diff_reject_reason(d, rec, cfg)
            bl = _bl(entry, d["characteristic"], d["difficulty"])
            row = {
                "key": key,
                "index": i,
                "characteristic": d["characteristic"],
                "difficulty": d["difficulty"],
                "kept": reason is None,
                "reason": reason,
                **{k: d[k] for k in ("label", "njs", "njs_offset", "notes", "bombs",
                                     "obstacles", "arcs", "chains", "rotations", "nps",
                                     "note_off_grid_ratio", "snap_off_grid_ratio",
                                     "onset_ratio", "onset_best_shift", "onset_best_ratio",
                                     "rule_errors_per_100", "rule_warnings_per_100", "tokens")},
                **{f"flag_{k}": v for k, v in d["flags"].items()},
                **{f"tech_{f}": (d["tech"] or {}).get(f, float("nan")) for f in FEATURES},
                "rules": json.dumps(d["rules"]),
                **{f"bl_{k}": bl.get(k) for k in ("status", "stars", "tech_rating",
                                                   "pass_rating", "acc_rating")},
            }  # fmt: skip
            diff_rows.append(row)
            kept += reason is None
        for s in rec.get("skipped", []):
            diff_rows.append({"key": key, "index": -1, **s, "kept": False})
        light = rec.get("light") or {}
        map_rows.append(
            {
                **base,
                "status": "kept" if kept else "rejected",
                "reason": None if kept else "no_kept_difficulties",
                "song_name": rec["song_name"],
                "song_sub_name": rec["song_sub_name"],
                "song_author": rec["song_author"],
                "mapper": rec["mapper"],
                "environment": rec["environment"],
                "duration_s": rec["duration_s"],
                "bpm": rec["timing"]["bpm"],
                "bpm_changes": len(rec["timing"]["bpm_changes"]),
                "bpm_source": rec["bpm_source"],
                "bpm_remapped": any("disagree on BPM" in w for w in rec["read_warnings"]),
                "song_time_offset": rec["timing"]["song_time_offset"],
                "phase": rec["timing"]["phase"],
                "n_windows": rec["n_windows"],
                "audio_windows": rec["audio_windows"],
                "kept_diffs": kept,
                "light_source": light.get("source"),
                "light_events": light.get("basic_events", 0) + light.get("boost_events", 0),
                "light_tokens": light.get("tokens", 0),
                "light_chroma": bool(light.get("uses_chroma", False)),
                "light_dropped_box_groups": (light.get("dropped") or {}).get("event_box_group", 0),
                **drift,
            }
        )
        if kept:
            recs[key] = rec

    kept_maps = sorted(recs)
    # --- song groups and splits
    fps, titles, artists, durs = [], [], [], []
    for key in kept_maps:
        with np.load(work / MAPS_DIR / f"{key}.npz") as z:
            fps.append(z["fingerprint"])
        r = recs[key]
        titles.append(r["song_name"])
        artists.append(r["song_author"])
        durs.append(r["duration_s"])
    groups = group_songs(kept_maps, fps, titles, artists, durs)
    group_of = dict(zip(kept_maps, groups.group, strict=True))
    split_of = {k: split_for(g, cfg.val_fraction, cfg.test_fraction) for k, g in group_of.items()}
    for row in map_rows:
        if row["key"] in group_of:
            row["song_group"] = group_of[row["key"]]
            row["split"] = split_of[row["key"]]

    # --- tech score: calibrate on rated kept difficulties, then rank within difficulty
    kept_rows = [r for r in diff_rows if r.get("kept")]
    x = feature_matrix(
        [None if np.isnan(r["tech_angle_change"]) else {f: r[f"tech_{f}"] for f in FEATURES}
         for r in kept_rows]
    )  # fmt: skip
    y = np.array([np.nan if r["bl_tech_rating"] is None else r["bl_tech_rating"]
                  for r in kept_rows], dtype=np.float64)  # fmt: skip
    diffs = [r["difficulty"] for r in kept_rows]
    score, cals = calibrate_by_difficulty(x, y, diffs, alpha=cfg.tech_ridge_alpha)
    style = style_by_difficulty(score, diffs)
    for r, s, st in zip(kept_rows, score, style, strict=True):
        r["tech_score"] = float(s)
        r["style"] = float(st)

    # --- intensity: percentile within (characteristic, difficulty, tempo)
    from bsmapper.data.intensity import bpm_bucket, intensity_labels

    for r in kept_rows:
        r["bpm"] = recs[r["key"]]["timing"]["bpm"]
    intensity, intensity_cal = intensity_labels(kept_rows)
    for r, x in zip(kept_rows, intensity, strict=True):
        r["intensity"] = float(x)
        r["bpm_bucket"] = bpm_bucket(r["bpm"])

    # --- shards
    _clear(out_dir)
    from bsmapper.data.structure import References, find_references
    from bsmapper.tokenizer import intensity_bucket, nps_bucket, style_bucket
    from bsmapper.tokenizer.grid import WINDOW_TICKS

    rng = np.random.default_rng(0)
    sample_p = min(1.0, cfg.repetition_sample / max(len(kept_rows), 1))
    repetition_samples: list[tuple[str, str, np.ndarray, np.ndarray, References]] = []
    ref_maps = ref_windows = 0

    by_key: dict[str, list[dict]] = {}
    for r in kept_rows:
        by_key.setdefault(r["key"], []).append(r)
    map_by_key = {r["key"]: r for r in map_rows}
    manifest_rows: list[dict[str, Any]] = []
    shard_counts: Counter[str] = Counter()
    for split in SPLITS:
        split_keys = [k for k in kept_maps if split_of[k] == split]
        for s, start in enumerate(range(0, len(split_keys), cfg.maps_per_shard)):
            shard_keys = split_keys[start : start + cfg.maps_per_shard]
            name = f"{split}-{s:03d}"
            total_rows = sum(recs[k]["n_windows"] * WINDOW_TICKS for k in shard_keys)
            feats = np.lib.format.open_memmap(
                out_dir / "features" / f"{name}.npy",
                mode="w+",
                dtype=np.float16,
                shape=(total_rows, FEATURE_DIM),
            )
            tok_rows: dict[str, list] = {"key": [], "characteristic": [], "difficulty": [],
                                         "tokens": [], "window_offsets": []}  # fmt: skip
            light_rows: dict[str, list] = {"key": [], "tokens": [], "window_offsets": []}
            row0 = 0
            for key in shard_keys:
                m = map_by_key[key]
                with np.load(work / MAPS_DIR / f"{key}.npz") as z:
                    f = z["features"]
                    feats[row0 : row0 + len(f)] = f
                    n_win = recs[key]["n_windows"]
                    if "structure" in z.files:
                        refs = find_references(z["structure"].astype(np.float32), n_win)
                    else:
                        refs = References(np.full(n_win, -1), np.full(n_win, np.nan))
                    ref_maps += 1
                    ref_windows += int((refs.kept(cfg.ref_threshold) >= 0).sum())
                    m.update(feature_shard=name, feature_row=row0, feature_rows=len(f))
                    if "light" in z.files:
                        m.update(light_shard=name, light_row=len(light_rows["key"]))
                        light_rows["key"].append(key)
                        light_rows["tokens"].append(z["light"])
                        light_rows["window_offsets"].append(z["light_off"])
                        lengths = np.diff(z["light_off"])
                        m["light_max_window_tokens"] = int(lengths.max()) if len(lengths) else 0
                    for r in by_key[key]:
                        i = r["index"]
                        tokens, offsets = z[f"tok{i}"], z[f"off{i}"]
                        mrow = {
                            "key": key,
                            "split": split,
                            "song_group": group_of[key],
                            **{k: v for k, v in r.items() if k not in ("kept", "reason",
                                                                       "index")},
                            "nps_bucket": nps_bucket(r["nps"]),
                            "style_bucket": style_bucket(r["style"])
                            if np.isfinite(r["style"]) else -1,
                            "intensity_bucket": intensity_bucket(r["intensity"]),
                            "ref_start": refs.start.tolist(),
                            "ref_score": [float(x) for x in refs.score],
                            "max_window_tokens": int(np.diff(offsets).max()),
                            "n_windows": recs[key]["n_windows"],
                            "feature_shard": name,
                            "feature_row": row0,
                            "token_shard": name,
                            "token_row": len(tok_rows["key"]),
                            "light_shard": m.get("light_shard"),
                            "light_row": m.get("light_row", -1),
                            **{k: m.get(k) for k in ("gold", "ranked", "curated", "bpm",
                                                     "duration_s", "song_name", "song_author",
                                                     "mapper", "uploaded", "phase",
                                                     "bpm_remapped")},
                        }  # fmt: skip
                        manifest_rows.append(mrow)
                        tok_rows["key"].append(key)
                        tok_rows["characteristic"].append(r["characteristic"])
                        tok_rows["difficulty"].append(r["difficulty"])
                        tok_rows["tokens"].append(tokens)
                        tok_rows["window_offsets"].append(offsets)
                        if rng.random() < sample_p:
                            repetition_samples.append(
                                (r["characteristic"], r["difficulty"], tokens, offsets, refs)
                            )
                row0 += len(f)
            feats.flush()
            del feats
            pq.write_table(_token_table(pa, tok_rows), out_dir / "tokens" / f"{name}.parquet")
            if light_rows["key"]:
                pq.write_table(_token_table(pa, light_rows), out_dir / "lights" / f"{name}.parquet")
            shard_counts[split] += 1

    pq.write_table(pa.Table.from_pylist(manifest_rows), out_dir / MANIFEST)
    pq.write_table(pa.Table.from_pylist(_uniform(diff_rows)), out_dir / "difficulties.parquet")
    pq.write_table(pa.Table.from_pylist(_uniform(map_rows)), out_dir / "maps.parquet")

    from bsmapper.data.stats import build_stats

    stats = build_stats(map_rows, diff_rows, manifest_rows, cals, groups)
    stats["intensity"] = intensity_cal
    stats["structure"] = structure_stats(repetition_samples, cfg.ref_threshold)
    stats["structure"]["maps"] = ref_maps
    stats["structure"]["windows_with_reference"] = ref_windows
    stats["structure"]["windows"] = int(sum(recs[k]["n_windows"] for k in kept_maps))
    meta = {
        "dataset_version": DATASET_VERSION,
        "builder_version": BUILDER_VERSION,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_dir": str(raw_dir),
        "ticks_per_beat": 24,
        "window_ticks": WINDOW_TICKS,
        "feature_dim": FEATURE_DIM,
        "feature_columns": [f"mel_{i}" for i in range(N_MELS)] + list(FEATURE_COLUMNS[1:]),
        "feature_dtype": "float16",
        "note_vocab_size": _vocab_size("NOTE_VOCAB"),
        "light_vocab_size": _vocab_size("LIGHT_VOCAB"),
        "shards": dict(shard_counts),
        "pack_config": asdict(cfg),
        "tech_calibration": {d: c.to_dict() for d, c in cals.items()},
        "stats": stats,
    }
    (out_dir / "dataset.json").write_text(json.dumps(meta, indent=1, default=_json_default))
    return meta


def structure_stats(samples, threshold: float) -> dict[str, Any]:
    """Human repetition on matched windows: the share of copies, mirrors, same-rhythm
    and different windows per difficulty (at ``threshold``), and by score bin (all
    difficulties pooled) to tune the threshold."""
    from bsmapper.data.structure import REPETITION_KINDS, window_repetition

    bins = np.round(np.arange(0.5, 1.0001, 0.05), 2)
    by_bin: dict[str, Counter[str]] = {}
    f1: dict[str, list[float]] = {}
    by_diff: dict[str, Counter[str]] = {}
    for char, diff, tokens, offsets, refs in samples:
        for _, score, kind, scores in window_repetition(tokens, offsets, refs):
            if kind is None:
                continue
            b = bins[max(0, np.searchsorted(bins, score, "right") - 1)]
            label = f"{b:.2f}" if score >= bins[0] else f"<{bins[0]:.2f}"
            by_bin.setdefault(label, Counter())[kind] += 1
            f1.setdefault(label, []).append(max(scores["copy"], scores["mirror"]))
            if score >= threshold and char == "Standard":
                by_diff.setdefault(diff, Counter())[kind] += 1

    def shares(c: Counter[str]) -> dict[str, float]:
        n = sum(c.values())
        return {"n": n, **{k: c[k] / n if n else float("nan") for k in REPETITION_KINDS}}

    return {
        "threshold": threshold,
        "sampled_difficulties": len(samples),
        "by_score": {
            k: {**shares(by_bin[k]), "mean_note_f1": float(np.mean(f1[k]))} for k in sorted(by_bin)
        },
        "standard_by_difficulty": {d: shares(c) for d, c in sorted(by_diff.items())},
    }


def _token_table(pa, rows: dict[str, list]):
    cols = {}
    for k, v in rows.items():
        if k == "tokens":
            cols[k] = pa.array([np.asarray(t, dtype=np.int16) for t in v], pa.list_(pa.int16()))
        elif k == "window_offsets":
            cols[k] = pa.array([np.asarray(t, dtype=np.int32) for t in v], pa.list_(pa.int32()))
        else:
            cols[k] = pa.array(v, pa.string())
    return pa.table(cols)


def _uniform(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give every row every column (pyarrow infers the schema from all rows)."""
    cols: dict[str, None] = {}
    for r in rows:
        cols.update(dict.fromkeys(r))
    return [{c: r.get(c) for c in cols} for r in rows]


def _vocab_size(name: str) -> int:
    from bsmapper import tokenizer

    return len(getattr(tokenizer, name))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
