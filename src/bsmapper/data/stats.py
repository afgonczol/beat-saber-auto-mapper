"""Build statistics for dataset.json and the dataset card (docs/DATASET.md)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from bsmapper.data.songs import SPLITS, SongGroups
from bsmapper.data.tech import FEATURES, TechCalibration, pearson, spearman
from bsmapper.schema import Characteristic, Difficulty

CHARS = [c.value for c in Characteristic if c.value not in ("NoArrows", "Lightshow")]
DIFFS = [d.value for d in Difficulty]


def _q(values: list[float], qs=(5, 25, 50, 75, 95)) -> list[float] | None:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    return [round(float(x), 4) for x in np.percentile(v, qs)] if len(v) else None


def summary_lines(stats: dict[str, Any]) -> list[str]:
    """A short human-readable summary for the CLI."""
    m, d = stats["maps"], stats["difficulties"]
    lines = [
        f"Maps: {m['kept']} kept of {m['seen']}; rejected: "
        + (", ".join(f"{k}={v}" for k, v in sorted(m["rejected"].items())) or "none"),
        f"Difficulties: {d['kept']} kept of {d['seen']}; rejected: "
        + (", ".join(f"{k}={v}" for k, v in sorted(d["rejected"].items())) or "none"),
    ]
    for split, s in stats["splits"].items():
        lines.append(
            f"  {split}: {s['maps']} maps, {s['songs']} songs, {s['difficulties']} difficulties, "
            f"{s['note_tokens']:,} note tokens, {s['light_tokens']:,} light tokens"
        )
    cals = stats["tech"]["calibration"]
    lines.append(
        "Tech score vs BeatLeader techRating within each difficulty (cross-validated "
        "Spearman): "
        + ", ".join(f"{d} {c['spearman_cv']:.2f} (n={c['n']})" for d, c in cals.items())
    )
    g = stats["grid_drift"]
    if g.get("maps"):
        lines.append(
            f"Grid drift: {g['maps']} maps tracked, {g['relation']}, median |error| quantiles "
            f"{g['median_abs_ticks_quantiles']} ticks"
        )
    return lines


def build_stats(
    map_rows: list[dict[str, Any]],
    diff_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    cals: dict[str, TechCalibration],
    groups: SongGroups,
) -> dict[str, Any]:
    kept_maps = [m for m in map_rows if m["status"] == "kept"]
    stats: dict[str, Any] = {
        "maps": {
            "seen": len(map_rows),
            "kept": len(kept_maps),
            "rejected": dict(Counter(m["reason"] for m in map_rows if m["status"] != "kept")),
        },
        "difficulties": {
            "seen": len(diff_rows),
            "kept": sum(1 for d in diff_rows if d.get("kept")),
            # Difficulties of rejected maps are not listed: only maps that were read are.
            "rejected": dict(Counter(d["reason"] for d in diff_rows if not d.get("kept"))),
        },
    }
    # Rejection reasons per characteristic (playable ones) for the card.
    rej = defaultdict(Counter)
    for d in diff_rows:
        if not d.get("kept"):
            rej[d["characteristic"]][d["reason"]] += 1
    stats["difficulties"]["rejected_by_characteristic"] = {k: dict(v) for k, v in rej.items()}

    # --- splits
    split_stats = {}
    for split in SPLITS:
        rows = [r for r in manifest_rows if r["split"] == split]
        maps = {r["key"] for r in rows}
        lights = [m for m in kept_maps if m.get("split") == split and m.get("light_tokens")]
        split_stats[split] = {
            "maps": len(maps),
            "songs": len({r["song_group"] for r in rows}),
            "difficulties": len(rows),
            "windows": sum(m["n_windows"] for m in kept_maps if m.get("split") == split),
            "note_tokens": int(sum(r["tokens"] for r in rows)),
            "maps_with_lights": len(lights),
            "light_tokens": int(sum(m["light_tokens"] for m in lights)),
        }
    stats["splits"] = split_stats

    # --- per characteristic x difficulty
    table: dict[str, dict[str, dict[str, int]]] = {}
    for c in CHARS:
        for d in DIFFS:
            rows = [r for r in manifest_rows if r["characteristic"] == c and r["difficulty"] == d]
            if not rows:
                continue
            cell = {"difficulties": len(rows), "note_tokens": int(sum(r["tokens"] for r in rows))}
            for split in SPLITS:
                cell[split] = sum(1 for r in rows if r["split"] == split)
            table.setdefault(c, {})[d] = cell
    stats["by_characteristic_difficulty"] = table

    # --- tech score
    rated = [r for r in manifest_rows if r.get("bl_tech_rating") is not None]
    tech: dict[str, Any] = {
        "calibration": {d: cals[d].to_dict() for d in DIFFS if d in cals},
        "rated_difficulties": len(rated),
        "difficulties_without_features": sum(
            1 for r in manifest_rows if not np.isfinite(r["tech_score"])
        ),
    }
    if rated:
        t = np.array([r["bl_tech_rating"] for r in rated])
        n = np.array([r["nps"] for r in rated])
        tech["bl_tech_vs_nps_pearson"] = pearson(t, n)
        per_diff = {}
        for d in DIFFS:
            sub = [r for r in rated if r["difficulty"] == d and np.isfinite(r["tech_score"])]
            if len(sub) >= 10:
                s_, t_, n_ = (np.array([r[k] for r in sub]) for k in ("tech_score",
                              "bl_tech_rating", "nps"))  # fmt: skip
                per_diff[d] = {
                    "n": len(sub),
                    "spearman_train": spearman(s_, t_),
                    "bl_tech_vs_nps_spearman": spearman(t_, n_),
                    "style_vs_nps_spearman": spearman(np.array([r["style"] for r in sub]), n_),
                }
        tech["by_difficulty"] = per_diff
        # Within-difficulty agreement per characteristic: mean over difficulties with >= 10.
        per_char = {}
        for c in CHARS:
            vals, total = [], 0
            for d in DIFFS:
                sub = [r for r in rated if r["characteristic"] == c and r["difficulty"] == d
                       and np.isfinite(r["tech_score"])]  # fmt: skip
                if len(sub) >= 10:
                    vals.append(spearman(np.array([r["tech_score"] for r in sub]),
                                         np.array([r["bl_tech_rating"] for r in sub])))  # fmt: skip
                    total += len(sub)
            if vals:
                per_char[c] = {"n": total, "mean_within_difficulty_spearman": float(np.mean(vals))}
        tech["by_characteristic"] = per_char
    tech["feature_quantiles"] = {f: _q([r[f"tech_{f}"] for r in manifest_rows]) for f in FEATURES}
    stats["tech"] = tech

    # --- grid drift (detected vs map grid)
    tracked = [m for m in kept_maps if m.get("drift_relation") is not None]
    drift: dict[str, Any] = {"maps": len(tracked)}
    if tracked:
        drift["relation"] = dict(Counter(m["drift_relation"] for m in tracked))
        same = [m for m in tracked if m["drift_relation"] in ("same", "double", "half")]
        drift["median_abs_ticks_quantiles"] = _q([m["drift_median_abs_ticks"] for m in same])
        drift["p95_abs_ticks_quantiles"] = _q([m["drift_p95_abs_ticks"] for m in same])
        drift["signed_median_ticks_quantiles"] = _q([m["drift_median_ticks"] for m in same])
        drift["maps_90pct_within_1_tick"] = sum(m["drift_within_1_tick"] >= 0.9 for m in same)
        drift["maps_90pct_within_2_ticks"] = sum(m["drift_within_2_ticks"] >= 0.9 for m in same)
        drift["coverage_quantiles"] = _q([m["detected_coverage"] for m in tracked])
        drift["low_coverage_maps"] = sum(m["detected_coverage"] < 0.7 for m in tracked)
        drift["map_has_bpm_changes"] = sum(m["bpm_changes"] > 0 for m in tracked)
    stats["grid_drift"] = drift

    # --- timing and lights
    stats["timing"] = {
        "phase_corrected_maps": sum(abs(m["phase"]) > 0 for m in kept_maps),
        "bpm_change_maps": sum(m["bpm_changes"] > 0 for m in kept_maps),
        "bpm_remapped_maps": sum(bool(m["bpm_remapped"]) for m in kept_maps),
        "nonzero_song_time_offset_maps": sum(m["song_time_offset"] != 0 for m in kept_maps),
        "duration_s_quantiles": _q([m["duration_s"] for m in kept_maps]),
    }
    lit = [m for m in kept_maps if m.get("light_tokens")]
    stats["lights"] = {
        "maps_with_lights": len(lit),
        "chroma_maps": sum(m["light_chroma"] for m in lit),
        "maps_with_dropped_box_groups": sum(m["light_dropped_box_groups"] > 0 for m in lit),
        "tokens_per_map_quantiles": _q([m["light_tokens"] for m in lit]),
        "max_window_tokens_quantiles": _q([m.get("light_max_window_tokens", 0) for m in lit]),
    }
    shifts = Counter(r["onset_best_shift"] for r in manifest_rows)
    stats["onset_alignment"] = {
        "best_shift_ticks": dict(sorted(shifts.items())),
        "share_best_shift_0_to_2": sum(shifts[k] for k in (0, 1, 2)) / max(len(manifest_rows), 1),
        "share_best_shift_half_beat": sum(v for k, v in shifts.items() if abs(k) >= 10)
        / max(len(manifest_rows), 1),
        "best_ratio_quantiles": _q([r["onset_best_ratio"] for r in manifest_rows]),
    }
    stats["note_tokens"] = {
        "per_difficulty_quantiles": _q([r["tokens"] for r in manifest_rows]),
        "max_window_tokens_quantiles": _q([r["max_window_tokens"] for r in manifest_rows]),
    }
    stats["songs"] = {
        "groups": len(set(groups.group)),
        "multi_map_groups": sum(1 for c in Counter(groups.group).values() if c > 1),
        "fingerprint_links": groups.fingerprint_links,
        "metadata_links": groups.metadata_links,
    }
    return stats
