"""Run an evaluation: every system on every song, then aggregate.

    ckpt = Path("checkpoints/note_v0/best.pt")
    cfg = EvalConfig(data=Path("data/eval_v0_test"), checkpoints=[ckpt])
    summary = run_eval(cfg)          # writes records.jsonl, summary.json, report.html

Records (one JSON line per system x song x difficulty, plus one per lightshow and style
sample) are appended as they are computed, so ``resume=True`` skips the songs a system
already did. ``aggregate`` turns the records into the summary.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bsmapper.data.mapgrid import N_MELS
from bsmapper.eval import repetition as rep
from bsmapper.eval.data import EvalSong, GroupKey, load_songs, song_analysis
from bsmapper.eval.lighting import ONSET_TOLERANCE, lighting_metrics, near, onset_peaks
from bsmapper.eval.metrics import (
    DISTRIBUTIONS,
    calibrations,
    difficulty_metrics,
    distributions,
    jensen_shannon,
    onset_scores,
    onset_ticks,
    pool,
)
from bsmapper.eval.reference import LEVELS, HumanReference
from bsmapper.eval.store import MAPS_DIR, SavedSystem, save_output, saved_systems
from bsmapper.eval.systems import HUMAN, HumanSystem, PipelineSystem
from bsmapper.rules import RuleChecker
from bsmapper.schema import Characteristic, Difficulty
from bsmapper.tokenizer import notes_per_second

log = logging.getLogger(__name__)

RECORDS = "records.jsonl"
SUMMARY = "summary.json"
REPORT = "report.html"
SUMMARY_VERSION = 1
ALL = "all"

# The headline numbers: (key, label, better) with better "high", "low" or "human".
HEADLINE: tuple[tuple[str, str, str], ...] = (
    ("onset_f1_1", "onset F1 ±1 tick", "high"),
    ("onset_f1_2", "onset F1 ±2 ticks", "high"),
    ("nps", "NPS", "human"),
    ("nps_ratio", "NPS / human NPS", "human"),
    ("nps_abs_error", "|NPS error|", "low"),
    ("nps_percentile_median", "NPS percentile (median)", "human"),
    ("step_offset", "difficulty step offset", "human"),
    ("audio_sync", "note onsets on audio peaks", "human"),
    ("errors_per_100", "rule errors /100 notes", "low"),
    ("warnings_per_100", "rule warnings /100 notes", "low"),
    ("jsd_direction", "JSD direction", "low"),
    ("jsd_position", "JSD position", "low"),
    ("jsd_pattern_2gram", "JSD pattern 2-gram", "low"),
    ("jsd_pattern_3gram", "JSD pattern 3-gram", "low"),
    ("jsd_rhythm", "JSD rhythm", "low"),
    ("red_share", "red share", "human"),
    ("crossover_rate", "crossover rate", "human"),
    ("tech_score", "tech score", "human"),
    ("rep_reuse", "repeats reused (exact+mirror)", "human"),
    ("rep_distance", "repetition distance", "low"),
)
LIGHT_HEADLINE: tuple[tuple[str, str, str], ...] = (
    ("events_per_bar", "light events / bar", "human"),
    ("onset_precision", "events on onsets", "high"),
    ("onset_lift", "events on onsets / chance", "high"),
    ("onset_recall", "strong onsets lit", "high"),
    ("on_beat", "events on beats", "human"),
    ("on_downbeat", "events on bar lines", "human"),
    ("section_response", "section changes answered", "high"),
    ("section_lift", "section answer / bar-line answer", "high"),
    ("section_energy_r", "density ~ section energy (r)", "high"),
)


@dataclass(slots=True)
class EvalConfig:
    data: Path
    out: Path = Path("out/eval/run")
    split: str = "test"
    stub: bool = True
    checkpoints: list[Path] = field(default_factory=list)
    intensities: list[float | None] = field(default_factory=lambda: [None])
    keys: list[str] | None = None
    limit: int | None = None
    modes: list[Characteristic] | None = None
    difficulties: list[Difficulty] | None = None
    seed: int = 0
    device: str | None = None
    lighting: bool = True
    style_songs: int = 12  # songs for the style sweep (0 = off)
    styles: tuple[float, ...] = (0.0, 0.5, 1.0)
    style_difficulty: Difficulty = Difficulty.EXPERT
    batch_difficulties: int = 2
    resume: bool = False
    rescore: Path | None = None  # replay the saved maps of this earlier run folder
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return json.loads(json.dumps(d, default=str))


# --- records -------------------------------------------------------------------------


def group_name(key: GroupKey) -> str:
    return f"{key[0].value}/{key[1].value}"


def _clean(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, np.floating | np.integer):
        return _clean(x.item())
    return x


def difficulty_records(
    system: str,
    song: EvalSong,
    out,
    pairs: list[tuple[int, int]],
    reference: HumanReference,
    checker: RuleChecker,
    cals: dict,
) -> list[dict[str, Any]]:
    beatmap = song.beatmap()
    peaks = onset_peaks(song.features[:, N_MELS])
    records = []
    for gk, dmap in out.dmaps.items():
        human = song.humans.get(gk)
        mode, diff = gk
        m = difficulty_metrics(dmap, beatmap, checker, cals.get(diff.value))
        if human is not None:
            human_nps = notes_per_second(human, beatmap)
            m["human_nps"] = human_nps
            m["nps_error"] = m["nps"] - human_nps
            m["nps_abs_error"] = abs(m["nps_error"])
            m["nps_ratio"] = m["nps"] / human_nps if human_nps > 0 else float("nan")
            if system != HUMAN:
                m.update(onset_scores(human, dmap))
        m.update(audio_sync(dmap, peaks))
        m["nps_percentile"] = reference.nps_percentile(mode, diff, song.bpm, m["nps"])
        m["step_offset"] = reference.step_offset(mode, diff, song.bpm, m["nps"])
        m["equivalent_level"] = m["step_offset"] + LEVELS.index(diff)
        m["style_percentile"] = reference.style_percentile(diff, m["tech_score"])
        records.append(
            {
                "type": "difficulty",
                "system": system,
                "key": song.key,
                "group": group_name(gk),
                "characteristic": mode.value,
                "difficulty": diff.value,
                "bpm": song.bpm,
                "metrics": m,
                "dist": {k: dict(v) for k, v in distributions(dmap).items()},
                "repetition": dict(rep.repetition_counts(dmap, pairs)),
            }
        )
    return records


def audio_sync(dmap, peaks) -> dict[str, float]:
    """Share of note onsets on an audio onset peak, and that over chance (see lighting)."""
    ticks = onset_ticks(dmap)
    if not len(ticks):
        return {"audio_sync": float("nan"), "audio_sync_lift": float("nan")}
    hit = float(near(ticks, peaks, ONSET_TOLERANCE).mean())
    chance = float(near(np.arange(ticks[0], ticks[-1] + 1), peaks, ONSET_TOLERANCE).mean())
    return {"audio_sync": hit, "audio_sync_lift": hit / chance if chance > 0 else float("nan")}


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def build_systems(cfg: EvalConfig) -> list:
    from bsmapper.eval.systems import model_system, pick_device, stub_system

    systems: list = [HumanSystem()]
    if cfg.rescore is not None:
        names = saved_systems(cfg.rescore)
        if not names:
            raise ValueError(f"{cfg.rescore} has no saved maps ({MAPS_DIR}/)")
        return systems + [SavedSystem(cfg.rescore, n) for n in names]
    if cfg.stub:
        systems.append(stub_system(cfg.seed))
    if cfg.checkpoints:
        device = pick_device(cfg.device)
        for ckpt in cfg.checkpoints:
            for intensity in cfg.intensities:
                systems.append(
                    model_system(
                        ckpt,
                        intensity=intensity,
                        device=device,
                        seed=cfg.seed,
                        batch_difficulties=cfg.batch_difficulties,
                        lighting=cfg.lighting,
                    )
                )
    return systems


def run_eval(
    cfg: EvalConfig,
    systems: list | None = None,
    progress: Callable[[str], None] = lambda msg: None,
) -> dict[str, Any]:
    """Score every system on every song of the split and write the outputs."""
    from bsmapper.data import ProcessedDataset

    cfg.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    songs = load_songs(
        cfg.data,
        cfg.split,
        keys=cfg.keys,
        limit=cfg.limit,
        modes=cfg.modes,
        difficulties=cfg.difficulties,
    )
    if not songs:
        raise ValueError(f"no {cfg.split} songs in {cfg.data} match the filters")
    ds_meta = json.loads((Path(cfg.data) / "dataset.json").read_text(encoding="utf-8"))
    reference = HumanReference(ProcessedDataset(cfg.data).rows())
    cals = calibrations(ds_meta)
    checker = RuleChecker()
    systems = systems if systems is not None else build_systems(cfg)

    path = cfg.out / RECORDS
    old = read_records(path) if cfg.resume else []
    failed = {(r["system"], r["key"]) for r in old if r["type"] == "song" and r.get("error")}
    if failed:  # retry the songs that failed last time
        old = [r for r in old if (r["system"], r["key"]) not in failed]
        path.write_text("".join(f"{json.dumps(r)}\n" for r in old), encoding="utf-8")
    done = {(r["system"], r["key"], r["type"]) for r in old}
    mode = "a" if cfg.resume else "w"
    style_keys = {s.key for s in _style_songs(songs, cfg)}
    with path.open(mode, encoding="utf-8") as fh:

        def write(records: Iterable[dict[str, Any]]) -> None:
            for r in records:
                fh.write(json.dumps(_clean(r)) + "\n")
            fh.flush()

        for i, song in enumerate(songs):
            analysis = song_analysis(song)
            pairs = rep.repeated_pairs(song.features)
            groups = sorted(song.humans, key=lambda g: (g[0].value, g[1].rank))
            for system in systems:
                if (system.name, song.key, "song") in done:
                    continue
                t = time.perf_counter()
                try:
                    out = system.run(song, analysis, groups, lighting=cfg.lighting)
                except Exception as exc:  # one failing song must not end a long run
                    log.exception("%s failed on %s", system.name, song.key)
                    write([_song_record(system.name, song, error=str(exc))])
                    continue
                if isinstance(system, PipelineSystem):
                    save_output(cfg.out, system.name, song.key, out)
                records = difficulty_records(
                    system.name, song, out, pairs, reference, checker, cals
                )
                if cfg.lighting:
                    lm = lighting_metrics(out.lightshow, song.features, analysis, song.beatmap())
                    records.append(
                        {
                            "type": "lighting",
                            "system": system.name,
                            "key": song.key,
                            "source": out.lighting_source,
                            "metrics": lm,
                        }
                    )
                if system.name != HUMAN and song.key in style_keys:
                    records += _style_records(system, song, analysis, cfg, cals, reference)
                records.append(
                    _song_record(
                        system.name,
                        song,
                        seconds=time.perf_counter() - t,
                        pairs=len(pairs),
                        notes=out.notes,
                    )
                )
                write(records)
            progress(f"[{i + 1}/{len(songs)}] {song.key} {song.song_name[:40]}")
    records = read_records(path)
    summary = aggregate(records)
    summary["meta"] = {
        "version": SUMMARY_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": cfg.label or cfg.out.name,
        "config": cfg.to_dict(),
        "commit": _git_commit(),
        "dataset": {
            "path": str(cfg.data),
            "created": ds_meta.get("created"),
            "split": cfg.split,
            "songs": len(songs),
            "difficulties": sum(len(s.humans) for s in songs),
            "keys": [s.key for s in songs],
        },
        "systems": [_describe(s) for s in systems],
        "seconds": time.perf_counter() - started,
        "repetition": {
            "segment_beats": rep.SEGMENT_BEATS,
            "similarity": rep.SIMILARITY,
            "match_f1": rep.MATCH_F1,
        },
    }
    (cfg.out / SUMMARY).write_text(json.dumps(_clean(summary), indent=1), encoding="utf-8")
    from bsmapper.eval.report import write_report

    write_report(summary, cfg.out / REPORT)
    return summary


def _describe(system) -> dict[str, Any]:
    if hasattr(system, "describe"):
        return system.describe()
    return {"name": system.name, "generator": system.name}


def _song_record(system: str, song: EvalSong, **extra) -> dict[str, Any]:
    return {"type": "song", "system": system, "key": song.key, "song": song.song_name, **extra}


def _style_songs(songs: list[EvalSong], cfg: EvalConfig) -> list[EvalSong]:
    if cfg.style_songs <= 0:
        return []
    gk = (Characteristic.STANDARD, cfg.style_difficulty)
    return [s for s in songs if gk in s.humans][: cfg.style_songs]


def _style_records(system, song, analysis, cfg: EvalConfig, cals, reference) -> list[dict]:
    """The style sweep: one difficulty at each requested style, without lights."""
    gk = (Characteristic.STANDARD, cfg.style_difficulty)
    out = []
    for style in cfg.styles:
        try:
            res = system.run(song, analysis, [gk], style=style, lighting=False)
        except Exception:
            log.exception("style %s failed for %s on %s", style, system.name, song.key)
            continue
        if isinstance(system, PipelineSystem):
            save_output(cfg.out, system.name, song.key, res, style)
        dmap = res.dmaps.get(gk)
        if dmap is None:
            continue
        m = difficulty_metrics(dmap, song.beatmap(), RuleChecker(), cals.get(gk[1].value))
        out.append(
            {
                "type": "style",
                "system": system.name,
                "key": song.key,
                "style": style,
                "tech_score": m["tech_score"],
                "style_percentile": reference.style_percentile(gk[1], m["tech_score"]),
                "nps": m["nps"],
            }
        )
    return out


# --- aggregation ---------------------------------------------------------------------


def _nanmean(values: list[float]) -> float:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def _nanmedian(values: list[float]) -> float:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else float("nan")


def _groups_of(record: dict[str, Any]) -> list[str]:
    return [record["group"], f"{record['characteristic']}/{ALL}", ALL]


def _group_order(name: str) -> tuple:
    mode, _, diff = name.partition("/")
    modes = [c.value for c in Characteristic]
    diffs = [d.value for d in Difficulty] + [ALL]
    if name == ALL:
        return (len(modes) + 1, 0)
    return (modes.index(mode) if mode in modes else len(modes), diffs.index(diff))


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The summary: per system and group, mean metrics and distances to the human maps."""
    diff_recs = [r for r in records if r["type"] == "difficulty"]
    systems = list(dict.fromkeys(r["system"] for r in records))
    by: dict[tuple[str, str], list[dict]] = {}
    for r in diff_recs:
        for g in _groups_of(r):
            by.setdefault((r["system"], g), []).append(r)
    groups = sorted({g for _, g in by}, key=_group_order)

    def pooled(recs: list[dict], name: str) -> Counter:
        return pool(r["dist"][name] for r in recs)

    def rep_counts(recs: list[dict]) -> Counter:
        return pool(r["repetition"] for r in recs)

    out: dict[str, Any] = {"systems": {}, "groups": groups}
    for system in systems:
        sys_out: dict[str, Any] = {}
        for g in groups:
            recs = by.get((system, g), [])
            if not recs:
                continue
            humans = by.get((HUMAN, g), [])
            keys = {(r["key"], r["group"]) for r in recs}
            humans = [h for h in humans if (h["key"], h["group"]) in keys]
            metric_names = sorted({k for r in recs for k in r["metrics"]})
            stats: dict[str, Any] = {"n": len(recs)}
            for k in metric_names:
                stats[k] = _nanmean([r["metrics"].get(k) for r in recs])
            stats["nps_percentile_median"] = _nanmedian(
                [r["metrics"].get("nps_percentile") for r in recs]
            )
            stats["step_offset_median"] = _nanmedian(
                [r["metrics"].get("step_offset") for r in recs]
            )
            if humans:
                for name in DISTRIBUTIONS:
                    if system == HUMAN:
                        even = [h for h in humans if _song_half(h["key"]) == 0]
                        odd = [h for h in humans if _song_half(h["key"]) == 1]
                        stats[f"jsd_{name}"] = jensen_shannon(pooled(even, name), pooled(odd, name))
                    else:
                        stats[f"jsd_{name}"] = jensen_shannon(
                            pooled(recs, name), pooled(humans, name)
                        )
            counts = rep_counts(recs)
            sh = rep.shares(counts)
            stats["rep_pairs"] = sum(counts[c] for c in rep.CLASSES)
            for c in rep.CLASSES:
                stats[f"rep_{c}"] = sh[c]
            stats["rep_reuse"] = sh["exact"] + sh["mirrored"]
            if humans:
                stats["rep_distance"] = rep.share_distance(sh, rep.shares(rep_counts(humans)))
            sys_out[g] = stats
        out["systems"][system] = {"groups": sys_out}

    # Lighting: one record per song and system.
    light = [r for r in records if r["type"] == "lighting" and r.get("metrics")]
    for system in systems:
        recs = [r for r in light if r["system"] == system]
        if not recs:
            continue
        names = sorted({k for r in recs for k in r["metrics"]})
        stats = {"n": len(recs)}
        stats.update({k: _nanmean([r["metrics"].get(k) for r in recs]) for k in names})
        stats["sources"] = dict(Counter(r.get("source", "") for r in recs))
        out["systems"][system]["lighting"] = stats

    # Style sweep: does the tech score follow the requested style?
    style = [r for r in records if r["type"] == "style"]
    for system in systems:
        recs = [r for r in style if r["system"] == system]
        if not recs:
            continue
        styles = sorted({r["style"] for r in recs})
        per = {
            f"{s:g}": {
                "tech_score": _nanmean([r["tech_score"] for r in recs if r["style"] == s]),
                "style_percentile": _nanmean(
                    [r["style_percentile"] for r in recs if r["style"] == s]
                ),
                "nps": _nanmean([r["nps"] for r in recs if r["style"] == s]),
                "n": sum(r["style"] == s for r in recs),
            }
            for s in styles
        }
        ok = [r for r in recs if r["tech_score"] is not None and math.isfinite(r["tech_score"])]
        from bsmapper.data.tech import spearman

        rho = (
            spearman(np.array([r["style"] for r in ok]), np.array([r["tech_score"] for r in ok]))
            if len(ok) >= 3
            else float("nan")
        )
        lo, hi = f"{styles[0]:g}", f"{styles[-1]:g}"
        out["systems"][system]["style"] = {
            "by_style": per,
            "spearman": rho,
            "tech_delta": per[hi]["tech_score"] - per[lo]["tech_score"],
            "percentile_delta": per[hi]["style_percentile"] - per[lo]["style_percentile"],
        }

    songs = [r for r in records if r["type"] == "song"]
    for system in systems:
        recs = [r for r in songs if r["system"] == system]
        out["systems"][system]["run"] = {
            "songs": len(recs),
            "failed": sum(1 for r in recs if r.get("error")),
            "seconds_per_song": _nanmean([r.get("seconds") for r in recs]),
            "notes": sorted({n for r in recs for n in r.get("notes") or []}),
        }
    return out


def _song_half(key: str) -> int:
    """A stable split of the songs into two halves, for the human-vs-human floor."""
    return int(key, 16) % 2 if all(c in "0123456789abcdef" for c in key.lower()) else len(key) % 2
