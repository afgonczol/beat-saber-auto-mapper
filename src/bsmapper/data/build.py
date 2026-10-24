"""Dataset builder (WS3b): downloaded map zips -> training shards.

Three stages, all resumable:

1. **Process** (CPU, ``workers`` processes): per map, read it (WS1), decode the
   audio and compute frame features (WS2), resample them onto the map's own grid
   (``mapgrid``), snap every difficulty (WS4 ``snap_map``), compute stats, rule
   issues (WS5) and tech features, tokenize notes and the lightshow (WS4) and
   fingerprint the audio and compute its per-beat structure features
   (``structure``). Writes ``work/maps/<key>.npz`` + ``<key>.json``.
2. **Track** (one process, uses the GPU): run the WS2 analyzer to get the
   *detected* beat grid, written to ``work/track/<key>.json``. It only feeds the
   grid-drift stat, so it runs alongside stage 1 and can be skipped.
3. **Pack** (``pack.pack``): apply the filters, calibrate the tech score, group
   maps by song, split, and write the shards and ``manifest.parquet``.

A map whose ``.json`` exists with the current ``BUILDER_VERSION`` is not processed
again, so an interrupted build continues where it stopped. Filters and the tech
calibration live in the pack stage, so changing them never needs stage 1 again.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import tempfile
import time
import traceback
import zipfile
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np

from bsmapper.schema import Beatmap, Characteristic, DifficultyMap, Lightshow

if TYPE_CHECKING:
    from bsmapper.audio.beats import BeatTracker

log = logging.getLogger(__name__)

BUILDER_VERSION = 3  # bump when stage 1 output changes; such maps are processed again
TRACK_VERSION = 1  # the same for stage 2 (detected grids)
PLAYABLE = (
    Characteristic.STANDARD,
    Characteristic.ONE_SABER,
    Characteristic.DEGREE_90,
    Characteristic.DEGREE_360,
)
WORK_DIR = "work"
MAPS_DIR = "maps"
TRACK_DIR = "track"


@dataclass(slots=True)
class BuildConfig:
    raw_dir: Path
    out_dir: Path
    workers: int = max((os.cpu_count() or 2) - 2, 1)
    tracker: str = "beat_this"  # "beat_this", "librosa", "auto" or "none"
    keys: tuple[str, ...] = ()  # only these maps (default: the whole manifest)
    limit: int | None = None  # only the first N manifest maps
    retry_errors: bool = False  # process maps again whose previous attempt crashed
    pack: bool = True
    # Recycling workers (max_tasks_per_child) deadlocks the pool on CPython 3.12 once the
    # first batch of workers retires (gh-115634), so it stays off by default.
    max_tasks_per_child: int | None = None

    @property
    def work_dir(self) -> Path:
        return self.out_dir / WORK_DIR


# --- stage 1: one map ------------------------------------------------------------


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, obj: Any) -> None:
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_map_audio(zip_path: Path, audio_name: str):
    """The map's song as mono float32 at the analyzer's sample rate."""
    from bsmapper.audio.decode import load_audio
    from bsmapper.formats.source import MapSource

    with MapSource(zip_path) as source:
        data = source.read_bytes(audio_name)
    suffix = PurePosixPath(audio_name).suffix or ".egg"
    with tempfile.TemporaryDirectory(prefix="bsmapper-") as td:
        path = Path(td) / f"song{suffix}"
        path.write_bytes(data)
        return load_audio(path)


def _audio_name(zip_path: Path) -> str:
    from bsmapper.formats import info as info_mod
    from bsmapper.formats.report import ReadReport
    from bsmapper.formats.source import INFO_NAME, MapSource

    with MapSource(zip_path) as source:
        return info_mod.parse(source.load_json(INFO_NAME), ReadReport()).beatmap.audio_path


def _last_window(dmap: DifficultyMap) -> int:
    from bsmapper.tokenizer.grid import WINDOW_BEATS

    beats = [o.beat for objs in (dmap.notes, dmap.bombs, dmap.obstacles, dmap.arcs,
                                 dmap.chains, dmap.rotations) for o in objs]  # fmt: skip
    return int(max(beats) // WINDOW_BEATS) if beats else 0


def _flat(windows) -> tuple[np.ndarray, np.ndarray]:
    """Window bodies as one int16 token array plus int32 offsets (len = windows + 1)."""
    lengths = [len(w.body) for w in windows]
    offsets = np.zeros(len(windows) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(lengths)
    tokens = np.fromiter((t for w in windows for t in w.body), dtype=np.int16, count=offsets[-1])
    return tokens, offsets


def _pick_lightshow(beatmap: Beatmap) -> tuple[str, Lightshow] | None:
    """The richest lightshow of the map: the shared one or a difficulty's own."""
    options: list[tuple[str, Lightshow]] = [("shared", beatmap.lightshow)]
    options += [
        (f"{d.characteristic}/{d.difficulty}", d.lightshow)
        for d in beatmap.difficulties
        if d.lightshow is not None
    ]
    name, ls = max(options, key=lambda o: len(o[1].basic_events) + len(o[1].boost_events))
    return (name, ls) if ls.basic_events or ls.boost_events else None


def _difficulty(dmap: DifficultyMap, timing, checker) -> tuple[dict, DifficultyMap]:
    """Stats for one (phase-shifted) difficulty and its snapped version."""
    from bsmapper.data.mapgrid import off_grid_ratio
    from bsmapper.data.tech import tech_features
    from bsmapper.rules import summarize
    from bsmapper.rules.spans import SPAN_RULES
    from bsmapper.tokenizer import notes_per_second, snap_map

    snapped, rep = snap_map(dmap)
    hit_beats = np.array([o.beat for o in (*dmap.notes, *dmap.bombs)], dtype=np.float64)
    timing_map = timing.beatmap()
    # The arc and chain rules (#12) are for generated maps; leave them out so the dataset
    # filter stays what datasets v0 and v1 were built with.
    issues = [i for i in checker.check(snapped, timing_map) if i.rule not in SPAN_RULES]
    errors = [i for i in issues if i.severity == "error"]
    per100 = max(len(snapped.notes), 1) / 100.0
    stats = {
        "characteristic": str(dmap.characteristic),
        "difficulty": str(dmap.difficulty),
        "label": dmap.label,
        "njs": dmap.njs,
        "njs_offset": dmap.njs_offset,
        "notes": len(snapped.notes),
        "bombs": len(snapped.bombs),
        "obstacles": len(snapped.obstacles),
        "arcs": len(snapped.arcs),
        "chains": len(snapped.chains),
        "rotations": len(snapped.rotations),
        "nps": notes_per_second(snapped, timing_map),
        "note_off_grid_ratio": off_grid_ratio(hit_beats),
        "snap_off_grid_ratio": rep.off_grid_ratio,
        "snap_max_error_ticks": rep.max_error_ticks,
        "snap_dropped": dict(rep.dropped),
        "rule_errors_per_100": len(errors) / per100,
        "rule_warnings_per_100": (len(issues) - len(errors)) / per100,
        "rules": {r: v["count"] for r, v in summarize(issues, len(snapped.notes)).items()},
        "tech": tech_features(snapped),
        "flags": {
            "bombs": bool(snapped.bombs),
            "walls": bool(snapped.obstacles),
            "arcs": bool(snapped.arcs),
            "chains": bool(snapped.chains),
        },
        "last_window": _last_window(snapped),
    }
    return stats, snapped


def process_map(key: str, zip_path: str, work_dir: str) -> str:
    """Stage 1 for one map. Writes its .npz/.json and returns the status."""
    out = Path(work_dir) / MAPS_DIR
    out.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"key": key, "version": BUILDER_VERSION, "status": "ok"}
    t0 = time.perf_counter()
    try:
        rec.update(_process(key, Path(zip_path), out))
    except Exception as e:  # one broken map must not stop the build
        rec.update(status="rejected", reason="error", error=f"{type(e).__name__}: {e}")
        rec["traceback"] = traceback.format_exc(limit=5)
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    _write_json(out / f"{key}.json", rec)
    return rec["status"] if rec["status"] == "ok" else f"rejected:{rec['reason']}"


class _Reject(Exception):
    def __init__(self, reason: str, **info: Any) -> None:
        super().__init__(reason)
        self.reason, self.info = reason, info


def _process(key: str, zip_path: Path, out: Path) -> dict[str, Any]:
    try:
        return _process_inner(key, zip_path, out)
    except _Reject as r:
        return {"status": "rejected", "reason": r.reason, **r.info}


def _process_inner(key: str, zip_path: Path, out: Path) -> dict[str, Any]:
    from bsmapper.audio.decode import AudioDecodeError
    from bsmapper.audio.features import FeatureConfig, compute_frame_features
    from bsmapper.data.mapgrid import (
        N_MELS,
        MapTiming,
        audio_windows,
        grid_phase,
        map_grid_features,
        onset_alignment,
        shift_lightshow,
        shift_map,
    )
    from bsmapper.data.songs import fingerprint
    from bsmapper.data.structure import (
        CHROMA_HOP,
        frame_chroma,
        per_beat_mean,
        structure_features,
    )
    from bsmapper.formats import MapFormatError, read_map_report
    from bsmapper.rules import RuleChecker
    from bsmapper.tokenizer import (
        Conditioning,
        LightingTokenizer,
        MapTokenizer,
        prepare_lightshow,
    )
    from bsmapper.tokenizer.grid import WINDOW_BEATS

    try:
        beatmap, report = read_map_report(zip_path)
    except (MapFormatError, OSError, zipfile.BadZipFile, ValueError, KeyError) as e:
        raise _Reject("unreadable", error=f"{type(e).__name__}: {e}") from e
    meta = {
        "song_name": beatmap.song_name,
        "song_sub_name": beatmap.song_sub_name,
        "song_author": beatmap.song_author,
        "mapper": beatmap.mapper,
        "environment": beatmap.environment,
        "info_version": report.info_version,
        "difficulty_versions": report.difficulty_versions,
        "bpm_source": report.bpm_source,
        "read_dropped": dict(report.dropped),
        "read_warnings": report.warnings[:20],
    }
    if report.needs_mods:
        mods = [m for m, on in (("noodle", report.noodle), ("me", report.mapping_extensions)) if on]
        raise _Reject("needs_mods", mods=mods, **meta)
    skipped = [
        {"characteristic": str(d.characteristic), "difficulty": str(d.difficulty),
         "reason": "characteristic"}
        for d in beatmap.difficulties
        if d.characteristic not in PLAYABLE
    ]  # fmt: skip
    diffs = [d for d in beatmap.difficulties if d.characteristic in PLAYABLE]
    if not diffs:
        raise _Reject("no_playable_difficulties", skipped=skipped, **meta)

    note_beats = np.array([n.beat for d in diffs for n in d.notes], dtype=np.float64)
    phase, off_before, off_after = grid_phase(note_beats)
    timing = MapTiming.of(beatmap, phase)

    try:
        y = load_map_audio(zip_path, beatmap.audio_path)
    except FileNotFoundError as e:
        raise _Reject("audio_missing", error=str(e), **meta) from e
    except AudioDecodeError as e:
        raise _Reject("audio_decode", error=str(e)[:500], **meta) from e
    cfg = FeatureConfig()
    duration = len(y) / cfg.sr
    if duration < 5.0:
        raise _Reject("audio_too_short", duration_s=duration, **meta)
    frame = compute_frame_features(y, cfg)
    chroma_frames = frame_chroma(y, cfg.sr)
    del y

    checker = RuleChecker()
    n_windows = audio_windows(timing, duration)
    stats, snapped = [], []
    for d in diffs:
        s, m = _difficulty(shift_map(d, -phase), timing, checker)
        stats.append(s)
        snapped.append(m)
    n_audio_windows = n_windows
    n_windows = max([n_windows] + [s["last_window"] + 1 for s in stats])

    arrays: dict[str, np.ndarray] = {}
    tok = MapTokenizer()
    for i, (s, m) in enumerate(zip(stats, snapped, strict=True)):
        windows = tok.encode(m, Conditioning.from_map(m), n_windows=n_windows)
        arrays[f"tok{i}"], arrays[f"off{i}"] = _flat(windows)
        s["tokens"] = int(arrays[f"off{i}"][-1])

    light: dict[str, Any] | None = None
    picked = _pick_lightshow(beatmap)
    if picked is not None:
        source, ls = picked
        ls, lrep = prepare_lightshow(shift_lightshow(ls, -phase))
        end = n_windows * WINDOW_BEATS
        past = sum(e.beat >= end for e in ls.basic_events) + sum(
            b.beat >= end for b in ls.boost_events
        )
        ls.basic_events = [e for e in ls.basic_events if e.beat < end]
        ls.boost_events = [b for b in ls.boost_events if b.beat < end]
        windows = LightingTokenizer().encode(ls, n_windows=n_windows)
        arrays["light"], arrays["light_off"] = _flat(windows)
        light = {
            "source": source,
            "basic_events": len(ls.basic_events),
            "boost_events": len(ls.boost_events),
            "uses_chroma": ls.uses_chroma,
            "dropped": dict(lrep.dropped),
            "past_end": int(past),
            "tokens": int(arrays["light_off"][-1]),
        }

    arrays["features"] = map_grid_features(frame, cfg.frame_rate, timing, duration, n_windows)
    onset = arrays["features"][:, N_MELS].astype(np.float32)
    for s, m in zip(stats, snapped, strict=True):
        s.update(onset_alignment(onset, [n.beat for n in m.notes]))
    arrays["fingerprint"] = fingerprint(frame["mel"])
    chroma = per_beat_mean(
        chroma_frames, cfg.sr / CHROMA_HOP, timing.seconds, n_windows * WINDOW_BEATS
    )
    arrays["structure"] = structure_features(arrays["features"], chroma).astype(np.float16)
    tmp = out / f"{key}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, out / f"{key}.npz")
    return {
        **meta,
        "duration_s": duration,
        "timing": timing.to_dict(),
        "phase_off_grid_before": off_before,
        "phase_off_grid_after": off_after,
        "n_windows": n_windows,
        "audio_windows": n_audio_windows,
        "diffs": stats,
        "skipped": skipped,
        "light": light,
    }


# --- stage 2: detected grid --------------------------------------------------------


def track_maps(items: Sequence[tuple[str, str]], work_dir: str, tracker: str | BeatTracker) -> None:
    """Stage 2: the WS2 analyzer's grid for each (key, zip) without a track file yet."""
    from bsmapper.audio import Analyzer

    logging.basicConfig(level=logging.WARNING)
    out = Path(work_dir) / TRACK_DIR
    out.mkdir(parents=True, exist_ok=True)
    analyzer = Analyzer(tracker=tracker, beat_features=False)
    t_start = time.perf_counter()
    for n, (key, zip_path) in enumerate(items, 1):
        rec: dict[str, Any] = {"key": key, "version": TRACK_VERSION}
        t0 = time.perf_counter()
        try:
            y = load_map_audio(Path(zip_path), _audio_name(Path(zip_path)))
            res = analyzer.analyze_signal(y)
            g = res.grid
            rec.update(
                status="ok",
                tracker=res.tracker,
                bpm=g.bpm,
                offset_s=g.offset_s,
                bpm_changes=[[c.beat, c.bpm] for c in g.bpm_changes],
                beats_per_bar=g.beats_per_bar,
                coverage=res.fit.coverage,
                phase_shift_s=res.phase_shift_s,
                duration_s=res.analysis.duration_s,
            )
        except Exception as e:
            rec.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        _write_json(out / f"{key}.json", rec)
        if n % 100 == 0 or n == len(items):
            rate = n / (time.perf_counter() - t_start)
            print(f"[track] {n}/{len(items)} maps, {rate:.2f} maps/s", flush=True)


# --- orchestration -----------------------------------------------------------------


@dataclass(slots=True)
class BuildResult:
    processed: Counter[str] = field(default_factory=Counter)
    tracked: int = 0
    skipped_done: int = 0
    seconds: float = 0.0
    pack: dict[str, Any] | None = None


def _done(path: Path, retry_errors: bool, version: int = BUILDER_VERSION) -> bool:
    rec = read_json(path)
    if rec is None or rec.get("version") != version:
        return False
    return not (retry_errors and rec.get("reason") == "error")


def select_entries(cfg: BuildConfig) -> list[tuple[str, str]]:
    """(key, absolute zip path) for every manifest map the build covers."""
    from bsmapper.data.manifest import read_manifest

    entries = read_manifest(cfg.raw_dir)
    if cfg.keys:
        wanted = set(cfg.keys)
        entries = [e for e in entries if e.record.key in wanted]
    if cfg.limit is not None:
        entries = entries[: cfg.limit]
    return [(e.record.key, str(cfg.raw_dir / e.zip_path)) for e in entries]


def _limit_threads() -> None:
    # Each worker is one process; keep BLAS/FFT libraries from oversubscribing the CPU.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def run_build(cfg: BuildConfig, progress_every: int = 50) -> BuildResult:
    t_start = time.perf_counter()
    items = select_entries(cfg)
    maps_dir, track_dir = cfg.work_dir / MAPS_DIR, cfg.work_dir / TRACK_DIR
    maps_dir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not _done(maps_dir / f"{it[0]}.json", cfg.retry_errors)]
    result = BuildResult(skipped_done=len(items) - len(todo))
    track_todo: list[tuple[str, str]] = []
    if cfg.tracker != "none":
        track_todo = [
            it for it in items if not _done(track_dir / f"{it[0]}.json", False, TRACK_VERSION)
        ]
    log.info(
        "%d maps: %d to process, %d already done; %d to beat-track (%s), %d workers",
        len(items), len(todo), result.skipped_done, len(track_todo), cfg.tracker, cfg.workers,
    )  # fmt: skip

    _limit_threads()
    ctx = mp.get_context("spawn")
    tracker_proc = None
    if track_todo:
        tracker_proc = ctx.Process(
            target=track_maps,
            args=(track_todo, str(cfg.work_dir), cfg.tracker),
            name="bsmapper-track",
        )
        tracker_proc.start()

    if todo:
        with ProcessPoolExecutor(
            max_workers=cfg.workers, mp_context=ctx, max_tasks_per_child=cfg.max_tasks_per_child
        ) as pool:
            futures = {pool.submit(process_map, k, z, str(cfg.work_dir)): k for k, z in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                key = futures[fut]
                try:
                    status = fut.result()
                except Exception as e:  # the worker process died (e.g. out of memory)
                    status = "rejected:error"
                    _write_json(
                        maps_dir / f"{key}.json",
                        {"key": key, "version": BUILDER_VERSION, "status": "rejected",
                         "reason": "error", "error": f"worker crashed: {e}"},
                    )  # fmt: skip
                result.processed[status] += 1
                if n % progress_every == 0 or n == len(todo):
                    el = time.perf_counter() - t_start
                    log.info(
                        "[process] %d/%d maps, %.2f maps/s, ETA %.0f min (%s)",
                        n, len(todo), n / el, (len(todo) - n) / max(n / el, 1e-9) / 60,
                        ", ".join(f"{k}={v}" for k, v in result.processed.most_common()),
                    )  # fmt: skip
    if tracker_proc is not None:
        log.info("waiting for beat tracking to finish ...")
        tracker_proc.join()
        result.tracked = len(track_todo)
    if cfg.pack:
        from bsmapper.data.pack import pack

        result.pack = pack(cfg.out_dir, cfg.raw_dir, keys=[k for k, _ in items])
    result.seconds = time.perf_counter() - t_start
    return result


__all__ = [
    "BUILDER_VERSION",
    "PLAYABLE",
    "BuildConfig",
    "BuildResult",
    "process_map",
    "read_json",
    "run_build",
    "track_maps",
]
