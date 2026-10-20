"""Score beat trackers against human-mapped grids from Beat Saber map folders.

    python -m bsmapper.audio.benchmark "<...>/Beat Saber_Data/CustomLevels" --limit 20

For each map with a single BPM, the human grid is ``t = beat * 60 / bpm`` (plus
``_songTimeOffset``). Many mappers leave beat 0 off the music and place notes
at a constant fractional offset instead, so the reference phase is taken from
the notes: the mode of note positions modulo half a beat. We fit our grid with
each tracker and measure where our beats fall on that reference, in ticks
(1/24 beat). A tracker that picks double
or half the human tempo is scored on the finer of the two grids, since both map
the same song; the BPM ratio is reported separately.

Info.dat is parsed minimally here (BPM and song file only); full parsing is WS1's.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bsmapper.audio.analyzer import ONSET_LATENCY_S, Analyzer
from bsmapper.audio.beats import make_tracker
from bsmapper.audio.decode import SAMPLE_RATE, load_audio
from bsmapper.audio.grid import BeatGrid


@dataclass(slots=True)
class HumanGrid:
    name: str
    audio: Path
    bpm: float
    offset_s: float
    has_bpm_changes: bool
    note_phase: float = 0.0  # beats; where the notes sit relative to the BPM grid

    @property
    def ref_offset_s(self) -> float:
        return self.offset_s + self.note_phase * 60.0 / self.bpm


def read_human_grid(map_dir: Path) -> HumanGrid | None:
    info_path = next((p for p in map_dir.iterdir() if p.name.lower() == "info.dat"), None)
    if info_path is None:
        return None
    info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    if "audio" in info:  # v4
        bpm, song, offset = info["audio"]["bpm"], info["audio"]["songFilename"], 0.0
        diff_files = [d["beatmapDataFilename"] for d in info.get("difficultyBeatmaps", [])]
        bpm_info = map_dir / info["audio"].get("audioDataFilename", "BPMInfo.dat")
        changes = bpm_info.is_file() and _v4_has_changes(bpm_info)
    else:
        bpm, song = info["_beatsPerMinute"], info["_songFilename"]
        offset = float(info.get("_songTimeOffset", 0.0))
        diff_files = [
            d["_beatmapFilename"]
            for s in info.get("_difficultyBeatmapSets", [])
            for d in s.get("_difficultyBeatmaps", [])
        ]
        changes = False
    notes: list[float] = []
    for name in diff_files:
        path = map_dir / name
        if not path.is_file():
            continue
        d = json.loads(path.read_text(encoding="utf-8-sig"))
        changes = changes or _diff_has_changes(d)
        notes += [n["_time"] for n in d.get("_notes", []) if n.get("_type") != 3]
        notes += [n.get("b", 0.0) for n in d.get("colorNotes", [])]
    grid = HumanGrid(map_dir.name, map_dir / song, float(bpm), offset, changes)
    grid.note_phase = note_phase(np.array(notes, dtype=np.float64))
    return grid


def note_phase(note_beats: np.ndarray, bins: int = 48) -> float:
    """Most common note position modulo half a beat, in beats, wrapped to [-0.25, 0.25)."""
    if len(note_beats) < 16:
        return 0.0
    x = (note_beats % 0.5) / 0.5  # 0..1 within a half beat
    hist = np.bincount((x * bins).astype(np.int64) % bins, minlength=bins)
    mode = (np.argmax(hist) + 0.5) / bins
    d = (x - mode + 0.5) % 1.0 - 0.5  # circular distance to the mode
    near = np.abs(d) <= 2 / bins
    phase = (mode + d[near].mean()) * 0.5
    return float((phase + 0.25) % 0.5 - 0.25)


def _v4_has_changes(path: Path) -> bool:
    return len(json.loads(path.read_text(encoding="utf-8-sig")).get("bpmData", [])) > 1


def _diff_has_changes(d: dict) -> bool:
    if d.get("bpmEvents"):  # v3
        return len(d["bpmEvents"]) > 1 or d["bpmEvents"][0].get("b", 0) > 0
    if any(e.get("_type") == 100 for e in d.get("_events", [])):  # v2 official
        return True
    if d.get("_BPMChanges"):  # MediocreMapper
        return True
    custom = d.get("_customData") or {}
    return bool(custom.get("_BPMChanges") or custom.get("_bpmChanges"))


@dataclass(slots=True)
class Score:
    map: str
    tracker: str
    human_bpm: float
    note_phase_ticks: float
    bpm: float
    bpm_ratio: float
    n_bpm_changes: int
    coverage: float  # tracked beats on the fitted grid
    shift_ms: float  # onset-based phase refinement
    median_ticks: float  # signed: > 0 means our beats are late
    median_abs_ticks: float
    p95_abs_ticks: float
    within_1_tick: float
    seconds: float


def score_grid(
    grid: BeatGrid, human: HumanGrid, duration_s: float
) -> tuple[float, float, float, float]:
    """Phase error of our beats on the human grid in ticks.

    Returns (median, median |err|, p95 |err|, share of beats within 1 tick).
    """
    ratio = grid.bpm / human.bpm
    # Compare on the finer grid when the tracker is at double/half (or 2/3, 3/2) tempo.
    unit = 1.0
    for r, u in ((2.0, 0.5), (1.5, 0.5), (4.0, 0.25)):
        if abs(ratio / r - 1) < 0.02:
            unit = u
    if abs(human.note_phase) * 24 >= 1:
        # The mapper's grid is off the music, so the note phase only tells us the
        # beat modulo half a beat: score modulo half a beat as well.
        unit = min(unit, 0.5)
    t = grid.beat_times(duration_s)
    hb = (t - human.ref_offset_s) * human.bpm / 60.0  # position on the human grid, in beats
    err = (hb / unit - np.round(hb / unit)) * unit * 24  # ticks
    a = np.abs(err)
    return (
        float(np.median(err)),
        float(np.median(a)),
        float(np.percentile(a, 95)),
        float(np.mean(a <= 1.0)),
    )


def run(levels: Path, analyzers: list[Analyzer], limit: int | None) -> list[Score]:
    scores: list[Score] = []
    maps = sorted(p for p in levels.iterdir() if p.is_dir())
    done = 0
    for map_dir in maps:
        try:
            human = read_human_grid(map_dir)
        except (OSError, ValueError, KeyError) as e:
            print(f"skip {map_dir.name}: {e}")
            continue
        if human is None or human.has_bpm_changes or not human.audio.is_file():
            continue
        y = load_audio(human.audio)
        duration = len(y) / SAMPLE_RATE
        for an in analyzers:
            t0 = time.perf_counter()
            res = an.analyze_signal(y)
            grid = res.grid
            el = time.perf_counter() - t0
            signed, med, p95, within = score_grid(grid, human, duration)
            s = Score(
                map=human.name,
                tracker=res.tracker,
                human_bpm=human.bpm,
                note_phase_ticks=round(human.note_phase * 24, 2),
                bpm=round(grid.bpm, 3),
                bpm_ratio=round(grid.bpm / human.bpm, 4),
                n_bpm_changes=len(grid.bpm_changes),
                coverage=round(res.fit.coverage, 3),
                shift_ms=round(res.phase_shift_s * 1000, 1),
                median_ticks=round(signed, 2),
                median_abs_ticks=round(med, 2),
                p95_abs_ticks=round(p95, 2),
                within_1_tick=round(within, 3),
                seconds=round(el, 2),
            )
            print(s)
            scores.append(s)
        done += 1
        if limit and done >= limit:
            break
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("levels", type=Path, help="folder of map folders (e.g. CustomLevels)")
    ap.add_argument("--trackers", default="beat_this,librosa")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write per-map scores as JSON lines")
    ap.add_argument("--no-refine", action="store_true", help="skip onset phase refinement")
    ap.add_argument("--latency-ms", type=float, default=ONSET_LATENCY_S * 1000)
    args = ap.parse_args()
    trackers = [make_tracker(n) for n in args.trackers.split(",")]
    analyzers = [
        Analyzer(
            tracker=tr,
            beat_features=False,
            refine_phase=not args.no_refine,
            onset_latency_s=args.latency_ms / 1000,
        )
        for tr in trackers
    ]
    scores = run(args.levels, analyzers, args.limit)
    if args.out:
        args.out.write_text("\n".join(json.dumps(asdict(s)) for s in scores) + "\n")
    for tr in trackers:
        rows = [s for s in scores if s.tracker == tr.name]
        if not rows:
            continue
        same = np.mean([abs(s.bpm_ratio - 1) < 0.005 for s in rows])
        octave = np.mean([min(abs(s.bpm_ratio / r - 1) for r in (0.5, 1, 2)) < 0.005 for s in rows])
        good = np.mean([s.within_1_tick >= 0.9 for s in rows])
        print(
            f"{tr.name:10s} maps={len(rows)} same_bpm={same:.0%} bpm_or_octave={octave:.0%} "
            f"maps_with_90%_beats_within_1_tick={good:.0%} "
            f"median_p95_ticks={np.median([s.p95_abs_ticks for s in rows]):.2f} "
            f"median_signed_ticks={np.median([s.median_ticks for s in rows]):+.2f} "
            f"mean_seconds={np.mean([s.seconds for s in rows]):.1f}"
        )


if __name__ == "__main__":
    main()
