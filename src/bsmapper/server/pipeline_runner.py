"""``PipelineJobRunner``: the real WS7 pipeline behind the server's ``JobRunner`` interface.

Analysis results are kept per job (a few songs, least recently used dropped), so
regenerating one difficulty skips the audio analysis. Each job writes its map folder to
``work_dir/map`` and the zip next to it; the packaged, lead-in padded ``song.egg`` is what
the preview plays, so its beats line up with the map.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

from bsmapper.generation import GenerateOptions, Pipeline, replace_difficulty, safe_folder_name
from bsmapper.generation.pipeline import (
    MAX_NPS_RATIO,
    RETRY_NPS_RATIO,
    DifficultyResult,
    Song,
    song_notes,
)
from bsmapper.schema import Beatmap
from bsmapper.server.models import DifficultyWarning, JobOptions, RegenerateRequest
from bsmapper.server.runner import JobInput, JobResult, ProgressFn
from bsmapper.tokenizer import notes_per_second

log = logging.getLogger(__name__)

STUB_NOTE = "Notes come from the placeholder stub generator, not the ML model (WS6 pending)."


def _options(options: JobOptions) -> GenerateOptions:
    return GenerateOptions(
        modes=list(options.modes),
        difficulties=list(options.difficulties),
        lighting=options.lighting,
        style=options.style,
        seed=options.seed,
    )


def _warnings(results: list[DifficultyResult]) -> list[DifficultyWarning]:
    return [
        DifficultyWarning(
            characteristic=r.dmap.characteristic, difficulty=r.dmap.difficulty, issue=issue
        )
        for r in results
        for issue in r.report.issues
    ]


class PipelineJobRunner:
    """Runs ``generation.Pipeline`` for server jobs (one at a time, on the worker thread)."""

    def __init__(self, pipeline: Pipeline | None = None, max_cached_songs: int = 4) -> None:
        self.pipeline = pipeline or Pipeline()
        self.max_cached_songs = max_cached_songs
        self._songs: OrderedDict[str, Song] = OrderedDict()

    # --- JobRunner -------------------------------------------------------------------

    def run(self, job: JobInput, progress: ProgressFn) -> JobResult:
        opts = _options(job.options)
        progress("analyze", 0.02, "Analysing audio")
        song = self._song(job, refresh=True)
        progress("analyze", 0.1, f"{song.bpm:g} BPM, {song.analysis.duration_s:.0f} s")
        beatmap, results = self.pipeline.build(song, opts, progress)
        notes = self._notes(song)
        progress("package", 0.85, "Packaging zip")
        zip_path, egg = self._package(job, song, beatmap)
        progress("package", 1.0, "Done")
        return JobResult(
            beatmap, song.song_analysis.duration_s, zip_path, _warnings(results), notes, egg
        )

    def regenerate(
        self, job: JobInput, previous: JobResult, request: RegenerateRequest, progress: ProgressFn
    ) -> JobResult:
        key = (request.characteristic, request.difficulty)
        opts = _options(job.options)
        progress("analyze", 0.05, "Loading analysis")
        song = self._song(job)
        progress("generate", 0.1, f"Regenerating {key[0]} {key[1]} (seed {request.seed})")
        beatmap = previous.beatmap
        overrides = {"seed": request.seed, "temperature": request.temperature}
        res = self.pipeline.generate_difficulty(song, beatmap, *key, opts, **overrides)
        harder = [
            d
            for d in beatmap.difficulties
            if d.characteristic == key[0] and d.difficulty.rank > key[1].rank
        ]
        if harder:
            above = min(harder, key=lambda d: d.difficulty.rank)
            above_nps = notes_per_second(above, beatmap)
            if above_nps > 0 and res.nps >= MAX_NPS_RATIO * above_nps:
                res = self.pipeline.generate_difficulty(
                    song, beatmap, *key, opts, target_nps=RETRY_NPS_RATIO * above_nps, **overrides
                )
                res.attempts = 2
        progress("rules", 0.7, f"Rules: {res.report.summary()}")
        new_map = replace_difficulty(beatmap, res.dmap)
        warnings = [
            w for w in previous.warnings if (w.characteristic, w.difficulty) != key
        ] + _warnings([res])
        progress("package", 0.85, "Packaging zip")
        zip_path, egg = self._package(job, song, new_map)
        progress("package", 1.0, "Done")
        return JobResult(
            new_map, previous.duration_s, zip_path, warnings, list(previous.notes), egg
        )

    # --- pieces ---------------------------------------------------------------------------

    def _song(self, job: JobInput, refresh: bool = False) -> Song:
        song = None if refresh else self._songs.get(job.job_id)
        if song is None:
            song = self.pipeline.analyze(job.audio_path)
        self._songs[job.job_id] = song
        self._songs.move_to_end(job.job_id)
        while len(self._songs) > self.max_cached_songs:
            self._songs.popitem(last=False)
        return song

    def _notes(self, song: Song) -> list[str]:
        notes = song_notes(song)
        if getattr(self.pipeline.generator, "name", "") == "stub":
            notes.insert(0, STUB_NOTE)
        return notes

    def _package(self, job: JobInput, song: Song, beatmap: Beatmap) -> tuple[Path, Path]:
        map_dir = job.work_dir / "map"
        zip_path = job.work_dir / f"{safe_folder_name(song.title, song.artist)}.zip"
        for old in job.work_dir.glob("*.zip"):
            old.unlink()
        self.pipeline.package(song, beatmap, map_dir, zip_path)
        return zip_path, map_dir / "song.egg"
