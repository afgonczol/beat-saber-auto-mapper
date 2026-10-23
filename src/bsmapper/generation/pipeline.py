"""End-to-end generation: audio file -> playable map folder and zip.

    pipeline = Pipeline()                       # StubGenerator until WS6 lands
    result = pipeline.run(Path("song.mp3"), Path("out"), GenerateOptions(lighting="v3"))
    result.map_dir, result.zip_path

Stages (the names the web UI shows): ``analyze`` (WS2 analysis, lead-in padding),
``generate`` (one ``NoteGenerator`` call per mode x difficulty, Expert+ first),
``rules`` (snap, dedupe, rule fixes, NJS/offset: ``postprocess``), ``lighting``
(WS8 L0 lightshow) and ``package`` (WS1 writer, song.egg, cover, zip).

**Cross-difficulty consistency**: difficulties of a mode are generated from the hardest
down. If a lower one comes out at least as dense as the one above it, it is generated
again with ``target_nps`` below the harder one's, so NPS strictly rises with difficulty.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bsmapper.generation.density import object_flags
from bsmapper.generation.njs import JumpOverride
from bsmapper.generation.postprocess import PostReport, postprocess
from bsmapper.generation.stub import StubGenerator
from bsmapper.generation.timing import LEAD_IN_EPS_S, map_beat_shift, song_time_analysis
from bsmapper.interfaces import (
    AudioAnalysis,
    FormatVersion,
    GenerationRequest,
    LightingGenerator,
    LightingStyle,
    NoteGenerator,
)
from bsmapper.rules import RuleChecker
from bsmapper.schema import Beatmap, BpmChange, Characteristic, Difficulty, DifficultyMap
from bsmapper.tokenizer import notes_per_second

log = logging.getLogger(__name__)

# progress(stage, fraction 0..1 of the whole run, message); same shape as the server's.
ProgressFn = Callable[[str, float, str], None]

# A lower difficulty must stay below this share of the next harder one's NPS ...
MAX_NPS_RATIO = 0.95
# ... and is regenerated at this share when it does not.
RETRY_NPS_RATIO = 0.8
PREVIEW_S = 10.0
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _no_progress(stage: str, fraction: float, message: str = "") -> None:
    pass


def safe_folder_name(title: str, artist: str = "") -> str:
    """A Windows-safe map folder name, ``"Title - Artist"``."""
    name = f"{title} - {artist}" if artist else title
    name = _UNSAFE.sub("", name).strip().rstrip(". ")
    return name[:80].rstrip(". ") or "bsmapper map"


@dataclass(slots=True)
class GenerateOptions:
    modes: Sequence[Characteristic] = (Characteristic.STANDARD,)
    difficulties: Sequence[Difficulty] = tuple(Difficulty)
    lighting: LightingStyle = "v3"
    style: float | None = None  # 0 flowy .. 1 technical
    seed: int = 0
    temperature: float = 1.0
    njs: float | None = None  # NJS for every difficulty (overrides the table)
    # Per-difficulty NJS / reaction-time overrides; they win over ``njs``.
    jump: dict[Difficulty, JumpOverride] = field(default_factory=dict)
    # Lowest-priority defaults, e.g. ``njs.configured_jumps()`` from ~/.bsmapper/config.toml.
    jump_defaults: dict[Difficulty, JumpOverride] = field(default_factory=dict)
    environment: str | None = None  # None: picked for the lighting style
    version: FormatVersion = "v3"
    # "model" needs a pipeline with a light model; "auto" uses it when there is one.
    lighting_source: str = "auto"
    # Object types; None = what most human maps of that mode and difficulty use
    # (``density.object_flags``).
    bombs: bool | None = None
    walls: bool | None = None
    arcs: bool | None = None
    chains: bool | None = None

    def jump_for(self, difficulty: Difficulty) -> JumpOverride:
        """NJS / reaction time for ``difficulty``: ``jump``, then ``njs``, then
        ``jump_defaults`` (the user config); None fields fall back to ``njs.JUMP_TABLE``."""
        explicit = self.jump.get(difficulty, JumpOverride()).over(JumpOverride(self.njs))
        return explicit.over(self.jump_defaults.get(difficulty))


@dataclass(slots=True)
class Song:
    """An analysed song, everything later stages need."""

    audio_path: Path
    analysis: AudioAnalysis  # in the original audio's time
    song_analysis: AudioAnalysis  # song time: padded, beat_times[i] is map beat i
    bpm: float
    bpm_changes: list[BpmChange]  # in map beats
    beats_per_bar: int
    lead_in_s: float
    confidence: float  # share of tracked beats on the fitted grid
    title: str
    artist: str

    def empty_beatmap(self, mapper: str = "bsmapper") -> Beatmap:
        return Beatmap(
            song_name=self.title,
            song_author=self.artist,
            mapper=mapper,
            bpm=self.bpm,
            bpm_changes=list(self.bpm_changes),
            preview_start=self.preview_start(),
            preview_duration=PREVIEW_S,
        )

    def preview_start(self) -> float:
        """Start of the most energetic section, in song seconds."""
        a = self.song_analysis
        latest = max(a.duration_s - PREVIEW_S, 0.0)
        sections = [s for s in a.sections if s.start_s >= 5.0] or a.sections
        if not sections:
            return min(a.duration_s / 3, latest)
        best = max(sections, key=lambda s: s.energy)
        return round(min(best.start_s, latest), 3)


@dataclass(slots=True)
class DifficultyResult:
    dmap: DifficultyMap
    report: PostReport
    nps: float
    attempts: int = 1
    seconds: float = 0.0  # generation time (a batch's time is shared out evenly)

    def per_100(self, severity: str) -> float:
        """Rule issues of ``severity`` per 100 notes after post-processing."""
        n = sum(i.severity == severity for i in self.report.issues)
        return 100.0 * n / max(len(self.dmap.notes), 1)


@dataclass(slots=True)
class PipelineResult:
    beatmap: Beatmap
    song: Song
    difficulties: list[DifficultyResult]
    map_dir: Path
    zip_path: Path | None
    notes: list[str] = field(default_factory=list)


def pad_audio(src: Path, dst: Path, lead_in_s: float, ffmpeg: str = "ffmpeg") -> Path:
    """Prepend ``lead_in_s`` of silence to ``src``, written as WAV to ``dst``."""
    delay_ms = max(round(lead_in_s * 1000.0, 3), 0.0)
    cmd = [ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-i", str(src), "-vn"]
    cmd += ["-af", f"adelay=delays={delay_ms}:all=1", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found; install it or pass ffmpeg=<path>") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed: {exc.stderr.strip()}") from exc
    return dst


class Pipeline:
    """Analysis, note generation, post-processing, lighting and packaging.

    ``generator`` defaults to ``StubGenerator`` (not ML; see its module) and ``lighting``
    to WS8's ``RuleLightingGenerator``. Both are swappable, as are the analyzer and the
    rules checker. With ``light_model`` (a WS6 model with a lighting decoder) the lights
    come from the model (L1) unless the options ask for rules; any failure falls back to
    the L0 rules. ``runtime.make_pipeline`` picks all of this from a checkpoint.
    """

    def __init__(
        self,
        generator: NoteGenerator | None = None,
        *,
        analyzer=None,
        lighting: LightingGenerator | None = None,
        light_model=None,
        checker: RuleChecker | None = None,
        tracker: str = "auto",
        ffmpeg: str = "ffmpeg",
    ) -> None:
        self.checker = checker or RuleChecker()
        self.generator = generator or StubGenerator(checker=self.checker)
        self._analyzer = analyzer
        self.tracker = tracker
        self.lighting = lighting
        self.light_model = light_model
        self.ffmpeg = ffmpeg
        self.setup = None  # runtime.GeneratorSetup when built by make_pipeline
        self.last_lighting = ""  # "model" or "rules": what the last build used
        self.last_notes: list[str] = []  # remarks from the last build (fallbacks)

    @property
    def mapper(self) -> str:
        name = getattr(self.generator, "name", "")
        return "bsmapper" if name in ("", "model") else f"bsmapper ({name})"

    # --- stages ---------------------------------------------------------------------

    def analyze(self, audio: Path, title: str | None = None, artist: str | None = None) -> Song:
        from bsmapper.audio import Analyzer
        from bsmapper.formats import read_tags

        if self._analyzer is None:
            self._analyzer = Analyzer(tracker=self.tracker)
        result = self._analyzer.analyze_full(audio)
        grid = result.grid
        lead_in = grid.lead_in_s()
        shift = map_beat_shift(lead_in, grid.beats_per_bar)
        tags = read_tags(Path(audio))
        log.info(
            "%.2f BPM, offset %.3f s, lead-in %.3f s, grid coverage %.0f%%, %d sections",
            grid.bpm,
            grid.offset_s,
            lead_in,
            100 * result.fit.coverage,
            len(result.analysis.sections),
        )
        return Song(
            audio_path=Path(audio),
            analysis=result.analysis,
            song_analysis=song_time_analysis(result.analysis, lead_in, grid.beats_per_bar),
            bpm=grid.bpm,
            bpm_changes=[BpmChange(c.beat + shift, c.bpm) for c in grid.bpm_changes],
            beats_per_bar=grid.beats_per_bar,
            lead_in_s=lead_in,
            confidence=result.fit.coverage,
            title=title or tags.title,
            artist=artist if artist is not None else tags.artist,
        )

    def generate_difficulty(
        self,
        song: Song,
        beatmap: Beatmap,
        characteristic: Characteristic,
        difficulty: Difficulty,
        opts: GenerateOptions,
        *,
        seed: int | None = None,
        temperature: float | None = None,
        target_nps: float | None = None,
    ) -> DifficultyResult:
        """Generate and post-process one difficulty (not added to ``beatmap``)."""
        request = _request(characteristic, difficulty, opts, seed, temperature, target_nps)
        start = time.perf_counter()
        raw = self.generator.generate(song.song_analysis, request)
        res = self._finish(raw, beatmap, opts)
        res.seconds = time.perf_counter() - start
        return res

    def _finish(self, raw: DifficultyMap, beatmap: Beatmap, opts) -> DifficultyResult:
        jump = opts.jump_for(raw.difficulty)
        dmap, report = postprocess(raw, beatmap, self.checker, jump.njs, jump.reaction_s)
        return DifficultyResult(dmap, report, notes_per_second(dmap, beatmap))

    def _first_pass(
        self, song, beatmap, characteristic, diffs, opts, progress, overrides
    ) -> dict[Difficulty, DifficultyResult]:
        """Every difficulty once: one batch if the generator decodes batches."""
        many = getattr(self.generator, "generate_many", None)
        if many is None or len(diffs) < 2:
            out = {}
            for diff in diffs:
                if progress:
                    progress(diff)
                out[diff] = self.generate_difficulty(
                    song, beatmap, characteristic, diff, opts, **overrides
                )
            return out
        for diff in diffs:
            if progress:
                progress(diff)
        requests = [_request(characteristic, d, opts, **overrides) for d in diffs]
        start = time.perf_counter()
        raws = many(song.song_analysis, requests)
        share = (time.perf_counter() - start) / len(diffs)
        out = {}
        for diff, raw in zip(diffs, raws, strict=True):
            out[diff] = self._finish(raw, beatmap, opts)
            out[diff].seconds = share
        return out

    def generate_mode(
        self,
        song: Song,
        beatmap: Beatmap,
        characteristic: Characteristic,
        difficulties: Sequence[Difficulty],
        opts: GenerateOptions,
        progress: Callable[[Difficulty], None] | None = None,
        **overrides,
    ) -> list[DifficultyResult]:
        """All difficulties of one mode, hardest first, with strictly rising NPS.
        Returned easiest first."""
        diffs = sorted(set(difficulties), key=lambda d: d.rank, reverse=True)
        first = self._first_pass(song, beatmap, characteristic, diffs, opts, progress, overrides)
        results: list[DifficultyResult] = []
        harder: DifficultyResult | None = None
        for diff in diffs:
            res = first[diff]
            if harder is not None and harder.nps > 0 and res.nps >= MAX_NPS_RATIO * harder.nps:
                target = RETRY_NPS_RATIO * harder.nps
                log.info(
                    "%s %s came out at %.2f NPS (>= %s's %.2f); retrying at %.2f",
                    characteristic,
                    diff,
                    res.nps,
                    harder.dmap.difficulty,
                    harder.nps,
                    target,
                )
                spent = res.seconds
                res = self.generate_difficulty(
                    song, beatmap, characteristic, diff, opts, target_nps=target, **overrides
                )
                res.attempts, res.seconds = 2, res.seconds + spent
            results.append(res)
            harder = res
        return results[::-1]

    def light(self, song: Song, beatmap: Beatmap, opts: GenerateOptions) -> tuple[str, str | None]:
        """Light ``beatmap``. Returns the source used ("model"/"rules") and a fallback note."""
        from bsmapper.lighting import RuleLightingGenerator

        source = opts.lighting_source
        if source == "model" and self.light_model is None:
            raise ValueError("lighting_source='model' needs a pipeline with a light model")
        note = None
        if self.light_model is not None and source in ("auto", "model"):
            from bsmapper.generation.lighting import ModelLightingGenerator

            model_gen = ModelLightingGenerator(
                self.light_model, time_offset_s=song.lead_in_s, seed=opts.seed
            )
            try:
                self._light_with(model_gen, song, beatmap, opts)
                return "model", None
            except Exception as exc:  # any model failure: the L0 rules still light the map
                log.warning("model lighting failed (%s); using the rule lightshow", exc)
                note = f"Model lighting failed ({exc}); used the rule-based lightshow."
        rules = self.lighting or RuleLightingGenerator(time_offset_s=song.lead_in_s)
        self._light_with(rules, song, beatmap, opts)
        return "rules", note

    @staticmethod
    def _light_with(generator, song: Song, beatmap: Beatmap, opts: GenerateOptions) -> None:
        from bsmapper.lighting import light_beatmap

        light_beatmap(
            beatmap, song.analysis, opts.lighting, environment=opts.environment, generator=generator
        )

    def package(
        self,
        song: Song,
        beatmap: Beatmap,
        map_dir: Path,
        zip_path: Path | None = None,
        version: FormatVersion = "v3",
    ) -> Path:
        """Write the map folder (and zip). Returns the zip if one was asked for."""
        from bsmapper.formats import extract_cover, package_map

        _clear_map_files(map_dir)
        with tempfile.TemporaryDirectory(prefix="bsmapper-") as tmp:
            cover = extract_cover(song.audio_path, Path(tmp))
            audio = song.audio_path
            if song.lead_in_s > LEAD_IN_EPS_S:
                audio = pad_audio(audio, Path(tmp) / "padded.wav", song.lead_in_s, self.ffmpeg)
            return package_map(
                beatmap,
                audio,
                map_dir,
                version=version,
                cover=cover,
                zip_path=zip_path,
                ffmpeg=self.ffmpeg,
            )

    # --- everything -------------------------------------------------------------------

    def build(
        self, song: Song, opts: GenerateOptions, progress: ProgressFn = _no_progress
    ) -> tuple[Beatmap, list[DifficultyResult]]:
        """Generate every requested difficulty and light the map (no files written)."""
        beatmap = self.song_beatmap(song)
        pairs = len(opts.modes) * len(set(opts.difficulties))
        done = 0
        results: list[DifficultyResult] = []
        for mode in dict.fromkeys(opts.modes):

            def step(diff: Difficulty, mode: Characteristic = mode) -> None:
                nonlocal done
                progress("generate", 0.1 + 0.6 * done / max(pairs, 1), f"Generating {mode} {diff}")
                done += 1

            results += self.generate_mode(song, beatmap, mode, opts.difficulties, opts, step)
        beatmap.difficulties = [r.dmap for r in results]
        progress("rules", 0.72, _rules_message(results))
        progress("lighting", 0.75, f"Lighting ({opts.lighting})")
        self.last_lighting, note = self.light(song, beatmap, opts)
        self.last_notes = [note] if note else []
        return beatmap, results

    def song_beatmap(self, song: Song) -> Beatmap:
        return song.empty_beatmap(self.mapper)

    def run(
        self,
        audio: Path,
        out_root: Path,
        opts: GenerateOptions | None = None,
        *,
        zip_output: bool = True,
        progress: ProgressFn = _no_progress,
    ) -> PipelineResult:
        opts = opts or GenerateOptions()
        progress("analyze", 0.02, "Analysing audio")
        song = self.analyze(Path(audio))
        notes = song_notes(song)
        progress("analyze", 0.1, f"{song.bpm:g} BPM, {song.analysis.duration_s:.0f} s")
        beatmap, results = self.build(song, opts, progress)
        notes += self.last_notes
        notes += [n for r in results if (n := _difficulty_note(r))]

        progress("package", 0.85, "Packaging")
        name = safe_folder_name(song.title, song.artist)
        map_dir = Path(out_root) / name
        zip_path = Path(out_root) / f"{name}.zip" if zip_output else None
        self.package(song, beatmap, map_dir, zip_path, opts.version)
        progress("package", 1.0, "Done")
        return PipelineResult(beatmap, song, results, map_dir, zip_path, notes)


def _request(characteristic, difficulty, opts, seed=None, temperature=None, target_nps=None):
    flags = object_flags(difficulty, characteristic)
    for key, value in (
        ("allow_bombs", opts.bombs),
        ("allow_obstacles", opts.walls),
        ("allow_arcs", opts.arcs),
        ("allow_chains", opts.chains),
    ):
        if value is not None:
            flags[key] = value
    return GenerationRequest(
        characteristic=characteristic,
        difficulty=difficulty,
        target_nps=target_nps,
        style=opts.style,
        njs=opts.jump_for(difficulty).njs,
        seed=opts.seed if seed is None else seed,
        temperature=opts.temperature if temperature is None else temperature,
        **flags,
    )


def song_notes(song: Song) -> list[str]:
    notes = []
    if song.confidence < 0.7:
        notes.append(
            f"Beat grid confidence is low ({100 * song.confidence:.0f}% of beats fit "
            f"{song.bpm:g} BPM); notes may drift off the music."
        )
    return notes


def _difficulty_note(res: DifficultyResult) -> str | None:
    errors = res.report.errors
    if errors:
        return f"{res.dmap.characteristic} {res.dmap.difficulty}: {len(errors)} rule errors remain"
    return None


def _rules_message(results: list[DifficultyResult]) -> str:
    fixed = sum(sum(r.report.repaired.values()) + sum(r.report.dropped.values()) for r in results)
    warnings = sum(len(r.report.issues) for r in results)
    return f"Rules: {fixed} objects fixed or dropped, {warnings} warnings"


def _clear_map_files(map_dir: Path) -> None:
    """Remove files of an earlier run (difficulties of modes no longer requested)."""
    if not map_dir.is_dir():
        return
    for path in map_dir.iterdir():
        if path.is_file() and (path.suffix in (".dat", ".egg") or path.stem == "cover"):
            path.unlink()


def replace_difficulty(beatmap: Beatmap, dmap: DifficultyMap) -> Beatmap:
    """A copy of ``beatmap`` with ``dmap`` in place of the same mode and difficulty."""
    diffs = list(beatmap.difficulties)
    key = (dmap.characteristic, dmap.difficulty)
    for i, d in enumerate(diffs):
        if (d.characteristic, d.difficulty) == key:
            diffs[i] = dmap
            break
    else:
        diffs.append(dmap)
        diffs.sort(key=lambda d: (list(Characteristic).index(d.characteristic), d.difficulty.rank))
    return dataclasses.replace(beatmap, difficulties=diffs)
