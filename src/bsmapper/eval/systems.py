"""The systems an evaluation compares: human maps, the stub and model checkpoints.

Generated systems run the WS7 pipeline's own stages on the eval song (``Pipeline.
generate_mode``: hardest first with strictly rising NPS, then ``postprocess``; then
``Pipeline.light``), so a score describes what ``bsmapper generate`` would hand the
player on that grid. Only the difficulties the human map has are generated.

To keep GPU memory low next to a training run, a mode's difficulties are decoded
``batch_difficulties`` at a time (the model adds one density-boost copy per difficulty,
so 2 means batches of 4). Between chunks the same NPS rule as ``generate_mode`` applies.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from bsmapper.eval.data import EvalSong, GroupKey
from bsmapper.interfaces import AudioAnalysis
from bsmapper.schema import Characteristic, Difficulty, DifficultyMap, Lightshow

log = logging.getLogger(__name__)

HUMAN = "human"
STUB = "stub"


@dataclass(slots=True)
class SystemOutput:
    dmaps: dict[GroupKey, DifficultyMap]
    lightshow: Lightshow | None = None
    lighting_source: str = ""
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


class HumanSystem:
    name = HUMAN
    kind = "human"

    def run(self, song: EvalSong, analysis: AudioAnalysis, groups, style=None, lighting=True):
        dmaps = {g: song.humans[g] for g in groups if g in song.humans}
        return SystemOutput(dmaps, song.lightshow if lighting else None, "human")


def checkpoint_label(path: Path | str) -> str:
    """``checkpoints/note_v0/best.pt`` -> ``note_v0``; other files -> ``<dir>/<stem>``."""
    p = Path(path)
    if p.stem in ("best", "last"):
        return p.parent.name
    return f"{p.parent.name}/{p.stem}"


def intensity_supported() -> bool:
    from bsmapper.generation import GenerateOptions

    return "intensity" in {f.name for f in dataclasses.fields(GenerateOptions)}


class PipelineSystem:
    """A generator (stub or model) run through the WS7 pipeline stages."""

    kind = "generated"

    def __init__(
        self,
        pipeline,
        name: str,
        *,
        intensity: float | None = None,
        seed: int = 0,
        batch_difficulties: int = 2,
        checkpoint: Path | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.name = name
        self.intensity = intensity
        self.seed = seed
        self.batch_difficulties = batch_difficulties
        self.checkpoint = checkpoint

    def describe(self) -> dict:
        setup = getattr(self.pipeline, "setup", None)
        return {
            "name": self.name,
            "generator": getattr(setup, "generator", getattr(self.pipeline.generator, "name", "")),
            "lighting": getattr(setup, "lighting", "rules"),
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "intensity": self.intensity,
            "seed": self.seed,
        }

    def _song(self, song: EvalSong, analysis: AudioAnalysis):
        from bsmapper.generation import Song

        beatmap = song.beatmap()
        return Song(
            audio_path=Path(f"{song.key}.egg"),
            analysis=analysis,
            song_analysis=analysis,
            bpm=beatmap.bpm,
            bpm_changes=list(beatmap.bpm_changes),
            beats_per_bar=4,
            lead_in_s=0.0,
            confidence=1.0,
            title=song.song_name,
            artist=song.song_author,
        )

    def options(self, style: float | None):
        from bsmapper.generation import GenerateOptions

        opts = GenerateOptions(lighting="vanilla", style=style, seed=self.seed)
        if self.intensity is not None:
            if not intensity_supported():
                raise ValueError("this bsmapper build has no intensity option")
            opts = dataclasses.replace(opts, intensity=self.intensity)  # type: ignore[call-arg]
        return opts

    def run(self, song: EvalSong, analysis: AudioAnalysis, groups, style=None, lighting=True):
        start = time.perf_counter()
        gsong = self._song(song, analysis)
        opts = self.options(style)
        beatmap = self.pipeline.song_beatmap(gsong)
        modes: dict[Characteristic, list[Difficulty]] = {}
        for mode, diff in groups:
            modes.setdefault(mode, []).append(diff)
        dmaps: dict[GroupKey, DifficultyMap] = {}
        for mode, diffs in modes.items():
            for res in self._generate_mode(gsong, beatmap, mode, diffs, opts):
                dmaps[(mode, res.dmap.difficulty)] = res.dmap
        beatmap.difficulties = list(dmaps.values())
        out = SystemOutput(dmaps)
        if lighting:
            out.lighting_source, note = self.pipeline.light(gsong, beatmap, opts)
            out.lightshow = beatmap.lightshow
            if note:
                out.notes.append(note)
        out.seconds = time.perf_counter() - start
        return out

    def _generate_mode(self, song, beatmap, mode, diffs, opts):
        from bsmapper.generation.pipeline import MAX_NPS_RATIO, RETRY_NPS_RATIO

        p = self.pipeline
        diffs = sorted(set(diffs), key=lambda d: d.rank, reverse=True)
        size = self.batch_difficulties if self.batch_difficulties > 0 else len(diffs)
        results = []  # hardest first
        for i in range(0, len(diffs), size):
            chunk = p.generate_mode(song, beatmap, mode, diffs[i : i + size], opts)[::-1]
            if results and results[-1].nps > 0:
                harder, res = results[-1], chunk[0]
                if res.nps >= MAX_NPS_RATIO * harder.nps:
                    target = RETRY_NPS_RATIO * harder.nps
                    chunk[0] = p.generate_difficulty(
                        song, beatmap, mode, res.dmap.difficulty, opts, target_nps=target
                    )
            results += chunk
        return results


def model_system(
    checkpoint: Path,
    *,
    intensity: float | None = None,
    device: str | None = None,
    seed: int = 0,
    batch_difficulties: int = 2,
    lighting: bool = True,
) -> PipelineSystem:
    from bsmapper.generation import make_pipeline

    pipeline = make_pipeline("model", "model" if lighting else "rules", checkpoint, device)
    name = checkpoint_label(checkpoint)
    if intensity is not None:
        name += f"@i{intensity:g}"
    return PipelineSystem(
        pipeline,
        name,
        intensity=intensity,
        seed=seed,
        batch_difficulties=batch_difficulties,
        checkpoint=Path(checkpoint),
    )


def stub_system(seed: int = 0) -> PipelineSystem:
    from bsmapper.generation import make_pipeline

    return PipelineSystem(make_pipeline("stub", "rules"), STUB, seed=seed, batch_difficulties=0)


def gpu_free_mb() -> int | None:
    """Free memory on the first GPU according to nvidia-smi, or None without one."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def pick_device(device: str | None, min_free_mb: int = 4096) -> str:
    """``device`` if given; else CUDA when it has ``min_free_mb`` free, else the CPU."""
    if device:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    free = gpu_free_mb()
    if free is not None and free < min_free_mb:
        log.warning("only %d MB free on the GPU; evaluating on the CPU", free)
        return "cpu"
    return "cuda"
