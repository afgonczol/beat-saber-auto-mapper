"""Dev script: a playable map folder with the L0 lightshow and a trivial note track.

    python -m bsmapper.lighting.demo song.mp3 out/Song --style v3 --environment WeaveEnvironment

It analyses the song, pads it with ``BeatGrid.lead_in_s()`` of silence so the map sits
on an integer beat grid, lights it, adds one Expert Standard difficulty (alternating
red/blue notes every two beats, for orientation only) and packages it with
``bsmapper.formats``. Copy the folder into ``Beat Saber_Data/CustomLevels`` to watch it.
Needs ffmpeg on PATH and the ``audio`` extra.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

from bsmapper.interfaces import AudioAnalysis, LightingStyle
from bsmapper.lighting.generator import STYLES, RuleLightingGenerator, light_beatmap
from bsmapper.schema import (
    Beatmap,
    BpmChange,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
)

log = logging.getLogger(__name__)


def trivial_notes(first_beat: float, last_beat: float, step: float = 2.0) -> list[Note]:
    """Alternating red/blue notes every ``step`` beats; each hand alternates down/up."""
    notes = []
    beat, k = first_beat, 0
    while beat <= last_beat:
        color = Color.RED if k % 2 == 0 else Color.BLUE
        swing = (k // 2) % 2  # this hand's 0th, 1st, ... note: down, up, down, ...
        direction = CutDirection.DOWN if swing == 0 else CutDirection.UP
        x = 1 if color == Color.RED else 2
        notes.append(Note(beat, x, 0, color, direction))
        beat, k = beat + step, k + 1
    return notes


def demo_beatmap(
    analysis: AudioAnalysis,
    *,
    bpm_changes: list[BpmChange] | None = None,
    beats_per_bar: int = 4,
    lead_in_s: float = 0.0,
    style: LightingStyle = "v3",
    environment: str | None = None,
    song_name: str = "",
    song_author: str = "",
) -> Beatmap:
    """The lit demo beatmap for an analysis (no audio needed).

    Grid beat b plays at map beat b + ``beats_per_bar`` when the audio is padded by
    ``lead_in_s`` (> 0), and at b otherwise.
    """
    shift = beats_per_bar if lead_in_s > 1e-6 else 0
    beatmap = Beatmap(
        song_name=song_name,
        song_author=song_author,
        bpm=analysis.bpm,
        bpm_changes=[BpmChange(c.beat + shift, c.bpm) for c in bpm_changes or []],
    )
    generator = RuleLightingGenerator(time_offset_s=lead_in_s)
    light_beatmap(beatmap, analysis, style, environment=environment, generator=generator)

    first = float(shift)
    last = beatmap.seconds_to_beat(analysis.duration_s + lead_in_s) - beats_per_bar
    dmap = DifficultyMap(
        Characteristic.STANDARD,
        Difficulty.EXPERT,
        njs=16.0,
        notes=trivial_notes(first, last),
        label=f"Lighting demo ({style}, {beatmap.environment.removesuffix('Environment')})",
    )
    beatmap.difficulties.append(dmap)
    return beatmap


def pad_audio(src: Path, dst: Path, lead_in_s: float, ffmpeg: str = "ffmpeg") -> Path:
    """Prepend ``lead_in_s`` of silence to ``src`` (as WAV at ``dst``)."""
    delay_ms = max(int(round(lead_in_s * 1000)), 0)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src)]
    cmd += ["-af", f"adelay=delays={delay_ms}:all=1", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def build_demo_map(
    song: Path,
    out_dir: Path,
    *,
    style: LightingStyle = "v3",
    environment: str | None = None,
    zip_path: Path | None = None,
    tracker: str = "auto",
    ffmpeg: str = "ffmpeg",
) -> Path:
    from bsmapper.audio import Analyzer
    from bsmapper.formats import extract_cover, package_map, read_tags

    song, out_dir = Path(song), Path(out_dir)
    result = Analyzer(tracker=tracker).analyze_full(song)
    grid = result.grid
    lead_in = grid.lead_in_s()
    log.info(
        "%.2f BPM, offset %.3f s, lead-in %.3f s, coverage %.0f%%, %d sections",
        grid.bpm,
        grid.offset_s,
        lead_in,
        100 * result.fit.coverage,
        len(result.analysis.sections),
    )
    tags = read_tags(song)
    beatmap = demo_beatmap(
        result.analysis,
        bpm_changes=grid.bpm_changes,
        beats_per_bar=grid.beats_per_bar,
        lead_in_s=lead_in,
        style=style,
        environment=environment,
        song_name=tags.title,
        song_author=tags.artist,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cover = extract_cover(song, out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        audio = pad_audio(song, Path(tmp) / "padded.wav", lead_in, ffmpeg) if lead_in else song
        return package_map(beatmap, audio, out_dir, cover=cover, zip_path=zip_path, ffmpeg=ffmpeg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("song", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--style", choices=STYLES, default="v3")
    parser.add_argument("--environment", help="Info.dat environment name")
    parser.add_argument("--zip", type=Path, help="also write a zip here")
    parser.add_argument("--tracker", default="auto", help="beat tracker: auto, beat_this, librosa")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    path = build_demo_map(
        args.song,
        args.out_dir,
        style=args.style,
        environment=args.environment,
        zip_path=args.zip,
        tracker=args.tracker,
    )
    print(path)


if __name__ == "__main__":
    main()
