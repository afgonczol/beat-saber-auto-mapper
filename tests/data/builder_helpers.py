"""Tiny raw datasets for the builder tests: fixture maps zipped with synthetic audio."""

from __future__ import annotations

import io
import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np

from bsmapper.data.filters import BeatLeaderRating, DiffInfo, MapRecord
from bsmapper.data.manifest import MANIFEST_NAME, ManifestEntry
from bsmapper.formats import read_map_report, write_map
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

FIXTURES = Path(__file__).parents[1] / "fixtures" / "maps"
SR = 22050


def synth_audio(beatmap: Beatmap, seconds: float, seed: int = 0) -> np.ndarray:
    """Clicks on the map's beats over a seeded random tone melody (unique per seed)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    y = np.zeros(n, dtype=np.float64)
    step = int(0.25 * SR)
    for start in range(0, n, step):
        f = rng.uniform(200, 2000)
        seg = slice(start, min(start + step, n))
        y[seg] += 0.2 * np.sin(2 * np.pi * f * t[seg])
    click = rng.standard_normal(int(0.03 * SR)) * np.exp(-np.linspace(0, 8, int(0.03 * SR)))
    beat = 0.0
    while (s := beatmap.beat_to_seconds(beat) + beatmap.song_time_offset) < seconds - 0.05:
        i = int(s * SR)
        y[i : i + len(click)] += 0.8 * click[: n - i]
        beat += 1.0
    return (y / np.abs(y).max() * 0.9).astype(np.float32)


def wav_bytes(y: np.ndarray) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def zip_folder(folder: Path, zip_path: Path, audio: bytes | None, audio_name: str) -> None:
    with zipfile.ZipFile(zip_path, "w") as z:
        for p in folder.iterdir():
            if p.is_file() and p.name != audio_name:
                z.write(p, p.name)
        if audio is not None:
            z.writestr(audio_name, audio)


def generated_map(seed: int, bpm: float = 120.0, beats: int = 64) -> Beatmap:
    """A playable Standard map with enough notes for the default filters."""
    r = random.Random(seed)
    diffs = []
    for diff, step in ((Difficulty.HARD, 1.0), (Difficulty.EXPERT, 0.5)):
        notes = []
        beat = 4.0
        down = True
        while beat < beats:
            for color, x in ((Color.RED, r.choice([0, 1])), (Color.BLUE, r.choice([2, 3]))):
                d = CutDirection.DOWN if down else CutDirection.UP
                if r.random() < 0.2:
                    d = CutDirection.DOWN_LEFT if down else CutDirection.UP_RIGHT
                notes.append(Note(beat, x, 0 if down else 2, color, d))
            down = not down
            beat += step
        diffs.append(DifficultyMap(Characteristic.STANDARD, diff, notes=notes))
    return Beatmap(
        song_name=f"Generated {seed}",
        song_author=f"Artist {seed}",
        bpm=bpm,
        bpm_changes=[BpmChange(0.0, bpm)],
        difficulties=diffs,
    )


def manifest_line(key: str, beatmap: Beatmap, zip_rel: str, seconds: float) -> str:
    diffs = tuple(
        DiffInfo(
            characteristic=str(d.characteristic),
            difficulty=str(d.difficulty),
            notes=len(d.notes),
            beatleader=BeatLeaderRating(
                str(d.characteristic), str(d.difficulty), tech_rating=1.0 + 0.1 * i
            ),
        )
        for i, d in enumerate(beatmap.difficulties)
    )
    rec = MapRecord(
        key=key, hash=key * 4, name=beatmap.song_name, song_name=beatmap.song_name,
        song_sub_name="", song_author=beatmap.song_author, level_author="tester",
        uploader="tester", bpm=beatmap.bpm, duration=int(seconds),
        uploaded="2024-01-01T00:00:00Z", score=0.9, upvotes=100, downvotes=1,
        automapper=False, declared_ai="None", ranked_scoresaber=False,
        ranked_beatleader=False, curated=True, download_url="", diffs=diffs,
    )  # fmt: skip
    return ManifestEntry(rec, zip_rel, 0, True, ["beatsaver_curated"], "2024-01-01Z").to_json()


def make_raw(
    root: Path,
    fixtures: tuple[str, ...] = ("v2", "v3", "v4", "v2_modded"),
    generated: int = 0,
    seconds: float = 20.0,
) -> Path:
    """data/raw-like folder: zips/<key>.zip with audio, and manifest.jsonl."""
    raw = root / "raw"
    (raw / "zips").mkdir(parents=True)
    lines = []
    for i, name in enumerate(fixtures):
        beatmap, _ = read_map_report(FIXTURES / name)
        audio = wav_bytes(synth_audio(beatmap, seconds, seed=i))
        zip_folder(FIXTURES / name, raw / "zips" / f"{name}.zip", audio, beatmap.audio_path)
        lines.append(manifest_line(name, beatmap, f"zips/{name}.zip", seconds))
    for i in range(generated):
        key = f"gen{i}"
        beatmap = generated_map(i)
        folder = root / "src" / key
        write_map(beatmap, folder, "v3")
        audio = wav_bytes(synth_audio(beatmap, 36.0, seed=100 + i))
        zip_folder(folder, raw / "zips" / f"{key}.zip", audio, beatmap.audio_path)
        lines.append(manifest_line(key, beatmap, f"zips/{key}.zip", 36.0))
        shutil.rmtree(folder)
    (raw / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return raw


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
