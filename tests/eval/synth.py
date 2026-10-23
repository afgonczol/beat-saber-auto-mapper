"""A tiny built dataset on disk for the eval tests: no audio, no model.

Each song is a 120 BPM click track (onset peaks on every beat, 1 tick late like the real
features, weaker ones on half beats) in four 16-beat sections A B A B. Section A always
has the same mel spectrum per beat, and so does B, so the second A and B repeat the
first. The human maps put notes on beats (Expert also on half beats) as a function of
the position within the section, so they repeat exactly where the music does. Lights
fire on every beat and change colour at each section.

``write_dataset`` writes the layout ``ProcessedDataset`` reads, plus ``train`` manifest
rows (no shards) for the human NPS reference: median NPS 1, 2, 3, 4, 5 for Easy..Expert+.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bsmapper.data.tech import FEATURES
from bsmapper.schema import (
    BasicEvent,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Lightshow,
    Note,
)
from bsmapper.tokenizer import LightingTokenizer, MapTokenizer

BPM = 120.0
TPB = 24
SECTION_BEATS = 16
N_WINDOWS = 4  # 64 beats, 32 s
N_BEATS = N_WINDOWS * 16
DIFFS = (Difficulty.HARD, Difficulty.EXPERT)
DOWNS = (CutDirection.DOWN, CutDirection.DOWN_LEFT, CutDirection.DOWN_RIGHT)


def click_features(seed: int = 0) -> np.ndarray:
    """(N_BEATS * 24, 83) float32: mel 0..79, onset, flux, rms."""
    rng = np.random.default_rng(seed)
    spectra = {s: rng.uniform(-60, -10, (SECTION_BEATS, 80)) for s in "AB"}
    n = N_BEATS * TPB
    feats = np.zeros((n, 83), dtype=np.float32)
    for b in range(N_BEATS):
        section = "AB"[(b // SECTION_BEATS) % 2]
        feats[b * TPB : (b + 1) * TPB, :80] = spectra[section][b % SECTION_BEATS]
    feats[1::TPB, 80] = 1.0  # onset peaks one tick after each beat
    feats[TPB // 2 + 1 :: TPB, 80] = 0.5
    feats[:, 81] = feats[:, 80]
    feats[:, 82] = 0.6
    return feats


def human_map(diff: Difficulty, mode: Characteristic = Characteristic.STANDARD) -> DifficultyMap:
    """Alternating hands and down/up swings; the same notes in every repeat of a section."""
    step = 0.5 if diff == Difficulty.EXPERT else 1.0
    notes = []
    for i in range(int((N_BEATS - 4) / step)):
        beat = 2.0 + i * step  # first note at beat 2
        pos = (beat % SECTION_BEATS) / step
        section = int(beat // SECTION_BEATS) % 2
        color = Color.RED if int(pos) % 2 == 0 else Color.BLUE
        down = int(pos) % 4 < 2
        x = (1 if color == Color.RED else 2) + (section if color == Color.BLUE else -section)
        d = CutDirection.DOWN if down else CutDirection.UP
        notes.append(Note(beat, x, 0 if down else 1, color, d))
    return DifficultyMap(mode, diff, njs=16.0, notes=notes)


def lightshow() -> Lightshow:
    events = []
    for b in range(N_BEATS):
        blue = (b // SECTION_BEATS) % 2
        events.append(BasicEvent(float(b), 0, 1 if blue else 5))
        if b % SECTION_BEATS == 0:
            events.append(BasicEvent(float(b), 8, 0))
    return Lightshow(basic_events=events)


def _table(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def write_dataset(root: Path, n_songs: int = 2) -> Path:
    """The dataset folder with ``n_songs`` test songs (keys 'a0', 'a1', ...)."""
    root.mkdir(parents=True, exist_ok=True)
    tok, ltok = MapTokenizer(), LightingTokenizer()
    manifest, maps, tokens, lights, feats, timing = [], [], [], [], [], {}
    for s in range(n_songs):
        key = f"a{s}"
        f = click_features(seed=s)
        row0 = sum(len(x) for x in feats)
        feats.append(f)
        lw = ltok.encode(lightshow(), n_windows=N_WINDOWS)
        lights.append(_bodies(key, lw))
        for diff in DIFFS:
            dmap = human_map(diff)
            windows = tok.encode(dmap, n_windows=N_WINDOWS)
            tokens.append(
                {**_bodies(key, windows), "characteristic": "Standard", "difficulty": diff.value}
            )
            n_notes = len(dmap.notes)
            nps = n_notes / ((dmap.notes[-1].beat - dmap.notes[0].beat) * 60 / BPM)
            manifest.append(
                {
                    "key": key,
                    "split": "test",
                    "characteristic": "Standard",
                    "difficulty": diff.value,
                    "njs": 16.0,
                    "njs_offset": 0.0,
                    "nps": nps,
                    "style": 0.5,
                    "tech_score": 0.0,
                    "flag_bombs": False,
                    "flag_walls": True,
                    "flag_arcs": False,
                    "flag_chains": False,
                    "n_windows": N_WINDOWS,
                    "feature_shard": "test-000",
                    "feature_row": row0,
                    "token_shard": "test-000",
                    "token_row": len(tokens) - 1,
                    "light_shard": "test-000",
                    "light_row": s,
                    "bpm": BPM,
                }
            )
        maps.append(
            {
                "key": key,
                "song_name": f"Click {s}",
                "song_author": "synth",
                "mapper": "test",
                "duration_s": N_BEATS * 60 / BPM,
                "bpm": BPM,
                "bpm_changes": 0,
                "song_time_offset": 0.0,
                "phase": 0.0,
            }
        )
        timing[key] = {"bpm": BPM, "bpm_changes": [], "song_time_offset": 0.0, "phase": 0.0}
    rng = np.random.default_rng(0)
    for level, diff in enumerate(Difficulty):
        for i in range(30):
            manifest.append(
                {
                    **manifest[0],
                    "key": f"t{level}{i}",
                    "split": "train",
                    "difficulty": diff.value,
                    "nps": level + 1.0 + rng.normal(0, 0.3),
                    "bpm": 110.0 + i,
                    "tech_score": float(rng.normal()),
                }
            )
    _table(manifest, root / "manifest.parquet")
    _table(maps, root / "maps.parquet")
    _table(tokens, root / "tokens" / "test-000.parquet")
    _table(lights, root / "lights" / "test-000.parquet")
    (root / "features").mkdir(exist_ok=True)
    np.save(root / "features" / "test-000.npy", np.concatenate(feats).astype(np.float16))
    (root / "timing.json").write_text(json.dumps(timing), encoding="utf-8")
    cal = {
        "features": list(FEATURES),
        "mean": [0.0] * len(FEATURES),
        "std": [1.0] * len(FEATURES),
        "weights": [1.0 / len(FEATURES)] * len(FEATURES),
        "intercept": 0.0,
    }
    meta = {"created": "test", "tech_calibration": {d.value: cal for d in Difficulty}}
    (root / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")
    return root


def _bodies(key: str, windows) -> dict:
    tokens, offsets = [], [0]
    for w in windows:
        tokens += list(w.body)
        offsets.append(len(tokens))
    return {"key": key, "tokens": tokens, "window_offsets": offsets}
