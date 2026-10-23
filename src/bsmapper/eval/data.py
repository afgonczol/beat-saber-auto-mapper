"""Held-out songs for evaluation, read from a built dataset (``bsmapper data build``).

Every song is evaluated on **the human map's own beat grid** (its BPM, BPM changes,
offset and phase, as WS3b aligned it). ``song_analysis`` turns the dataset's
precomputed feature block into a song-time ``AudioAnalysis`` whose ``beat_times[i]`` is
map beat ``i``, so generators see exactly the features the model was trained on and a
generated note at beat ``b`` sits at the same point in the audio as a human note at
``b``. Beat tracking is left out on purpose: a v0-vs-v1 comparison then measures the
note model only. No audio is decoded.

``snapshot_split`` copies one split (features, tokens, lights, manifest, timing) into a
small standalone folder, so later dataset rebuilds cannot change a test set that
earlier runs were scored on.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bsmapper.audio.sections import segment_sections
from bsmapper.data.mapgrid import N_MELS, MapTiming
from bsmapper.interfaces import AudioAnalysis
from bsmapper.schema import Beatmap, Characteristic, Difficulty, DifficultyMap, Lightshow
from bsmapper.tokenizer import TICKS_PER_BEAT, LightingTokenizer, MapTokenizer

TIMING_FILE = "timing.json"  # {key: MapTiming.to_dict()} written by snapshot_split
FRAME_RATE_CHANGES = 100.0  # frame rate of the synthetic analysis for maps with BPM changes

GroupKey = tuple[Characteristic, Difficulty]


@dataclass(slots=True)
class EvalSong:
    """One held-out song: its feature block, timing and every human difficulty."""

    key: str
    song_name: str
    song_author: str
    mapper: str
    timing: MapTiming
    duration_s: float  # audio length
    features: np.ndarray  # (n_windows * 384, 83) float32, row j = map beat j / 24
    humans: dict[GroupKey, DifficultyMap]
    rows: dict[GroupKey, dict[str, Any]]  # manifest row per difficulty
    lightshow: Lightshow | None = None  # human basic events (v3 box groups are not stored)
    timing_exact: bool = True  # False: BPM changes were missing, constant BPM assumed
    notes: list[str] = field(default_factory=list)

    @property
    def bpm(self) -> float:
        return self.timing.bpm

    def beatmap(self) -> Beatmap:
        """Metadata-free tempo map (phase corrected): beat 0 is song time 0."""
        return self.timing.beatmap()

    @property
    def start_s(self) -> float:
        """Audio time of map beat 0."""
        return float(self.timing.seconds(0.0))

    @property
    def song_duration_s(self) -> float:
        """Length of the song in song time (from map beat 0 to the end of the audio)."""
        return self.duration_s - self.start_s


def _load_timings(root: Path, keys: set[str]) -> dict[str, dict]:
    path = root / TIMING_FILE
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    out = {}
    work = root / "work" / "maps"
    for key in keys:
        p = work / f"{key}.json"
        if p.is_file():
            out[key] = json.loads(p.read_text(encoding="utf-8"))["timing"]
    return out


def load_songs(
    root: Path | str,
    split: str = "test",
    *,
    keys: list[str] | None = None,
    limit: int | None = None,
    modes: list[Characteristic] | None = None,
    difficulties: list[Difficulty] | None = None,
) -> list[EvalSong]:
    """The songs of ``split`` in key order (``limit`` keeps the first N after filtering)."""
    import pyarrow.parquet as pq

    from bsmapper.data import ProcessedDataset

    ds = ProcessedDataset(root)
    root = Path(root)
    rows = ds.rows(split)
    if modes:
        rows = [r for r in rows if Characteristic(r["characteristic"]) in modes]
    if difficulties:
        rows = [r for r in rows if Difficulty(r["difficulty"]) in difficulties]
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(r["key"], []).append(r)
    wanted = sorted(by_key)
    if keys:
        wanted = [k for k in wanted if k in set(keys)]
    if limit:
        wanted = wanted[:limit]
    maps = {
        r["key"]: r for r in pq.read_table(root / "maps.parquet").to_pylist() if r["key"] in wanted
    }
    timings = _load_timings(root, set(wanted))
    tok = MapTokenizer()
    songs = []
    for key in wanted:
        m = maps[key]
        notes = []
        exact = True
        if key in timings:
            timing = MapTiming.from_dict(timings[key])
        else:
            timing = MapTiming(float(m["bpm"]), (), float(m["song_time_offset"]), float(m["phase"]))
            if m.get("bpm_changes"):
                exact = False
                notes.append("BPM changes missing from the dataset; constant BPM assumed")
        humans, mrows = {}, {}
        for r in by_key[key]:
            gk = (Characteristic(r["characteristic"]), Difficulty(r["difficulty"]))
            meta = DifficultyMap(gk[0], gk[1], njs=float(r["njs"]), njs_offset=r["njs_offset"])
            humans[gk] = tok.decode(ds.note_windows(r), meta=meta)
            mrows[gk] = r
        first = by_key[key][0]
        lights = ds.light_windows(first)
        lightshow = LightingTokenizer().decode(lights) if lights else None
        songs.append(
            EvalSong(
                key=key,
                song_name=m["song_name"] or "",
                song_author=m["song_author"] or "",
                mapper=m["mapper"] or "",
                timing=timing,
                duration_s=float(m["duration_s"]),
                features=np.asarray(ds.features(first), dtype=np.float32),
                humans=humans,
                rows=mrows,
                lightshow=lightshow if lightshow and lightshow.basic_events else None,
                timing_exact=exact,
                notes=notes,
            )
        )
    return songs


def song_analysis(song: EvalSong, sections: bool = True) -> AudioAnalysis:
    """A song-time ``AudioAnalysis`` on the human map's grid, from the feature block.

    Constant-tempo maps get one frame per tick (frame rate ``bpm * 24 / 60``), so the
    generators' resampling onto ticks gives back the dataset's features exactly. Maps
    with BPM changes are resampled to 100 frames/s along their tempo map.
    """
    beatmap = song.beatmap()
    feats = song.features
    n_rows = feats.shape[0]
    duration = max(song.song_duration_s, 1.0)
    duration = min(duration, beatmap.beat_to_seconds(n_rows / TICKS_PER_BEAT))
    if not beatmap.bpm_changes:
        frame_rate = beatmap.bpm * TICKS_PER_BEAT / 60.0
        frames = feats
    else:
        frame_rate = FRAME_RATE_CHANGES
        t = np.arange(int(duration * frame_rate) + 1) / frame_rate
        rows = np.array([beatmap.seconds_to_beat(float(s)) for s in t]) * TICKS_PER_BEAT
        rows = np.clip(rows, 0, n_rows - 1)
        i0 = np.minimum(rows.astype(np.int64), n_rows - 2)
        w = (rows - i0)[:, None]
        frames = feats[i0] * (1 - w) + feats[i0 + 1] * w
    frame_features = {
        "mel": np.ascontiguousarray(frames[:, :N_MELS]),
        "onset_strength": np.ascontiguousarray(frames[:, N_MELS]),
        "spectral_flux": np.ascontiguousarray(frames[:, N_MELS + 1]),
        "rms": np.ascontiguousarray(frames[:, N_MELS + 2]),
    }
    n_beats = int(beatmap.seconds_to_beat(duration)) + 1
    n_beats = max(2, min(n_beats, n_rows // TICKS_PER_BEAT))
    beat_times = np.array([beatmap.beat_to_seconds(float(b)) for b in range(n_beats)])
    beat_feats = {
        "mel": feats[:, :N_MELS],
        "onset_strength": feats[:, N_MELS],
        "spectral_flux": feats[:, N_MELS + 1],
        "rms": feats[:, N_MELS + 2],
    }
    secs = []
    if sections:
        secs = segment_sections(frame_features, frame_rate, beat_times, duration)
    return AudioAnalysis(
        duration_s=float(duration),
        bpm=float(beatmap.bpm),
        beat_times=beat_times,
        downbeat_times=beat_times[::4],
        sections=secs,
        features=frame_features,
        frame_rate=float(frame_rate),
        beat_features=beat_feats,
        ticks_per_beat=TICKS_PER_BEAT,
    )


def snapshot_split(src: Path | str, dst: Path | str, split: str = "test") -> dict[str, int]:
    """Copy ``split`` of the dataset at ``src`` into ``dst`` as a standalone dataset.

    It keeps ``dataset.json``, the three parquet tables (all rows; only ``split``'s shards
    are copied, so the other splits' rows stay as the NPS / tech-score reference) and a
    ``timing.json`` with every ``split`` map's exact tempo map from ``work/maps``.
    """
    import pyarrow.parquet as pq

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.json", "manifest.parquet", "maps.parquet", "difficulties.parquet"):
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    files = 0
    for kind, ext in (("features", "npy"), ("tokens", "parquet"), ("lights", "parquet")):
        (dst / kind).mkdir(exist_ok=True)
        for f in sorted((src / kind).glob(f"{split}-*.{ext}")):
            shutil.copy2(f, dst / kind / f.name)
            files += 1
    manifest = pq.read_table(src / "manifest.parquet").to_pylist()
    keys = {r["key"] for r in manifest if r["split"] == split}
    timings = {k: v for k, v in _load_timings(src, keys).items() if k in keys}
    (dst / TIMING_FILE).write_text(json.dumps(timings, indent=1), encoding="utf-8")
    return {"maps": len(keys), "shard_files": files, "timings": len(timings)}
