"""A stand-in for the WS7 pipeline: returns a canned map shaped to the uploaded song.

The canned map is the 16-beat v3 block in ``mock_map/``, read with ``bsmapper.formats``. It
is tiled over the song's length, thinned per difficulty, adapted per mode (one saber, no
arrows, 90/360 rotations), varied by seed/style/temperature, and given a few fake rule
warnings. Nothing here is musical; it only exercises the server and the UI.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import random
import re
import shutil
import time
from pathlib import Path

from bsmapper.formats import (
    audio_duration,
    package_map,
    read_map,
    read_tags,
    write_map,
    write_placeholder_cover,
    zip_map,
)
from bsmapper.interfaces import Issue, LightingStyle
from bsmapper.schema import (
    BasicEvent,
    Beatmap,
    Characteristic,
    Color,
    ColorBoostEvent,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Lightshow,
    Note,
    RotationEvent,
)
from bsmapper.server.models import DifficultyWarning, RegenerateRequest
from bsmapper.server.runner import JobInput, JobResult, ProgressFn

log = logging.getLogger(__name__)

MOCK_MAP_DIR = Path(__file__).with_name("mock_map")
BLOCK_BEATS = 16
LEAD_IN_BEATS = 4
FALLBACK_DURATION_S = 60.0

NJS = {
    Difficulty.EASY: 10.0,
    Difficulty.NORMAL: 10.0,
    Difficulty.HARD: 12.0,
    Difficulty.EXPERT: 16.0,
    Difficulty.EXPERT_PLUS: 18.0,
}
MIRROR = {
    CutDirection.LEFT: CutDirection.RIGHT,
    CutDirection.RIGHT: CutDirection.LEFT,
    CutDirection.UP_LEFT: CutDirection.UP_RIGHT,
    CutDirection.UP_RIGHT: CutDirection.UP_LEFT,
    CutDirection.DOWN_LEFT: CutDirection.DOWN_RIGHT,
    CutDirection.DOWN_RIGHT: CutDirection.DOWN_LEFT,
}
# Outward diagonal of a straight swing, per hand (the "technical" variation).
DIAGONAL = {
    (Color.RED, CutDirection.DOWN): CutDirection.DOWN_LEFT,
    (Color.RED, CutDirection.UP): CutDirection.UP_LEFT,
    (Color.BLUE, CutDirection.DOWN): CutDirection.DOWN_RIGHT,
    (Color.BLUE, CutDirection.UP): CutDirection.UP_RIGHT,
}


class MockJobRunner:
    """``JobRunner`` returning a canned map. ``step_delay`` makes progress visible in the UI."""

    def __init__(self, step_delay: float = 0.4) -> None:
        self.step_delay = step_delay
        self.template = read_map(MOCK_MAP_DIR)
        block = self.template.difficulties[0]
        self.block = block
        self.block_lightshow = block.lightshow or self.template.lightshow

    def describe(self) -> tuple[str, str]:
        return "mock", "canned maps for UI development"

    # --- JobRunner -------------------------------------------------------------------

    def run(self, job: JobInput, progress: ProgressFn) -> JobResult:
        opts = job.options
        progress("analyze", 0.02, "Reading audio")
        duration, notes = self._duration(job.audio_path)
        tags = read_tags(job.audio_path)
        self._sleep()
        beatmap = dataclasses.replace(
            self.template,
            song_name=tags.title or job.audio_path.stem,
            song_author=tags.artist,
            mapper="bsmapper (mock)",
            preview_start=min(12.0, duration / 3),
            difficulties=[],
            lightshow=Lightshow(),
        )
        last_beat = math.floor(beatmap.seconds_to_beat(duration)) - 2
        progress("analyze", 0.1, f"{duration:.0f} s at {beatmap.bpm:g} BPM (mock)")

        pairs = [(m, d) for m in opts.modes for d in _sorted_diffs(opts.difficulties)]
        for i, (mode, diff) in enumerate(pairs):
            progress("generate", 0.1 + 0.6 * i / len(pairs), f"Generating {mode} {diff}")
            rng = random.Random(f"{opts.seed}:{mode}:{diff}")
            beatmap.difficulties.append(
                self._difficulty(mode, diff, last_beat, rng, opts.style, temperature=1.0)
            )
            self._sleep()
        progress("lighting", 0.7, f"Lighting ({opts.lighting})")
        beatmap.lightshow = self._lightshow(last_beat, opts.lighting)
        self._sleep()

        progress("rules", 0.8, "Checking playability")
        warnings = [w for dmap in beatmap.difficulties for w in _mock_warnings(dmap)]
        self._sleep()

        progress("package", 0.9, "Packaging zip")
        zip_path = self._package(beatmap, job, notes)
        return JobResult(beatmap, duration, zip_path, warnings, notes)

    def regenerate(
        self, job: JobInput, previous: JobResult, request: RegenerateRequest, progress: ProgressFn
    ) -> JobResult:
        key = (request.characteristic, request.difficulty)
        progress("generate", 0.1, f"Regenerating {key[0]} {key[1]} (seed {request.seed})")
        beatmap = dataclasses.replace(
            previous.beatmap, difficulties=list(previous.beatmap.difficulties)
        )
        last_beat = math.floor(beatmap.seconds_to_beat(previous.duration_s)) - 2
        rng = random.Random(f"{request.seed}:{key[0]}:{key[1]}:{request.temperature}")
        new = self._difficulty(
            *key, last_beat, rng, job.options.style, temperature=request.temperature
        )
        for i, dmap in enumerate(beatmap.difficulties):
            if (dmap.characteristic, dmap.difficulty) == key:
                beatmap.difficulties[i] = new
                break
        else:
            beatmap.difficulties.append(new)
        self._sleep()
        progress("rules", 0.7, "Checking playability")
        warnings = [
            w for w in previous.warnings if (w.characteristic, w.difficulty) != key
        ] + _mock_warnings(new)
        self._sleep()
        progress("package", 0.9, "Packaging zip")
        notes = [n for n in previous.notes if not n.startswith("Packaged without")]
        zip_path = self._package(beatmap, job, notes)
        return JobResult(beatmap, previous.duration_s, zip_path, warnings, notes)

    # --- pieces ---------------------------------------------------------------------

    def _sleep(self) -> None:
        if self.step_delay > 0:
            time.sleep(self.step_delay)

    def _duration(self, audio: Path) -> tuple[float, list[str]]:
        seconds = audio_duration(audio)
        if seconds and seconds > 5:
            return seconds, []
        return FALLBACK_DURATION_S, [
            f"Could not read the song length; the mock map assumes {FALLBACK_DURATION_S:.0f} s."
        ]

    def _difficulty(
        self,
        mode: Characteristic,
        diff: Difficulty,
        last_beat: float,
        rng: random.Random,
        style: float | None,
        temperature: float,
    ) -> DifficultyMap:
        dmap = DifficultyMap(mode, diff, njs=NJS[diff])
        p_mirror = min(1.0, 0.5 * temperature)
        p_diag = (0.15 if style is None else 0.6 * style) * temperature
        start = float(LEAD_IN_BEATS)
        while start + BLOCK_BEATS <= last_beat:
            mirror = rng.random() < p_mirror
            for n in self.block.notes:
                if not _keep_note(n, diff):
                    continue
                n = _mirror(n) if mirror else n
                if (n.color, n.direction) in DIAGONAL and rng.random() < p_diag:
                    n = dataclasses.replace(n, direction=DIAGONAL[(n.color, n.direction)])
                dmap.notes.append(dataclasses.replace(n, beat=n.beat + start))
            if diff not in (Difficulty.EASY, Difficulty.NORMAL):
                dmap.bombs += [
                    dataclasses.replace(b, beat=b.beat + start) for b in self.block.bombs
                ]
            dmap.obstacles += [
                dataclasses.replace(o, beat=o.beat + start, x=4 - o.x - o.width if mirror else o.x)
                for o in self.block.obstacles
            ]
            if diff in (Difficulty.EXPERT, Difficulty.EXPERT_PLUS):
                dmap.arcs += [
                    _mirror(a, beat=a.beat + start, tail_beat=a.tail_beat + start)
                    if mirror
                    else dataclasses.replace(a, beat=a.beat + start, tail_beat=a.tail_beat + start)
                    for a in self.block.arcs
                ]
            if diff == Difficulty.EXPERT_PLUS:
                dmap.chains += [
                    _mirror(c, beat=c.beat + start, tail_beat=c.tail_beat + start)
                    if mirror
                    else dataclasses.replace(c, beat=c.beat + start, tail_beat=c.tail_beat + start)
                    for c in self.block.chains
                ]
            start += BLOCK_BEATS
        return _adapt_to_mode(dmap, last_beat)

    def _lightshow(self, last_beat: float, style: LightingStyle) -> Lightshow:
        ls = Lightshow(uses_chroma=style == "chroma")
        start = 0.0
        block = 0
        while start + BLOCK_BEATS <= last_beat + BLOCK_BEATS:
            for e in self.block_lightshow.basic_events:
                if e.beat + start > last_beat + 2:
                    continue
                color = None
                if style == "chroma" and e.type <= 4 and e.value:
                    hue = ((e.beat + start) / 64.0) % 1.0
                    color = (*_hsv(hue), 1.0)
                ls.basic_events.append(dataclasses.replace(e, beat=e.beat + start, color=color))
            ls.boost_events += [
                ColorBoostEvent(b.beat + start, b.on)
                for b in self.block_lightshow.boost_events
                if block % 2 == 1
            ]
            start += BLOCK_BEATS
            block += 1
        ls.basic_events.append(BasicEvent(last_beat + 2, 0, 0))
        return ls

    def _package(self, beatmap: Beatmap, job: JobInput, notes: list[str]) -> Path:
        map_dir = job.work_dir / "map"
        shutil.rmtree(map_dir, ignore_errors=True)
        zip_path = job.work_dir / f"{_safe_name(beatmap.song_name)}.zip"
        zip_path.unlink(missing_ok=True)
        try:
            return package_map(beatmap, job.audio_path, map_dir, zip_path=zip_path)
        except RuntimeError as exc:  # ffmpeg missing or the upload is not audio
            log.warning("packaging %s without audio: %s", job.job_id, exc)
            notes.append(f"Packaged without song.egg: {exc}")
            shutil.rmtree(map_dir, ignore_errors=True)
            cover = write_placeholder_cover(map_dir / "cover.png", beatmap.song_name)
            write_map(dataclasses.replace(beatmap, cover_path=cover.name), map_dir)
            return zip_map(map_dir, zip_path)


def _sorted_diffs(diffs: list[Difficulty]) -> list[Difficulty]:
    return sorted(set(diffs), key=lambda d: d.rank)


def _keep_note(n: Note, diff: Difficulty) -> bool:
    on_beat = n.beat == int(n.beat)
    if diff == Difficulty.EASY:
        return on_beat and int(n.beat) % 2 == 0 and n.color == Color(int(n.beat) // 2 % 2)
    if diff == Difficulty.NORMAL:
        return on_beat and n.color == Color(int(n.beat) % 2)
    if diff == Difficulty.HARD:
        return on_beat
    return True


def _mirror(obj, **changes):
    """Mirror a note/arc/chain left-right, swapping colors."""
    fields = {"x": 3 - obj.x, "color": Color(1 - obj.color)}
    if hasattr(obj, "direction"):
        fields["direction"] = MIRROR.get(obj.direction, obj.direction)
    if hasattr(obj, "tail_x"):
        fields["tail_x"] = 3 - obj.tail_x
    if hasattr(obj, "tail_direction"):
        fields["tail_direction"] = MIRROR.get(obj.tail_direction, obj.tail_direction)
    if hasattr(obj, "angle_offset"):
        fields["angle_offset"] = -obj.angle_offset
    return dataclasses.replace(obj, **fields, **changes)


def _adapt_to_mode(dmap: DifficultyMap, last_beat: float) -> DifficultyMap:
    mode = dmap.characteristic
    if mode == Characteristic.ONE_SABER:
        dmap.notes = [n for n in dmap.notes if n.color == Color.BLUE]
        dmap.arcs = [a for a in dmap.arcs if a.color == Color.BLUE]
        dmap.chains = [c for c in dmap.chains if c.color == Color.BLUE]
    elif mode == Characteristic.NO_ARROWS:
        dmap.notes = [dataclasses.replace(n, direction=CutDirection.ANY) for n in dmap.notes]
        dmap.arcs, dmap.chains = [], []
    elif mode in (Characteristic.DEGREE_90, Characteristic.DEGREE_360):
        # 90: sway within +-45 degrees. 360: keep turning, reversing every 32 beats.
        angle = 0
        beat = float(LEAD_IN_BEATS + 2)
        while beat < last_beat:
            if mode == Characteristic.DEGREE_90:
                step = 15 if angle < 45 and (int(beat) // 8) % 2 == 0 else -15
                step = -15 if angle >= 45 else (15 if angle <= -45 else step)
            else:
                step = 15 if (int(beat) // 32) % 2 == 0 else -15
            angle += step
            dmap.rotations.append(RotationEvent(beat, step))
            beat += 2 if mode == Characteristic.DEGREE_90 else 4
    return dmap


def _mock_warnings(dmap: DifficultyMap) -> list[DifficultyWarning]:
    first = LEAD_IN_BEATS
    issues: list[Issue] = []
    if dmap.notes:
        issues.append(
            Issue(
                "vision_block",
                first + 6,
                "warning",
                "(mock) Notes at beat 6 may hide the next pair.",
            )
        )
    if dmap.difficulty in (Difficulty.EXPERT, Difficulty.EXPERT_PLUS) and len(dmap.notes) > 20:
        issues.append(
            Issue("nps_ceiling", first + 20, "warning", "(mock) Peak density above the target NPS.")
        )
    if dmap.difficulty == Difficulty.EXPERT_PLUS and len(dmap.notes) > 40:
        issues.append(
            Issue("parity", first + 30, "error", "(mock) Dot note followed by a reset swing.")
        )
    if dmap.characteristic == Characteristic.DEGREE_360 and dmap.rotations:
        issues.append(
            Issue("rotation", dmap.rotations[0].beat, "warning", "(mock) Fast turn after a wall.")
        )
    return [
        DifficultyWarning(characteristic=dmap.characteristic, difficulty=dmap.difficulty, issue=i)
        for i in issues
    ]


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\- ]+", "", name).strip()[:60] or "map"


def _hsv(h: float) -> tuple[float, float, float]:
    import colorsys

    return colorsys.hsv_to_rgb(h, 0.85, 1.0)
