"""``StubGenerator``: a placeholder ``NoteGenerator``. **Not the ML model.**

It exists only so the pipeline, the web UI, the lighting and in-game loading can be
tested end to end before WS6 has a trained model. It is never offered as a mode.

What it does:

1. Candidate times on a per-difficulty grid (1, 2 or 4 steps per beat), scored by the
   beat-synchronous onset strength, the section energy and an on-beat bonus. Silent
   stretches (the lead-in bar, breaks) get no candidates.
2. Keeps the best-scoring candidates up to a target NPS that rises from Easy to Expert+.
3. Alternates hands, and each hand alternates forehand (down) and backhand (up) swings:
   naive parity. Every placed object is checked against the ``bsmapper.rules`` state
   (``MapState.candidates_mask``), so anything that would break a hard rule is never
   placed; the first allowed candidate by weight wins.
4. Lean phrases: every few bars both hands may drift one column left or right, onto the
   outer lanes, and on Normal and up a full-height side wall often closes the other side,
   so the player has to lean (most maps make you move; centred notes never do).
5. For 90/360 it inserts rotation events every few bars (90 sways within ±45°).

``style`` (0 flowy .. 1 technical) shifts weight from straight down/up cuts on the beat
towards diagonals, top-row and off-beat notes. ``seed`` and ``temperature`` vary choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bsmapper.generation.njs import njs_and_offset
from bsmapper.generation.timing import seconds_to_beats, tempo_beatmap
from bsmapper.interfaces import AudioAnalysis, GenerationRequest
from bsmapper.rules import MapState, RuleChecker
from bsmapper.schema import (
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
    Obstacle,
    RotationEvent,
)

STUB_LABEL = "Stub"  # difficulty labels start with it; the mapper name says "stub" too
SILENCE_RMS = 0.06  # beat-averaged RMS (song-relative) below this counts as silence
FIRST_NOTE_S = 1.5  # no notes before this (the player needs time to see the first ones)
END_MARGIN_S = 1.0
DEFAULT_STYLE = 0.3
ONE_SABER_DENSITY = 0.6  # one hand does all the work


@dataclass(frozen=True, slots=True)
class StubProfile:
    steps_per_beat: int  # candidate grid resolution
    nps: float  # default target notes per second over the mapped part of the song
    per_beat: float  # ... but at most this many notes per beat (slow songs stay calm)
    double_share: float  # chance that an eligible downbeat is hit with both hands
    rotation_bars: int  # bars between rotation events (90/360)
    phrase_bars: int = 2  # bars per lean phrase
    lean_share: float = 0.5  # chance that a phrase leans left or right
    wall_share: float = 0.5  # chance that a leaning phrase gets a side wall


PROFILES: dict[Difficulty, StubProfile] = {
    Difficulty.EASY: StubProfile(1, 1.0, 0.5, 0.0, 4, 4, 0.25, 0.0),
    Difficulty.NORMAL: StubProfile(1, 1.7, 0.9, 0.0, 4, 4, 0.4, 0.3),
    Difficulty.HARD: StubProfile(2, 2.6, 1.25, 0.05, 2, 2, 0.5, 0.45),
    Difficulty.EXPERT: StubProfile(2, 3.6, 1.75, 0.1, 2, 2, 0.6, 0.55),
    Difficulty.EXPERT_PLUS: StubProfile(4, 4.8, 2.5, 0.15, 1, 2, 0.7, 0.6),
}

_DOWN = {CutDirection.DOWN, CutDirection.DOWN_LEFT, CutDirection.DOWN_RIGHT}
_UP = {CutDirection.UP, CutDirection.UP_LEFT, CutDirection.UP_RIGHT}
# (straight, outward diagonal, inward diagonal) per hand and swing.
_DIRS = {
    (Color.RED, True): (CutDirection.DOWN, CutDirection.DOWN_LEFT, CutDirection.DOWN_RIGHT),
    (Color.RED, False): (CutDirection.UP, CutDirection.UP_LEFT, CutDirection.UP_RIGHT),
    (Color.BLUE, True): (CutDirection.DOWN, CutDirection.DOWN_RIGHT, CutDirection.DOWN_LEFT),
    (Color.BLUE, False): (CutDirection.UP, CutDirection.UP_RIGHT, CutDirection.UP_LEFT),
}
# Preferred columns per hand and lean (-1 = body to the left, +1 = to the right).
_COLUMNS = {
    (Color.RED, 0): (1, 0),
    (Color.BLUE, 0): (2, 3),
    (Color.RED, -1): (0, 1),
    (Color.BLUE, -1): (1, 2),
    (Color.RED, 1): (2, 1),
    (Color.BLUE, 1): (3, 2),
}
WALL_HEIGHT = 5  # full height: it has to be leaned around, not ducked


def _feature(analysis: AudioAnalysis, name: str, n_ticks: int) -> np.ndarray:
    """A beat-synchronous feature (1-D) at map ticks, resampled from frames if needed."""
    arr = analysis.beat_features.get(name)
    if arr is None:
        frames = analysis.features.get(name)
        if frames is None or len(frames) < 2 or len(analysis.beat_times) < 2:
            return np.zeros(n_ticks, dtype=np.float32)
        tpb = analysis.ticks_per_beat
        beats = np.arange(n_ticks) / tpb
        times = np.interp(beats, np.arange(len(analysis.beat_times)), analysis.beat_times)
        idx = np.clip(np.round(times * analysis.frame_rate).astype(int), 0, len(frames) - 1)
        arr = frames[idx]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if len(arr) < n_ticks:
        arr = np.concatenate([arr, np.zeros(n_ticks - len(arr), dtype=np.float32)])
    return arr[:n_ticks]


def stub_label(difficulty: Difficulty) -> str:
    """``"Stub Expert+"``: what editors and SongCore show instead of the difficulty name."""
    return f"{STUB_LABEL} {difficulty.value.replace('Plus', '+')}"


def _beats_per_bar(analysis: AudioAnalysis) -> int:
    if len(analysis.downbeat_times) < 2 or len(analysis.beat_times) < 2:
        return 4
    ratio = np.median(np.diff(analysis.downbeat_times)) / np.median(np.diff(analysis.beat_times))
    return max(1, round(float(ratio)))


class StubGenerator:
    """Interface-only ``NoteGenerator``: beats and onsets with naive alternating parity.

    Clearly a stub: difficulty labels read ``"Stub <difficulty>"``. Expects a *song-time*
    analysis (``generation.timing.song_time_analysis``): ``beat_times[i]`` is map beat i.
    """

    name = "stub"

    def __init__(
        self,
        checker: RuleChecker | None = None,
        profiles: dict[Difficulty, StubProfile] | None = None,
    ) -> None:
        self.checker = checker or RuleChecker()
        self.profiles = profiles or PROFILES

    def generate(self, analysis: AudioAnalysis, request: GenerationRequest) -> DifficultyMap:
        beatmap = tempo_beatmap(analysis)
        njs, offset = njs_and_offset(request.difficulty, beatmap.bpm, request.njs)
        dmap = DifficultyMap(request.characteristic, request.difficulty, njs=njs, njs_offset=offset)
        dmap.label = stub_label(request.difficulty)
        if request.characteristic == Characteristic.LIGHTSHOW:
            return dmap

        profile = self.profiles[request.difficulty]
        mode_index = list(Characteristic).index(request.characteristic)
        rng = np.random.default_rng([request.seed, mode_index, request.difficulty.rank])
        temperature = max(request.temperature, 0.05)
        tech = DEFAULT_STYLE if request.style is None else float(np.clip(request.style, 0, 1))
        target_nps = request.target_nps or self.default_nps(
            request.difficulty, request.characteristic, beatmap.bpm
        )

        times = self._select_times(analysis, profile, target_nps, tech, temperature, rng)
        state = self.checker.state(dmap, beatmap)
        walls = request.allow_obstacles
        self._place(dmap, state, analysis, times, profile, tech, temperature, rng, walls)
        return dmap

    def default_nps(self, difficulty: Difficulty, mode: Characteristic, bpm: float) -> float:
        """Target NPS when the request has none: the profile's, capped per beat."""
        profile = self.profiles[difficulty]
        nps = min(profile.nps, profile.per_beat * bpm / 60.0)
        return nps * ONE_SABER_DENSITY if mode == Characteristic.ONE_SABER else nps

    # --- rhythm -------------------------------------------------------------------

    def _select_times(
        self,
        analysis: AudioAnalysis,
        profile: StubProfile,
        target_nps: float,
        tech: float,
        temperature: float,
        rng: np.random.Generator,
    ) -> list[tuple[float, float]]:
        """(beat, score) of the chosen note times, in time order."""
        tpb = analysis.ticks_per_beat
        n_beats = len(analysis.beat_times)
        if n_beats < 2:
            return []
        first = math.ceil(float(seconds_to_beats(analysis, FIRST_NOTE_S)))
        last = float(seconds_to_beats(analysis, analysis.duration_s - END_MARGIN_S))
        last = min(last, n_beats - 1.0)
        if last <= first:
            return []

        n_ticks = n_beats * tpb
        onset = _feature(analysis, "onset_strength", n_ticks)
        rms = _feature(analysis, "rms", n_ticks)
        beat_rms = np.convolve(rms, np.ones(tpb) / tpb, mode="same")
        padded = np.concatenate([onset[:1], onset, onset[-1:], onset[-1:]])
        peak = np.maximum.reduce([padded[k : k + n_ticks] for k in range(4)])  # ticks -1..+2
        bar = _beats_per_bar(analysis)

        step = tpb // profile.steps_per_beat
        ticks = np.arange(first * tpb, int(last * tpb) + 1, step)
        ticks = ticks[beat_rms[ticks] >= SILENCE_RMS]
        if len(ticks) == 0:
            return []
        beats = ticks / tpb
        energy = self._energy(analysis, beats)
        on_beat = ticks % tpb == 0
        downbeat = on_beat & ((ticks // tpb) % bar == 0)
        half = (ticks % (tpb // 2) == 0) & ~on_beat
        score = peak[ticks] * (0.6 + 0.8 * energy)
        score += on_beat * 0.35 * (1.0 - 0.6 * tech) + downbeat * 0.15 + half * 0.1 * (1 - tech)
        score += rng.normal(0.0, 0.05 * temperature, len(score))

        span_s = float(np.interp(beats[-1], np.arange(n_beats), analysis.beat_times)) - float(
            np.interp(beats[0], np.arange(n_beats), analysis.beat_times)
        )
        n_keep = min(len(ticks), max(1, round(target_nps * max(span_s, 1.0))))
        keep = np.sort(np.argsort(-score, kind="stable")[:n_keep])
        return [(float(beats[i]), float(score[i])) for i in keep]

    @staticmethod
    def _energy(analysis: AudioAnalysis, beats: np.ndarray) -> np.ndarray:
        if not analysis.sections:
            return np.full(len(beats), 0.5)
        idx = np.arange(len(analysis.beat_times))
        t = np.interp(beats, idx, analysis.beat_times)
        out = np.full(len(beats), 0.5)
        for s in analysis.sections:
            out[(t >= s.start_s) & (t < s.end_s)] = float(s.energy)
        return out

    # --- placement ------------------------------------------------------------------

    def _place(
        self,
        dmap: DifficultyMap,
        state: MapState,
        analysis: AudioAnalysis,
        times: list[tuple[float, float]],
        profile: StubProfile,
        tech: float,
        temperature: float,
        rng: np.random.Generator,
        walls: bool = True,
    ) -> None:
        mode = dmap.characteristic
        one_hand = mode == Characteristic.ONE_SABER
        dots = mode == Characteristic.NO_ARROWS
        rotates = mode in (Characteristic.DEGREE_90, Characteristic.DEGREE_360)
        bar = _beats_per_bar(analysis)

        down = {Color.RED: True, Color.BLUE: True}
        last: dict[Color, Note] = {}
        hand = Color.BLUE  # the first single note is red
        next_rotation = times[0][0] + bar if times else 0.0
        angle, turn = 0, 1
        scores = np.array([s for _, s in times]) if times else np.zeros(0)
        strong = float(np.quantile(scores, 0.8)) if len(scores) else 0.0
        prev_beat = -math.inf
        phrase = profile.phrase_bars * bar
        phrase_end, lean, last_lean = -math.inf, 0, 1

        for beat, score in times:
            if beat >= phrase_end - 1e-6:
                start = beat - (beat - times[0][0]) % phrase
                phrase_end = start + phrase
                lean = 0
                if rng.random() < profile.lean_share:
                    # Usually swap sides, sometimes stay: the player moves back and forth.
                    lean = -last_lean if rng.random() < 0.75 else last_lean
                    last_lean = lean
                    if walls and rng.random() < profile.wall_share:
                        self._side_wall(dmap, state, beat, phrase_end - beat - 1.0, lean)
            if rotates and beat >= next_rotation - 1e-6:
                angle, turn = self._rotate(dmap, state, beat, score >= strong, angle, turn, rng)
                next_rotation = beat + profile.rotation_bars * bar

            double = (
                not one_hand
                and down[Color.RED] == down[Color.BLUE]
                and abs(beat - round(beat)) < 1e-6
                and round(beat) % bar == 0
                and beat - prev_beat >= 1.0 - 1e-6
                and score >= strong
                and rng.random() < profile.double_share
            )
            if one_hand:
                order = [Color.BLUE]
            elif double:
                order = [Color.RED, Color.BLUE]
            else:
                other = Color.RED if hand == Color.BLUE else Color.BLUE
                order = [other, hand]  # the other hand first, this one as a fallback
            placed = False
            for color in order:
                prev = last.get(color)
                cols = _COLUMNS[(color, 0 if one_hand else lean)]
                swing = (down[color], prev, double, dots)
                note = self._pick(state, beat, color, *swing, tech, temperature, rng, cols)
                if note is None:
                    continue
                state.add(note)
                dmap.notes.append(note)
                down[color] = not down[color]
                last[color] = note
                hand, placed = color, True
                if not double:
                    break
            if placed:
                prev_beat = beat

    def _pick(
        self,
        state: MapState,
        beat: float,
        color: Color,
        down: bool,
        last: Note | None,
        double: bool,
        dots: bool,
        tech: float,
        temperature: float,
        rng: np.random.Generator,
        columns: tuple[int, int],
    ) -> Note | None:
        straight, outward, inward = _DIRS[(color, down)]
        if dots:
            dirs = [(CutDirection.ANY, 1.0)]
        else:
            dirs = [(straight, 1.0 - 0.6 * tech), (outward, 0.25 + 0.5 * tech)]
            if not double:
                dirs.append((inward, 0.05 + 0.3 * tech))
        first, second = columns
        cols = [(first, 1.0), (second, 0.7 + 0.3 * tech)] if not double else [(first, 1.0)]
        rows = [(0, 1.0), (2, 0.1 + 0.4 * tech)] if down else [(0, 1.0), (1, 0.3 + 0.4 * tech)]
        if double:
            rows = rows[:1]

        candidates: list[Note] = []
        weights: list[float] = []
        for x, wx in cols:
            for y, wy in rows:
                if y == 1 and x in (1, 2):
                    continue  # centre middle cells block the view of what follows
                for d, wd in dirs:
                    w = wx * wy * wd
                    if last is not None and last.x == x:
                        w *= 1.5  # keep the hand where it is: smoother, slower swings
                    candidates.append(Note(beat, x, y, color, d))
                    weights.append(w)
        allowed = state.candidates_mask(candidates)
        if not allowed.any():
            return self._fallback(state, beat, color, down, dots, columns)
        w = np.array(weights)[allowed] ** (1.0 / temperature)
        options = [c for c, ok in zip(candidates, allowed, strict=True) if ok]
        return options[int(rng.choice(len(options), p=w / w.sum()))]

    @staticmethod
    def _fallback(
        state: MapState, beat: float, color: Color, down: bool, dots: bool, columns
    ) -> Note | None:
        """Any allowed note on the hand's side with the wanted swing, from the full mask."""
        mask = state.note_mask(beat, color)
        wanted = [CutDirection.ANY] if dots else sorted(_DOWN if down else _UP)
        for x in dict.fromkeys([*columns, *_COLUMNS[(color, 0)]]):
            for y in (0, 2, 1):
                for d in wanted:
                    if mask[x, y, d] and not (y == 1 and x in (1, 2)):
                        return Note(beat, x, y, color, d)
        return None

    @staticmethod
    def _side_wall(
        dmap: DifficultyMap, state: MapState, beat: float, duration: float, lean: int
    ) -> None:
        """A full-height wall over the two columns the player leans away from."""
        if duration < 1.0 or beat < state.last_beat - 1e-6:
            return
        x = 2 if lean < 0 else 0
        wall = Obstacle(beat, duration, x, 0, 2, WALL_HEIGHT)
        if state.allowed(wall):
            state.add(wall)
            dmap.obstacles.append(wall)

    @staticmethod
    def _rotate(
        dmap: DifficultyMap,
        state: MapState,
        beat: float,
        strong: bool,
        angle: int,
        turn: int,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """Add one rotation event; returns the new (total angle, turning direction)."""
        if dmap.characteristic == Characteristic.DEGREE_90:
            limit = 45
            if abs(angle + 15 * turn) > limit or rng.random() < 0.2:
                turn = -turn
            sizes = [15]
        else:
            if rng.random() < 0.15:
                turn = -turn
            sizes = [30, 15] if strong else [15]
        for size in sizes:
            step = size * turn
            if dmap.characteristic == Characteristic.DEGREE_90 and abs(angle + step) > 45:
                continue
            event = RotationEvent(beat, step)
            if state.allowed(event):
                state.add(event)
                dmap.rotations.append(event)
                return angle + step, turn
        return angle, turn
