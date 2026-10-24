"""``ModelGenerator``: notes from the WS6 note model, decoded under the WS5 rules.

    generator = ModelGenerator(load_model("checkpoints/note_v0/best.pt", device="cuda"))
    dmaps = generator.generate_many(song_analysis, requests)   # one batch per mode

Decoding is WS6's ``NoteSampler``: 16-beat windows, the previous window's tail as context,
grammar masks, temperature / top-p / seed and classifier-free guidance on STYLE. This
module adds what generation needs on top:

- **conditioning**: ``Conditioning.from_request``. ``target_nps`` is the NPS wanted in
  the map; when it is None it comes from ``intensity.plan`` (the human NPS percentile for
  the request's intensity). The model is conditioned on it times a per-model gain
  (``density.gains_for``): v0 undershoots its conditioning, v1 follows it closely. The
  intensity goes into note_v1's INTENSITY token. Above 0.75 a label is also conditioned
  on the next difficulty's DIFF token, with its rules ceilings relaxed to match
  (``intensity``, ``ModelSettings.intensity_shift``); note_v0 has only the shift.
- **density feedback**: every difficulty is decoded next to a copy conditioned on a
  higher NPS (``ModelSettings.density_boosts``). After each window the candidate that best
  fits the window's expected note count (wanted NPS x the music's loudness there) and
  leaves no long gaps over loud music (``density.GAP_LIMIT_BEATS``) is kept, and every
  copy continues from it: its tokens become the next window's context and its rules state
  is copied, so a resampled window is seamless.
- **request mask**: object types the request forbids (bombs, walls, arcs, chains),
  red notes in One Saber, arrows in No Arrows, rotations outside 90/360.
- **rules mask** (``mask_fn``): one ``bsmapper.rules.MapState`` per sequence, fed every
  object as soon as the parser completes it. Before each token, note/bomb heads, cells
  and directions that would break a hard rule at the current tick are masked out.
- **features**: the song-time analysis resampled onto the map's tempo map with WS3b's
  ``map_grid_features``, exactly the layout the model was trained on.

The torch-dependent parts import lazily, so the stub path never needs the ``ml`` extra.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from bsmapper.generation.density import (
    GAP_LIMIT_BEATS,
    Loudness,
    conditioning_nps,
    gains_for,
    loud_gaps,
)
from bsmapper.generation.intensity import SHIFT_ABOVE, checker_for, conditioning_cap
from bsmapper.generation.intensity import plan as intensity_plan
from bsmapper.generation.njs import njs_and_offset
from bsmapper.generation.timing import tempo_beatmap
from bsmapper.interfaces import AudioAnalysis, GenerationRequest
from bsmapper.rules import MapState, RuleChecker
from bsmapper.schema import (
    GRID_COLUMNS,
    GRID_ROWS,
    Beatmap,
    Bomb,
    Characteristic,
    Color,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Note,
    RotationEvent,
)
from bsmapper.tokenizer import (
    NOTE_GRAMMAR,
    NOTE_VOCAB,
    TICKS_PER_BEAT,
    WINDOW_TICKS,
    Conditioning,
    MapTokenizer,
    Parser,
    events_to_map,
)

if TYPE_CHECKING:
    from bsmapper.models import MapperModel, StepState

ROTATING = (Characteristic.DEGREE_90, Characteristic.DEGREE_360)


@dataclass(frozen=True, slots=True)
class ModelSettings:
    top_p: float = 0.95
    style_guidance: float = 1.5  # classifier-free guidance on STYLE (1 = off)
    rule_masks: bool = True
    context_events: int = 16
    max_body_tokens: int = 1024
    # Extra candidates per difficulty, as multiples of its conditioning NPS; () = off.
    density_boosts: tuple[float, ...] = (1.6,)
    # A boosted window must beat the base one's score by this much to be kept.
    boost_margin: float = 0.05
    # Above intensity 0.75, also condition a label on the next difficulty's DIFF token (#32).
    # Re-checked for note_v1, which has an INTENSITY token: at 0.8, the token alone left
    # Expert at 0.82-1.33x the human median on three songs, the token plus the shift at
    # 0.95-1.33x, with no rule errors either way. So it stays on for both models.
    # None = on for every model.
    intensity_shift: bool | None = None
    # Structure memory (note_v1): reference windows from the song's self-similarity.
    references: bool = True
    # Conditioning NPS = wanted x gain per difficulty; None = calibrated per model version
    # (``density.gains_for``).
    nps_gain: dict | None = None
    # Logit offset on the ARC head token where arcs are allowed (#12): about e**bias times
    # the odds of an arc, and more in the end, since arcs in the context beget arcs. On
    # Stronger, Toxic and a trap song (Hard to Expert+), 0 gave a median of 10 arcs a minute
    # (0 to 18), 0.3 gave 20 (1 to 46) and 0.7 gave 53 (10 to 96). Human maps that use arcs
    # have a median of 13 to 18 and a p75 of 23 to 29. 0 = the model's own rate.
    arc_bias: float = 0.3


def song_features(
    analysis: AudioAnalysis, beatmap: Beatmap, time_offset_s: float = 0.0
) -> np.ndarray:
    """(n_windows * 384, 83) model input: ``analysis`` frames on ``beatmap``'s tempo map.

    ``time_offset_s`` is silence prepended to ``analysis``'s audio in the packaged song
    (0 for a song-time analysis, ``lead_in_s`` for the original one).
    """
    from bsmapper.data.mapgrid import MapTiming, audio_windows, map_grid_features

    timing = MapTiming(beatmap.bpm, tuple(beatmap.bpm_changes), -time_offset_s)
    n_windows = audio_windows(timing, analysis.duration_s)
    feats = map_grid_features(
        analysis.features, analysis.frame_rate, timing, analysis.duration_s, n_windows
    )
    return feats.astype(np.float32)


class _Vocab:
    """Token ids used by the request and rules masks."""

    def __init__(self) -> None:
        v = NOTE_VOCAB
        self.size = len(v)
        self.note_heads = {c: v.id("NOTE", c) for c in Color}
        self.pos = {v.value(i): i for i in v.ids("POS")}
        self.dirs = {v.value(i): i for i in v.ids("DIR")}
        self.rot = {v.value(i): i for i in v.ids("ROT")}
        self.bomb = v.id("BOMB")
        self.wall = v.id("WALL")
        self.arc = v.id("ARC")
        self.chain = v.id("CHAIN")


_V = _Vocab()


def request_mask(request: GenerationRequest) -> np.ndarray:
    """Tokens the request rules out entirely (object types, colours, rotations)."""
    m = np.ones(_V.size, dtype=bool)
    mode = request.characteristic
    if mode == Characteristic.LIGHTSHOW:
        m[[*_V.note_heads.values(), _V.bomb, _V.wall, _V.arc, _V.chain]] = False
    for allowed, token in (
        (request.allow_bombs, _V.bomb),
        (request.allow_obstacles, _V.wall),
        (request.allow_arcs, _V.arc),
        (request.allow_chains, _V.chain),
    ):
        if not allowed:
            m[token] = False
    if mode == Characteristic.ONE_SABER:
        m[_V.note_heads[Color.RED]] = False
    if mode == Characteristic.NO_ARROWS:
        m[_V.arc] = m[_V.chain] = False
    if mode not in ROTATING:
        m[list(_V.rot.values())] = False
    return m


class RuleMasker:
    """``mask_fn`` / ``on_window`` for ``NoteSampler``: request and rules masks per sequence."""

    def __init__(
        self,
        requests: Sequence[GenerationRequest],
        metas: Sequence[DifficultyMap],
        beatmap: Beatmap,
        checker: RuleChecker | Sequence[RuleChecker],
        rules: bool = True,
    ) -> None:
        self.requests = list(requests)
        self.static = [request_mask(r) for r in requests]
        checkers = list(checker) if isinstance(checker, Sequence) else [checker] * len(metas)
        self.states = [c.state(m, beatmap) for c, m in zip(checkers, metas, strict=True)]
        self.rules = rules
        self._seen: list[tuple[int, int]] = [(0, 0)] * len(requests)  # (window, events fed)
        self._cache: dict[tuple, bool] = {}
        self.masked_steps = 0

    # --- NoteSampler callbacks ------------------------------------------------------

    def mask(self, step: StepState) -> np.ndarray:
        b = step.sequence
        self._sync(b, step.window, step.parser.events)
        mask = self.static[b].copy()
        if self.rules:
            ruled = mask & self._rules(b, step.parser, step.window * WINDOW_TICKS)
            if ruled.any() and not np.array_equal(ruled, mask):
                self.masked_steps += 1
                mask = ruled
        return mask

    def on_window(self, k: int, bodies: list[list[int]]) -> None:
        """Feed what each window's last token completed (a note waiting for ANGLE)."""
        for b, body in enumerate(bodies):
            self._sync(b, k, NOTE_GRAMMAR.parse(body))
            self._seen[b] = (k + 1, 0)

    def adopt(self, row: int, source: int) -> None:
        """Make ``row`` continue from ``source``'s state (after a window was chosen)."""
        self.states[row] = self.states[source].clone()
        self._seen[row] = self._seen[source]
        self._cache.clear()

    # --- internals ------------------------------------------------------------------

    def _sync(self, b: int, window: int, events) -> None:
        seen_window, fed = self._seen[b]
        if seen_window != window:
            fed = 0
        state, req = self.states[b], self.requests[b]
        for event in events[fed:]:
            dmap = events_to_map(
                [(window * WINDOW_TICKS + event.tick, event)], req.characteristic, req.difficulty
            )
            for group in (dmap.rotations, dmap.obstacles, dmap.bombs, dmap.notes):
                for obj in group:
                    if obj.beat >= state.last_beat - 1e-4:
                        state.add(obj)
        self._seen[b] = (window, len(events))

    def _rules(self, b: int, parser: Parser, start_tick: int) -> np.ndarray:
        mask = np.ones(_V.size, dtype=bool)
        if parser.time is None:
            return mask
        state = self.states[b]
        beat = (start_tick + parser.time) / TICKS_PER_BEAT
        partial = parser.partial
        dots = self.requests[b].characteristic == Characteristic.NO_ARROWS
        if partial is None or not parser.in_object:
            for color, token in _V.note_heads.items():
                mask[token] = self._any_note(b, state, beat, color, None, dots)
            mask[_V.bomb] = any(
                state.allowed(Bomb(beat, x, y))
                for x in range(GRID_COLUMNS)
                for y in range(GRID_ROWS)
            )
            for degrees, token in _V.rot.items():
                mask[token] = state.allowed(RotationEvent(beat, degrees))
            return mask
        kind, head, values = partial
        if kind == "NOTE" and len(values) == 0:
            for cell, token in _V.pos.items():
                mask[token] = self._any_note(b, state, beat, head, cell, dots)
        elif kind == "NOTE" and len(values) == 1:
            x, y = values[0]
            for d, token in _V.dirs.items():
                ok = not dots or d == CutDirection.ANY
                mask[token] = ok and state.allowed(Note(beat, x, y, head, d))
        elif kind == "BOMB" and len(values) == 0:
            for (x, y), token in _V.pos.items():
                mask[token] = state.allowed(Bomb(beat, x, y))
        return mask

    def _any_note(self, b, state: MapState, beat, color, cell, dots: bool) -> bool:
        """Is any note of ``color`` (in ``cell``, if given) allowed? Cached per state."""
        key = (b, beat, color, cell, len(state.notes), len(state.bombs), state.last_beat)
        hit = self._cache.get(key)
        if hit is None:
            if len(self._cache) > 4096:
                self._cache.clear()
            cells = (
                [cell] if cell else [(x, y) for y in range(GRID_ROWS) for x in range(GRID_COLUMNS)]
            )
            dirs = [CutDirection.ANY] if dots else list(CutDirection)
            hit = any(state.allowed(Note(beat, x, y, color, d)) for x, y in cells for d in dirs)
            self._cache[key] = hit
        return hit


@dataclass(slots=True)
class DensityFeedback:
    """Per-window choice between a difficulty's candidates (base NPS and boosted copies).

    ``groups[g]`` lists the batch rows of request ``g``, base first. After each window the
    best candidate is kept and copied into the other rows (tokens and rules state).
    """

    masker: RuleMasker
    groups: list[list[int]]
    wanted: list[float]  # NPS wanted per request
    gap_limits: list[float]
    loudness: Loudness
    beatmap: Beatmap
    margin: float = 0.05
    boosted: list[int] | None = None  # windows taken from a boosted candidate, per request
    last_note: list[float | None] | None = None
    windows: int = 0

    def __post_init__(self) -> None:
        self.boosted = [0] * len(self.groups)
        self.last_note = [None] * len(self.groups)

    def on_window(self, k: int, bodies: list[list[int]]) -> None:
        self.masker.on_window(k, bodies)
        self.windows = k + 1
        for g, rows in enumerate(self.groups):
            cands = [_note_beats(k, bodies[r]) for r in rows]
            best = 0
            if len(rows) > 1:
                scores = [self.score(g, k, c) + self.margin * j for j, c in enumerate(cands)]
                best = int(np.argmin(scores))
            if best:
                self.boosted[g] += 1  # type: ignore[index]
            chosen = rows[best]
            for r in rows:
                if r != chosen:
                    bodies[r] = list(bodies[chosen])
                    self.masker.adopt(r, chosen)
            if cands[best]:
                self.last_note[g] = cands[best][-1]  # type: ignore[index]

    def score(self, g: int, k: int, beats: list[float]) -> float:
        """Lower is better: distance from the expected note count, plus loud gaps."""
        start, end = k * 16.0, (k + 1) * 16.0
        seconds = self.beatmap.beat_to_seconds(end) - self.beatmap.beat_to_seconds(start)
        expected = self.wanted[g] * seconds * self.loudness.activity(start, end)
        density = abs(len(beats) - expected) / max(expected, 2.0)
        last = self.last_note[g]  # type: ignore[index]
        seq = [last if last is not None else start, *beats, end]
        limit = self.gap_limits[g]
        gaps = loud_gaps(seq, self.loudness, limit)
        return density + 0.5 * sum((gap - limit) / limit for gap in gaps)


def _note_beats(k: int, body: list[int]) -> list[float]:
    ticks = {e.tick for e in NOTE_GRAMMAR.parse(body) if e.kind == "NOTE"}
    return [(k * WINDOW_TICKS + t) / TICKS_PER_BEAT for t in sorted(ticks)]


class ModelGenerator:
    """``NoteGenerator`` on the WS6 note model. Expects a song-time analysis (``timing``).

    ``generate_many`` decodes several requests over the same song as one batch (the
    pipeline sends all difficulties of a mode together). Requests are grouped by seed and
    temperature, since a batch shares its sampling options.
    """

    name = "model"

    def __init__(
        self,
        model: MapperModel,
        settings: ModelSettings | None = None,
        checker: RuleChecker | None = None,
    ) -> None:
        self.model = model
        self.settings = settings or ModelSettings()
        self.checker = checker or RuleChecker()
        self.tokenizer = MapTokenizer(context_events=self.settings.context_events)
        self.last_masked_steps = 0
        # Per request of the last generate_many: (windows from a boosted candidate, windows).
        self.last_boosted: list[tuple[int, int]] = []
        self.last_plans: list = []  # intensity.IntensityPlan per request of the last call
        self.last_refs: np.ndarray | None = None  # reference start ticks of the last call

    @property
    def version(self) -> int:
        """Tokenizer version of the checkpoint (1 = note_v0, 2 = note_v1)."""
        from bsmapper.tokenizer import version_for_vocab

        return version_for_vocab(self.model.cfg.note_vocab)

    @property
    def shifts_intensity(self) -> bool:
        return self.settings.intensity_shift is not False

    @property
    def needs_chroma(self) -> bool:
        """Does this model use the structure memory (so the pipeline adds chroma)?"""
        return self.version >= 2 and self.settings.references

    def generate(self, analysis: AudioAnalysis, request: GenerationRequest) -> DifficultyMap:
        return self.generate_many(analysis, [request])[0]

    def generate_many(
        self, analysis: AudioAnalysis, requests: Sequence[GenerationRequest]
    ) -> list[DifficultyMap]:
        beatmap = tempo_beatmap(analysis)
        plans = [self._plan(r, beatmap.bpm) for r in requests]
        requests = [
            r if r.target_nps is not None else replace(r, target_nps=p.wanted_nps)
            for r, p in zip(requests, plans, strict=True)
        ]
        self.last_plans = plans
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(requests):
            groups.setdefault((r.seed, r.temperature, r.repetition), []).append(i)
        feats = song_features(analysis, beatmap)
        refs = reference_starts(analysis, beatmap, feats) if self.needs_chroma else None
        self.last_refs = refs
        out: list[DifficultyMap | None] = [None] * len(requests)
        self.last_masked_steps = 0
        self.last_boosted = [(0, 0)] * len(requests)
        for (seed, temperature, repetition), idx in groups.items():
            dmaps, boosted = self._decode(
                feats, beatmap, [requests[i] for i in idx], seed, temperature, refs, repetition
            )
            for i, dmap, b in zip(idx, dmaps, boosted, strict=True):
                out[i], self.last_boosted[i] = dmap, b
        return out  # type: ignore[return-value]

    def _decode(
        self,
        feats: np.ndarray,
        beatmap: Beatmap,
        requests: list[GenerationRequest],
        seed: int,
        temperature: float,
        refs: np.ndarray | None = None,
        repetition: float | None = None,
    ) -> tuple[list[DifficultyMap], list[tuple[int, int]]]:
        from bsmapper.models import NoteSampler, SamplingOptions, WindowRefs
        from bsmapper.models.sampling import repetition_guidance

        s = self.settings
        gains = s.nps_gain or gains_for(self.version)
        rows: list[GenerationRequest] = []  # every candidate, base first per request
        conds_as: list[Difficulty] = []  # the DIFF token each row is conditioned on
        groups: list[list[int]] = []
        for r in requests:
            as_ = self._plan(r, beatmap.bpm)
            cond_diff = as_.condition_as
            # Above 0.75 the targets reach the top NPS buckets; keep the boosted copy within
            # what human maps of that difficulty do (intensity 0.5 is left as in #31).
            cap = math.inf
            if as_.intensity > SHIFT_ABOVE:
                cap = conditioning_cap(cond_diff, r.characteristic, beatmap.bpm, gains)
            base = min(conditioning_nps(cond_diff, r.target_nps or 0.0, gains), cap)
            factors = (1.0, *s.density_boosts) if r.target_nps else (1.0,)
            groups.append(list(range(len(rows), len(rows) + len(factors))))
            rows += [replace(r, target_nps=round(min(base * f, cap), 2)) for f in factors]
            conds_as += [cond_diff] * len(factors)
        metas = []
        for r in rows:  # NJS / reaction time follow the label
            njs, offset = njs_and_offset(r.difficulty, beatmap.bpm, r.njs)
            metas.append(DifficultyMap(r.characteristic, r.difficulty, njs=njs, njs_offset=offset))
        # Relaxed ceilings only where a label is decoded as the next difficulty.
        shift = self.shifts_intensity
        checkers = [checker_for(self.checker, r.intensity) if shift else self.checker for r in rows]
        masker = RuleMasker(rows, metas, beatmap, checkers, rules=s.rule_masks)
        feedback = DensityFeedback(
            masker,
            groups,
            [r.target_nps or 0.0 for r in requests],
            [GAP_LIMIT_BEATS[conds_as[g[0]]] for g in groups],
            Loudness(feats[:, -1]),
            beatmap,
            s.boost_margin,
        )
        options = SamplingOptions(
            temperature=temperature,
            top_p=s.top_p,
            seed=seed,
            style_guidance=s.style_guidance,
            context_events=s.context_events,
            max_body_tokens=s.max_body_tokens,
            repetition_guidance=repetition_guidance(repetition),
            token_bias=((_V.arc, s.arc_bias),) if s.arc_bias else (),
        )
        conds = [
            Conditioning.from_request(replace(r, difficulty=d))
            for r, d in zip(rows, conds_as, strict=True)
        ]
        n_windows = max(1, math.ceil(feats.shape[0] / WINDOW_TICKS))
        sampler = NoteSampler(self.model)
        callbacks = {"mask_fn": masker.mask, "on_window": feedback.on_window}
        if refs is not None:
            callbacks["ref_fn"] = WindowRefs(refs)
        windows = sampler.generate(feats, conds, options, n_windows=n_windows, **callbacks)
        self.last_masked_steps += masker.masked_steps
        dmaps = []
        for g in groups:
            meta = metas[g[0]]
            decoded = self.tokenizer.decode(windows[g[0]], meta=meta)
            # The prefix names the conditioning difficulty; the map keeps its label.
            dmaps.append(replace(decoded, difficulty=meta.difficulty))
        boosted = [(n, feedback.windows) for n in feedback.boosted]  # type: ignore[union-attr]
        return dmaps, boosted

    def _plan(self, request: GenerationRequest, bpm: float):
        return intensity_plan(
            request.difficulty,
            request.characteristic,
            bpm,
            request.intensity,
            model_intensity=not self.shifts_intensity,
        )


def chroma_frames(audio_path, frame_rate: float, n_frames: int) -> np.ndarray:
    """(n_frames, 12) chroma of an audio file at the analysis frame rate (for the
    structure memory), computed as the dataset build does (``data.structure``)."""
    from bsmapper.audio.decode import load_audio
    from bsmapper.audio.features import FeatureConfig
    from bsmapper.data.structure import CHROMA_HOP, frame_chroma

    sr = FeatureConfig().sr
    chroma = frame_chroma(load_audio(audio_path, sr), sr)
    if len(chroma) == 0:
        return np.zeros((n_frames, 12), dtype=np.float32)
    idx = np.floor(np.arange(n_frames) / frame_rate * sr / CHROMA_HOP).astype(np.int64)
    return chroma[np.clip(idx, 0, len(chroma) - 1)].astype(np.float32)


def reference_starts(
    analysis: AudioAnalysis, beatmap: Beatmap, feats: np.ndarray
) -> np.ndarray | None:
    """Reference start tick per window (-1 = none), or None without a chroma feature."""
    chroma = analysis.features.get("chroma")
    if chroma is None or len(chroma) < 2:
        return None
    from bsmapper.data.structure import find_references, per_beat_mean, structure_features

    n_windows = feats.shape[0] // WINDOW_TICKS
    n_beats = n_windows * 16
    beat_s = np.vectorize(beatmap.beat_to_seconds)
    beats = per_beat_mean(chroma, analysis.frame_rate, beat_s, n_beats)
    refs = find_references(structure_features(feats, beats), n_windows)
    return refs.kept()


def describe(model: Any) -> str:
    """A short description of a loaded model for logs."""
    n = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    return f"{n / 1e6:.1f}M params on {device}"
