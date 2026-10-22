"""``ModelGenerator``: notes from the WS6 note model, decoded under the WS5 rules.

    generator = ModelGenerator(load_model("checkpoints/note_v0/best.pt", device="cuda"))
    dmaps = generator.generate_many(song_analysis, requests)   # one batch per mode

Decoding is WS6's ``NoteSampler``: 16-beat windows, the previous window's tail as context,
grammar masks, temperature / top-p / seed and classifier-free guidance on STYLE. This
module adds what generation needs on top:

- **conditioning**: ``Conditioning.from_request``; a request without ``target_nps`` gets
  the per-difficulty heuristic from ``density``, scaled by ``NPS_CONDITIONING_GAIN``
  (the v0 model undershoots what it is conditioned on).
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

from bsmapper.generation.density import NPS_CONDITIONING_GAIN, target_nps
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
        checker: RuleChecker,
        rules: bool = True,
    ) -> None:
        self.requests = list(requests)
        self.static = [request_mask(r) for r in requests]
        self.states = [checker.state(m, beatmap) for m in metas]
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

    def generate(self, analysis: AudioAnalysis, request: GenerationRequest) -> DifficultyMap:
        return self.generate_many(analysis, [request])[0]

    def generate_many(
        self, analysis: AudioAnalysis, requests: Sequence[GenerationRequest]
    ) -> list[DifficultyMap]:
        beatmap = tempo_beatmap(analysis)
        requests = [self._with_nps(r, beatmap.bpm) for r in requests]
        groups: dict[tuple[int, float], list[int]] = {}
        for i, r in enumerate(requests):
            groups.setdefault((r.seed, r.temperature), []).append(i)
        feats = song_features(analysis, beatmap)
        out: list[DifficultyMap | None] = [None] * len(requests)
        self.last_masked_steps = 0
        for (seed, temperature), idx in groups.items():
            dmaps = self._decode(feats, beatmap, [requests[i] for i in idx], seed, temperature)
            for i, dmap in zip(idx, dmaps, strict=True):
                out[i] = dmap
        return out  # type: ignore[return-value]

    @staticmethod
    def _with_nps(request: GenerationRequest, bpm: float) -> GenerationRequest:
        if request.target_nps is not None:
            return request
        nps = target_nps(request.difficulty, request.characteristic, bpm)
        return replace(request, target_nps=round(nps * NPS_CONDITIONING_GAIN, 2))

    def _decode(
        self,
        feats: np.ndarray,
        beatmap: Beatmap,
        requests: list[GenerationRequest],
        seed: int,
        temperature: float,
    ) -> list[DifficultyMap]:
        from bsmapper.models import NoteSampler, SamplingOptions

        s = self.settings
        metas = []
        for r in requests:
            njs, offset = njs_and_offset(r.difficulty, beatmap.bpm, r.njs)
            metas.append(DifficultyMap(r.characteristic, r.difficulty, njs=njs, njs_offset=offset))
        masker = RuleMasker(requests, metas, beatmap, self.checker, rules=s.rule_masks)
        options = SamplingOptions(
            temperature=temperature,
            top_p=s.top_p,
            seed=seed,
            style_guidance=s.style_guidance,
            context_events=s.context_events,
            max_body_tokens=s.max_body_tokens,
        )
        conds = [Conditioning.from_request(r) for r in requests]
        n_windows = max(1, math.ceil(feats.shape[0] / WINDOW_TICKS))
        sampler = NoteSampler(self.model)
        callbacks = {"mask_fn": masker.mask, "on_window": masker.on_window}
        windows = sampler.generate(feats, conds, options, n_windows=n_windows, **callbacks)
        self.last_masked_steps += masker.masked_steps
        return [self.tokenizer.decode(w, meta=m) for w, m in zip(windows, metas, strict=True)]


def describe(model: Any) -> str:
    """A short description of a loaded model for logs."""
    n = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    return f"{n / 1e6:.1f}M params on {device}"
