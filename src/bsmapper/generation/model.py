"""``ModelGenerator``: windowed autoregressive decoding for the WS6 note model.

The model itself is not here yet. This module is the decoding loop that will drive it,
written against a small ``NoteSampler`` protocol so WS6 only has to implement two
methods (encode one window of audio, return next-token logits):

    for each 16-beat window k:
        ids = BOS prefix context SEP              # tokenizer.Conditioning + tail_context
        until EOS:
            logits = sampler.next_logits(memory, ids)
            mask   = grammar mask                  # NOTE_GRAMMAR.parser().allowed_mask()
                   & request mask                  # no bombs / walls / ... if disallowed
                   & rules mask                    # bsmapper.rules MapState at this tick
            logits += density bias                 # towards request.target_nps
            sample with temperature / top-p / seeded RNG, feed parser and rules state
    tokenizer.MapTokenizer().decode(windows)

Every complete object is added to the ``MapState`` as soon as the parser emits it, so
the masks always reflect what has been placed (WS5's incremental API).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from bsmapper.generation.njs import njs_and_offset
from bsmapper.generation.timing import tempo_beatmap
from bsmapper.interfaces import AudioAnalysis, GenerationRequest
from bsmapper.rules import MapState, RuleChecker
from bsmapper.schema import (
    Characteristic,
    Color,
    CutDirection,
    DifficultyMap,
    Note,
    RotationEvent,
)
from bsmapper.tokenizer import (
    BOS,
    EOS,
    NOTE_GRAMMAR,
    NOTE_VOCAB,
    SEP,
    TICKS_PER_BEAT,
    WINDOW_BEATS,
    WINDOW_TICKS,
    Conditioning,
    MapTokenizer,
    Parser,
    TokenWindow,
    events_to_map,
    tail_context,
)


class NoteSampler(Protocol):
    """What the WS6 note model provides to ``ModelGenerator``."""

    def encode(self, analysis: AudioAnalysis, start_beat: float, n_beats: int) -> Any:
        """Encode the audio of one window (map beats ``start_beat`` .. ``+ n_beats``).

        ``analysis`` is in song time: row ``j`` of ``beat_features`` is map beat j / 24.
        The return value is opaque to the decoder and passed back to ``next_logits``.
        """
        ...

    def next_logits(self, memory: Any, input_ids: Sequence[int]) -> np.ndarray:
        """Logits over ``NOTE_VOCAB`` for the token after ``input_ids``
        (``BOS prefix context SEP body...``, as in ``TokenWindow.input_ids``)."""
        ...


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    top_p: float = 0.95
    max_tokens_per_window: int = 1024
    context_events: int = 16
    rule_masks: bool = True
    # Logit bias on note heads per unit of tanh((expected - placed notes) / scale).
    density_gain: float = 2.0
    density_scale: float = 4.0


@dataclass(slots=True)
class DecodeStats:
    windows: int = 0
    tokens: int = 0
    forced_ends: int = 0  # windows cut at max_tokens_per_window
    rule_masked_steps: int = 0  # steps where the rules removed at least one option
    fallback_steps: int = 0  # steps where the rules left nothing and were ignored


def sample_token(
    logits: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    """Sample one id from ``logits`` restricted to ``mask``, with temperature and top-p."""
    allowed = np.flatnonzero(mask)
    if len(allowed) == 0:
        raise ValueError("no token is allowed")
    if temperature <= 1e-6:
        return int(allowed[np.argmax(logits[allowed])])
    z = logits[allowed].astype(np.float64) / temperature
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p, kind="stable")
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, top_p) + 1)]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return int(allowed[rng.choice(len(allowed), p=p)])


class _Masks:
    """Static token-id sets used to build the per-step masks."""

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

    def request_mask(self, request: GenerationRequest) -> np.ndarray:
        """Tokens the request rules out entirely (object types, colours, directions)."""
        m = np.ones(self.size, dtype=bool)
        mode = request.characteristic
        if mode == Characteristic.LIGHTSHOW:
            m[[*self.note_heads.values(), self.bomb, self.wall, self.arc, self.chain]] = False
        if not request.allow_bombs:
            m[self.bomb] = False
        if not request.allow_obstacles:
            m[self.wall] = False
        if not request.allow_arcs:
            m[self.arc] = False
        if not request.allow_chains:
            m[self.chain] = False
        if mode == Characteristic.ONE_SABER:
            m[self.note_heads[Color.RED]] = False
        if mode not in (Characteristic.DEGREE_90, Characteristic.DEGREE_360):
            m[list(self.rot.values())] = False
        return m


class ModelGenerator:
    """``NoteGenerator`` that decodes a ``NoteSampler`` window by window.

    Without a sampler (until WS6 lands) ``generate`` raises; the pipeline uses
    ``StubGenerator`` instead. Expects a song-time analysis (``timing``).
    """

    name = "model"

    def __init__(
        self,
        sampler: NoteSampler | None = None,
        sampling: SamplingConfig | None = None,
        checker: RuleChecker | None = None,
    ) -> None:
        self.sampler = sampler
        self.sampling = sampling or SamplingConfig()
        self.checker = checker or RuleChecker()
        self.tokenizer = MapTokenizer(context_events=self.sampling.context_events)
        self._masks = _Masks()
        self.last_stats = DecodeStats()
        self._note_masks: dict[tuple, np.ndarray] = {}

    def generate(self, analysis: AudioAnalysis, request: GenerationRequest) -> DifficultyMap:
        if self.sampler is None:
            raise RuntimeError(
                "ModelGenerator has no note model yet (WS6); use StubGenerator for now"
            )
        beatmap = tempo_beatmap(analysis)
        njs, offset = njs_and_offset(request.difficulty, beatmap.bpm, request.njs)
        meta = DifficultyMap(request.characteristic, request.difficulty, njs=njs, njs_offset=offset)
        state = self.checker.state(meta, beatmap)
        mode_index = list(Characteristic).index(request.characteristic)
        rng = np.random.default_rng([request.seed, mode_index, request.difficulty.rank])
        prefix = Conditioning.from_request(request).prefix_ids()
        static = self._masks.request_mask(request)
        stats = self.last_stats = DecodeStats()

        n_windows = max(1, math.ceil(len(analysis.beat_times) / WINDOW_BEATS))
        windows: list[TokenWindow] = []
        prev_body: list[int] = []
        for k in range(n_windows):
            context = tail_context(prev_body, NOTE_GRAMMAR, self.sampling.context_events)
            memory = self.sampler.encode(analysis, k * WINDOW_BEATS, WINDOW_BEATS)
            body = self._decode_window(
                k, memory, prefix, context, static, state, request, rng, stats
            )
            windows.append(TokenWindow(k, body, context, list(prefix)))
            prev_body = body
        stats.windows = len(windows)
        return self.tokenizer.decode(windows, meta=meta)

    # --- one window -------------------------------------------------------------------

    def _decode_window(
        self,
        k: int,
        memory: Any,
        prefix: list[int],
        context: list[int],
        static: np.ndarray,
        state: MapState,
        request: GenerationRequest,
        rng: np.random.Generator,
        stats: DecodeStats,
    ) -> list[int]:
        cfg = self.sampling
        ids = [BOS, *prefix, *context, SEP]
        body: list[int] = []
        parser = NOTE_GRAMMAR.parser()
        start_tick = k * WINDOW_TICKS
        obj_start = 0  # body index where the object being read started
        self._note_masks.clear()
        for _ in range(cfg.max_tokens_per_window):
            mask = parser.allowed_mask() & static
            if cfg.rule_masks:
                ruled = mask & self._rules_mask(parser, state, start_tick, request)
                if not ruled.any():
                    stats.fallback_steps += 1
                elif (ruled != mask).any():
                    stats.rule_masked_steps += 1
                    mask = ruled
            if not mask.any():  # only possible inside an object the request forbids
                break
            logits = np.asarray(self.sampler.next_logits(memory, ids), dtype=np.float64)
            logits = logits + self._density_bias(parser, state, start_tick, request)
            token = sample_token(logits, mask, rng, request.temperature, cfg.top_p)
            stats.tokens += 1
            if token == EOS:
                break
            if not parser.in_object:
                obj_start = len(body)
            for event in parser.feed(token):
                self._commit(state, start_tick + event.tick, event, request)
            ids.append(token)
            body.append(token)
        else:
            stats.forced_ends += 1
        if parser.in_object:  # cut off mid-object: drop the unfinished object
            return body[:obj_start]
        return self._close(parser, body, state, start_tick, request)

    def _close(
        self,
        parser: Parser,
        body: list[int],
        state: MapState,
        start_tick: int,
        request: GenerationRequest,
    ) -> list[int]:
        """End the window: flush an object still waiting for an optional modifier."""
        before = len(parser.events)
        for event in parser.finish()[before:]:
            self._commit(state, start_tick + event.tick, event, request)
        return body

    def _commit(self, state: MapState, tick: int, event, request: GenerationRequest) -> None:
        """Feed an emitted event to the rules state (walls/arcs/chains included)."""
        dmap = events_to_map([(tick, event)], request.characteristic, request.difficulty)
        for group in (dmap.rotations, dmap.obstacles, dmap.bombs, dmap.notes):
            for obj in group:
                if obj.beat >= state.last_beat - 1e-4:
                    state.add(obj)

    # --- masks and bias ------------------------------------------------------------------

    def _rules_mask(
        self, parser: Parser, state: MapState, start_tick: int, request: GenerationRequest
    ) -> np.ndarray:
        m = self._masks
        mask = np.ones(m.size, dtype=bool)
        if parser.time is None:
            return mask
        beat = (start_tick + parser.time) / TICKS_PER_BEAT
        partial = parser.partial
        if partial is None or not parser.in_object:
            # Between objects: a head may only start if something of it can be placed.
            for color, tok in m.note_heads.items():
                mask[tok] = bool(self._note_mask(state, beat, color).any())
            mask[m.bomb] = bool(state.bomb_mask(beat).any())
            for degrees, tok in m.rot.items():
                mask[tok] = state.allowed(RotationEvent(beat, degrees))
            return mask
        kind, head, values = partial
        if kind == "NOTE":
            cells = self._note_mask(state, beat, head)
            if len(values) == 0:
                for (x, y), tok in m.pos.items():
                    mask[tok] = bool(cells[x, y].any())
            elif len(values) == 1:
                x, y = values[0]
                for d, tok in m.dirs.items():
                    ok = bool(cells[x, y, d])
                    if request.characteristic == Characteristic.NO_ARROWS:
                        ok = ok and d == CutDirection.ANY
                    mask[tok] = ok
        elif kind == "BOMB" and len(values) == 0:
            cells = state.bomb_mask(beat)
            for (x, y), tok in m.pos.items():
                mask[tok] = bool(cells[x, y])
        return mask

    def _note_mask(self, state: MapState, beat: float, color: Color) -> np.ndarray:
        """``state.note_mask``, cached until the state changes."""
        key = (beat, color, len(state.notes), len(state.bombs), state.last_beat)
        mask = self._note_masks.get(key)
        if mask is None:
            if len(self._note_masks) > 64:
                self._note_masks.clear()
            mask = self._note_masks[key] = state.note_mask(beat, color)
        return mask

    def _density_bias(
        self, parser: Parser, state: MapState, start_tick: int, request: GenerationRequest
    ) -> np.ndarray:
        """Push note heads up when behind ``target_nps``, down when ahead of it."""
        bias = np.zeros(self._masks.size)
        cfg = self.sampling
        if request.target_nps is None or cfg.density_gain == 0 or parser.time is None:
            return bias
        if parser.in_object:
            return bias
        beat = (start_tick + parser.time) / TICKS_PER_BEAT
        notes: list[Note] = state.notes
        first = notes[0].beat if notes else beat
        elapsed = state.seconds(beat) - state.seconds(first)
        deficit = request.target_nps * elapsed - len(notes)
        push = cfg.density_gain * math.tanh(deficit / cfg.density_scale)
        bias[list(self._masks.note_heads.values())] = push
        return bias
