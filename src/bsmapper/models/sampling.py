"""Window-by-window autoregressive decoding, the API WS7 builds generation on.

    model = load_model("checkpoints/note_v0/last.pt", device="cuda")
    sampler = NoteSampler(model)
    windows = sampler.generate(
        features,                                  # (frames, 83) beat-synchronous, 24/beat
        [Conditioning.from_request(r) for r in requests],
        SamplingOptions(temperature=0.9, top_p=0.95, seed=1),
        mask_fn=my_rule_mask,                      # optional, see StepState
    )
    dmap = MapTokenizer().decode(windows[0], meta=...)

Every step samples from the model's logits restricted to the tokens the event
grammar allows next (``Parser.allowed_mask``), intersected with ``mask_fn``'s
mask when one is given. If that intersection is empty, the grammar mask alone is
used, so a rule mask can never make decoding stall. Window ``k``'s decoder prompt
is ``BOS prefix tail(k-1) SEP``, where the tail is built from the *generated*
window ``k-1``. Several conditionings (e.g. all difficulties of one mode) decode
in one batch over the same audio.

Sampling runs on the CPU in NumPy with ``seed``, so a seed gives the same map on
any device as long as the logits agree.

The prompt layout follows the checkpoint's tokenizer version (from its vocabulary
size): a note_v0 checkpoint (v1) gets the 8-token prefix and no reference blocks, so
it decodes exactly as before. A v2 checkpoint (note_v1) also gets ``INTENSITY`` and,
when ``ref_fn`` gives one, a reference block ``REF <events> REF_END`` built from the
already generated windows (``WindowRefs``). ``SamplingOptions.repetition_guidance``
scales the reference's effect by classifier-free guidance against the same prompt
without the block.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from bsmapper.models.mapper import MapperModel
from bsmapper.tokenizer import (
    BOS,
    DEFAULT_MAX_REF_TOKENS,
    EOS,
    LIGHT_GRAMMAR,
    NOTE_GRAMMAR,
    PAD,
    REF,
    REF_END,
    SEP,
    WINDOW_TICKS,
    Conditioning,
    Grammar,
    Parser,
    TokenWindow,
    span_tokens,
    tail_context,
    version_for_vocab,
)


@dataclass(slots=True)
class SamplingOptions:
    temperature: float = 1.0  # 0 = greedy
    top_p: float = 0.95
    seed: int = 0
    # Classifier-free guidance on STYLE: logits = uncond + scale * (cond - uncond), where
    # uncond has STYLE_NONE. 1.0 = off. Only applies to conditionings with a style.
    style_guidance: float = 1.0
    # Classifier-free guidance on the reference window (note_v1): logits = noref +
    # scale * (ref - noref). 1.0 = the learned behaviour, 0 = ignore the reference,
    # > 1 = repeat more. See ``repetition_guidance`` for the 0..1 knob.
    repetition_guidance: float = 1.0
    context_events: int = 16
    max_body_tokens: int = 1024
    # (token id, logit offset) added before sampling, e.g. to make arcs more likely; a
    # bias of b multiplies a token's odds by about e**b wherever the grammar allows it.
    token_bias: tuple[tuple[int, float], ...] = ()


@dataclass(slots=True)
class StepState:
    """What a mask callback sees before each sampled token."""

    window: int  # window index; its first tick is window * 384
    sequence: int  # index into the batch of conditionings
    body: list[int]  # body tokens sampled so far in this window
    parser: Parser  # grammar state: .time (tick in window), .events, .partial


MaskFn = Callable[[StepState], np.ndarray | None]  # bool over the vocabulary, True = allowed
WindowFn = Callable[[int, list[list[int]]], None]  # (window, bodies) after each window
# (window, sequence, that sequence's generated bodies so far) -> reference events or None.
RefFn = Callable[[int, int, list[list[int]]], list[int] | None]


def repetition_guidance(repetition: float | None) -> float:
    """``GenerationRequest.repetition`` (0..1) as a reference guidance scale.

    None -> 1.0 (the learned, human-like amount). Otherwise ``2 * repetition``:
    0 ignores the reference, 0.5 is the learned amount and 1.0 doubles its effect.
    """
    if repetition is None:
        return 1.0
    return 2.0 * min(max(repetition, 0.0), 1.0)


class WindowRefs:
    """A ``RefFn`` from reference start ticks, e.g. ``find_references(...).kept()``.

    Window ``k``'s reference is the events of the already generated windows in
    [starts[k], starts[k] + 16 beats), with TIME relative to the span's start.
    """

    def __init__(
        self,
        starts: Sequence[int | None],
        max_tokens: int = DEFAULT_MAX_REF_TOKENS,
        grammar: Grammar = NOTE_GRAMMAR,
    ) -> None:
        self.starts = [None if s is None or s < 0 else int(s) for s in starts]
        self.max_tokens, self.grammar = max_tokens, grammar

    def __call__(self, window: int, sequence: int, bodies: list[list[int]]) -> list[int] | None:
        start = self.starts[window] if window < len(self.starts) else None
        if start is None or start + WINDOW_TICKS > window * WINDOW_TICKS:
            return None
        return span_tokens(bodies, start, self.grammar, self.max_tokens)


def pad_features(features: np.ndarray, n_windows: int | None = None) -> np.ndarray:
    """Pad (frames, F) features to whole windows, with each column's minimum (silence)."""
    frames = features.shape[0]
    n = n_windows if n_windows is not None else max(1, math.ceil(frames / WINDOW_TICKS))
    total = n * WINDOW_TICKS
    if frames >= total:
        return np.asarray(features[:total], dtype=np.float32)
    fill = features.min(axis=0) if frames else np.zeros(features.shape[1])
    out = np.empty((total, features.shape[1]), dtype=np.float32)
    out[:frames] = features
    out[frames:] = fill
    return out


def sample_token(
    logits: np.ndarray, allowed: np.ndarray, temperature: float, top_p: float, rng
) -> int:
    """Sample one id from ``logits`` restricted to ``allowed`` (temperature, nucleus)."""
    ids = np.flatnonzero(allowed)
    if len(ids) == 0:
        raise ValueError("no token is allowed")
    lg = logits[ids].astype(np.float64)
    if temperature <= 0:
        return int(ids[np.argmax(lg)])
    lg = (lg - lg.max()) / temperature
    p = np.exp(lg)
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p, kind="stable")
        cum = np.cumsum(p[order])
        keep = order[: int(np.searchsorted(cum, top_p) + 1)]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return int(ids[rng.choice(len(ids), p=p)])


class _WindowDecoder:
    """Decodes one window for a batch of prompts with a KV cache."""

    def __init__(self, decoder, grammar: Grammar, device: torch.device) -> None:
        self.decoder, self.grammar, self.device = decoder, grammar, device
        self.vocab_size = len(grammar.vocab)
        self.model_vocab = decoder.embed.num_embeddings

    @torch.no_grad()
    def run(
        self,
        memory: torch.Tensor,
        memory_valid: torch.Tensor | None,
        prompts: list[list[int]],
        window: int,
        opts: SamplingOptions,
        rng,
        mask_fn: MaskFn | None,
        guidance: list[tuple[int, int, float]] | None = None,
    ) -> list[list[int]]:
        """Bodies (without EOS) for ``prompts``.

        ``guidance`` lists (cond row, uncond row, scale): the extra uncond rows of the
        batch are only used to adjust their cond row's logits.
        """
        n = len(prompts)
        width = max(len(p) for p in prompts)
        tokens = torch.full((n, width), PAD, dtype=torch.long)
        valid = torch.zeros((n, width), dtype=torch.bool)
        for b, p in enumerate(prompts):
            tokens[b, width - len(p) :] = torch.as_tensor(p)
            valid[b, width - len(p) :] = True
        cache = self.decoder.start(memory, memory_valid)
        logits = self.decoder.step(tokens.to(self.device), cache, valid.to(self.device))[:, -1]

        uncond_rows = {u for _, u, _ in guidance or ()}
        rows = [b for b in range(n) if b not in uncond_rows]
        # Past ``limit`` only EOS may start; the margin lets an open object finish.
        hard_limit = self.decoder.max_len - width
        limit = max(0, min(opts.max_body_tokens, hard_limit - 32))
        parsers = {b: self.grammar.parser() for b in rows}
        bodies: dict[int, list[int]] = {b: [] for b in rows}
        active = set(rows)
        while active:
            lg = _guide(logits.float().cpu().numpy(), guidance or ())
            if lg.shape[1] < self.vocab_size:  # an older checkpoint: newer tokens never sampled
                pad = np.full((n, self.vocab_size - lg.shape[1]), -np.inf, dtype=lg.dtype)
                lg = np.concatenate([lg, pad], axis=1)
            for token, offset in opts.token_bias:
                lg[:, token] += offset
            chosen = np.full(n, PAD, dtype=np.int64)
            for b in sorted(active):
                parser, body = parsers[b], bodies[b]
                allowed = parser.allowed_mask()
                allowed[self.model_vocab :] = False
                if len(body) >= limit and not parser.in_object:
                    allowed = np.zeros_like(allowed)
                    allowed[EOS] = True
                elif mask_fn is not None and len(body) < limit:
                    extra = mask_fn(StepState(window, b, body, parser))
                    if extra is not None:
                        both = allowed & extra
                        if both.any():
                            allowed = both
                token = sample_token(lg[b], allowed, opts.temperature, opts.top_p, rng)
                parser.feed(token)
                chosen[b] = token
                if token == EOS:
                    active.discard(b)
                    continue
                body.append(token)
                if len(body) >= hard_limit:  # out of positions inside an object
                    active.discard(b)
                    bodies[b] = _complete_prefix(body, self.grammar)
            for c, u, _ in guidance or ():
                chosen[u] = chosen[c]
            if not active:
                break
            step_valid = torch.zeros(n, 1, dtype=torch.bool)
            cond_of = {u: c for c, u, _ in guidance or ()}
            for b in range(n):
                step_valid[b, 0] = cond_of.get(b, b) in active
            step_tokens = torch.as_tensor(chosen).view(n, 1)
            step_tokens[~step_valid] = PAD
            logits = self.decoder.step(
                step_tokens.to(self.device), cache, step_valid.to(self.device)
            )[:, -1]
        return [bodies[b] for b in rows]


def _guide(lg: np.ndarray, guidance: Sequence[tuple[int, int, float]]) -> np.ndarray:
    """Apply classifier-free guidance; a row may be guided by several uncond rows."""
    per_row: dict[int, list[tuple[int, float]]] = {}
    for c, u, scale in guidance:
        per_row.setdefault(c, []).append((u, scale))
    if not per_row:
        return lg
    orig = lg.copy()
    for c, terms in per_row.items():
        if len(terms) == 1:  # the original formula, bit for bit
            u, scale = terms[0]
            lg[c] = orig[u] + scale * (orig[c] - orig[u])
        else:
            lg[c] = orig[c] + sum((scale - 1.0) * (orig[c] - orig[u]) for u, scale in terms)
    return lg


def _complete_prefix(body: list[int], grammar: Grammar) -> list[int]:
    """The longest prefix of ``body`` that ends on a complete event."""
    parser = grammar.parser()
    end = 0
    for i, token in enumerate(body):
        parser.feed(token)
        if not parser.in_object:
            end = i + 1
    return body[:end]


class NoteSampler:
    def __init__(self, model: MapperModel, device: torch.device | str | None = None) -> None:
        self.model = model.eval()
        self.device = torch.device(device) if device else next(model.parameters()).device
        self._decoder = _WindowDecoder(model.notes, NOTE_GRAMMAR, self.device)
        self.version = version_for_vocab(model.cfg.note_vocab)  # tokenizer version

    @property
    def supports_references(self) -> bool:
        return self.version >= 2

    @torch.no_grad()
    def generate(
        self,
        features: np.ndarray,
        conditionings: Conditioning | Sequence[Conditioning],
        options: SamplingOptions | None = None,
        mask_fn: MaskFn | None = None,
        on_window: WindowFn | None = None,
        n_windows: int | None = None,
        ref_fn: RefFn | None = None,
    ) -> list[list[TokenWindow]]:
        """One list of TokenWindows per conditioning (decode with ``MapTokenizer``).

        ``ref_fn`` gives window references (note_v1 checkpoints; ignored otherwise).
        """
        opts = options or SamplingOptions()
        conds = [conditionings] if isinstance(conditionings, Conditioning) else list(conditionings)
        feats = pad_features(features, n_windows)
        n = feats.shape[0] // WINDOW_TICKS
        rng = np.random.default_rng(opts.seed)
        v = self.version
        prefixes = [c.prefix_ids(version=v) for c in conds]
        guided = [
            (b, c.drop_style().prefix_ids(version=v))
            for b, c in enumerate(conds)
            if opts.style_guidance != 1.0 and c.style is not None
        ]
        if not self.supports_references or opts.repetition_guidance == 0.0:
            ref_fn = None
        out: list[list[TokenWindow]] = [[] for _ in conds]
        history: list[list[list[int]]] = [[] for _ in conds]  # generated bodies per sequence
        prev: list[list[int]] = [[] for _ in conds]
        for k in range(n):
            memory = self._encode(feats, k)
            contexts = [
                tail_context(p, NOTE_GRAMMAR, opts.context_events) if k else [] for p in prev
            ]
            refs = [ref_fn(k, b, history[b]) if ref_fn else None for b in range(len(conds))]
            blocks = [[] if r is None else [REF, *r, REF_END] for r in refs]
            prompts = [
                [BOS, *pre, *blk, *ctx, SEP]
                for pre, blk, ctx in zip(prefixes, blocks, contexts, strict=True)
            ]
            guidance = []
            for b, uncond in guided:
                guidance.append((b, len(prompts), opts.style_guidance))
                prompts.append([BOS, *uncond, *blocks[b], *contexts[b], SEP])
            if opts.repetition_guidance != 1.0:
                for b, r in enumerate(refs):
                    if r is not None:
                        guidance.append((b, len(prompts), opts.repetition_guidance))
                        prompts.append([BOS, *prefixes[b], *contexts[b], SEP])
            bodies = self._decoder.run(
                memory.expand(len(prompts), -1, -1), None, prompts, k, opts, rng, mask_fn, guidance
            )
            if on_window is not None:
                on_window(k, bodies)
            for b, body in enumerate(bodies):
                out[b].append(TokenWindow(k, body, contexts[b], prefixes[b], refs[b]))
                history[b].append(body)
            prev = bodies
        return out

    def _encode(self, feats: np.ndarray, k: int) -> torch.Tensor:
        window = feats[k * WINDOW_TICKS : (k + 1) * WINDOW_TICKS]
        x = torch.as_tensor(window, dtype=torch.float32, device=self.device)[None]
        return self.model.encode_audio(x)


class LightSampler:
    """Lighting windows for a song, following generated (or human) note windows."""

    def __init__(self, model: MapperModel, device: torch.device | str | None = None) -> None:
        if not model.cfg.has_lighting:
            raise ValueError("this checkpoint has no lighting decoder")
        self.model = model.eval()
        self.device = torch.device(device) if device else next(model.parameters()).device
        self._decoder = _WindowDecoder(model.lights, LIGHT_GRAMMAR, self.device)

    @torch.no_grad()
    def generate(
        self,
        features: np.ndarray,
        note_windows: Sequence[TokenWindow],
        options: SamplingOptions | None = None,
        mask_fn: MaskFn | None = None,
    ) -> list[TokenWindow]:
        opts = options or SamplingOptions()
        feats = pad_features(features, len(note_windows))
        rng = np.random.default_rng(opts.seed)
        mem_len = self.model.cfg.note_memory_len
        out: list[TokenWindow] = []
        prev: list[int] = []
        for k, nw in enumerate(note_windows):
            window = feats[k * WINDOW_TICKS : (k + 1) * WINDOW_TICKS]
            x = torch.as_tensor(window, dtype=torch.float32, device=self.device)[None]
            audio = self.model.encode_audio(x)
            notes = [BOS, *nw.prefix, SEP, *nw.body, EOS][:mem_len]
            note_tokens = torch.as_tensor([notes], dtype=torch.long, device=self.device)
            memory, valid = self.model.light_memory(audio, note_tokens)
            context = tail_context(prev, LIGHT_GRAMMAR, opts.context_events) if k else []
            (body,) = self._decoder.run(
                memory, valid, [[BOS, *context, SEP]], k, opts, rng, mask_fn
            )
            out.append(TokenWindow(k, body, context))
            prev = body
        return out


def with_seed(options: SamplingOptions, seed: int) -> SamplingOptions:
    return replace(options, seed=seed)
