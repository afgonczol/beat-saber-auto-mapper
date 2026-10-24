"""The note model and the lighting model, sharing one audio encoder.

    audio features (B, 384, 83) ─► AudioEncoder ─► memory (B, 384, d)
    BOS prefix context SEP body ─► note decoder (x-attn: memory) ─► next note token
    BOS context SEP body        ─► light decoder (x-attn: memory + NoteMemory) ─► next light

``NoteMemory`` encodes the window's note tokens (``BOS prefix SEP body EOS``) so the
lights follow the notes as well as the music. Token sequences are exactly
``TokenWindow.input_ids()`` from the tokenizer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from bsmapper.models.config import ModelConfig
from bsmapper.models.layers import (
    DecoderBlock,
    DecoderCache,
    EncoderBlock,
    LayerCache,
    causal_mask,
    padding_mask,
    run_blocks,
)
from bsmapper.tokenizer import PAD


class AudioEncoder(nn.Module):
    """Beat-synchronous features (one frame per tick) -> contextual frame embeddings."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Per-feature normalization, set from training data (see set_feature_stats).
        self.register_buffer("feat_mean", torch.zeros(cfg.feature_dim))
        self.register_buffer("feat_std", torch.ones(cfg.feature_dim))
        k = cfg.conv_kernel
        self.stem = nn.Sequential(
            nn.Conv1d(cfg.feature_dim, cfg.d_model, k, padding=k // 2),
            nn.GELU(),
            nn.Conv1d(cfg.d_model, cfg.d_model, k, padding=k // 2),
            nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, cfg.audio_frames, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.qk_norm)
            for _ in range(cfg.encoder_layers)
        )
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[1] != self.cfg.audio_frames:
            raise ValueError(f"expected {self.cfg.audio_frames} frames, got {features.shape[1]}")
        x = (features.float() - self.feat_mean) / self.feat_std
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(x + self.pos)
        x = run_blocks(self.blocks, x, self.cfg.grad_checkpointing, None)
        return self.ln(x)


class NoteMemory(nn.Module):
    """Bidirectional encoder over a window's note tokens, for the lighting decoder."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.note_vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.note_memory_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            EncoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.qk_norm)
            for _ in range(cfg.note_memory_layers)
        )
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        tokens = tokens[:, : self.cfg.note_memory_len]
        valid = tokens != PAD
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens) + self.pos(pos)
        x = run_blocks(self.blocks, x, self.cfg.grad_checkpointing, padding_mask(valid))
        return self.ln(x), valid


class TokenDecoder(nn.Module):
    """Causal decoder over one vocabulary, cross-attending to a memory."""

    def __init__(self, cfg: ModelConfig, vocab: int, layers: int, max_len: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_len = max_len
        self.embed = nn.Embedding(vocab, cfg.d_model)
        self.pos = nn.Embedding(max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.qk_norm)
            for _ in range(layers)
        )
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.embed.weight

    def forward(self, tokens: Tensor, memory: Tensor, memory_valid: Tensor | None = None):
        """Teacher-forced logits (B, L, V) for right-padded ``tokens`` (PAD = 0).

        Logits at PAD positions are meaningless; real positions never see the pads.
        """
        length = tokens.shape[1]
        if length > self.max_len:
            raise ValueError(f"sequence of {length} tokens is longer than max_len {self.max_len}")
        pos = torch.arange(length, device=tokens.device)
        x = self.drop(self.embed(tokens) + self.pos(pos))
        x = run_blocks(
            self.blocks, x, self.cfg.grad_checkpointing, memory, None, padding_mask(memory_valid)
        )
        return self.head(self.ln(x))

    # --- incremental decoding ---------------------------------------------------

    def start(self, memory: Tensor, memory_valid: Tensor | None = None) -> DecoderCache:
        b = memory.shape[0]
        return DecoderCache(
            layers=[LayerCache() for _ in self.blocks],
            key_valid=torch.zeros(b, 0, dtype=torch.bool, device=memory.device),
            next_pos=torch.zeros(b, dtype=torch.long, device=memory.device),
            memory=memory,
            memory_valid=memory_valid,
        )

    def step(self, tokens: Tensor, cache: DecoderCache, valid: Tensor | None = None) -> Tensor:
        """Append ``tokens`` (B, n) to the cache and return their logits (B, n, V).

        ``valid`` marks real tokens; ``False`` entries (left padding of a prompt, or
        a finished sequence) take no position and are hidden from later tokens.
        """
        if valid is None:
            valid = torch.ones_like(tokens, dtype=torch.bool)
        before = torch.cumsum(valid.long(), dim=1) - valid.long()
        pos = (cache.next_pos[:, None] + before).clamp(max=self.max_len - 1)
        cache.next_pos = cache.next_pos + valid.long().sum(dim=1)
        cache.key_valid = torch.cat([cache.key_valid, valid], dim=1)
        self_mask = causal_mask(cache.key_valid, tokens.shape[1])
        cross_mask = padding_mask(cache.memory_valid)
        x = self.embed(tokens) + self.pos(pos)
        for block, layer_cache in zip(self.blocks, cache.layers, strict=True):
            x = block(x, cache.memory, self_mask, cross_mask, layer_cache)
        return self.head(self.ln(x))


class MapperModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.light_styles: dict | None = None  # the checkpoint's StyleModel (light prompt v2)
        self.encoder = AudioEncoder(cfg)
        self.notes = TokenDecoder(cfg, cfg.note_vocab, cfg.decoder_layers, cfg.max_len)
        if cfg.has_lighting:
            self.note_memory = NoteMemory(cfg)
            self.lights = TokenDecoder(cfg, cfg.light_vocab, cfg.light_layers, cfg.light_max_len)
            self.memory_type = nn.Parameter(torch.zeros(2, cfg.d_model))  # audio, notes
        self.apply(_init_weights)

    def encode_audio(self, features: Tensor) -> Tensor:
        return self.encoder(features)

    def note_logits(self, audio_memory: Tensor, tokens: Tensor) -> Tensor:
        return self.notes(tokens, audio_memory)

    def light_memory(self, audio_memory: Tensor, note_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Memory for the lighting decoder: audio frames followed by the note track."""
        if not self.cfg.has_lighting:
            raise RuntimeError("this model has no lighting decoder (light_layers = 0)")
        note_mem, note_valid = self.note_memory(note_tokens)
        b, t, _ = audio_memory.shape
        memory = torch.cat(
            [audio_memory + self.memory_type[0], note_mem + self.memory_type[1]], dim=1
        )
        audio_valid = torch.ones(b, t, dtype=torch.bool, device=memory.device)
        return memory, torch.cat([audio_valid, note_valid], dim=1)

    def light_logits(self, audio_memory: Tensor, note_tokens: Tensor, tokens: Tensor) -> Tensor:
        memory, valid = self.light_memory(audio_memory, note_tokens)
        return self.lights(tokens, memory, valid)

    def set_feature_stats(self, mean, std) -> None:
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-3)
        self.encoder.feat_mean.copy_(mean)
        self.encoder.feat_std.copy_(std)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear | nn.Conv1d):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, AudioEncoder):
        nn.init.normal_(module.pos, std=0.02)
