"""Pre-norm transformer blocks with a key/value cache for incremental decoding.

Masks follow ``F.scaled_dot_product_attention``: boolean, ``True`` = may attend,
broadcast to ``(batch, heads, queries, keys)``.

``qk_norm`` (note_v1) applies a LayerNorm per head to queries and keys before the
dot product. That bounds the attention logits, which grew without limit in the first
note_v1 run and made it diverge right after warmup. The norms run in the compute dtype
(bf16 under autocast) while their kernel accumulates the statistics in fp32; letting
autocast return fp32 q/k cost about 2 GB more activations for nothing. SDPA's flash and
memory-efficient kernels keep the softmax statistics in fp32 too. Blocks without it
(note_v0) have exactly the old parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(slots=True)
class LayerCache:
    """Self-attention keys/values so far, and the cross-attention ones (fixed per window)."""

    k: Tensor | None = None
    v: Tensor | None = None
    cross_k: Tensor | None = None
    cross_v: Tensor | None = None


@dataclass(slots=True)
class DecoderCache:
    """Incremental decoding state for a batch. Sequences may be left-padded."""

    layers: list[LayerCache]
    key_valid: Tensor  # (B, L) which cached positions are real tokens
    next_pos: Tensor  # (B,) position id of each sequence's next real token
    memory: Tensor
    memory_valid: Tensor | None = None
    stats: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.key_valid.shape[1]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, qk_norm: bool = False) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.q_norm = nn.LayerNorm(d_model // n_heads) if qk_norm else None
        self.k_norm = nn.LayerNorm(d_model // n_heads) if qk_norm else None

    def _split(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        return x.view(b, n, self.n_heads, d // self.n_heads).transpose(1, 2)

    def project_kv(self, x: Tensor) -> tuple[Tensor, Tensor]:
        k, v = self.kv(x).chunk(2, dim=-1)
        k, v = self._split(k), self._split(v)
        if self.k_norm is not None:
            k = _head_norm(self.k_norm, k, v.dtype)
        return k, v

    def attend(
        self, x: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, causal: bool = False
    ) -> Tensor:
        q = self._split(self.q(x))
        if self.q_norm is not None:
            q = _head_norm(self.q_norm, q, v.dtype)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        b, h, n, dh = y.shape
        return self.out(y.transpose(1, 2).reshape(b, n, h * dh))

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        k, v = self.project_kv(x if context is None else context)
        return self.attend(x, k, v, mask, causal)


def _head_norm(norm: nn.LayerNorm, x: Tensor, dtype: torch.dtype) -> Tensor:
    """``norm(x)`` in ``dtype`` even under autocast (which would upcast it to fp32)."""
    with torch.autocast(x.device.type, enabled=False):
        w, b = norm.weight.to(dtype), norm.bias.to(dtype)
        return F.layer_norm(x.to(dtype), norm.normalized_shape, w, b, norm.eps)


class FeedForward(nn.Sequential):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )


class EncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, qk_norm: bool = False
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, qk_norm)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.drop(self.attn(self.ln1(x), mask=mask))
        return x + self.drop(self.ff(self.ln2(x)))


class DecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, qk_norm: bool = False
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, qk_norm)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, qk_norm)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_mask: Tensor | None,
        cross_mask: Tensor | None,
        cache: LayerCache | None = None,
    ) -> Tensor:
        h = self.ln1(x)
        if cache is None:
            # Teacher forcing on right-padded sequences: a plain causal mask is exact for
            # every real token (pads only follow them), and lets SDPA use flash attention.
            x = x + self.drop(self.self_attn(h, mask=self_mask, causal=self_mask is None))
        else:
            k, v = self.self_attn.project_kv(h)
            if cache.k is not None:
                k = torch.cat([cache.k, k], dim=2)
                v = torch.cat([cache.v, v], dim=2)
            cache.k, cache.v = k, v
            x = x + self.self_attn.attend(h, k, v, self_mask)
        h = self.ln2(x)
        if cache is None:
            x = x + self.drop(self.cross_attn(h, memory, mask=cross_mask))
        else:
            if cache.cross_k is None:
                cache.cross_k, cache.cross_v = self.cross_attn.project_kv(memory)
            x = x + self.cross_attn.attend(h, cache.cross_k, cache.cross_v, cross_mask)
        return x + self.drop(self.ff(self.ln3(x)))


def padding_mask(key_valid: Tensor | None) -> Tensor | None:
    """(B, K) valid keys -> (B, 1, 1, K) attention mask."""
    return None if key_valid is None else key_valid[:, None, None, :]


def causal_mask(key_valid: Tensor, n_queries: int) -> Tensor:
    """Causal mask for the last ``n_queries`` of ``key_valid.shape[1]`` positions.

    Padded keys are hidden, except that every position may see itself, so a
    padded query never has an empty row (which would give NaNs).
    """
    b, k = key_valid.shape
    q_pos = torch.arange(k - n_queries, k, device=key_valid.device)
    k_pos = torch.arange(k, device=key_valid.device)
    causal = k_pos[None, :] <= q_pos[:, None]  # (Q, K)
    own = k_pos[None, :] == q_pos[:, None]
    return ((causal[None] & key_valid[:, None, :]) | own[None])[:, None]  # (B, 1, Q, K)


def run_blocks(blocks: nn.ModuleList, x: Tensor, use_checkpoint: bool, *args) -> Tensor:
    for block in blocks:
        if use_checkpoint and torch.is_grad_enabled():
            x = checkpoint(block, x, *args, use_reentrant=False)
        else:
            x = block(x, *args)
    return x
