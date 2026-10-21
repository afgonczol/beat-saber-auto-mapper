"""Model hyper-parameters. Stored inside every checkpoint, so a checkpoint rebuilds itself."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from bsmapper.tokenizer import LIGHT_VOCAB, NOTE_VOCAB, WINDOW_TICKS

FEATURE_DIM = 83  # 80 log-mel bins + onset strength, spectral flux, RMS (WS3b layout)


@dataclass(slots=True)
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    # Audio encoder: beat-synchronous features, one frame per tick.
    feature_dim: int = FEATURE_DIM
    audio_frames: int = WINDOW_TICKS
    conv_kernel: int = 5
    encoder_layers: int = 6
    # Note decoder.
    note_vocab: int = len(NOTE_VOCAB)
    decoder_layers: int = 8
    max_len: int = 1536  # longest BOS prefix context SEP body EOS sequence
    # Lighting decoder (0 layers = no lighting head).
    light_vocab: int = len(LIGHT_VOCAB)
    light_layers: int = 6
    light_max_len: int = 1536
    note_memory_layers: int = 2  # small encoder over the note track the lights follow
    note_memory_len: int = 1024
    tie_embeddings: bool = True
    grad_checkpointing: bool = False

    @property
    def d_ff(self) -> int:
        return self.d_model * self.ff_mult

    @property
    def has_lighting(self) -> bool:
        return self.light_layers > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**d)
