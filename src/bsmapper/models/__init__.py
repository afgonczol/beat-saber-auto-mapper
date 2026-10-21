"""WS6: audio encoder, note decoder and lighting decoder (needs the ``ml`` extra).

    model = load_model("checkpoints/note_v0/last.pt", device="cuda")
    windows = NoteSampler(model).generate(features, [conditioning], SamplingOptions(seed=1))

See docs/MODEL.md for the architecture and the sampling API.
"""

from bsmapper.models.checkpoint import load_model, model_state
from bsmapper.models.config import FEATURE_DIM, ModelConfig
from bsmapper.models.mapper import AudioEncoder, MapperModel, NoteMemory, TokenDecoder
from bsmapper.models.sampling import (
    LightSampler,
    MaskFn,
    NoteSampler,
    SamplingOptions,
    StepState,
    pad_features,
    sample_token,
)

__all__ = [
    "FEATURE_DIM",
    "AudioEncoder",
    "LightSampler",
    "MapperModel",
    "MaskFn",
    "ModelConfig",
    "NoteMemory",
    "NoteSampler",
    "SamplingOptions",
    "StepState",
    "TokenDecoder",
    "load_model",
    "model_state",
    "pad_features",
    "sample_token",
]
