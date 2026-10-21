"""WS7: end-to-end generation pipeline, sampling, post-processing, stub generator.

    from bsmapper.generation import GenerateOptions, Pipeline
    result = Pipeline().run(Path("song.mp3"), Path("out"), GenerateOptions(lighting="v3"))

``Pipeline()`` uses ``StubGenerator`` (placeholder notes, **not** the ML model) until
WS6's model plugs into ``ModelGenerator`` through the ``NoteSampler`` protocol.
"""

from bsmapper.generation.install import GameConfig, install_map, load_config
from bsmapper.generation.model import (
    DecodeStats,
    ModelGenerator,
    NoteSampler,
    SamplingConfig,
    sample_token,
)
from bsmapper.generation.njs import JUMP_TABLE, njs_and_offset, reaction_time
from bsmapper.generation.pipeline import (
    DifficultyResult,
    GenerateOptions,
    Pipeline,
    PipelineResult,
    Song,
    replace_difficulty,
    safe_folder_name,
)
from bsmapper.generation.postprocess import PostReport, dedupe, fix_with_rules, postprocess
from bsmapper.generation.stub import PROFILES, STUB_LABEL, StubGenerator, StubProfile
from bsmapper.generation.timing import song_time_analysis, tempo_beatmap

__all__ = [
    "JUMP_TABLE",
    "PROFILES",
    "STUB_LABEL",
    "DecodeStats",
    "DifficultyResult",
    "GameConfig",
    "GenerateOptions",
    "ModelGenerator",
    "NoteSampler",
    "Pipeline",
    "PipelineResult",
    "PostReport",
    "SamplingConfig",
    "Song",
    "StubGenerator",
    "StubProfile",
    "dedupe",
    "fix_with_rules",
    "install_map",
    "load_config",
    "njs_and_offset",
    "postprocess",
    "reaction_time",
    "replace_difficulty",
    "safe_folder_name",
    "sample_token",
    "song_time_analysis",
    "tempo_beatmap",
]
