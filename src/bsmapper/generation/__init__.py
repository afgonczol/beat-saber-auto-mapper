"""WS7: end-to-end generation pipeline, sampling, post-processing, stub generator.

    from bsmapper.generation import GenerateOptions, make_pipeline
    pipeline = make_pipeline()          # the WS6 model if a checkpoint is found, else the stub
    result = pipeline.run(Path("song.mp3"), Path("out"), GenerateOptions(lighting="v3"))

``ModelGenerator`` decodes the WS6 note model under the WS5 rules and
``ModelLightingGenerator`` its lighting decoder (L1). ``StubGenerator`` makes
placeholder notes (**not** ML) when there is no model. See docs/GENERATION.md.
"""

from bsmapper.generation.density import target_nps
from bsmapper.generation.install import GameConfig, install_map, load_config
from bsmapper.generation.model import ModelGenerator, ModelSettings, RuleMasker, song_features
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
from bsmapper.generation.runtime import GeneratorSetup, find_checkpoint, make_pipeline, resolve
from bsmapper.generation.stub import PROFILES, STUB_LABEL, StubGenerator, StubProfile
from bsmapper.generation.timing import song_time_analysis, tempo_beatmap

__all__ = [
    "JUMP_TABLE",
    "PROFILES",
    "STUB_LABEL",
    "DifficultyResult",
    "GameConfig",
    "GenerateOptions",
    "GeneratorSetup",
    "ModelGenerator",
    "ModelSettings",
    "Pipeline",
    "PipelineResult",
    "PostReport",
    "RuleMasker",
    "Song",
    "StubGenerator",
    "StubProfile",
    "dedupe",
    "find_checkpoint",
    "fix_with_rules",
    "install_map",
    "load_config",
    "make_pipeline",
    "njs_and_offset",
    "postprocess",
    "reaction_time",
    "replace_difficulty",
    "resolve",
    "safe_folder_name",
    "song_features",
    "song_time_analysis",
    "target_nps",
    "tempo_beatmap",
]
