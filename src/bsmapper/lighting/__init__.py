"""WS8: rule-based baseline, ML wrapper and vanilla/v3/Chroma renderers.

generator = RuleLightingGenerator()
lightshow = generator.generate(analysis, beatmap, "v3")  # interfaces.LightingGenerator
light_beatmap(beatmap, analysis, "v3")  # also picks a suitable environment
RuleLightingGenerator(effects=EffectRenderer(Path("checkpoints/light_effects")))  # WS8c

See docs/LIGHTING.md.
"""

from bsmapper.lighting.effects import EffectLibrary, EffectRenderer, available_libraries
from bsmapper.lighting.generator import (
    DEFAULT_LEGACY_ENVIRONMENT,
    DEFAULT_V3_ENVIRONMENT,
    STYLES,
    RuleLightingGenerator,
    environment_for,
    light_beatmap,
    render,
)
from bsmapper.lighting.profiles import (
    EnvironmentProfile,
    available_environments,
    get_profile,
    load_profiles,
)
from bsmapper.lighting.render import render_chroma, render_v3, render_vanilla
from bsmapper.lighting.rules import RuleConfig, RuleLightshow, SectionPlan, rule_lightshow

__all__ = [
    "DEFAULT_LEGACY_ENVIRONMENT",
    "DEFAULT_V3_ENVIRONMENT",
    "STYLES",
    "EffectLibrary",
    "EffectRenderer",
    "EnvironmentProfile",
    "RuleConfig",
    "RuleLightingGenerator",
    "RuleLightshow",
    "SectionPlan",
    "available_environments",
    "available_libraries",
    "environment_for",
    "get_profile",
    "light_beatmap",
    "load_profiles",
    "render",
    "render_chroma",
    "render_v3",
    "render_vanilla",
    "rule_lightshow",
]
