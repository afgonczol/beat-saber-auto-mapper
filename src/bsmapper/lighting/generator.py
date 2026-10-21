"""``RuleLightingGenerator``: the L0 baseline behind ``interfaces.LightingGenerator``."""

from __future__ import annotations

import logging

from bsmapper.interfaces import AudioAnalysis, LightingStyle
from bsmapper.lighting.profiles import get_profile
from bsmapper.lighting.render import render_chroma, render_v3, render_vanilla
from bsmapper.lighting.rules import RuleConfig, RuleLightshow, rule_lightshow
from bsmapper.schema import Beatmap, Lightshow

log = logging.getLogger(__name__)

DEFAULT_V3_ENVIRONMENT = "WeaveEnvironment"
DEFAULT_LEGACY_ENVIRONMENT = "DefaultEnvironment"  # The First
STYLES: tuple[LightingStyle, ...] = ("vanilla", "v3", "chroma")


class RuleLightingGenerator:
    """Section-aware rule lightshow (L0), rendered in the requested style.

    ``generate`` lights for ``beatmap.environment``. For ``"v3"`` that environment needs
    a profile (see ``profiles``); without one the result falls back to vanilla events,
    since event box groups do nothing in legacy environments. ``light_beatmap`` picks a
    suitable environment and sets it on the beatmap.

    ``time_offset_s`` is silence prepended to the analysed audio in the packaged song
    (for example ``BeatGrid.lead_in_s()``).
    """

    def __init__(self, *, time_offset_s: float = 0.0, config: RuleConfig | None = None) -> None:
        self.time_offset_s = time_offset_s
        self.config = config or RuleConfig()

    def plan(self, analysis: AudioAnalysis, beatmap: Beatmap) -> RuleLightshow:
        return rule_lightshow(
            analysis, beatmap, time_offset_s=self.time_offset_s, config=self.config
        )

    def generate(
        self, analysis: AudioAnalysis, beatmap: Beatmap, style: LightingStyle
    ) -> Lightshow:
        return render(self.plan(analysis, beatmap), style, beatmap.environment)


def render(plan: RuleLightshow, style: LightingStyle, environment: str) -> Lightshow:
    """Render a planned lightshow in ``style`` for ``environment``."""
    if style not in STYLES:
        raise ValueError(f"unknown lighting style {style!r}; expected one of {STYLES}")
    profile = get_profile(environment)
    if style == "v3":
        if profile is not None:
            return render_v3(plan.lightshow, profile)
        log.warning("no v3 lighting profile for %s; writing vanilla events instead", environment)
        return render_vanilla(plan.lightshow)
    if profile is not None:
        log.warning(
            "%s uses group lighting and ignores most basic events; use a legacy "
            "environment for %s lighting",
            environment,
            style,
        )
    if style == "chroma":
        return render_chroma(plan.lightshow, plan.sections)
    return render_vanilla(plan.lightshow)


def environment_for(style: LightingStyle, current: str, requested: str | None = None) -> str:
    """The environment a map should use for ``style``.

    ``requested`` wins when it suits the style. Otherwise ``current`` is kept when it
    suits, else the default for the style: a profiled group-lighting environment for v3,
    a legacy environment for vanilla and Chroma.
    """
    for candidate in (requested, current):
        if candidate is None:
            continue
        has_profile = get_profile(candidate) is not None
        if has_profile == (style == "v3"):
            return candidate
        if candidate == requested:
            log.warning("environment %s does not suit %s lighting", candidate, style)
    return DEFAULT_V3_ENVIRONMENT if style == "v3" else DEFAULT_LEGACY_ENVIRONMENT


def light_beatmap(
    beatmap: Beatmap,
    analysis: AudioAnalysis,
    style: LightingStyle = "v3",
    *,
    environment: str | None = None,
    generator: RuleLightingGenerator | None = None,
) -> Beatmap:
    """Pick the environment for ``style``, generate the lightshow and store both on
    ``beatmap`` (shared v4-style lightshow; difficulties keep ``lightshow=None``)."""
    generator = generator or RuleLightingGenerator()
    beatmap.environment = environment_for(style, beatmap.environment, environment)
    beatmap.lightshow = generator.generate(analysis, beatmap, style)
    return beatmap
