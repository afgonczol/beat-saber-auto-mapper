"""ML lighting (L1): basic events from the WS6 lighting decoder, rendered by WS8.

``ModelLightingGenerator`` is an ``interfaces.LightingGenerator``. It decodes a vanilla
lightshow with ``LightSampler``, following the note track of the map's hardest difficulty
(re-encoded after post-processing, so the lights follow the notes the player sees). That
lightshow goes through WS8's ``render``, the same path the L0 rules take: v3 group
lighting (with its rest pose and capped laser sweeps), Chroma (section palettes from the
L0 planner) or vanilla. ``light_with_fallback`` uses the L0 rules when the model fails
or its lightshow is too sparse to be a show.

Readability limits (``lighting.readability``) apply as a post-filter on the decoded
events (holds, colour holds, moment caps, motion gaps). The decode-time mask
(``ReadabilityMask``) is opt-in (``ReadabilityConfig.decode_mask``). Section
intensities from the L0 planner set how much change each section allows. Loud 16-beat
windows the model leaves dark get the L0 rule lightshow's light events.
``readability=None`` turns all of this off.

A checkpoint whose lighting decoder has the v2 light prompt (WS8b) is also given a
lighting style and a section-level scene plan per window:

* ``light_style``: ``"auto"`` draws a style from the checkpoint's auto-pick for this
  song (seeded, so re-rolls vary), ``"none"`` leaves it to the model (``LSTYLE_NONE``),
  or a style name (``lighting.styles.STYLE_PROTOTYPES``: pulse, calm, scenes, flash,
  accent);
* scenes come from ``lighting.scenes.plan_scenes`` over the L0 planner's sections;
* light reference blocks repeat the generated lights of the span each window's music
  repeats (the same structure references the notes use).

v1 checkpoints (note_v1) ignore all three and decode as before.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bsmapper.generation.model import reference_starts, song_features
from bsmapper.interfaces import AudioAnalysis, LightingStyle
from bsmapper.lighting.events import LightColor
from bsmapper.lighting.readability import (
    IntensityAt,
    ReadabilityConfig,
    ReadabilityMask,
    limit_lightshow,
    section_intensity,
)
from bsmapper.lighting.scenes import SectionSpan, plan_scenes
from bsmapper.lighting.styles import StyleModel, song_vector, window_loudness
from bsmapper.schema import Beatmap, Characteristic, Difficulty, DifficultyMap, Lightshow
from bsmapper.tokenizer import (
    LIGHT_GRAMMAR,
    Conditioning,
    LightingTokenizer,
    LightPrompt,
    MapTokenizer,
    notes_per_second,
    version_for_vocab,
)

if TYPE_CHECKING:
    from bsmapper.lighting.rules import RuleLightshow
    from bsmapper.models import MapperModel

log = logging.getLogger(__name__)

MIN_EVENTS_PER_BAR = 0.5  # sparser than this and the L0 rules take over
LIGHT_TYPES = range(5)  # back, ring, left, right, centre lights
DEFAULT_READABILITY = ReadabilityConfig()
LIGHT_STYLE_CHOICES = ("auto", "none", "pulse", "calm", "scenes", "flash", "accent")
MAX_LIGHT_REF_TOKENS = 384  # as in training (DataConfig.max_light_ref_tokens)


def lead_difficulty(beatmap: Beatmap) -> DifficultyMap | None:
    """The note track lights follow: the hardest Standard difficulty, else the hardest."""
    diffs = [d for d in beatmap.difficulties if d.notes]
    if not diffs:
        return None
    standard = [d for d in diffs if d.characteristic == Characteristic.STANDARD]
    return max(standard or diffs, key=lambda d: d.difficulty.rank)


class ModelLightingGenerator:
    """L1 lighting. ``time_offset_s`` is the silence prepended to the analysed audio."""

    name = "model"

    def __init__(
        self,
        model: MapperModel,
        *,
        time_offset_s: float = 0.0,
        temperature: float = 0.9,
        top_p: float = 0.95,
        seed: int = 0,
        max_body_tokens: int = 1024,
        readability: ReadabilityConfig | None = DEFAULT_READABILITY,
        light_style: str = "auto",
        style_guidance: float = 1.0,
        references: bool = True,
    ) -> None:
        if not model.cfg.has_lighting:
            raise ValueError("this checkpoint has no lighting decoder")
        self.model = model
        self.time_offset_s = time_offset_s
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        self.max_body_tokens = max_body_tokens
        self.readability = readability
        self.light_style = light_style
        self.style_guidance = style_guidance
        self.references = references
        styles = getattr(model, "light_styles", None)
        self.styles = StyleModel.from_dict(styles) if styles else None
        if light_style not in ("auto", "none") and self.styles is not None:
            self.styles.index(light_style)  # an unknown name fails here, not mid-song
        self.last_style: str | None = None  # the style the last lightshow was asked for

    def basic_lightshow(self, analysis: AudioAnalysis, beatmap: Beatmap) -> Lightshow:
        """The decoded vanilla lightshow (before rendering and the readability post-filter).

        With ``readability.decode_mask``, the decoder runs under ``ReadabilityMask``.
        """
        from bsmapper.models import LightSampler, SamplingOptions

        feats = song_features(analysis, beatmap, self.time_offset_s)
        n_windows = feats.shape[0] // 384
        lead = lead_difficulty(beatmap)
        if lead is None:  # a lightshow-only map: follow an empty note track
            lead = DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT_PLUS)
            cond = Conditioning.from_map(lead)
        else:
            cond = Conditioning.from_map(lead, nps=notes_per_second(lead, beatmap))
        note_windows = MapTokenizer().encode(lead, cond, n_windows=n_windows)
        # The note track's prefix must be in the checkpoint's layout (note_v0 has no
        # INTENSITY token; its vocabulary would not even have the id).
        prefix = cond.prefix_ids(version=version_for_vocab(self.model.cfg.note_vocab))
        for w in note_windows:
            w.prefix = list(prefix)
        options = SamplingOptions(
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            max_body_tokens=self.max_body_tokens,
        )
        mask = None
        if self.readability is not None and self.readability.decode_mask:
            intensity_at = self.intensity_at(analysis, beatmap)
            mask = ReadabilityMask(beatmap.beat_to_seconds, intensity_at, self.readability)
        sampler = LightSampler(self.model)
        extra: dict = {}
        if sampler.supports_prompts:
            extra = self.prompt_plan(analysis, beatmap, feats)
        windows = sampler.generate(feats, note_windows[:n_windows], options, mask_fn=mask, **extra)
        return LightingTokenizer().decode(windows)

    def choose_style(self, feats, bpm: float) -> int | None:
        """The style index for this song, or None for ``LSTYLE_NONE``."""
        if self.styles is None or self.light_style == "none":
            return None
        if self.light_style == "auto":
            return self.styles.pick(song_vector(feats, bpm), seed=self.seed)
        return self.styles.index(self.light_style)

    def prompt_plan(self, analysis: AudioAnalysis, beatmap: Beatmap, feats) -> dict:
        """``LightSampler.generate`` keyword arguments for a v2 lighting decoder: one
        ``LightPrompt`` per window (style + section scene) and the light references."""
        from bsmapper.lighting import RuleLightingGenerator
        from bsmapper.models import WindowRefs

        n_windows = feats.shape[0] // 384
        style = self.choose_style(feats, beatmap.bpm)
        self.last_style = "none" if style is None else self.styles.names[style]  # type: ignore[union-attr]
        refs = reference_starts(analysis, beatmap, feats) if self.references else None
        prompts = [LightPrompt(style)] * n_windows
        if self.styles is not None:
            plan = RuleLightingGenerator(time_offset_s=self.time_offset_s).plan(analysis, beatmap)
            spans = [
                SectionSpan(s.start, s.end, s.primary == LightColor.RED) for s in plan.sections
            ]
            scenes = plan_scenes(
                self.styles, style, window_loudness(feats, n_windows), spans, refs, self.seed
            )
            prompts = [
                p if sc is None else p.with_scene(sc) for p, sc in zip(prompts, scenes, strict=True)
            ]
        log.info("light style %s, %d windows", self.last_style, n_windows)
        out: dict = {"prompts": prompts, "style_guidance": self.style_guidance}
        if refs is not None:
            out["ref_fn"] = WindowRefs(refs, MAX_LIGHT_REF_TOKENS, LIGHT_GRAMMAR)
        return out

    def intensity_at(self, analysis: AudioAnalysis, beatmap: Beatmap) -> IntensityAt:
        """Section intensity per map beat, from the L0 planner's sections."""
        from bsmapper.lighting import RuleLightingGenerator

        return _intensity(
            RuleLightingGenerator(time_offset_s=self.time_offset_s).plan(analysis, beatmap)
        )

    def generate(
        self, analysis: AudioAnalysis, beatmap: Beatmap, style: LightingStyle
    ) -> Lightshow:
        from bsmapper.lighting import RuleLightingGenerator, render
        from bsmapper.lighting.rules import RuleLightshow

        basic = self.basic_lightshow(analysis, beatmap)
        check_density(basic, beatmap, analysis.duration_s + self.time_offset_s)
        # Section plans (palettes, boosts, intensities) come from the L0 planner; events
        # from the model.
        sections = RuleLightingGenerator(time_offset_s=self.time_offset_s).plan(analysis, beatmap)
        if self.readability is not None:
            basic, report = limit_lightshow(
                basic,
                beatmap.beat_to_seconds,
                _intensity(sections),
                self.readability,
                fallback=sections.lightshow,
            )
            log.info("readability filter: %s", report.summary())
        plan = RuleLightshow(basic, sections.sections, sections.end_beat)
        return render(plan, style, beatmap.environment)


def configured_light_style(path=None) -> str:
    """``[generate] light_style`` from ``~/.bsmapper/config.toml``, else ``"auto"``."""
    import tomllib

    from bsmapper.generation.install import config_path

    cfg = path or config_path()
    if not cfg.is_file():
        return "auto"
    with cfg.open("rb") as f:
        value = tomllib.load(f).get("generate", {}).get("light_style")
    if value is None:
        return "auto"
    value = str(value).lower()
    if value not in LIGHT_STYLE_CHOICES:
        raise ValueError(
            f"{cfg}: [generate] light_style must be one of {', '.join(LIGHT_STYLE_CHOICES)}"
        )
    return value


def _intensity(plan: RuleLightshow) -> IntensityAt:
    return section_intensity([s.start for s in plan.sections], [s.intensity for s in plan.sections])


class SparseLightshowError(ValueError):
    pass


def check_density(lightshow: Lightshow, beatmap: Beatmap, duration_s: float) -> None:
    """Raise ``SparseLightshowError`` if there are too few light events to be a show."""
    lights = [e for e in lightshow.basic_events if e.type in LIGHT_TYPES]
    bars = max(beatmap.seconds_to_beat(duration_s) / 4.0, 1.0)
    if len(lights) / bars < MIN_EVENTS_PER_BAR:
        raise SparseLightshowError(
            f"model lightshow has {len(lights)} light events over {bars:.0f} bars"
        )
