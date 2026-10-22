"""ML lighting (L1): basic events from the WS6 lighting decoder, rendered by WS8.

``ModelLightingGenerator`` is an ``interfaces.LightingGenerator``. It decodes a vanilla
lightshow with ``LightSampler``, following the note track of the map's hardest difficulty
(re-encoded after post-processing, so the lights follow the notes the player sees). That
lightshow goes through WS8's ``render``, the same path the L0 rules take: v3 group
lighting (with its rest pose and capped laser sweeps), Chroma (section palettes from the
L0 planner) or vanilla. ``light_with_fallback`` uses the L0 rules when the model fails
or its lightshow is too sparse to be a show.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bsmapper.generation.model import song_features
from bsmapper.interfaces import AudioAnalysis, LightingStyle
from bsmapper.schema import Beatmap, Characteristic, Difficulty, DifficultyMap, Lightshow
from bsmapper.tokenizer import Conditioning, LightingTokenizer, MapTokenizer, notes_per_second

if TYPE_CHECKING:
    from bsmapper.models import MapperModel

log = logging.getLogger(__name__)

MIN_EVENTS_PER_BAR = 0.5  # sparser than this and the L0 rules take over
LIGHT_TYPES = range(5)  # back, ring, left, right, centre lights


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
    ) -> None:
        if not model.cfg.has_lighting:
            raise ValueError("this checkpoint has no lighting decoder")
        self.model = model
        self.time_offset_s = time_offset_s
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        self.max_body_tokens = max_body_tokens

    def basic_lightshow(self, analysis: AudioAnalysis, beatmap: Beatmap) -> Lightshow:
        """The decoded vanilla lightshow (before rendering)."""
        from bsmapper.models import LightSampler, SamplingOptions

        feats = song_features(analysis, beatmap, self.time_offset_s)
        n_windows = feats.shape[0] // 384
        lead = lead_difficulty(beatmap)
        if lead is None:  # a lightshow-only map: follow an empty note track
            empty = DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT_PLUS)
            note_windows = MapTokenizer().encode(empty, n_windows=n_windows)
        else:
            cond = Conditioning.from_map(lead, nps=notes_per_second(lead, beatmap))
            note_windows = MapTokenizer().encode(lead, cond, n_windows=n_windows)
        options = SamplingOptions(
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            max_body_tokens=self.max_body_tokens,
        )
        windows = LightSampler(self.model).generate(feats, note_windows[:n_windows], options)
        return LightingTokenizer().decode(windows)

    def generate(
        self, analysis: AudioAnalysis, beatmap: Beatmap, style: LightingStyle
    ) -> Lightshow:
        from bsmapper.lighting import RuleLightingGenerator, render
        from bsmapper.lighting.rules import RuleLightshow

        basic = self.basic_lightshow(analysis, beatmap)
        check_density(basic, beatmap, analysis.duration_s + self.time_offset_s)
        # Section plans (palettes, boosts) come from the L0 planner; events from the model.
        sections = RuleLightingGenerator(time_offset_s=self.time_offset_s).plan(analysis, beatmap)
        plan = RuleLightshow(basic, sections.sections, sections.end_beat)
        return render(plan, style, beatmap.environment)


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
