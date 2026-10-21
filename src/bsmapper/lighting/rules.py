"""L0 rule baseline: a section-aware vanilla lightshow from an ``AudioAnalysis``.

The output is style-agnostic: vanilla basic events (``events``) plus colour boost, and a
``SectionPlan`` per section with its mood and palette. ``render`` turns that into the
vanilla, v3 group lighting or Chroma style.

Rules, per section:
- section start: ring spin, laser speed from the mood, centre lights in the section's
  lead colour; a ring zoom when the energy jumps or the section is high-energy;
- downbeats: back lights (white on phrase starts in high-energy sections), ring lights;
- onsets: left/right lasers alternate, thinned to a minimum gap that depends on the mood;
- drops (and high-energy sections entered with a big jump): colour boost on for the whole
  section.

Moods (calm/mid/high) come from the section label and its *intensity*: the energy blended
with its min-max rescaled value, so songs whose sections are all loud still get contrast.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from bsmapper.interfaces import AudioAnalysis, Section
from bsmapper.lighting.events import (
    BACK_LASERS,
    CENTER_LIGHTS,
    LASER_SPEED_TYPES,
    LEFT_LASERS,
    LIGHT_TYPES,
    OFF,
    RIGHT_LASERS,
    RING_LIGHTS,
    RING_SPIN,
    RING_ZOOM,
    Action,
    LightColor,
    light_value,
)
from bsmapper.lighting.palettes import CALM, HIGH, MID, Palette, choose_palettes
from bsmapper.schema import BasicEvent, Beatmap, ColorBoostEvent, Lightshow

TICKS_PER_BEAT = 24
CALM_LABELS = frozenset({"intro", "outro", "break"})


@dataclass(slots=True)
class RuleConfig:
    calm_energy: float = 0.4  # below this a section is calm whatever its label
    high_energy: float = 0.7  # at or above this a section is high-energy
    # Section energies are relative to the loudest section, and loud masters squash them
    # together (0.9..1.0). Moods use an intensity that blends the energy with its
    # min-max rescaled value; this is the weight of the rescaled part.
    relative_weight: float = 0.5
    zoom_jump: float = 0.25  # ring zoom when the intensity changes this much
    boost_jump: float = 0.3  # boost a high-energy section entered with this big a jump
    # Minimum gap between laser hits, in beats.
    onset_gap: dict[str, float] = field(default_factory=lambda: {CALM: 1.0, MID: 0.5, HIGH: 0.25})
    laser_speed: dict[str, int] = field(default_factory=lambda: {CALM: 1, MID: 3, HIGH: 6})
    peak_window_s: float = 0.05  # onset peaks are maxima within +- this window
    seed: int = 0  # rotates palette choice


@dataclass(frozen=True, slots=True)
class SectionPlan:
    start: float  # beats
    end: float
    start_s: float  # seconds in the analysed audio
    end_s: float
    label: str
    energy: float  # as analysed
    intensity: float  # energy blended with its song-relative value, 0..1
    mood: str
    palette: Palette
    primary: LightColor  # leads back and centre lights
    secondary: LightColor  # leads ring lights
    boost: bool


@dataclass(slots=True)
class RuleLightshow:
    """A vanilla lightshow plus the section plans that the renderers need."""

    lightshow: Lightshow
    sections: list[SectionPlan]
    end_beat: float


def snap(beat: float) -> float:
    return round(beat * TICKS_PER_BEAT) / TICKS_PER_BEAT


def beat_clock(beatmap: Beatmap, time_offset_s: float = 0.0) -> Callable[[float], float]:
    """Seconds in the analysed audio -> map beats, snapped to the 24-tick grid.

    ``time_offset_s`` is silence prepended to the audio before packaging (for example
    ``BeatGrid.lead_in_s()``), so analysis time t plays at t + offset in the map.
    """

    def to_beat(seconds: float) -> float:
        return snap(beatmap.seconds_to_beat(float(seconds) + time_offset_s))

    return to_beat


def section_mood(label: str, intensity: float, config: RuleConfig) -> str:
    if label == "drop" or intensity >= config.high_energy:
        return HIGH
    if label == "chorus" and intensity >= 0.55:
        return HIGH
    if intensity < config.calm_energy or (label in CALM_LABELS and intensity < 0.55):
        return CALM
    return MID


def intensities(energies: list[float], relative_weight: float = 0.5) -> list[float]:
    """Blend each energy with its min-max rescaled value across the song."""
    if not energies:
        return []
    lo, hi = min(energies), max(energies)
    spread = hi - lo
    rel = [(e - lo) / spread if spread > 1e-6 else e for e in energies]
    w = relative_weight
    return [
        float(np.clip((1 - w) * e + w * r, 0.0, 1.0)) for e, r in zip(energies, rel, strict=True)
    ]


def onset_peaks(
    analysis: AudioAnalysis, t0: float, t1: float, window_s: float = 0.05
) -> list[tuple[float, float]]:
    """(time_s, strength 0..1) of onset peaks in [t0, t1).

    Uses the frame-level ``onset_strength`` feature; without it, every beat is a peak.
    """
    env = analysis.features.get("onset_strength")
    if env is None or len(env) < 3:
        beats = analysis.beat_times[(analysis.beat_times >= t0) & (analysis.beat_times < t1)]
        return [(float(t), 0.5) for t in beats]
    env = np.asarray(env, dtype=np.float64).reshape(len(env), -1).mean(axis=1)
    top = float(env.max())
    if top <= 0:
        return []
    env = env / top
    fr = analysis.frame_rate
    i0 = max(int(np.floor(t0 * fr)), 0)
    i1 = min(int(np.ceil(t1 * fr)), len(env))
    seg = env[i0:i1]
    if seg.size < 3:
        return []
    half = max(int(round(window_s * fr)), 1)
    padded = np.pad(seg, half, constant_values=-1.0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    is_max = (seg >= windows.max(axis=1)) & (seg > 0)
    threshold = max(0.1, float(seg.mean() + 0.5 * seg.std()))
    idx = np.flatnonzero(is_max & (seg >= threshold))
    # Plateaus give several equal maxima; keep the first of each run.
    if idx.size:
        idx = idx[np.concatenate(([True], np.diff(idx) > half))]
    peaks = [((i0 + i) / fr, float(seg[i])) for i in idx]
    return [(t, s) for t, s in peaks if t0 <= t < t1]


def thin_hits(hits: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Keep the strongest hits so that no two are closer than ``min_gap`` beats."""
    kept: list[tuple[float, float]] = []
    for beat, strength in sorted(hits, key=lambda h: (-h[1], h[0])):
        if all(abs(beat - b) >= min_gap - 1e-9 for b, _ in kept):
            kept.append((beat, strength))
    return sorted(kept)


class _Events:
    """Basic events keyed by (beat, type); a later event on the same key wins."""

    def __init__(self) -> None:
        self.events: dict[tuple[float, int], BasicEvent] = {}

    def add(self, beat: float, type_: int, value: int, brightness: float = 1.0) -> None:
        self.events[(beat, type_)] = BasicEvent(beat, type_, value, round(brightness, 3))

    def light(
        self, beat: float, type_: int, color: LightColor, action: Action, brightness: float = 1.0
    ) -> None:
        self.add(beat, type_, light_value(color, action), brightness)

    def sorted(self, end_beat: float) -> list[BasicEvent]:
        return [e for k, e in sorted(self.events.items()) if e.beat <= end_beat]


def plan_sections(
    analysis: AudioAnalysis, to_beat: Callable[[float], float], config: RuleConfig
) -> list[SectionPlan]:
    sections = sorted(analysis.sections, key=lambda s: s.start_s)
    if not sections:  # treat the whole song as one mid-energy section
        sections = [Section(0.0, analysis.duration_s, "verse", 0.5)]
    levels = intensities([float(s.energy) for s in sections], config.relative_weight)
    moods = [
        section_mood(s.label, level, config) for s, level in zip(sections, levels, strict=True)
    ]
    palettes = choose_palettes(moods, seed=config.seed)
    plans = []
    for i, (s, mood, palette) in enumerate(zip(sections, moods, palettes, strict=True)):
        jump = levels[i] - levels[i - 1] if i > 0 else 0.0
        boost = s.label == "drop" or (mood == HIGH and jump >= config.boost_jump)
        primary = LightColor.RED if i % 2 == 0 else LightColor.BLUE
        secondary = LightColor.BLUE if primary == LightColor.RED else LightColor.RED
        plans.append(
            SectionPlan(
                start=to_beat(s.start_s),
                end=to_beat(s.end_s),
                start_s=s.start_s,
                end_s=s.end_s,
                label=s.label,
                energy=float(s.energy),
                intensity=levels[i],
                mood=mood,
                palette=palette,
                primary=primary,
                secondary=secondary,
                boost=boost,
            )
        )
    return plans


def rule_lightshow(
    analysis: AudioAnalysis,
    beatmap: Beatmap,
    *,
    time_offset_s: float = 0.0,
    config: RuleConfig | None = None,
) -> RuleLightshow:
    """Build the L0 vanilla lightshow for ``analysis`` on ``beatmap``'s tempo map."""
    config = config or RuleConfig()
    to_beat = beat_clock(beatmap, time_offset_s)
    plans = plan_sections(analysis, to_beat, config)
    end_beat = to_beat(analysis.duration_s)
    downbeats = sorted({to_beat(t) for t in analysis.downbeat_times})
    bar = float(np.median(np.diff(downbeats))) if len(downbeats) > 1 else 4.0

    ev = _Events()
    boosts: list[ColorBoostEvent] = []
    prev: SectionPlan | None = None
    for plan in plans:
        b0, b1 = plan.start, plan.end
        if b1 <= b0:
            continue
        mood = plan.mood

        # Section start.
        ev.add(b0, RING_SPIN, 0)
        if prev is not None and (
            mood == HIGH or abs(plan.intensity - prev.intensity) >= config.zoom_jump
        ):
            ev.add(b0, RING_ZOOM, 0)
        speed = config.laser_speed[mood] + (1 if plan.intensity >= 0.9 else 0)
        for type_ in LASER_SPEED_TYPES:
            ev.add(b0, type_, speed)
        center_action = Action.FLASH if mood == HIGH else Action.ON
        ev.light(b0, CENTER_LIGHTS, plan.primary, center_action, 0.4 + 0.6 * plan.intensity)
        if mood == CALM:  # calm sections start dark and build from the downbeats
            for type_ in (BACK_LASERS, RING_LIGHTS, LEFT_LASERS, RIGHT_LASERS):
                ev.add(b0, type_, OFF)
        if plan.boost != (prev.boost if prev else False):
            boosts.append(ColorBoostEvent(b0, plan.boost))

        # Downbeats: back and ring lights.
        bars = [d for d in downbeats if b0 <= d < b1]
        for k, d in enumerate(bars):
            if mood == CALM:
                if k % 2 == 0:
                    ev.light(d, BACK_LASERS, plan.primary, Action.FADE, 0.6)
                if k % 4 == 2:
                    ev.light(d, RING_LIGHTS, plan.secondary, Action.FADE, 0.5)
            elif mood == MID:
                ev.light(d, BACK_LASERS, plan.primary, Action.FLASH, 0.8)
                ring = plan.secondary if k % 2 == 0 else plan.primary
                ev.light(d, RING_LIGHTS, ring, Action.FLASH, 0.7)
            else:
                back = LightColor.WHITE if k % 4 == 0 else plan.primary
                ev.light(d, BACK_LASERS, back, Action.FLASH, 1.0)
                ev.light(d, RING_LIGHTS, plan.secondary, Action.FLASH, 1.0)
                half = snap(d + bar / 2)
                if half < b1:
                    ev.light(half, RING_LIGHTS, plan.primary, Action.FLASH, 0.9)

        # Onsets: lasers, alternating sides.
        hits = [
            (to_beat(t), s)
            for t, s in onset_peaks(analysis, plan.start_s, plan.end_s, config.peak_window_s)
        ]
        hits = thin_hits([(b, s) for b, s in hits if b0 <= b < b1], config.onset_gap[mood])
        action = Action.FADE if mood == CALM else Action.FLASH
        for j, (b, strength) in enumerate(hits):
            side = LEFT_LASERS if j % 2 == 0 else RIGHT_LASERS
            color = plan.primary if side == LEFT_LASERS else plan.secondary
            ev.light(b, side, color, action, 0.5 + 0.5 * min(max(strength, 0.0), 1.0))
        prev = plan

    # Song end: everything off.
    for type_ in LIGHT_TYPES:
        ev.add(end_beat, type_, OFF)
    for type_ in LASER_SPEED_TYPES:
        ev.add(end_beat, type_, 0)
    if boosts and boosts[-1].on:
        boosts.append(ColorBoostEvent(end_beat, False))

    lightshow = Lightshow(basic_events=ev.sorted(end_beat), boost_events=boosts)
    return RuleLightshow(lightshow=lightshow, sections=plans, end_beat=end_beat)
