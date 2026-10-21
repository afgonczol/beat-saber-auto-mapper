"""Section palettes.

Every section gets a palette. Vanilla and v3 output can only pick between the
environment's primary, secondary and white, so there a palette decides which of those
leads (``primary``/``secondary`` swap from section to section). Chroma uses the RGB values.
"""

from __future__ import annotations

from dataclasses import dataclass

RGB = tuple[float, float, float]

# Section moods. Labels from bsmapper.audio are hints, so energy decides as well.
CALM = "calm"
MID = "mid"
HIGH = "high"


@dataclass(frozen=True, slots=True)
class Palette:
    name: str
    mood: str
    left: RGB  # replaces the red/primary colour
    right: RGB  # replaces the blue/secondary colour
    accent: RGB  # replaces white


PALETTES: tuple[Palette, ...] = (
    Palette("dusk", CALM, (0.55, 0.25, 0.85), (0.15, 0.45, 0.95), (0.85, 0.80, 1.00)),
    Palette("ocean", CALM, (0.10, 0.65, 0.75), (0.15, 0.30, 0.90), (0.75, 0.95, 1.00)),
    Palette("forest", CALM, (0.30, 0.80, 0.40), (0.10, 0.55, 0.60), (0.90, 1.00, 0.85)),
    Palette("classic", MID, (0.90, 0.15, 0.20), (0.15, 0.50, 1.00), (1.00, 1.00, 1.00)),
    Palette("sunset", MID, (1.00, 0.45, 0.10), (0.85, 0.15, 0.55), (1.00, 0.90, 0.70)),
    Palette("mint", MID, (0.20, 0.95, 0.65), (0.60, 0.25, 0.95), (0.90, 1.00, 0.95)),
    Palette("inferno", HIGH, (1.00, 0.20, 0.05), (1.00, 0.70, 0.00), (1.00, 0.95, 0.80)),
    Palette("neon", HIGH, (1.00, 0.05, 0.75), (0.00, 0.95, 1.00), (1.00, 1.00, 1.00)),
    Palette("voltage", HIGH, (0.70, 1.00, 0.05), (0.20, 0.35, 1.00), (1.00, 1.00, 0.90)),
)


def palettes_for(mood: str) -> tuple[Palette, ...]:
    return tuple(p for p in PALETTES if p.mood == mood)


def choose_palettes(moods: list[str], seed: int = 0) -> list[Palette]:
    """One palette per section: matching the mood, never the same as the previous section.

    Sections with the same mood cycle through that mood's palettes, so a returning chorus
    gets a fresh look. ``seed`` rotates the starting point of every cycle.
    """
    counters: dict[str, int] = {}
    chosen: list[Palette] = []
    for mood in moods:
        options = palettes_for(mood) or PALETTES
        k = counters.get(mood, 0)
        palette = options[(k + seed) % len(options)]
        if chosen and palette == chosen[-1] and len(options) > 1:
            k += 1
            palette = options[(k + seed) % len(options)]
        counters[mood] = k + 1
        chosen.append(palette)
    return chosen
