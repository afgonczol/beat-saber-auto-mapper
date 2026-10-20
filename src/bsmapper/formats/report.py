"""What a reader found while parsing a map, beyond the ``Beatmap`` itself."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


class MapFormatError(ValueError):
    """The map cannot be parsed at all (missing or broken Info.dat)."""


@dataclass(slots=True)
class ReadReport:
    """Side information collected while reading a map.

    ``noodle`` / ``mapping_extensions`` mean the map needs those mods to play as
    intended. Objects that only make sense with a mod (off-grid notes, precision
    rotations, Noodle custom data) are dropped from the ``Beatmap`` and counted in
    ``dropped``. The dataset builder should skip maps where ``needs_mods`` is true.
    """

    info_version: str = ""
    # "<Characteristic>/<Difficulty>" -> version string of that difficulty file
    difficulty_versions: dict[str, str] = field(default_factory=dict)
    requirements: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    noodle: bool = False
    mapping_extensions: bool = False
    chroma: bool = False
    # Where BPM changes came from: "none", "events" (v2 type 100 / v3 bpmEvents),
    # "custom" (legacy MMA2/ChroMapper _BPMChanges) or "audio_data" (v4 BPMInfo.dat).
    bpm_source: str = "none"
    dropped: Counter[str] = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    @property
    def needs_mods(self) -> bool:
        return self.noodle or self.mapping_extensions

    def drop(self, reason: str, count: int = 1) -> None:
        if count:
            self.dropped[reason] += count

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
