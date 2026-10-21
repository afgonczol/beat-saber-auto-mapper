"""Environment profiles for v3 group lighting, loaded from ``profiles/<Environment>.json``.

A profile says which light groups an environment has and how the vanilla channels map
onto them:

- ``groups``: id -> name, light count, colour support, rotation axes (x, y, z);
- ``channels``: vanilla light type (0 back, 1 rings, 2/3 left/right lasers, 4 centre)
  -> the colour groups that play it, with a beat ``wave`` that ripples the hit across
  the group's lights;
- ``spin``: groups that turn on ring spins (type 8) and fan out on ring zooms (type 9);
- ``left_sweep`` / ``right_sweep``: groups that sweep with the laser speed (types 12/13);
  every motion may set ``rest`` (the angle moves return to, default 0 = the environment's
  own pose) and ``max_angle`` (how far sweeps swing either side of rest, default 30);
- ``basic_types``: basic event types the environment still reacts to; these are passed
  through unchanged (boost always is).

Adding an environment is a new JSON file; no code changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any


@dataclass(frozen=True, slots=True)
class LightGroup:
    id: int
    name: str
    count: int
    color: bool
    rotation: tuple[bool, bool, bool]


@dataclass(frozen=True, slots=True)
class Channel:
    groups: tuple[int, ...]
    wave: float = 0.0  # beats from the first to the last light of a group


@dataclass(frozen=True, slots=True)
class Motion:
    groups: tuple[int, ...]
    axis: int  # 0 = x, 1 = y, 2 = z
    spread: float = 0.0  # degrees between the first and last light
    rest: float = 0.0  # angle every move returns to; 0 is the environment's own pose
    max_angle: float = 30.0  # sweeps swing at most this far either side of rest


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    environment: str  # Info.dat environment name
    name: str
    groups: dict[int, LightGroup]
    channels: dict[int, Channel]
    basic_types: frozenset[int]
    spin: Motion | None = None
    left_sweep: Motion | None = None
    right_sweep: Motion | None = None

    def validate(self) -> list[str]:
        """Problems with the profile; empty when it is consistent."""
        problems = []
        for type_, channel in self.channels.items():
            for gid in channel.groups:
                group = self.groups.get(gid)
                if group is None:
                    problems.append(f"channel {type_}: unknown group {gid}")
                elif not group.color:
                    problems.append(f"channel {type_}: group {gid} has no colour")
        for label, motion in (
            ("spin", self.spin),
            ("left_sweep", self.left_sweep),
            ("right_sweep", self.right_sweep),
        ):
            if motion is None:
                continue
            if not 0 < motion.max_angle <= 90:
                problems.append(f"{label}: max_angle {motion.max_angle} is not in (0, 90]")
            if not 0 <= motion.axis <= 2:
                problems.append(f"{label}: axis {motion.axis} is not 0..2")
                continue
            for gid in motion.groups:
                group = self.groups.get(gid)
                if group is None:
                    problems.append(f"{label}: unknown group {gid}")
                elif not group.rotation[motion.axis]:
                    problems.append(f"{label}: group {gid} cannot rotate on axis {motion.axis}")
        return problems


def _motion(raw: Any) -> Motion | None:
    if not raw:
        return None
    return Motion(
        tuple(raw["groups"]),
        int(raw["axis"]),
        float(raw.get("spread", 0.0)),
        float(raw.get("rest", 0.0)),
        float(raw.get("max_angle", 30.0)),
    )


def parse_profile(data: dict[str, Any]) -> EnvironmentProfile:
    groups = {
        int(gid): LightGroup(
            int(gid), g.get("name", ""), int(g["count"]), bool(g["color"]), tuple(g["rotation"])
        )
        for gid, g in data["groups"].items()
    }
    channels = {
        int(type_): Channel(tuple(c["groups"]), float(c.get("wave", 0.0)))
        for type_, c in data["channels"].items()
    }
    return EnvironmentProfile(
        environment=data["environment"],
        name=data.get("name", data["environment"]),
        groups=groups,
        channels=channels,
        basic_types=frozenset(int(t) for t in data.get("basic_types", {})),
        spin=_motion(data.get("spin")),
        left_sweep=_motion(data.get("left_sweep")),
        right_sweep=_motion(data.get("right_sweep")),
    )


@cache
def load_profiles() -> dict[str, EnvironmentProfile]:
    """Every bundled profile, keyed by Info.dat environment name."""
    profiles = {}
    for entry in resources.files("bsmapper.lighting").joinpath("profiles").iterdir():
        if entry.name.endswith(".json"):
            profile = parse_profile(json.loads(entry.read_text(encoding="utf-8")))
            profiles[profile.environment] = profile
    return profiles


def get_profile(environment: str) -> EnvironmentProfile | None:
    return load_profiles().get(environment)


def available_environments() -> list[str]:
    return sorted(load_profiles())
