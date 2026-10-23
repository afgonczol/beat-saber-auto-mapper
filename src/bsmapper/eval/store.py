"""Generated maps of a run, saved so a run can be re-scored without generating again.

``<out>/maps/<system>/<key>.json.gz`` holds one song's difficulties and vanilla
lightshow; style-sweep maps are saved as ``<key>@style<s>.json.gz``. ``SavedSystem``
replays them (``bsmapper evaluate --rescore``), so a metric change costs seconds, not
hours of decoding.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bsmapper.eval.data import EvalSong, GroupKey
from bsmapper.eval.systems import SystemOutput
from bsmapper.schema import (
    Arc,
    BasicEvent,
    Bomb,
    Chain,
    Characteristic,
    Color,
    ColorBoostEvent,
    CutDirection,
    Difficulty,
    DifficultyMap,
    Lightshow,
    Note,
    Obstacle,
    RotationEvent,
)

MAPS_DIR = "maps"


def _file(root: Path, system: str, key: str, style: float | None) -> Path:
    name = key if style is None else f"{key}@style{style:g}"
    return root / MAPS_DIR / system.replace("/", "_") / f"{name}.json.gz"


def dmap_to_dict(dmap: DifficultyMap) -> dict[str, Any]:
    d = asdict(dmap)
    d["lightshow"] = None
    return d


def dmap_from_dict(d: dict[str, Any]) -> DifficultyMap:
    def notes(items):
        return [
            Note(**{**n, "color": Color(n["color"]), "direction": CutDirection(n["direction"])})
            for n in items
        ]

    def arcs(items):
        return [
            Arc(
                **{
                    **a,
                    "color": Color(a["color"]),
                    "direction": CutDirection(a["direction"]),
                    "tail_direction": CutDirection(a["tail_direction"]),
                }
            )
            for a in items
        ]

    def chains(items):
        return [
            Chain(**{**c, "color": Color(c["color"]), "direction": CutDirection(c["direction"])})
            for c in items
        ]

    return DifficultyMap(
        characteristic=Characteristic(d["characteristic"]),
        difficulty=Difficulty(d["difficulty"]),
        njs=d["njs"],
        njs_offset=d["njs_offset"],
        notes=notes(d["notes"]),
        bombs=[Bomb(**b) for b in d["bombs"]],
        obstacles=[Obstacle(**o) for o in d["obstacles"]],
        arcs=arcs(d["arcs"]),
        chains=chains(d["chains"]),
        rotations=[RotationEvent(**r) for r in d["rotations"]],
        label=d.get("label"),
    )


def lightshow_to_dict(ls: Lightshow) -> dict[str, Any]:
    return {
        "basic_events": [asdict(e) for e in ls.basic_events],
        "boost_events": [asdict(e) for e in ls.boost_events],
        "uses_chroma": ls.uses_chroma,
    }


def lightshow_from_dict(d: dict[str, Any]) -> Lightshow:
    basic = []
    for e in d["basic_events"]:
        color = tuple(e["color"]) if e.get("color") is not None else None
        basic.append(BasicEvent(**{**e, "color": color}))
    boosts = [ColorBoostEvent(**b) for b in d["boost_events"]]
    return Lightshow(basic, boosts, [], d.get("uses_chroma", False))


def save_output(
    root: Path, system: str, key: str, out: SystemOutput, style: float | None = None
) -> Path:
    path = _file(root, system, key, style)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "dmaps": [dmap_to_dict(d) for d in out.dmaps.values()],
        "lightshow": lightshow_to_dict(out.lightshow) if out.lightshow else None,
        "lighting_source": out.lighting_source,
        "seconds": out.seconds,
        "notes": out.notes,
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    return path


def load_output(path: Path) -> SystemOutput:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    dmaps: dict[GroupKey, DifficultyMap] = {}
    for d in doc["dmaps"]:
        dm = dmap_from_dict(d)
        dmaps[(dm.characteristic, dm.difficulty)] = dm
    ls = lightshow_from_dict(doc["lightshow"]) if doc.get("lightshow") else None
    return SystemOutput(
        dmaps, ls, doc.get("lighting_source", ""), doc.get("seconds", 0.0), doc.get("notes", [])
    )


def saved_systems(root: Path) -> list[str]:
    folder = root / MAPS_DIR
    return sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []


class SavedSystem:
    """Replays a system's saved maps (from ``root``'s earlier run) instead of generating."""

    kind = "saved"

    def __init__(self, root: Path, name: str) -> None:
        self.root, self.name = root, name

    def run(self, song: EvalSong, analysis, groups, style=None, lighting=True) -> SystemOutput:
        path = _file(self.root, self.name, song.key, style)
        if not path.is_file():
            raise FileNotFoundError(f"no saved maps for {self.name} on {song.key} ({path.name})")
        out = load_output(path)
        out.dmaps = {g: d for g, d in out.dmaps.items() if g in set(groups)}
        if not lighting:
            out.lightshow = None
        return out

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "generator": "saved maps", "path": str(self.root / MAPS_DIR)}
