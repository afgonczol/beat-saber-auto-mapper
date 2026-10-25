"""Group lighting from human effects (WS8c).

The light model (and the L0 rules) decide *when* lights hit and how busy each section
is, but they only speak the five basic channels. ``render_v3`` turns those into one
fixed box-group template, so every song looked alike. Human group-lighting maps use
much more: subsets and steps of a group, chases across its lights, white, rotation on
several axes, translations, and each environment's own geometry.

This module re-uses human group lighting instead. ``effects_build`` stores the human
lightshows of each group-lighting environment with per-bar descriptors. At generation:

1. ``song_targets``: per bar of the song, a target activity (the section's intensity
   blended with how busy the generated basic lightshow is there, both song-relative)
   and a 16-step rhythm (the generated light events' onsets).
2. ``phrases``: each section is cut into phrases of up to ``PHRASE_BARS`` bars.
3. ``plan_effects``: per phrase, the human passage (source map, start bar) of the same
   length whose activity and rhythm best match, sampled among the best few with the
   seed. A phrase prefers to continue the passage of the phrase before it in the same
   section, and sources the song already uses, so a song keeps the vocabulary of a few
   mappers the way a human lightshow keeps its mapper's, but avoids bars it already
   played; a section that repeats an earlier one replays that section's passages. The
   light style shifts the activity target and the preference for motion and white.
4. ``render_effects``: copies each passage's box groups, basic events and colour boosts
   to its phrase. At each splice the source's state at the cut (the lights, angles and
   positions its earlier events left) is re-created, and whatever the previous passage
   left on and the new source does not set is reset, so a passage looks the way it did
   in the human map.

Bars are ``BAR_BEATS`` beats from beat 0, which the beat grid puts on a downbeat.
"""

from __future__ import annotations

import bisect
import gzip
import json
import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bsmapper.lighting.events import (
    LIGHT_TYPES,
    Action,
    LightColor,
    decode_value,
    light_value,
)
from bsmapper.schema import BasicEvent, ColorBoostEvent, EventBoxGroup, Lightshow

log = logging.getLogger(__name__)

EFFECTS_VERSION = 1
INDEX_FILE = "index.json"
BAR_BEATS = 4.0
RHYTHM_STEPS = 16  # 16th notes in a 4/4 bar
# Basic events that light something in group-lighting environments, and the ones whose
# last value is a state (lights and laser speeds) rather than a trigger.
LIGHT_BASIC_TYPES = frozenset({0, 1, 2, 3, 4, 6, 7, 10, 11})
STATE_BASIC_TYPES = LIGHT_BASIC_TYPES | {12, 13}
SKIP_BASIC_TYPES = frozenset({5, 14, 15, 100})  # boost, lane rotations, BPM

PHRASE_BARS = 8
MIN_PHRASE_BARS = 2  # a shorter remainder joins the phrase before it
CARRY_GLIDE = 0.25  # beats to glide rotations / translations into a passage's state
MAX_CARRY = 8  # box groups replayed per light group to rebuild its state
TOP_K = 12
TEMPERATURE = 0.03
W_RHYTHM = 0.3
CONTINUE_BONUS = 0.04
NEW_SOURCE_PENALTY = 0.04  # a source not used yet in the song: keep a song to a few
REPLAY_PENALTY = 0.08  # per share of the passage already played elsewhere in the song
SILENT_RHYTHM = 0.5  # rhythm similarity for a target bar without light events
SWAP_RATIO = 1.5  # a passage leads with a colour when it has this much more of it
AUTO_MIN_MAPS = 30  # environments picked by default need this many source maps

# Light style -> (activity shift, motion weight, white weight).
STYLE_BIAS: dict[str, tuple[float, float, float]] = {
    "pulse": (0.0, 0.0, 0.0),
    "calm": (-0.15, -0.1, 0.0),
    "scenes": (-0.05, -0.05, 0.0),
    "flash": (0.15, 0.1, 0.0),
    "accent": (0.0, 0.0, 0.15),
}

_VALUE_KEY = {"rotation": "r", "translation": "t", "fx": "v"}
_EASE_KEY = {"rotation": "e", "translation": "e", "fx": "i"}
_EVENTS_KEY = {"color": "e", "rotation": "l", "translation": "l", "fx": "l"}
_OFF_FILTER = {"c": 0, "f": 1, "p": 1, "t": 0, "r": 0, "n": 0, "s": 0, "l": 0.0, "d": 0}


# --- the library ------------------------------------------------------------------


def _ranks(values: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value in the sorted ``pool`` (0..1)."""
    if not len(pool):
        return np.zeros(len(values))
    lo = np.searchsorted(pool, values, side="left")
    hi = np.searchsorted(pool, values, side="right")
    return (lo + hi) / (2.0 * len(pool))


def _unit_rows(m: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norm, out=np.zeros_like(m), where=norm > 0)


@dataclass(slots=True)
class SourceMap:
    key: str
    mapper: str
    song: str
    bars: int
    act: np.ndarray  # activity percentile in the environment's pool, per bar
    motion: np.ndarray  # motion percentile
    white: np.ndarray  # share of lit colour events that are white
    red: np.ndarray  # lit red / blue colour events
    blue: np.ndarray
    rhythm: np.ndarray  # (bars, RHYTHM_STEPS), unit rows (zero rows for silent bars)


class EffectLibrary:
    """One environment's human group lighting (``effects_build`` writes it)."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        doc = json.loads((self.folder / INDEX_FILE).read_text(encoding="utf-8"))
        if doc.get("version") != EFFECTS_VERSION:
            raise ValueError(f"{self.folder}: effect library version {doc.get('version')}")
        self.environment: str = doc["environment"]
        rows = doc["maps"]
        acts = [np.asarray(m["act"], dtype=float) for m in rows]
        motions = [np.asarray(m["motion"], dtype=float) for m in rows]
        act_pool = np.sort(np.concatenate(acts)) if acts else np.zeros(0)
        motion_pool = np.sort(np.concatenate(motions)) if motions else np.zeros(0)
        self.maps: list[SourceMap] = []
        for m, act, motion in zip(rows, acts, motions, strict=True):
            white = np.asarray(m["white"], dtype=float)
            red = np.asarray(m["red"], dtype=float)
            blue = np.asarray(m["blue"], dtype=float)
            lit = white + red + blue
            self.maps.append(
                SourceMap(
                    key=m["key"],
                    mapper=m.get("mapper", ""),
                    song=m.get("song", ""),
                    bars=int(m["bars"]),
                    act=_ranks(act, act_pool),
                    motion=_ranks(motion, motion_pool),
                    white=np.divide(white, lit, out=np.zeros_like(white), where=lit > 0),
                    red=red,
                    blue=blue,
                    rhythm=_unit_rows(
                        np.asarray(m["rhythm"], dtype=float).reshape(-1, RHYTHM_STEPS)
                    ),
                )
            )
        self._sources: dict[int, _Source] = {}

    def source(self, i: int) -> _Source:
        if i not in self._sources:
            path = self.folder / f"{self.maps[i].key}.json.gz"
            with gzip.open(path, "rt", encoding="utf-8") as f:
                self._sources[i] = _Source(json.load(f))
        return self._sources[i]


def available_libraries(root: Path | None) -> dict[str, Path]:
    """Environment -> library folder for every library under ``root``."""
    if root is None or not Path(root).is_dir():
        return {}
    out = {}
    for folder in sorted(Path(root).iterdir()):
        if (folder / INDEX_FILE).is_file():
            out[folder.name] = folder
    return out


class _Source:
    """A source map's events with indexes for slicing and for its state at a beat."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.groups: list[list] = data["groups"]  # [beat, kind, gid, boxes], by beat
        self.basic: list[list] = data["basic"]  # [beat, type, value, float], by beat
        self.boost: list[list] = data["boost"]  # [beat, on], by beat
        self.group_beats = [g[0] for g in self.groups]
        self.basic_beats = [e[0] for e in self.basic]
        self.boost_beats = [e[0] for e in self.boost]
        self.by_light: dict[tuple[str, int], list[int]] = defaultdict(list)
        for i, (_b, kind, gid, _boxes) in enumerate(self.groups):
            self.by_light[(kind, gid)].append(i)
        self.basic_by_type: dict[int, list[int]] = defaultdict(list)
        for i, e in enumerate(self.basic):
            self.basic_by_type[e[1]].append(i)

    def groups_between(self, a: float, z: float) -> list[list]:
        return self.groups[
            bisect.bisect_left(self.group_beats, a) : bisect.bisect_left(self.group_beats, z)
        ]

    def basic_between(self, a: float, z: float) -> list[list]:
        return self.basic[
            bisect.bisect_left(self.basic_beats, a) : bisect.bisect_left(self.basic_beats, z)
        ]

    def boost_between(self, a: float, z: float) -> list[list]:
        return self.boost[
            bisect.bisect_left(self.boost_beats, a) : bisect.bisect_left(self.boost_beats, z)
        ]

    def light_state(self, beat: float) -> dict[tuple[str, int], list[dict]]:
        """Boxes that rebuild every light group's state just before ``beat``."""
        out = {}
        for light, idx in self.by_light.items():
            k = bisect.bisect_left([self.groups[i][0] for i in idx], beat)
            boxes: list[dict] = []
            seen: set[int] = set()  # axes (0 for colour) set by the boxes so far
            covered: set[int] = set()  # axes some box sets for every light
            for i in reversed(idx[max(0, k - MAX_CARRY) : k]):
                group_boxes = self.groups[i][3]
                boxes[:0] = [b for b in (_state_box(light[0], box) for box in group_boxes) if b]
                for box in group_boxes:
                    seen.add(box.get("a", 0))
                    if _covers_all(box["f"]):
                        covered.add(box.get("a", 0))
                if seen <= covered:
                    break
            if boxes:
                out[light] = boxes
        return out

    def basic_state(self, beat: float) -> dict[int, list]:
        """Last value of each state basic type before ``beat``: {type: [type, value, float]}."""
        out = {}
        for type_, idx in self.basic_by_type.items():
            if type_ not in STATE_BASIC_TYPES:
                continue
            k = bisect.bisect_left([self.basic[i][0] for i in idx], beat)
            if k:
                _b, t, value, f = self.basic[idx[k - 1]]
                out[type_] = [t, _settled(t, value), f]
        return out

    def boost_state(self, beat: float) -> bool | None:
        k = bisect.bisect_left(self.boost_beats, beat)
        return bool(self.boost[k - 1][1]) if k else None


def _covers_all(f: dict) -> bool:
    """The filter selects every light of the group."""
    full_limit = f["l"] == 0 or f["l"] >= 1
    division = f["f"] == 1 and f["p"] <= 1
    step = f["f"] == 2 and f["p"] == 0 and f["t"] == 1
    return (division or step) and full_limit


def _state_box(kind: str, box: dict) -> dict | None:
    """``box`` reduced to the state it leaves: one instant colour event, or a short glide
    to the final angle / position / fx value. None when it only extends earlier events."""
    if kind == "color":
        events = [e for e in box["e"] if e["i"] != 2]
        if not events:
            return None
        last = events[-1]
        event = {**last, "b": 0.0, "i": 0}
        return {**box, "w": 0.0, "b": 1 if box["r"] else 0, "e": [event]}
    events = [e for e in box["l"] if e["p"] == 0]
    if not events:
        return None
    last = dict(events[-1])
    if kind == "rotation":
        last.update(l=0, o=0)
    hold = {**last, "b": 0.0, "p": 1}
    glide = {**last, "b": CARRY_GLIDE, "p": 0, _EASE_KEY[kind]: 0}
    return {**box, "w": 0.0, "l": [hold, glide]}


def _settled(type_: int, value: int) -> int:
    """The value a basic light event leaves the light in: flashes and transitions settle
    on, fades end dark."""
    if type_ not in LIGHT_BASIC_TYPES:
        return value
    decoded = decode_value(value)
    if decoded is None:
        return 0
    color, action = decoded
    return 0 if action == Action.FADE else light_value(color, Action.ON)


def _reset_boxes(kind: str) -> list[dict]:
    """Boxes that turn every light of a group off, or glide it back to rest on every axis."""
    if kind == "color":
        event = {"b": 0.0, "i": 0, "c": 0, "s": 0.0, "f": 0, "sb": 0.0, "sf": 0}
        box = {"f": dict(_OFF_FILTER), "w": 0.0, "d": 1, "r": 0.0, "t": 1, "b": 0, "i": 0}
        return [{**box, "e": [event]}]
    value, ease = _VALUE_KEY[kind], _EASE_KEY[kind]
    hold = {"b": 0.0, "p": 1, value: 0.0, ease: 0}
    glide = {"b": CARRY_GLIDE, "p": 0, value: 0.0, ease: 0}
    if kind == "rotation":
        hold.update(l=0, o=0)
        glide.update(l=0, o=0)
    box = {"f": dict(_OFF_FILTER), "w": 0.0, "d": 1, "s": 0.0, "t": 1, "b": 0, "i": 0}
    if kind == "fx":
        return [{**box, "l": [dict(hold), dict(glide)]}]
    return [{**box, "a": axis, "r": 0, "l": [dict(hold), dict(glide)]} for axis in range(3)]


# --- planning ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SectionInfo:
    start: float  # beats
    end: float
    intensity: float  # 0..1
    lead: LightColor | None = None  # the colour the section leads with


@dataclass(frozen=True, slots=True)
class Phrase:
    section: int
    start_bar: int
    bars: int


@dataclass(frozen=True, slots=True)
class Placement:
    start_bar: int  # in the song
    bars: int
    source: int  # index into EffectLibrary.maps
    src_bar: int
    swap: bool = False  # swap red and blue


def song_targets(
    lightshow: Lightshow, sections: Sequence[SectionInfo], n_bars: int
) -> tuple[np.ndarray, np.ndarray]:
    """(activity target per bar, rhythm per bar as unit rows) for a generated song."""
    counts = np.zeros(n_bars)
    rhythm = np.zeros((n_bars, RHYTHM_STEPS))
    for e in lightshow.basic_events:
        if e.type not in LIGHT_TYPES:
            continue
        bar = int(math.floor(e.beat / BAR_BEATS + 1e-9))
        if 0 <= bar < n_bars:
            counts[bar] += 1
            step = int(round((e.beat / BAR_BEATS - bar) * RHYTHM_STEPS)) % RHYTHM_STEPS
            rhythm[bar, step] += 1
    busy = _ranks(counts, np.sort(counts))
    intensity = np.full(n_bars, 0.5)
    for s in sections:
        a = max(0, int(round(s.start / BAR_BEATS)))
        z = min(n_bars, int(round(s.end / BAR_BEATS)))
        intensity[a:z] = s.intensity
    return 0.5 * busy + 0.5 * intensity, _unit_rows(rhythm)


def phrases(sections: Sequence[SectionInfo], n_bars: int) -> list[Phrase]:
    """Cut the song's bars into phrases of up to ``PHRASE_BARS`` bars per section."""
    starts = sorted({min(n_bars, max(0, int(round(s.start / BAR_BEATS)))) for s in sections})
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    owner = {}
    for i, s in enumerate(sections):
        owner.setdefault(min(n_bars, max(0, int(round(s.start / BAR_BEATS)))), i)
    out: list[Phrase] = []
    bounds = [*starts, n_bars]
    for a, z in zip(bounds, bounds[1:], strict=False):
        if z <= a:
            continue
        section = owner.get(a, out[-1].section if out else 0)
        sizes = [PHRASE_BARS] * ((z - a) // PHRASE_BARS)
        rest = (z - a) % PHRASE_BARS
        if rest >= MIN_PHRASE_BARS or not sizes:
            sizes.append(rest)
        else:
            sizes[-1] += rest
        bar = a
        for size in sizes:
            out.append(Phrase(section, bar, size))
            bar += size
    return out


def _lead_of(src: SourceMap, a: int, n: int) -> LightColor | None:
    red, blue = src.red[a : a + n].sum(), src.blue[a : a + n].sum()
    if red > SWAP_RATIO * blue:
        return LightColor.RED
    if blue > SWAP_RATIO * red:
        return LightColor.BLUE
    return None


def _costs(
    src: SourceMap, act: np.ndarray, rhythm: np.ndarray, bias: tuple[float, float, float]
) -> np.ndarray:
    """Cost of every start bar of ``src`` for a phrase with these targets."""
    n = len(act)
    length = src.bars - n + 1
    if length <= 0:
        return np.zeros(0)
    cost = np.zeros(length)
    sim = np.zeros(length)
    motion = np.zeros(length)
    white = np.zeros(length)
    silent_target = np.linalg.norm(rhythm, axis=1) == 0
    for j in range(n):
        cost += (src.act[j : j + length] - act[j]) ** 2
        # A bar the generated lights leave empty says nothing about rhythm.
        sim += SILENT_RHYTHM if silent_target[j] else src.rhythm[j : j + length] @ rhythm[j]
        motion += src.motion[j : j + length]
        white += src.white[j : j + length]
    _, w_motion, w_white = bias
    return cost / n + W_RHYTHM * (1.0 - sim / n) - w_motion * motion / n - w_white * white / n


def plan_effects(
    library: EffectLibrary,
    act: np.ndarray,
    rhythm: np.ndarray,
    song_phrases: Sequence[Phrase],
    sections: Sequence[SectionInfo],
    *,
    repeat_of: Sequence[int | None] | None = None,
    style: str | None = None,
    seed: int = 0,
) -> list[Placement]:
    """One human passage per phrase (see the module docstring)."""
    rng = np.random.default_rng(seed)
    bias = STYLE_BIAS.get(style or "", (0.0, 0.0, 0.0))
    act = np.clip(act + bias[0], 0.0, 1.0)
    by_section: dict[int, list[Placement]] = defaultdict(list)
    out: list[Placement] = []
    for ph in song_phrases:
        lead = sections[ph.section].lead if ph.section < len(sections) else None
        placement = None
        src_section = repeat_of[ph.section] if repeat_of and ph.section < len(repeat_of) else None
        if src_section is not None and src_section != ph.section:
            done = len(by_section[ph.section])
            template = by_section.get(src_section, [])
            if done < len(template):
                t = template[done]
                if t.src_bar + ph.bars <= library.maps[t.source].bars:
                    placement = Placement(ph.start_bar, ph.bars, t.source, t.src_bar, t.swap)
        if placement is None:
            placement = _choose(library, act, rhythm, ph, by_section[ph.section], out, bias, rng)
            if placement is None:
                continue
            src_lead = _lead_of(library.maps[placement.source], placement.src_bar, ph.bars)
            swap = lead is not None and src_lead is not None and src_lead != lead
            placement = Placement(ph.start_bar, ph.bars, placement.source, placement.src_bar, swap)
        by_section[ph.section].append(placement)
        out.append(placement)
    return out


def _choose(
    library: EffectLibrary,
    act: np.ndarray,
    rhythm: np.ndarray,
    ph: Phrase,
    before: list[Placement],
    played: list[Placement],
    bias: tuple[float, float, float],
    rng: np.random.Generator,
) -> Placement | None:
    a, z = ph.start_bar, ph.start_bar + ph.bars
    prev = before[-1] if before else None
    used = {p.source for p in played}
    candidates: list[tuple[float, int, int]] = []
    for i, src in enumerate(library.maps):
        cost = _costs(src, act[a:z], rhythm[a:z], bias)
        if not len(cost):
            continue
        if used and i not in used:
            cost += NEW_SOURCE_PENALTY
        starts = np.arange(len(cost))
        for p in played:
            if p.source == i:
                overlap = np.minimum(starts + ph.bars, p.src_bar + p.bars)
                overlap = np.maximum(overlap - np.maximum(starts, p.src_bar), 0)
                cost += REPLAY_PENALTY * overlap / ph.bars
        if prev is not None and prev.source == i:
            nxt = prev.src_bar + prev.bars
            if nxt < len(cost):
                cost[nxt] -= CONTINUE_BONUS
        best = np.argsort(cost)[:TOP_K]
        candidates += [(float(cost[s]), i, int(s)) for s in best]
    if not candidates:
        return None
    candidates.sort()
    top = candidates[:TOP_K]
    costs = np.array([c for c, _, _ in top])
    p = np.exp(-(costs - costs[0]) / TEMPERATURE)
    _, i, s = top[int(rng.choice(len(top), p=p / p.sum()))]
    return Placement(ph.start_bar, ph.bars, i, s)


# --- rendering --------------------------------------------------------------------


def _swap_box(kind: str, box: dict) -> dict:
    if kind != "color":
        return box
    swapped = {LightColor.RED: LightColor.BLUE, LightColor.BLUE: LightColor.RED}
    events = [{**e, "c": int(swapped.get(e["c"], e["c"]))} for e in box["e"]]
    return {**box, "e": events}


def _swap_value(type_: int, value: int) -> int:
    decoded = decode_value(value) if type_ in LIGHT_BASIC_TYPES else None
    if decoded is None or decoded[0] == LightColor.WHITE:
        return value
    color = LightColor.BLUE if decoded[0] == LightColor.RED else LightColor.RED
    return light_value(color, decoded[1])


@dataclass(slots=True)
class _Out:
    boxes: dict[tuple[float, str, int], list[dict]] = field(
        default_factory=lambda: defaultdict(list)
    )
    basic: list[BasicEvent] = field(default_factory=list)
    boost: list[ColorBoostEvent] = field(default_factory=list)
    lights: set[tuple[str, int]] = field(default_factory=set)  # groups set so far
    basic_types: set[int] = field(default_factory=set)

    def add(self, beat: float, kind: str, gid: int, boxes: list[dict]) -> None:
        self.boxes[(round(beat, 6), kind, gid)].extend(boxes)
        self.lights.add((kind, gid))


def render_effects(
    library: EffectLibrary, placements: Sequence[Placement], end_beat: float
) -> Lightshow:
    """Copy each placement's human passage into the song (see the module docstring)."""
    out = _Out()
    for p in placements:
        src = library.source(p.source)
        a = p.src_bar * BAR_BEATS
        z = a + p.bars * BAR_BEATS
        dest = p.start_bar * BAR_BEATS
        offset = dest - a

        def fix(kind: str, boxes: list[dict], swap: bool = p.swap) -> list[dict]:
            return [_swap_box(kind, b) for b in boxes] if swap else list(boxes)

        state = src.light_state(a)
        for kind, gid in sorted(out.lights - state.keys()):
            out.add(dest, kind, gid, _reset_boxes(kind))
        for (kind, gid), boxes in sorted(state.items()):
            out.add(dest, kind, gid, fix(kind, boxes))
        for beat, kind, gid, boxes in src.groups_between(a, z):
            out.add(beat + offset, kind, gid, fix(kind, boxes))

        basic_state = src.basic_state(a)
        for type_ in sorted(out.basic_types - basic_state.keys()):
            if type_ in LIGHT_BASIC_TYPES:
                out.basic.append(BasicEvent(dest, type_, 0, 0.0))
        for type_, value, f in basic_state.values():
            value = _swap_value(type_, value) if p.swap else value
            out.basic.append(BasicEvent(dest, type_, value, f))
            out.basic_types.add(type_)
        for beat, type_, value, f in src.basic_between(a, z):
            value = _swap_value(type_, value) if p.swap else value
            out.basic.append(BasicEvent(round(beat + offset, 6), type_, value, f))
            out.basic_types.add(type_)

        on = src.boost_state(a)
        if on is not None:
            out.boost.append(ColorBoostEvent(dest, on))
        for beat, on in src.boost_between(a, z):
            out.boost.append(ColorBoostEvent(round(beat + offset, 6), bool(on)))

    for kind, gid in sorted(out.lights):
        if kind == "color":
            out.add(end_beat, kind, gid, _reset_boxes(kind))
    for type_ in sorted(out.basic_types & LIGHT_BASIC_TYPES):
        out.basic.append(BasicEvent(end_beat, type_, 0, 0.0))
    groups = [
        EventBoxGroup(beat, gid, kind, tuple(boxes))  # type: ignore[arg-type]
        for (beat, kind, gid), boxes in sorted(out.boxes.items())
    ]
    out.basic.sort(key=lambda e: (e.beat, e.type))
    out.boost.sort(key=lambda e: e.beat)
    return Lightshow(basic_events=out.basic, boost_events=out.boost, event_box_groups=groups)


# --- the renderer -----------------------------------------------------------------


@dataclass(slots=True)
class EffectRenderer:
    """Renders v3 lighting from the human effect libraries under ``root``.

    ``style`` (a light style name) biases the choice; ``last_sources`` lists the human
    maps (key, mapper) the last render used.
    """

    root: Path
    seed: int = 0
    style: str | None = None
    libraries: dict[str, Path] = field(default_factory=dict)
    last_sources: list[tuple[str, str]] = field(default_factory=list)
    _loaded: dict[str, EffectLibrary] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.libraries = available_libraries(self.root)

    def supports(self, environment: str) -> bool:
        return environment in self.libraries

    def library(self, environment: str) -> EffectLibrary:
        if environment not in self._loaded:
            self._loaded[environment] = EffectLibrary(self.libraries[environment])
        return self._loaded[environment]

    def auto_environments(self) -> list[str]:
        """Environments with at least ``AUTO_MIN_MAPS`` source maps, else all of them."""
        sizes = {}
        for env, folder in self.libraries.items():
            index = json.loads((folder / INDEX_FILE).read_text(encoding="utf-8"))
            sizes[env] = len(index["maps"])
        big = sorted(e for e, n in sizes.items() if n >= AUTO_MIN_MAPS)
        return big or sorted(sizes)

    def pick_environment(self) -> str | None:
        envs = self.auto_environments()
        if not envs:
            return None
        return envs[int(np.random.default_rng(self.seed).integers(len(envs)))]

    def render(
        self,
        lightshow: Lightshow,
        sections: Sequence[SectionInfo],
        end_beat: float,
        environment: str,
        repeat_of: Sequence[int | None] | None = None,
    ) -> Lightshow:
        library = self.library(environment)
        n_bars = max(1, int(math.ceil(end_beat / BAR_BEATS - 1e-9)))
        act, rhythm = song_targets(lightshow, sections, n_bars)
        song_phrases = phrases(sections, n_bars)
        placements = plan_effects(
            library,
            act,
            rhythm,
            song_phrases,
            sections,
            repeat_of=repeat_of,
            style=self.style,
            seed=self.seed,
        )
        used = dict.fromkeys(p.source for p in placements)
        self.last_sources = [(library.maps[i].key, library.maps[i].mapper) for i in used]
        log.info(
            "%s effects: %d phrases from %d human maps (%s)",
            environment,
            len(placements),
            len(used),
            ", ".join(k for k, _ in self.last_sources),
        )
        return render_effects(library, placements, n_bars * BAR_BEATS)
