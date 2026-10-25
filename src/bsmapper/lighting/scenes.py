"""Section-level scene plans for the version 2 light prompt (WS8b, option b2).

A scene is ``(level, palette, flips)``: light events per beat, the colour mix and how
often lit events change colour (``tokenizer.lighting.window_scene``). In training each
window's scene is read from the human lightshow. At generation ``plan_scenes`` picks one
scene per L0 section and gives it to every window of that section, so a section keeps
one density and palette instead of drifting window by window:

* a section whose music repeats an earlier section (most of its windows' reference
  spans, ``data.structure``, start in that section) reuses that section's scene;
* any other section draws a scene from the human scenes of the chosen style at the
  section's loudness (``StyleModel.scene_table``), seeded;
* a loud section (loudness bin 2 or more) never draws a dark or near-dark scene, which
  fixes the dark intros and choruses of phase 1;
* single-colour and lead palettes follow the L0 section's lead colour, so palettes
  alternate between sections the way the rule planner alternates them;
* silent windows get a dark scene; a section with no human scene to draw from gets
  None (the ``_NONE`` tokens: no preference).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bsmapper.lighting.styles import LIVE_RMS, Scene, StyleModel, loudness_bin
from bsmapper.tokenizer import WINDOW_BEATS, WINDOW_TICKS
from bsmapper.tokenizer.vocab import LIGHT_PALETTES

_PAL = {name: i for i, name in enumerate(LIGHT_PALETTES)}
_SWAP = {
    _PAL["RED"]: _PAL["BLUE"],
    _PAL["BLUE"]: _PAL["RED"],
    _PAL["RED_LEAD"]: _PAL["BLUE_LEAD"],
    _PAL["BLUE_LEAD"]: _PAL["RED_LEAD"],
}
_RED_PALETTES = {_PAL["RED"], _PAL["RED_LEAD"]}
DARK_SCENE: Scene = (0, _PAL["DARK"], 0)
LOUD_BIN = 2  # sections this loud or louder get a lit scene
MIN_LOUD_LEVEL = 2


@dataclass(frozen=True, slots=True)
class SectionSpan:
    start: float  # beats
    end: float
    red_lead: bool | None = None  # the L0 section's lead colour; None = no preference


def window_sections(sections: Sequence[SectionSpan], n_windows: int) -> list[int]:
    """Section index of each window: the section holding the window's middle beat."""
    if not sections:
        return [0] * n_windows
    starts = [s.start for s in sections]
    out = []
    for k in range(n_windows):
        mid = (k + 0.5) * WINDOW_BEATS
        out.append(max(0, int(np.searchsorted(starts, mid, side="right")) - 1))
    return out


def _repeat_of(
    section: int, members: list[int], owner: list[int], refs: Sequence[int] | None
) -> int | None:
    """The earlier section most of ``members``' reference spans start in, if any."""
    if refs is None:
        return None
    votes: Counter[int] = Counter()
    for k in members:
        start = refs[k] if k < len(refs) else -1
        if start is None or start < 0:
            continue
        src = owner[min(start // WINDOW_TICKS, len(owner) - 1)]
        if src < section:
            votes[src] += 1
    if not votes:
        return None
    src, n = votes.most_common(1)[0]
    return src if n * 2 > len(members) else None


def _draw(
    model: StyleModel, style: int | None, bin_: int, rng: np.random.Generator
) -> Scene | None:
    rows = model.scene_table(style, bin_)
    if bin_ >= LOUD_BIN:
        rows = [(s, p) for s, p in rows if s[0] >= MIN_LOUD_LEVEL and s[1] != _PAL["DARK"]]
    if not rows:
        return None
    p = np.array([p for _, p in rows])
    return rows[int(rng.choice(len(rows), p=p / p.sum()))][0]


def _lead(scene: Scene, red_lead: bool | None) -> Scene:
    level, palette, flips = scene
    if red_lead is None or palette not in _SWAP or (palette in _RED_PALETTES) == red_lead:
        return scene
    return level, _SWAP[palette], flips


def plan_scenes(
    model: StyleModel,
    style: int | None,
    loudness: Sequence[float],
    sections: Sequence[SectionSpan],
    refs: Sequence[int] | None = None,
    seed: int = 0,
) -> list[Scene | None]:
    """One scene per window (``loudness``: each window's loudness relative to the song's
    loudest, ``styles.window_loudness``; ``refs``: reference start ticks, -1 = none)."""
    n = len(loudness)
    rng = np.random.default_rng(seed)
    owner = window_sections(sections, n)
    members: dict[int, list[int]] = {}
    for k, s in enumerate(owner):
        members.setdefault(s, []).append(k)
    chosen: dict[int, Scene | None] = {}
    for s in sorted(members):
        ks = members[s]
        src = _repeat_of(s, ks, owner, refs)
        if src is not None and chosen.get(src) is not None:
            chosen[s] = chosen[src]
            continue
        live = [loudness[k] for k in ks if loudness[k] > LIVE_RMS]
        if not live:
            chosen[s] = None
            continue
        scene = _draw(model, style, loudness_bin(float(np.median(live))), rng)
        red_lead = sections[s].red_lead if s < len(sections) else None
        chosen[s] = None if scene is None else _lead(scene, red_lead)
    return [DARK_SCENE if loudness[k] <= LIVE_RMS else chosen.get(owner[k]) for k in range(n)]
