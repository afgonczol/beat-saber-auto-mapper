"""The conditioning prefix: ``CHAR DIFF NPS STYLE BOMBS WALLS ARCS CHAINS``.

``NPS`` buckets the difficulty's notes per second and ``STYLE`` buckets the
0 (flowy) .. 1 (technical) slider into ``STYLE_0..4``. Both have a ``_NONE``
token for "no preference": training drops them at random (``drop_style``) so
the model works without them and supports classifier-free guidance. The four
flags say whether the map uses (training) or may use (generation) each object type.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, replace

from bsmapper.interfaces import GenerationRequest
from bsmapper.schema import Beatmap, Characteristic, Difficulty, DifficultyMap
from bsmapper.tokenizer.vocab import NOTE_VOCAB, NPS_EDGES, STYLE_BUCKETS, Vocab

PREFIX_GROUPS = ("CHAR", "DIFF", "NPS", "STYLE", "BOMBS", "WALLS", "ARCS", "CHAINS")
PREFIX_LEN = len(PREFIX_GROUPS)


def nps_bucket(nps: float) -> int:
    return bisect.bisect_right(NPS_EDGES, nps)


def nps_bucket_value(bucket: int) -> float:
    """A representative NPS for a bucket (its midpoint, or the top edge + 1)."""
    lo = NPS_EDGES[bucket - 1] if bucket else 0.0
    hi = NPS_EDGES[bucket] if bucket < len(NPS_EDGES) else lo + 2.0
    return (lo + hi) / 2


def style_bucket(style: float) -> int:
    return min(int(max(0.0, style) * STYLE_BUCKETS), STYLE_BUCKETS - 1)


def notes_per_second(
    dmap: DifficultyMap, beatmap: Beatmap, duration_s: float | None = None
) -> float:
    """Colour notes per second, over ``duration_s`` or else from first to last note."""
    if not dmap.notes:
        return 0.0
    if duration_s is None:
        beats = [n.beat for n in dmap.notes]
        duration_s = beatmap.beat_to_seconds(max(beats)) - beatmap.beat_to_seconds(min(beats))
    return len(dmap.notes) / duration_s if duration_s > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Conditioning:
    characteristic: Characteristic
    difficulty: Difficulty
    nps: float | None = None  # None = no preference
    style: float | None = None  # 0 flowy .. 1 technical; None = no preference
    bombs: bool = True
    walls: bool = True
    arcs: bool = True
    chains: bool = True

    @classmethod
    def from_request(cls, request: GenerationRequest) -> Conditioning:
        return cls(
            request.characteristic,
            request.difficulty,
            nps=request.target_nps,
            style=request.style,
            bombs=request.allow_bombs,
            walls=request.allow_obstacles,
            arcs=request.allow_arcs,
            chains=request.allow_chains,
        )

    @classmethod
    def from_map(
        cls, dmap: DifficultyMap, nps: float | None = None, style: float | None = None
    ) -> Conditioning:
        """Ground-truth conditioning for training; flags say which objects the map uses."""
        return cls(
            dmap.characteristic,
            dmap.difficulty,
            nps=nps,
            style=style,
            bombs=bool(dmap.bombs),
            walls=bool(dmap.obstacles),
            arcs=bool(dmap.arcs),
            chains=bool(dmap.chains),
        )

    def drop_style(self) -> Conditioning:
        return replace(self, style=None)

    def drop_nps(self) -> Conditioning:
        return replace(self, nps=None)

    def prefix_ids(self, vocab: Vocab = NOTE_VOCAB) -> list[int]:
        return [
            vocab.id("CHAR", self.characteristic),
            vocab.id("DIFF", self.difficulty),
            vocab.id("NPS", None if self.nps is None else nps_bucket(self.nps)),
            vocab.id("STYLE", None if self.style is None else style_bucket(self.style)),
            vocab.id("BOMBS", self.bombs),
            vocab.id("WALLS", self.walls),
            vocab.id("ARCS", self.arcs),
            vocab.id("CHAINS", self.chains),
        ]

    @classmethod
    def from_prefix(cls, ids: Sequence[int], vocab: Vocab = NOTE_VOCAB) -> Conditioning:
        """Read a prefix back. NPS and style come back as bucket midpoints."""
        if [vocab.group(i) for i in ids] != list(PREFIX_GROUPS):
            raise ValueError(f"not a conditioning prefix: {vocab.names(ids)}")
        char, diff, nps, style, bombs, walls, arcs, chains = (vocab.value(i) for i in ids)
        return cls(
            char,
            diff,
            nps=None if nps is None else nps_bucket_value(nps),
            style=None if style is None else (style + 0.5) / STYLE_BUCKETS,
            bombs=bombs,
            walls=walls,
            arcs=arcs,
            chains=chains,
        )
