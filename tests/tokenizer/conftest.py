"""Seeded random maps for property-style round-trip tests.

``on_grid`` maps use only values the vocabulary can express, so they must
round-trip exactly as they are. ``messy`` maps add everything real files
contain (off-grid times, odd angles and multipliers, objects off the grid or
before beat 0) and must round-trip exactly after ``snap_map``.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest

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
from bsmapper.tokenizer.grid import snap_beat, tick_to_beat
from bsmapper.tokenizer.vocab import LIGHT_EVENT_TYPES, PALETTE


class _Gen:
    def __init__(self, seed: int, messy: bool) -> None:
        self.r = random.Random(seed)
        self.messy = messy
        self.windows = self.r.randint(1, 6)

    def tick(self) -> int:
        # Cluster some objects on shared ticks (chords) and at window edges.
        r = self.r
        choice = r.random()
        if choice < 0.1:
            return r.choice([0, 383, 384, 767, 768]) % (self.windows * 384)
        return r.randrange(self.windows * 384)

    def beat(self) -> float:
        b = tick_to_beat(self.tick())
        if self.messy and self.r.random() < 0.4:
            b += self.r.uniform(-0.03, 0.03)
            if self.r.random() < 0.05:
                b = -abs(b) - 0.5
        return b

    def tail(self, beat: float) -> float:
        """A tail time after ``beat``; on the grid the way readers write beats."""
        tail = beat + self.span()
        if self.messy:
            return tail - 1.0 if self.r.random() < 0.05 else tail
        return snap_beat(tail)

    def span(self) -> float:
        r = self.r
        ticks = r.choice([0, r.randrange(48), r.randrange(24 * 20), r.randrange(24 * 70)])
        b = tick_to_beat(ticks)
        if self.messy and r.random() < 0.3:
            b += r.uniform(-0.02, 0.02)
        return b

    def cell(self) -> tuple[int, int]:
        if self.messy and self.r.random() < 0.03:
            return self.r.choice([(-1, 0), (4, 1), (2, 3), (1000, 1000)])
        return self.r.randrange(4), self.r.randrange(3)

    def frac(self) -> float:
        if self.messy:
            return self.r.choice([self.r.uniform(-0.5, 4.0), 1.0, 0.5])
        return self.r.randrange(61) / 20

    def note(self) -> Note:
        r = self.r
        x, y = self.cell()
        if self.messy:
            angle = r.choice([0, 0, r.randint(-90, 90)])
        else:
            angle = r.choice([0, 0, 0, r.randrange(-45, 46, 5)])
        return Note(self.beat(), x, y, r.choice(list(Color)), r.choice(list(CutDirection)), angle)

    def bomb(self) -> Bomb:
        return Bomb(self.beat(), *self.cell())

    def wall(self) -> Obstacle:
        r = self.r
        if self.messy and r.random() < 0.1:
            return Obstacle(self.beat(), r.uniform(-2, 2), r.randint(-1, 4), 0, r.randint(0, 5), 7)
        return Obstacle(
            self.beat(),
            self.span(),
            r.randrange(4),
            r.randrange(3),
            r.randint(1, 4),
            r.randint(1, 5),
        )

    def arc(self) -> Arc:
        r = self.r
        beat = self.beat()
        (x, y), (tx, ty) = self.cell(), self.cell()
        anchor = r.choice([0, 1, 2, 5]) if self.messy else r.randrange(3)
        return Arc(
            beat, r.choice(list(Color)), x, y, r.choice(list(CutDirection)),
            self.tail(beat), tx, ty, r.choice(list(CutDirection)), self.frac(), self.frac(), anchor,
        )  # fmt: skip

    def chain(self) -> Chain:
        r = self.r
        beat = self.beat()
        (x, y), (tx, ty) = self.cell(), self.cell()
        slices = r.randint(-2, 60) if self.messy else r.randint(1, 32)
        return Chain(
            beat, r.choice(list(Color)), x, y, r.choice(list(CutDirection)),
            self.tail(beat), tx, ty, slices, self.frac(),
        )  # fmt: skip

    def rotation(self) -> RotationEvent:
        r = self.r
        if self.messy:
            degrees = r.choice([r.randint(-400, 400), 15 * r.randint(-30, 30)])
        else:
            degrees = r.choice([d for d in range(-165, 181, 15) if d])
        return RotationEvent(self.beat(), degrees, early=r.random() < 0.7)

    def many(self, make: Callable, mean: int) -> list:
        items = [make() for _ in range(self.r.randint(0, 2 * mean))]
        if items and self.r.random() < 0.1:
            items.append(items[0])  # exact duplicates must survive too
        return items

    def difficulty_map(self) -> DifficultyMap:
        r = self.r
        return DifficultyMap(
            r.choice(list(Characteristic)),
            r.choice(list(Difficulty)),
            njs=r.choice([10.0, 16.0, 18.5]),
            njs_offset=r.choice([0.0, -0.25]),
            notes=self.many(self.note, 40),
            bombs=self.many(self.bomb, 6),
            obstacles=self.many(self.wall, 4),
            arcs=self.many(self.arc, 3),
            chains=self.many(self.chain, 3),
            rotations=self.many(self.rotation, 4),
            label=r.choice([None, "Random"]),
        )

    def lightshow(self) -> Lightshow:
        r = self.r
        basic = []
        for _ in range(r.randint(0, 80)):
            if self.messy:
                event = BasicEvent(
                    self.beat(),
                    r.choice([*LIGHT_EVENT_TYPES, 5, 99, 1000]),
                    r.randint(-1, 40),
                    r.choice([1.0, r.uniform(-0.2, 3.0)]),
                    r.choice([None, (r.random(), r.random(), r.random(), r.random())]),
                    r.choice([None, {"lightID": [1, 2]}]),
                )
            else:
                color = r.choice([None, (*r.choice(PALETTE), 1.0)])
                event = BasicEvent(
                    self.beat(),
                    r.choice(LIGHT_EVENT_TYPES),
                    r.randint(0, 31),
                    r.choice([1.0, r.randrange(41) / 20]),
                    color,
                )
            basic.append(event)
        boosts = [ColorBoostEvent(self.beat(), r.random() < 0.5) for _ in range(r.randint(0, 6))]
        basic.sort(key=lambda e: e.beat)
        boosts.sort(key=lambda e: e.beat)
        return Lightshow(basic, boosts, [], any(e.color is not None for e in basic))


@pytest.fixture
def random_map() -> Callable[..., DifficultyMap]:
    def make(seed: int, messy: bool = False) -> DifficultyMap:
        return _Gen(seed, messy).difficulty_map()

    return make


@pytest.fixture
def random_lightshow() -> Callable[..., Lightshow]:
    def make(seed: int, messy: bool = False) -> Lightshow:
        return _Gen(seed, messy).lightshow()

    return make
