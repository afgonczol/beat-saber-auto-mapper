"""Lighting styles (WS8b): what kind of show a human lightshow is, and which one to pick.

Human lightshows each keep one style for the whole song (docs/LIGHTING_V2.md, §2), and
one unconditioned model picks a style by chance. The version 2 light prompt names the
style (``LSTYLE_k``), so the model can be asked for one.

* ``map_style_features`` describes a lightshow by readability statistics on its loud
  windows: change rate, laser and ring shares, colour flips, action mix, holds, motion,
  sync with beats and onsets, white and boost use, how it follows the energy.
* ``is_sequenced`` flags Chroma lightID sequences and strobes, which the vanilla light
  vocabulary turns into noise. They are left out of style training. Once lightIDs are
  dropped, a lightID sequence shows as many events of one light type on the same tick
  (``stacked``): a clean, bimodal signal on dataset v1 (median 0.1% of events, p90 66%).
* ``StyleModel`` holds the k-means styles (standardisation, centroids, names), the
  auto-pick (softmax regression from song features, ``song_vector``, to style) and
  the scene tables of ``lighting.scenes`` (human scene frequencies per style and
  loudness). It is one JSON file, built by ``lighting.style_build`` and stored in the
  checkpoint, so generation needs nothing else.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bsmapper.lighting.events import LIGHT_TYPES
from bsmapper.lighting.metrics import constant_tempo, visible_changes, window_stats
from bsmapper.schema import Lightshow
from bsmapper.tokenizer import TICKS_PER_BEAT, WINDOW_TICKS

STYLE_MODEL_VERSION = 1
ONSET_COL, FLUX_COL, RMS_COL = 80, 81, 82  # beat-synchronous feature columns (WS3b)
LOUD = 0.7  # windows at least this loud (relative to the song's loudest) describe a style
LIVE_RMS = 0.02  # quieter windows are silence
MIN_LIT_WINDOWS = 4
MIN_RATE = 0.3  # visible changes per second on loud windows; sparser is not a show
# Chroma lightID sequences and strobes: the vanilla vocabulary cannot represent them.
SEQUENCED_STACKED = 0.1  # share of light events stacked on one (tick, type)
SEQUENCED_STROBE = 0.6  # share of changes less than 0.1 s apart
LOUDNESS_EDGES = (0.2, 0.4, 0.6, 0.8)  # window loudness bins for the scene tables

# (name, function of the raw per-map features) -> the vector k-means clusters.
STYLE_FEATURES: tuple[tuple[str, Callable[[dict], float]], ...] = (
    ("log_rate", lambda f: np.log1p(f["rate"])),
    ("laser_share", lambda f: f["laser_share"]),
    ("ring_share", lambda f: f["ring_share"]),
    ("color_ratio", lambda f: np.log1p(10 * f["color_ratio"])),
    ("fade", lambda f: f["fade"]),
    ("flash", lambda f: f["flash"]),
    ("on", lambda f: f["on"]),
    ("strobe_ratio", lambda f: f["strobe_ratio"]),
    ("log_hold", lambda f: np.log(max(f["hold_med"], 0.01))),
    ("log_spin", lambda f: np.log1p(f["spin"])),
    ("log_zoom", lambda f: np.log1p(f["zoom"])),
    ("log_laser_speed", lambda f: np.log1p(f["laser_speed"])),
    ("on_beat", lambda f: f["on_beat"]),
    ("off_grid", lambda f: f["off_grid"]),
    ("onset_lift", lambda f: min(f["onset_lift"], 5.0)),
    ("boost_time", lambda f: f["boost_time"]),
    ("white_share", lambda f: f["white_share"]),
    ("energy_follow", lambda f: f["energy_follow"]),
    ("consistency_cv", lambda f: min(f["consistency_cv"], 3.0)),
)
STYLE_FEATURE_NAMES = tuple(name for name, _ in STYLE_FEATURES)

# Named styles, matched to clusters by these z-score signatures (docs/LIGHTING_V2.md §2).
STYLE_PROTOTYPES: dict[str, dict[str, float]] = {
    "pulse": {"fade": 1.0, "log_laser_speed": 1.0, "on": -0.5, "on_beat": 0.5},
    "calm": {"log_rate": -1.0, "log_hold": 1.0, "log_laser_speed": -1.0, "fade": 0.5},
    "scenes": {"on": 1.5, "fade": -1.0, "color_ratio": -0.5},
    "flash": {"flash": 2.0, "log_zoom": 1.0, "log_rate": 0.5, "fade": -0.5},
    "accent": {"white_share": 2.0, "boost_time": 1.0},
}
STYLE_DESCRIPTIONS = {
    "pulse": "beat-pulsed fades with laser sweeps",
    "calm": "slow fade pulses, long holds, little laser motion",
    "scenes": "sustained on/off scenes, few colour flips",
    "flash": "fast flashes with ring zooms and spins",
    "accent": "white accents and colour boosts",
}


def _median(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def window_loudness(feats: np.ndarray, n_windows: int) -> np.ndarray:
    """Mean RMS per 16-beat window, relative to the loudest window (0..1)."""
    rms = np.asarray(feats[: n_windows * WINDOW_TICKS, RMS_COL], dtype=np.float32)
    rms = np.pad(rms, (0, n_windows * WINDOW_TICKS - len(rms)))
    energy = rms.reshape(n_windows, WINDOW_TICKS).mean(axis=1)
    peak = float(energy.max()) if len(energy) else 0.0
    return energy / peak if peak > 0 else np.zeros_like(energy)


def loudness_bin(rel: float) -> int:
    return int(np.searchsorted(LOUDNESS_EDGES, rel, side="right"))


def map_style_features(lightshow: Lightshow, feats: np.ndarray, bpm: float) -> dict | None:
    """Raw style features of one lightshow on a constant tempo, or None when it is too
    sparse or dark to describe a style. ``feats`` are its beat-synchronous features."""
    n_windows = max(1, int(np.ceil(len(feats) / WINDOW_TICKS)))
    b2s = constant_tempo(bpm)
    ws = [w.as_dict() for w in window_stats(lightshow, n_windows * 16, b2s)]
    energy = window_loudness(feats, n_windows)
    if not len(energy) or energy.max() <= 0:
        return None
    live = [k for k in range(n_windows) if energy[k] > LIVE_RMS]
    lit = [k for k in live if ws[k]["changes_per_s"] > 0]
    if len(lit) < MIN_LIT_WINDOWS:
        return None
    loud = [k for k in lit if energy[k] >= LOUD] or lit

    def med(key: str) -> float:
        return _median([ws[k][key] for k in loud])

    f: dict[str, Any] = {"rate": med("changes_per_s")}
    if not f["rate"] >= MIN_RATE:
        return None
    total = sum(med(f"ch{t}_per_s") for t in LIGHT_TYPES) or 1.0
    f["laser_share"] = (med("ch2_per_s") + med("ch3_per_s")) / total
    f["ring_share"] = med("ch1_per_s") / total
    f["color_ratio"] = med("color_switches_per_s") / max(f["rate"], 1e-6)
    for action in ("fade", "flash", "on", "transition", "off"):
        f[action] = med(f"{action}_share")
    f["strobe_ratio"] = med("strobe_per_s") / max(f["rate"], 1e-6)
    hold = med("hold_median_s")
    f["hold_med"] = hold if np.isfinite(hold) else 0.01
    f["spin"] = med("spin_per_s")
    f["zoom"] = med("zoom_per_s")
    f["laser_speed"] = med("laser_speed_per_s")
    rate = np.array([ws[k]["changes_per_s"] for k in live])
    f["energy_follow"] = _spearman(energy[live], rate) if len(live) > 4 else 0.0
    loud_rate = np.array([ws[k]["changes_per_s"] for k in loud])
    f["consistency_cv"] = float(loud_rate.std() / max(loud_rate.mean(), 1e-6))
    f.update(_sync_features(lightshow, feats, n_windows))
    return f


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _sync_features(lightshow: Lightshow, feats: np.ndarray, n_windows: int) -> dict[str, float]:
    changes = visible_changes(lightshow)
    beats = [c.beat for t in LIGHT_TYPES for c in changes[t]]
    onset = np.asarray(feats[:, ONSET_COL], dtype=np.float32)
    rms = np.asarray(feats[:, RMS_COL], dtype=np.float32)
    ticks = np.array([round(b * TICKS_PER_BEAT) for b in beats], dtype=np.int64)
    ticks = ticks[(ticks >= 0) & (ticks < len(onset))]
    out = {"on_beat": 0.0, "off_grid": 0.0, "onset_lift": 1.0}
    if len(ticks):
        frac = ticks % TICKS_PER_BEAT
        out["on_beat"] = float((frac == 0).mean())
        out["off_grid"] = float(((frac % 6) != 0).mean())  # finer than 1/4 beat
        peak = np.maximum(onset, np.maximum(np.roll(onset, 1), np.roll(onset, -1)))
        loud = rms > 0.05
        base = float(peak[loud].mean()) if loud.any() else 1.0
        out["onset_lift"] = float(peak[ticks].mean() / max(base, 1e-6))
    lit = [c.color for t in LIGHT_TYPES for c in changes[t] if c.lit]
    out["white_share"] = float(np.mean([c == 2 for c in lit])) if lit else 0.0
    n_beats = n_windows * 16
    on, t0, boosted = False, 0.0, 0.0
    for b in sorted(lightshow.boost_events, key=lambda b: b.beat):
        if b.on and not on:
            on, t0 = True, b.beat
        elif not b.on and on:
            on, boosted = False, boosted + b.beat - t0
    if on:
        boosted += n_beats - t0
    out["boost_time"] = boosted / max(n_beats, 1)
    out["chroma"] = float(any(e.color is not None for e in lightshow.basic_events))
    stacks = Counter(
        (round(e.beat * TICKS_PER_BEAT), e.type)
        for e in lightshow.basic_events
        if e.type in LIGHT_TYPES
    )
    stacked = sum(n for n in stacks.values() if n > 1)
    out["stacked"] = stacked / max(sum(stacks.values()), 1)
    return out


def is_sequenced(f: dict) -> bool:
    """A lightID sequence or a strobe show: noise once moved onto the vanilla vocabulary."""
    return f.get("stacked", 0.0) >= SEQUENCED_STACKED or f["strobe_ratio"] >= SEQUENCED_STROBE


def style_vector(f: dict) -> np.ndarray:
    return np.nan_to_num(np.array([fn(f) for _, fn in STYLE_FEATURES], dtype=float))


# --- song features for the auto-pick ------------------------------------------------

SONG_FEATURE_NAMES = (
    "log_bpm",
    "log_minutes",
    "rms_mean",
    "rms_std",
    "loud_share",
    "window_cv",
    "onset_mean",
    "onset_std",
    "onset_beat",
    "flux_mean",
    *(f"mel_{i}" for i in range(8)),
)


def song_vector(feats: np.ndarray, bpm: float) -> np.ndarray:
    """Song-level audio features the auto-pick uses (``SONG_FEATURE_NAMES``)."""
    x = np.asarray(feats, dtype=np.float32)
    n_windows = max(1, int(np.ceil(len(x) / WINDOW_TICKS)))
    rms = x[:, RMS_COL]
    live = rms > LIVE_RMS * max(float(rms.max()), 1e-6)
    xs = x[live] if live.sum() > TICKS_PER_BEAT else x
    rel = xs[:, RMS_COL] / max(float(rms.max()), 1e-6)
    onset = xs[:, ONSET_COL]
    on_beat = np.arange(len(x))[live if live.sum() > TICKS_PER_BEAT else slice(None)]
    beat_ticks = (on_beat % TICKS_PER_BEAT) == 0
    onset_beat = float(onset[beat_ticks].mean() / max(float(onset.mean()), 1e-6))
    windows = window_loudness(x, n_windows)
    windows = windows[windows > LIVE_RMS]
    mel = xs[:, :80].reshape(len(xs), 8, 10).mean(axis=2).mean(axis=0)
    return np.array(
        [
            np.log(max(bpm, 1.0)),
            np.log(max(len(x) / TICKS_PER_BEAT / max(bpm, 1.0), 0.1)),
            float(rel.mean()),
            float(rel.std()),
            float((rel >= LOUD).mean()),
            float(windows.std() / max(windows.mean(), 1e-6)) if len(windows) else 0.0,
            float(onset.mean()),
            float(onset.std()),
            onset_beat,
            float(xs[:, FLUX_COL].mean()),
            *(mel - mel.mean()),
        ],
        dtype=float,
    )


# --- fitting ------------------------------------------------------------------------


def kmeans(
    z: np.ndarray, k: int, n_init: int = 20, iters: int = 300, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """(centroids, labels) of the best of ``n_init`` k-means++ runs."""
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for _ in range(n_init):
        centers = [z[rng.integers(len(z))]]
        for _ in range(1, k):
            d = np.min([((z - c) ** 2).sum(axis=1) for c in centers], axis=0)
            centers.append(z[rng.choice(len(z), p=d / d.sum())])
        c = np.array(centers)
        for _ in range(iters):
            labels = ((z[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
            new = np.array(
                [z[labels == j].mean(axis=0) if (labels == j).any() else c[j] for j in range(k)]
            )
            if np.allclose(new, c):
                break
            c = new
        labels = ((z[:, None, :] - c[None]) ** 2).sum(axis=2).argmin(axis=1)
        inertia = float(((z - c[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, c, labels)
    assert best is not None
    return best[1], best[2]


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def fit_softmax(
    x: np.ndarray, y: np.ndarray, k: int, l2: float = 1e-2, steps: int = 2000, lr: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """(coef (F, k), intercept (k,)) of an L2-regularised softmax regression, fitted
    by full-batch gradient descent on standardised ``x``."""
    n, f = x.shape
    w, b = np.zeros((f, k)), np.zeros(k)
    onehot = np.eye(k)[y]
    for _ in range(steps):
        p = softmax(x @ w + b)
        g = (p - onehot) / n
        w -= lr * (x.T @ g + l2 * w)
        b -= lr * g.sum(axis=0)
    return w, b


def name_clusters(centers: np.ndarray) -> list[str]:
    """Style names for z-space centroids: the best one-to-one match to the prototypes.
    Clusters beyond the prototypes are named ``style<k>``."""
    index = {name: i for i, name in enumerate(STYLE_FEATURE_NAMES)}
    protos = list(STYLE_PROTOTYPES)
    score = np.array(
        [
            [sum(w * c[index[feat]] for feat, w in STYLE_PROTOTYPES[p].items()) for p in protos]
            for c in centers
        ]
    )
    k = len(centers)
    best: tuple[float, tuple[int, ...]] = (-np.inf, ())
    slots = list(range(len(protos))) + [-1] * max(0, k - len(protos))
    for perm in set(itertools.permutations(slots, k)):
        s = sum(score[i, p] for i, p in enumerate(perm) if p >= 0)
        if s > best[0]:
            best = (s, perm)
    return [protos[p] if p >= 0 else f"style{i}" for i, p in enumerate(best[1])]


# --- the model ------------------------------------------------------------------------

Scene = tuple[int, int, int]  # (level, palette, flips), see tokenizer.lighting.window_scene


@dataclass(slots=True)
class StyleModel:
    names: list[str]
    mean: list[float]  # style feature standardisation
    scale: list[float]
    centers: list[list[float]]  # (k, features) in z-space
    # Auto-pick: softmax regression from standardised song features.
    song_mean: list[float] = field(default_factory=list)
    song_scale: list[float] = field(default_factory=list)
    coef: list[list[float]] = field(default_factory=list)  # (song features, k)
    intercept: list[float] = field(default_factory=list)
    # Scene tables: style name ("*" = any) -> loudness bin -> [[level, palette, flips, p]].
    scenes: dict[str, dict[str, list[list[float]]]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)  # build report (sizes, accuracy)
    version: int = STYLE_MODEL_VERSION

    @property
    def k(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise ValueError(f"unknown light style {name!r}; choose from {self.names}") from None

    def assign(self, f: dict) -> int:
        """The nearest style of raw map features ``f``."""
        z = (style_vector(f) - np.asarray(self.mean)) / np.asarray(self.scale)
        return int(((np.asarray(self.centers) - z) ** 2).sum(axis=1).argmin())

    def song_probs(self, song: np.ndarray) -> np.ndarray:
        """Auto-pick probabilities per style for a ``song_vector``."""
        if not self.coef:
            return np.full(self.k, 1.0 / self.k)
        z = (song - np.asarray(self.song_mean)) / np.asarray(self.song_scale)
        return softmax(z @ np.asarray(self.coef) + np.asarray(self.intercept))

    def pick(self, song: np.ndarray, seed: int = 0) -> int:
        """A style for a song: drawn from the auto-pick's probabilities with ``seed``, so
        re-rolls vary but stay likely for the song."""
        p = self.song_probs(song)
        return int(np.random.default_rng(seed).choice(self.k, p=p / p.sum()))

    def scene_table(self, style: int | None, bin_: int) -> list[tuple[Scene, float]]:
        """(scene, probability) of human windows of ``style`` at loudness bin ``bin_``."""
        tables = self.scenes.get("*" if style is None else self.names[style]) or {}
        rows = tables.get(str(bin_)) or self.scenes.get("*", {}).get(str(bin_)) or []
        return [((int(r[0]), int(r[1]), int(r[2])), float(r[3])) for r in rows]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StyleModel:
        if d.get("version") != STYLE_MODEL_VERSION:
            raise ValueError(f"unsupported style model version {d.get('version')}")
        return cls(**d)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> StyleModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
