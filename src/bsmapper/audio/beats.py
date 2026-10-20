"""Beat and downbeat trackers.

Two backends behind one small protocol:

- :class:`BeatThisTracker` (default when installed): CPJKU's beat_this
  transformer (ISMIR 2024). It tracks beats and downbeats jointly and is much
  more accurate on real music; it needs torch and downloads its checkpoint on
  first use. See ``docs/audio.md`` for the comparison.
- :class:`LibrosaBeatTracker`: librosa's dynamic-programming tracker. It needs
  no model, so it is the fallback and the backend used in CI. librosa has no
  downbeat tracker, so downbeats come from a bass-onset heuristic.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class BeatTracker(Protocol):
    name: str

    def track(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (beat_times, downbeat_times) in seconds."""
        ...


class LibrosaBeatTracker:
    name = "librosa"

    def __init__(self, hop_length: int = 256, tightness: float = 400.0) -> None:
        self.hop_length = hop_length
        self.tightness = tightness

    def track(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        import librosa

        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        _, beats = librosa.beat.beat_track(
            onset_envelope=env,
            sr=sr,
            hop_length=self.hop_length,
            tightness=self.tightness,
            units="time",
        )
        beats = np.asarray(beats, dtype=np.float64)
        return beats, _downbeats_from_bass(y, sr, beats, self.hop_length)


def _downbeats_from_bass(
    y: np.ndarray, sr: int, beats: np.ndarray, hop_length: int, beats_per_bar: int = 4
) -> np.ndarray:
    """Pick the bar phase whose beats carry the most low-frequency onset energy.

    Kicks and bass notes cluster on downbeats in most popular music, so this is
    a decent guess when no downbeat model is available.
    """
    import librosa

    if len(beats) < 2 * beats_per_bar:
        return beats[:1]
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, fmax=200.0, n_mels=32)
    frames = np.clip(librosa.time_to_frames(beats, sr=sr, hop_length=hop_length), 0, len(env) - 1)
    # Take the max in a small window: the tracked beat can be a frame early or late.
    strength = np.array([env[max(f - 2, 0) : f + 3].max() for f in frames])
    phase = int(np.argmax([strength[p::beats_per_bar].mean() for p in range(beats_per_bar)]))
    return beats[phase::beats_per_bar]


class BeatThisTracker:
    name = "beat_this"

    def __init__(
        self, checkpoint: str = "final0", device: str | None = None, dbn: bool = False
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.dbn = dbn
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import beat_this.inference  # noqa: F401
        except ImportError:
            return False
        return True

    def track(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        from beat_this.inference import Audio2Frames

        if self._model is None:
            import torch
            from beat_this.inference import Audio2Beats

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = Audio2Beats(checkpoint_path=self.checkpoint, device=device, dbn=self.dbn)
        beat_logits, downbeat_logits = Audio2Frames.__call__(self._model, y, sr)
        beats, downbeats = self._model.frames2beats(beat_logits, downbeat_logits)
        fps = self._model.frames2beats.fps
        beats = _refine_peaks(np.asarray(beats, dtype=np.float64), beat_logits.cpu().numpy(), fps)
        downbeats = np.asarray(downbeats, dtype=np.float64)
        if len(beats) and len(downbeats):
            # beat_this puts each downbeat on a beat; keep them on the refined beats.
            idx = np.abs(beats[None, :] - downbeats[:, None]).argmin(axis=1)
            downbeats = np.unique(beats[idx])
        return beats, downbeats


def _refine_peaks(times: np.ndarray, logits: np.ndarray, fps: float) -> np.ndarray:
    """Sub-frame peak positions by parabolic interpolation of the logits.

    beat_this predicts at 50 fps, so raw beats are quantized to 20 ms. When the
    beat period is a whole number of frames, that error is the same on every
    beat and survives the grid fit; interpolation removes most of it.
    """
    if len(times) == 0 or len(logits) < 3:
        return times
    f = np.clip(np.round(times * fps).astype(np.int64), 1, len(logits) - 2)
    y0, y1, y2 = logits[f - 1], logits[f], logits[f + 1]
    denom = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(denom < 0, 0.5 * (y0 - y2) / denom, 0.0)
    return (f + np.clip(delta, -0.5, 0.5)) / fps


def make_tracker(name: str = "auto", **kwargs) -> BeatTracker:
    """``"auto"`` picks beat_this when it is installed, else librosa."""
    if name == "auto":
        if BeatThisTracker.available():
            return BeatThisTracker(**kwargs)
        log.warning("beat_this is not installed; falling back to librosa's beat tracker")
        return LibrosaBeatTracker()
    if name == "beat_this":
        return BeatThisTracker(**kwargs)
    if name == "librosa":
        return LibrosaBeatTracker(**kwargs)
    raise ValueError(f"unknown beat tracker {name!r} (expected auto, beat_this or librosa)")
