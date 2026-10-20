"""Frame-level features and their beat-synchronous resampling.

All frame features share one hop, so they line up row for row:

- ``mel``: log-mel power in dB relative to the song's loudest bin, clipped to
  [-80, 0]. Shape (frames, n_mels).
- ``onset_strength``: librosa's onset envelope on the log-mel (positive
  spectral difference, max-filtered across bands), scaled to the song's max.
- ``spectral_flux``: L2 norm of the positive difference of the magnitude
  spectrum, scaled to the song's max.
- ``rms``: RMS energy, scaled to the song's max.
- optional ``onset_percussive`` / ``rms_harmonic`` from an HPSS split.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bsmapper.audio.decode import SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    sr: int = SAMPLE_RATE
    hop_length: int = 220  # 22050 / 220 = 100.23 frames per second
    n_fft: int = 1024
    n_mels: int = 80
    fmin: float = 30.0
    fmax: float | None = None  # None = Nyquist
    top_db: float = 80.0
    hpss: bool = False

    @property
    def frame_rate(self) -> float:
        return self.sr / self.hop_length


def _scale_to_max(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(x)) if x.size else 0.0
    return (x / peak if peak > 0 else np.zeros_like(x)).astype(np.float32)


def compute_frame_features(
    y: np.ndarray, cfg: FeatureConfig | None = None
) -> dict[str, np.ndarray]:
    import librosa

    cfg = cfg or FeatureConfig()
    stft = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length, center=True)
    mag = np.abs(stft)
    n_frames = mag.shape[1]

    def log_mel(power: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            S=power, sr=cfg.sr, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
        )
        ref = max(float(mel.max()), 1e-10)
        return librosa.power_to_db(mel, ref=ref, top_db=cfg.top_db)

    mel_db = log_mel(mag**2)
    # n_fft must be passed even with S=: librosa uses it to undo the frame centering.
    onset = librosa.onset.onset_strength(
        S=mel_db, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length, lag=1, max_size=3
    )
    flux = np.zeros(n_frames, dtype=np.float32)
    flux[1:] = np.sqrt((np.maximum(np.diff(mag, axis=1), 0.0) ** 2).sum(axis=0))
    rms = librosa.feature.rms(S=mag, frame_length=cfg.n_fft)[0]

    feats = {
        "mel": mel_db.T.astype(np.float32),
        "onset_strength": _scale_to_max(onset),
        "spectral_flux": _scale_to_max(flux),
        "rms": _scale_to_max(rms),
    }
    if cfg.hpss:
        harm, perc = librosa.decompose.hpss(stft)
        feats["onset_percussive"] = _scale_to_max(
            librosa.onset.onset_strength(
                S=log_mel(np.abs(perc) ** 2),
                sr=cfg.sr,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                lag=1,
                max_size=3,
            )
        )
        feats["rms_harmonic"] = _scale_to_max(
            librosa.feature.rms(S=np.abs(harm), frame_length=cfg.n_fft)[0]
        )
    for name, arr in feats.items():
        assert arr.shape[0] == n_frames, (name, arr.shape)
    return feats


def beat_synchronous(
    features: dict[str, np.ndarray],
    frame_rate: float,
    beat_to_seconds: Callable[[np.ndarray], np.ndarray],
    n_beats: int,
    ticks_per_beat: int = 24,
) -> dict[str, np.ndarray]:
    """Resample frame features onto the tick grid: row ``j`` is beat ``j / ticks_per_beat``.

    Linear interpolation is enough here: a tick (60 / (bpm * 24) s) is at most
    about two frames long at normal tempos, so no peak falls between ticks.
    """
    ticks = np.arange(n_beats * ticks_per_beat) / ticks_per_beat
    frame_pos = beat_to_seconds(ticks) * frame_rate
    out: dict[str, np.ndarray] = {}
    for name, arr in features.items():
        if arr.shape[0] < 2:
            continue
        pos = np.clip(frame_pos, 0, arr.shape[0] - 1)
        i0 = np.minimum(pos.astype(np.int64), arr.shape[0] - 2)
        w = (pos - i0).reshape((-1,) + (1,) * (arr.ndim - 1))
        out[name] = (arr[i0] * (1 - w) + arr[i0 + 1] * w).astype(np.float32)
    return out
