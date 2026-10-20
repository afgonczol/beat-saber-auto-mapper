"""Decode any audio file to a mono float32 signal at a fixed sample rate.

ffmpeg is preferred because it is what WS1 uses to write ``song.egg``, so the
timeline we analyze matches the one the game plays (encoder delay, gapless
trimming). soundfile + soxr resampling is the fallback when ffmpeg is missing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050


class AudioDecodeError(RuntimeError):
    pass


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return the file as a mono float32 array at ``sr`` Hz."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    errors: list[str] = []
    if shutil.which("ffmpeg"):
        try:
            return _load_ffmpeg(path, sr)
        except AudioDecodeError as e:
            errors.append(f"ffmpeg: {e}")
    try:
        return _load_soundfile(path, sr)
    except Exception as e:  # soundfile raises its own LibsndfileError, among others
        errors.append(f"soundfile: {e}")
    raise AudioDecodeError(f"could not decode {path}: " + "; ".join(errors))


def _load_ffmpeg(path: Path, sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioDecodeError(proc.stderr.decode(errors="replace").strip()[-500:])
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if y.size == 0:
        raise AudioDecodeError("no audio samples decoded")
    return y


def _load_soundfile(path: Path, sr: int) -> np.ndarray:
    import soundfile as sf

    # Open via a file object so libsndfile sniffs the format from the header:
    # Beat Saber's ``.egg`` is plain Ogg Vorbis with an unknown extension.
    with path.open("rb") as f:
        y, file_sr = sf.read(f, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        import soxr

        y = soxr.resample(y, file_sr, sr, quality="HQ").astype(np.float32)
    return y
