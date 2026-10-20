"""Slow tests: beat_this (torch + checkpoint download) and real songs from local map folders.

Run with ``pytest -m slow tests/audio``. Real-song tests read map folders from
``BSMAPPER_MAPS_DIR`` (default: the Steam CustomLevels folder) and skip if absent.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("beat_this")

from bsmapper.audio import Analyzer, BeatThisTracker  # noqa: E402
from bsmapper.audio.benchmark import read_human_grid, score_grid  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_audio_analyzer import BPM, OFFSET, TICK_S, click_track  # noqa: E402

pytestmark = pytest.mark.slow

DEFAULT_MAPS = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\Beat Saber\Beat Saber_Data\CustomLevels"
)
MAPS_DIR = Path(os.environ.get("BSMAPPER_MAPS_DIR", DEFAULT_MAPS))


@pytest.fixture(scope="module")
def analyzer():
    return Analyzer(tracker=BeatThisTracker())


def test_beat_this_click_track(analyzer):
    res = analyzer.analyze_signal(click_track(loud_from=24.0 + OFFSET))
    assert res.tracker == "beat_this"
    assert res.grid.bpm == BPM
    assert abs(res.grid.offset_s - OFFSET) <= TICK_S
    assert res.fit.coverage > 0.9


def _reference_maps(limit: int = 10):
    """Single-BPM maps whose notes sit on the BPM grid (so the grid is a trustworthy reference)."""
    if not MAPS_DIR.is_dir():
        return []
    out = []
    for d in sorted(p for p in MAPS_DIR.iterdir() if p.is_dir()):
        try:
            human = read_human_grid(d)
        except (OSError, ValueError, KeyError):
            continue
        trusted = human and not human.has_bpm_changes and abs(human.note_phase) * 24 < 1
        if trusted and human.audio.is_file():
            out.append(human)
        if len(out) >= limit:
            break
    return out


def test_real_songs_within_one_tick(analyzer):
    maps = _reference_maps()
    if len(maps) < 5:
        pytest.skip(f"need map folders with songs in {MAPS_DIR}")
    good = []
    for human in maps:
        res = analyzer.analyze_full(human.audio)
        _, _, _, within = score_grid(res.grid, human, res.analysis.duration_s)
        good.append(within >= 0.9)
        print(f"{human.name}: bpm {res.grid.bpm} vs {human.bpm}, within 1 tick {within:.0%}")
    # Old maps are sometimes hand-timed loosely; demand 8 of 10, not all.
    assert np.mean(good) >= 0.8
