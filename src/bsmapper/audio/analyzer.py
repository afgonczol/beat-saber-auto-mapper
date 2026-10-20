"""The WS2 implementation of ``interfaces.AudioAnalyzer``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bsmapper.audio.beats import BeatTracker, make_tracker
from bsmapper.audio.decode import SAMPLE_RATE, load_audio
from bsmapper.audio.features import FeatureConfig, beat_synchronous, compute_frame_features
from bsmapper.audio.grid import BeatGrid, GridFit, fit_beat_grid, refine_offset
from bsmapper.audio.sections import segment_sections
from bsmapper.interfaces import AudioAnalysis

log = logging.getLogger(__name__)

TICKS_PER_BEAT = 24
# Below this share of tracked beats on the fitted grid, the grid is often wrong
# (tempo drift, live recordings, tracker confusion): see docs/audio.md.
LOW_CONFIDENCE_COVERAGE = 0.7
# Delay of the ``onset_strength`` feature's peaks behind where human mappers put
# the beat, measured with ``python -m bsmapper.audio.benchmark`` (docs/audio.md).
ONSET_LATENCY_S = 0.012


@dataclass(slots=True)
class AnalysisResult:
    """Everything the analyzer knows; ``analysis`` is the contract-level view.

    ``grid`` carries what ``AudioAnalysis`` has no field for yet (offset, BPM
    changes, beats per bar), and ``raw_beats`` / ``raw_downbeats`` are the
    tracker's unregularized output.
    """

    analysis: AudioAnalysis
    grid: BeatGrid
    fit: GridFit
    raw_beats: np.ndarray
    raw_downbeats: np.ndarray
    tracker: str
    phase_shift_s: float = 0.0  # applied by the onset-based offset refinement


class Analyzer:
    """Decode, track beats, fit a grid, compute features and sections.

    ``analysis.beat_times[i]`` is grid beat ``i``; beat 0 is the first downbeat
    at or after 0 s, so ``beat_times[0]`` is the offset. Row ``j`` of every
    ``beat_features`` array is beat ``j / 24``. ``bpm`` is the tempo at beat 0.
    """

    def __init__(
        self,
        tracker: str | BeatTracker = "auto",
        feature_config: FeatureConfig | None = None,
        beat_features: bool = True,
        allow_tempo_changes: bool = True,
        snap_bpm: bool = True,
        refine_phase: bool = True,
        onset_latency_s: float = ONSET_LATENCY_S,
    ) -> None:
        self.tracker = make_tracker(tracker) if isinstance(tracker, str) else tracker
        self.feature_config = feature_config or FeatureConfig()
        self.beat_features = beat_features
        self.allow_tempo_changes = allow_tempo_changes
        self.snap_bpm = snap_bpm
        self.refine_phase = refine_phase
        self.onset_latency_s = onset_latency_s

    def analyze(self, audio_path: Path) -> AudioAnalysis:
        return self.analyze_full(audio_path).analysis

    def analyze_full(self, audio_path: str | Path) -> AnalysisResult:
        return self.analyze_signal(load_audio(audio_path, self.feature_config.sr))

    def analyze_signal(self, y: np.ndarray, sr: int = SAMPLE_RATE) -> AnalysisResult:
        if sr != self.feature_config.sr:
            raise ValueError(f"expected {self.feature_config.sr} Hz audio, got {sr}")
        duration = len(y) / sr
        raw_beats, raw_downbeats = self.tracker.track(y, sr)
        fit = fit_beat_grid(
            raw_beats,
            raw_downbeats,
            allow_tempo_changes=self.allow_tempo_changes,
            snap=self.snap_bpm,
        )
        if fit.coverage < LOW_CONFIDENCE_COVERAGE:
            log.warning(
                "low beat-grid confidence: only %.0f%% of tracked beats fit %.2f BPM; "
                "consider a manual BPM/offset override",
                100 * fit.coverage,
                fit.grid.bpm,
            )
        cfg = self.feature_config
        feats = compute_frame_features(y, cfg)
        grid, shift = fit.grid, 0.0
        if self.refine_phase:
            grid, shift = refine_offset(
                grid, feats["onset_strength"], cfg.frame_rate, latency_s=self.onset_latency_s
            )
        beat_times = grid.beat_times(duration)
        sync = {}
        if self.beat_features:
            sync = beat_synchronous(
                feats, cfg.frame_rate, grid.beat_to_seconds, len(beat_times), TICKS_PER_BEAT
            )
        sections = segment_sections(
            feats, cfg.frame_rate, beat_times, duration, beats_per_bar=grid.beats_per_bar
        )
        analysis = AudioAnalysis(
            duration_s=duration,
            bpm=grid.bpm,
            beat_times=beat_times,
            downbeat_times=beat_times[:: grid.beats_per_bar],
            sections=sections,
            features=feats,
            frame_rate=cfg.frame_rate,
            beat_features=sync,
            ticks_per_beat=TICKS_PER_BEAT,
        )
        return AnalysisResult(
            analysis=analysis,
            grid=grid,
            fit=fit,
            raw_beats=raw_beats,
            raw_downbeats=raw_downbeats,
            tracker=self.tracker.name,
            phase_shift_s=shift,
        )
