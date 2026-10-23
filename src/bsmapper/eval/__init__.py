"""WS10: offline metrics against held-out human maps, reports and run-to-run diffs.

    from bsmapper.eval import EvalConfig, run_eval
    ckpt = Path("checkpoints/note_v0/best.pt")
    cfg = EvalConfig(data=Path("data/eval_v0_test"), checkpoints=[ckpt])
    summary = run_eval(cfg)      # out/eval/run/{records.jsonl, summary.json, report.html}

Generated maps are made on each human map's own beat grid (``data.song_analysis``),
through the WS7 pipeline's generate and post-process stages. See docs/EVAL.md.
"""

from bsmapper.eval.compare import compare_rows, format_compare, load_summary
from bsmapper.eval.data import EvalSong, load_songs, snapshot_split, song_analysis
from bsmapper.eval.lighting import lighting_metrics
from bsmapper.eval.metrics import (
    difficulty_metrics,
    distributions,
    jensen_shannon,
    onset_scores,
)
from bsmapper.eval.reference import HumanReference
from bsmapper.eval.repetition import repeated_pairs, repetition_counts
from bsmapper.eval.runner import EvalConfig, aggregate, run_eval
from bsmapper.eval.systems import HumanSystem, PipelineSystem, model_system, stub_system

__all__ = [
    "EvalConfig",
    "EvalSong",
    "HumanReference",
    "HumanSystem",
    "PipelineSystem",
    "aggregate",
    "compare_rows",
    "difficulty_metrics",
    "distributions",
    "format_compare",
    "jensen_shannon",
    "lighting_metrics",
    "load_songs",
    "load_summary",
    "model_system",
    "onset_scores",
    "repeated_pairs",
    "repetition_counts",
    "run_eval",
    "snapshot_split",
    "song_analysis",
    "stub_system",
]
