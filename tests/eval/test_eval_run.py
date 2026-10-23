"""End to end on a synthetic click-track dataset (CI-safe: no audio, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from synth import N_BEATS, write_dataset  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from bsmapper.cli import app  # noqa: E402
from bsmapper.eval import (  # noqa: E402
    EvalConfig,
    compare_rows,
    format_compare,
    load_songs,
    run_eval,
    snapshot_split,
    song_analysis,
)
from bsmapper.eval.compare import verdict  # noqa: E402
from bsmapper.eval.runner import RECORDS, read_records  # noqa: E402
from bsmapper.eval.systems import HUMAN, STUB, HumanSystem, checkpoint_label  # noqa: E402
from bsmapper.generation import song_features, tempo_beatmap  # noqa: E402
from bsmapper.schema import Characteristic, Difficulty  # noqa: E402


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    return write_dataset(tmp_path_factory.mktemp("ds") / "processed")


@pytest.fixture(scope="module")
def run(dataset, tmp_path_factory) -> tuple[EvalConfig, dict]:
    out = tmp_path_factory.mktemp("eval") / "run"
    cfg = EvalConfig(data=dataset, out=out, style_songs=1, styles=(0.0, 1.0), label="synthetic")
    return cfg, run_eval(cfg)


def test_songs_load_on_the_human_grid(dataset) -> None:
    songs = load_songs(dataset)
    assert [s.key for s in songs] == ["a0", "a1"]
    song = songs[0]
    assert set(song.humans) == {
        (Characteristic.STANDARD, Difficulty.HARD),
        (Characteristic.STANDARD, Difficulty.EXPERT),
    }
    assert song.lightshow is not None and song.timing_exact
    analysis = song_analysis(song)
    assert analysis.beat_times[1] == pytest.approx(0.5)
    assert len(analysis.sections) >= 1
    # The generators' feature resampling gives back the dataset's rows exactly.
    feats = song_features(analysis, tempo_beatmap(analysis))
    n = min(len(feats), len(song.features))
    assert np.allclose(feats[:n], song.features[:n], atol=1e-3)
    assert load_songs(dataset, keys=["a1"])[0].key == "a1"
    assert load_songs(dataset, difficulties=[Difficulty.EXPERT])[0].humans.keys() == {
        (Characteristic.STANDARD, Difficulty.EXPERT)
    }


def test_run_writes_summary_report_and_records(run) -> None:
    cfg, summary = run
    assert (cfg.out / "summary.json").is_file() and (cfg.out / RECORDS).is_file()
    html = (cfg.out / "report.html").read_text(encoding="utf-8")
    assert "<title>" in html and "Headline" in html and "Repetition" in html
    assert set(summary["systems"]) == {HUMAN, STUB}
    assert summary["meta"]["dataset"]["songs"] == 2
    groups = summary["systems"][STUB]["groups"]
    assert {"Standard/Hard", "Standard/Expert", "Standard/all", "all"} <= set(groups)


def test_human_reference_values(run) -> None:
    _, summary = run
    human = summary["systems"][HUMAN]["groups"]["all"]
    assert human["nps_ratio"] == pytest.approx(1.0)
    assert "onset_f1_2" not in human
    assert human["errors_per_100"] == 0.0
    assert human["rep_exact"] == 1.0  # the synthetic maps repeat every repeated section
    assert human["rep_distance"] == 0.0
    # Hard (2 NPS) sits a level below the reference ladder's Hard median (3).
    hard = summary["systems"][HUMAN]["groups"]["Standard/Hard"]
    assert hard["step_offset"] == pytest.approx(-1.0, abs=0.1)
    assert summary["systems"][HUMAN]["lighting"]["on_beat"] == 1.0


def test_stub_scores(run) -> None:
    _, summary = run
    stub = summary["systems"][STUB]
    g = stub["groups"]["all"]
    assert 0.3 < g["onset_f1_2"] <= 1.0  # the stub follows the clicks, so does the human
    assert g["errors_per_100"] == 0.0
    assert 0.0 <= g["jsd_direction"] <= 1.0 and 0.0 <= g["rep_distance"] <= 1.0
    assert stub["lighting"]["sources"] == {"rules": 2}
    style = stub["style"]
    assert set(style["by_style"]) == {"0", "1"} and np.isfinite(style["tech_delta"])
    assert stub["run"]["songs"] == 2 and stub["run"]["failed"] == 0


def test_resume_skips_finished_songs(run, dataset) -> None:
    cfg, summary = run
    before = len(read_records(cfg.out / RECORDS))
    calls = []

    class Counting(HumanSystem):
        def run(self, *args, **kw):
            calls.append(1)
            return super().run(*args, **kw)

    cfg2 = EvalConfig(data=dataset, out=cfg.out, resume=True, style_songs=0)
    again = run_eval(cfg2, systems=[Counting()])
    assert calls == [] and len(read_records(cfg.out / RECORDS)) == before
    assert again["systems"][STUB]["groups"]["all"] == summary["systems"][STUB]["groups"]["all"]


def test_compare_two_runs(run, tmp_path) -> None:
    cfg, summary = run
    b = json.loads(json.dumps(summary))
    b["systems"]["note_v1"] = json.loads(json.dumps(b["systems"][STUB]))
    b["systems"]["note_v1"]["groups"]["all"]["onset_f1_2"] += 0.1
    sys_a, sys_b, rows = compare_rows(summary, b)
    assert (sys_a, sys_b) == (STUB, "note_v1")
    f1 = next(r for r in rows if r["group"] == "all" and r["metric"] == "onset_f1_2")
    assert f1["delta"] == pytest.approx(0.1) and f1["verdict"] == "better"
    text = format_compare(summary, b)
    assert "B is better on 1 rows and worse on 0" in text
    assert verdict(3.0, 3.5, "human", 4.0) == "better"
    assert verdict(3.0, 5.5, "human", 4.0) == "worse"
    with pytest.raises(ValueError):
        compare_rows(summary, b, sys_b="nope")
    path_b = tmp_path / "b.json"
    path_b.write_text(json.dumps(b), encoding="utf-8")
    res = CliRunner().invoke(
        app, ["evaluate", "--compare", str(cfg.out / "summary.json"), str(path_b)]
    )
    assert res.exit_code == 0, res.output
    assert "onset F1 ±2 ticks" in res.output


def test_cli_runs_and_snapshots(dataset, tmp_path) -> None:
    snap = tmp_path / "snap"
    res = CliRunner().invoke(app, ["evaluate", "--data", str(dataset), "--snapshot", str(snap)])
    assert res.exit_code == 0, res.output
    counts = snapshot_split(dataset, tmp_path / "snap2")
    assert counts == {"maps": 2, "shard_files": 3, "timings": 2}
    out = tmp_path / "out"
    args = ["evaluate", "--data", str(snap), "--out", str(out), "--limit", "1"]
    res = CliRunner().invoke(app, [*args, "--no-lighting", "--style-songs", "0"])
    assert res.exit_code == 0, res.output
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["meta"]["dataset"]["keys"] == ["a0"]
    assert "lighting" not in summary["systems"][STUB]


def test_checkpoint_label() -> None:
    assert checkpoint_label("checkpoints/note_v0/best.pt") == "note_v0"
    assert checkpoint_label("checkpoints/note_v1/step_0010000.pt") == "note_v1/step_0010000"


def test_model_system_on_a_tiny_checkpoint(dataset, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from bsmapper.eval.systems import PipelineSystem
    from bsmapper.generation import ModelGenerator, ModelSettings, Pipeline
    from bsmapper.models import MapperModel, ModelConfig

    torch.manual_seed(0)
    cfg = ModelConfig(
        d_model=32,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        encoder_layers=1,
        decoder_layers=1,
        light_layers=1,
        note_memory_layers=1,
        max_len=192,
        light_max_len=192,
        note_memory_len=128,
    )
    model = MapperModel(cfg).eval()
    settings = ModelSettings(max_body_tokens=48, density_boosts=())
    pipeline = Pipeline(ModelGenerator(model, settings), light_model=model)
    system = PipelineSystem(pipeline, "tiny", batch_difficulties=1)
    out = tmp_path / "model"
    ec = EvalConfig(data=dataset, out=out, limit=1, style_songs=0, stub=False, lighting=False)
    summary = run_eval(ec, systems=[HumanSystem(), system])
    tiny = summary["systems"]["tiny"]
    assert tiny["run"]["songs"] == 1 and tiny["run"]["failed"] == 0
    assert tiny["groups"]["all"]["n"] == 2
    assert N_BEATS == 64


def test_rescore_replays_saved_maps(run, dataset, tmp_path) -> None:
    cfg, summary = run
    from bsmapper.eval.store import load_output, saved_systems

    assert saved_systems(cfg.out) == [STUB]
    out = load_output(cfg.out / "maps" / STUB / "a0.json.gz")
    assert out.dmaps and out.lightshow is not None and out.lighting_source == "rules"
    again = run_eval(
        EvalConfig(
            data=dataset, out=tmp_path / "re", rescore=cfg.out, style_songs=1, styles=(0.0, 1.0)
        )
    )
    for key in ("onset_f1_2", "nps", "errors_per_100", "jsd_pattern_3gram", "rep_exact"):
        a = summary["systems"][STUB]["groups"]["all"][key]
        assert again["systems"][STUB]["groups"]["all"][key] == pytest.approx(a)
    assert again["systems"][STUB]["lighting"]["on_beat"] == pytest.approx(
        summary["systems"][STUB]["lighting"]["on_beat"]
    )
    assert again["systems"][STUB]["style"]["tech_delta"] == pytest.approx(
        summary["systems"][STUB]["style"]["tech_delta"]
    )
