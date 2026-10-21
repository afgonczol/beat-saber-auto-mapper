"""Dataset builder on the fixture maps (zipped with synthetic audio)."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
from builder_helpers import FIXTURES, make_raw, read

pytest.importorskip("librosa")
pytest.importorskip("pyarrow")

from bsmapper.data.build import (  # noqa: E402
    BUILDER_VERSION,
    BuildConfig,
    process_map,
    run_build,
    track_maps,
)
from bsmapper.data.dataset import ProcessedDataset  # noqa: E402
from bsmapper.data.mapgrid import FEATURE_DIM, shift_map  # noqa: E402
from bsmapper.data.pack import PackConfig, pack  # noqa: E402
from bsmapper.formats import read_map_report  # noqa: E402
from bsmapper.tokenizer import MapTokenizer, canonical_order, snap_map  # noqa: E402
from bsmapper.tokenizer.grid import WINDOW_TICKS  # noqa: E402


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return make_raw(tmp_path_factory.mktemp("build"), generated=3)


def _process(raw, key, work):
    status = process_map(key, str(raw / "zips" / f"{key}.zip"), str(work))
    return status, read(work / "maps" / f"{key}.json")


def test_process_fixture_map(raw, tmp_path):
    status, rec = _process(raw, "v3", tmp_path)
    assert status == "ok" and rec["version"] == BUILDER_VERSION
    assert [(d["characteristic"], d["difficulty"]) for d in rec["diffs"]] == [
        ("Standard", "Expert"),
        ("Standard", "ExpertPlus"),
        ("90Degree", "Normal"),
    ]
    assert rec["timing"]["bpm"] == 128 and rec["timing"]["bpm_changes"]
    z = np.load(tmp_path / "maps" / "v3.npz")
    assert z["features"].shape == (rec["n_windows"] * WINDOW_TICKS, FEATURE_DIM)
    assert z["features"].dtype == np.float16
    assert z["fingerprint"].dtype == np.uint32 and len(z["fingerprint"]) > 100

    # The stored windows decode to exactly the snapped difficulty.
    beatmap, _ = read_map_report(FIXTURES / "v3")
    from bsmapper.data.dataset import _windows
    from bsmapper.tokenizer import NOTE_GRAMMAR

    for i, d in enumerate(rec["diffs"]):
        windows = _windows(z[f"tok{i}"], z[f"off{i}"], [], NOTE_GRAMMAR, 16)
        assert len(windows) == rec["n_windows"]
        dm = beatmap.difficulties[i]
        expected, _ = snap_map(shift_map(dm, -rec["timing"]["phase"]))
        assert MapTokenizer().decode(windows, meta=expected) == canonical_order(expected)
        assert d["tokens"] == len(z[f"tok{i}"])
    assert rec["light"]["basic_events"] == 5


def test_rejects_modded_unreadable_and_silent_maps(raw, tmp_path):
    assert _process(raw, "v2_modded", tmp_path)[0] == "rejected:needs_mods"

    broken = raw.parent / "broken"
    broken.mkdir(exist_ok=True)
    (broken / "junk.zip").write_bytes(b"not a zip")
    st = process_map("junk", str(broken / "junk.zip"), str(tmp_path))
    assert st == "rejected:unreadable"

    no_audio = broken / "noaudio.zip"
    with zipfile.ZipFile(no_audio, "w") as z:
        for p in (FIXTURES / "v2").iterdir():
            z.write(p, p.name)
    assert process_map("noaudio", str(no_audio), str(tmp_path)) == "rejected:audio_missing"


class ClickTracker:
    """Beats every 0.5 s: what a tracker would find in the generated maps' audio."""

    name = "clicks"

    def track(self, y, sr):
        beats = np.arange(0.0, len(y) / sr - 0.1, 0.5)
        return beats, beats[::4]


def test_track_maps_writes_detected_grid(raw, tmp_path):
    items = [("gen0", str(raw / "zips" / "gen0.zip")), ("junk", str(tmp_path / "missing.zip"))]
    track_maps(items, str(tmp_path), ClickTracker())
    rec = read(tmp_path / "track" / "gen0.json")
    assert rec["status"] == "ok" and rec["tracker"] == "clicks"
    assert rec["bpm"] == pytest.approx(120.0) and rec["coverage"] == 1.0
    assert read(tmp_path / "track" / "junk.json")["status"] == "error"

    out = tmp_path / "processed"
    _process(raw, "gen0", out / "work")
    (out / "work" / "track").mkdir(parents=True)
    (out / "work" / "track" / "gen0.json").write_bytes(
        (tmp_path / "track" / "gen0.json").read_bytes()
    )
    stats = pack(out, raw)["stats"]["grid_drift"]
    assert stats["maps"] == 1 and stats["relation"] == {"same": 1}
    assert stats["median_abs_ticks_quantiles"][2] < 1.0


def test_run_build_is_resumable_and_packs(raw, tmp_path):
    out = tmp_path / "processed"
    cfg = BuildConfig(raw_dir=raw, out_dir=out, workers=2, tracker="none", pack=False)
    first = run_build(cfg)
    assert sum(first.processed.values()) == 7
    assert first.processed["rejected:needs_mods"] == 1
    again = run_build(cfg)
    assert again.skipped_done == 7 and not again.processed

    meta = pack(out, raw, cfg=PackConfig(min_notes=3))
    stats = meta["stats"]
    assert stats["maps"]["seen"] == 7
    assert stats["maps"]["rejected"] == {"needs_mods": 1}
    assert stats["difficulties"]["kept"] == 2 + 3 + 2 + 3 * 2
    assert sum(s["maps"] for s in stats["splits"].values()) == 6
    for name in ("manifest.parquet", "maps.parquet", "difficulties.parquet", "dataset.json"):
        assert (out / name).is_file()

    ds = ProcessedDataset(out)
    rows = ds.rows()
    assert len(rows) == stats["difficulties"]["kept"]
    tok = MapTokenizer()
    for row in rows:
        windows = ds.note_windows(row)
        assert len(windows) == row["n_windows"]
        assert ds.features(row).shape == (row["n_windows"] * WINDOW_TICKS, FEATURE_DIM)
        # Context and prefix rebuilt by the reader match a fresh encode.
        dm = tok.decode(windows)
        fresh = tok.encode(dm, ds.conditioning(row), n_windows=row["n_windows"])
        assert [w.input_ids() for w in windows] == [w.input_ids() for w in fresh]
        assert 0 <= row["style"] <= 1 or row["style_bucket"] == -1
    gen = [r for r in rows if r["key"].startswith("gen")]
    assert {r["style_bucket"] for r in gen} <= set(range(5))
    assert all(r["light_shard"] is not None for r in rows if r["key"] in ("v2", "v3", "v4"))
    assert ds.light_windows(next(r for r in rows if r["key"] == "v3"))


def test_pack_default_filters_reject_tiny_fixture_maps(raw, tmp_path):
    out = tmp_path / "processed"
    for key in ("v2", "gen0", "gen1"):
        _process(raw, key, out / "work")
    stats = pack(out, raw)["stats"]
    assert stats["difficulties"]["rejected"]["too_few_notes"] == 2  # the v2 fixture's
    assert stats["maps"]["rejected"] == {"no_kept_difficulties": 1}
    assert stats["difficulties"]["kept"] == 4
