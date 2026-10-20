"""Real BeatSaver calls. Run with `pytest -m slow tests/data`."""

import pytest
from typer.testing import CliRunner

from bsmapper.cli import app
from bsmapper.data import BeatSaverClient, FilterConfig, MapRecord, check, read_manifest

pytestmark = pytest.mark.slow


def test_live_search_returns_filterable_maps():
    with BeatSaverClient(requests_per_second=1.0) as client:
        data = client.search(0, FilterConfig().search_params())
    recs = [r for r in map(MapRecord.from_api, data["docs"]) if r is not None]
    assert len(recs) >= 50
    passing = [r for r in recs if check(r, FilterConfig()) is None]
    assert len(passing) >= len(recs) // 2
    assert all(not r.automapper for r in passing)


def test_live_scrape_two_maps(tmp_path):
    out = tmp_path / "raw"
    args = ["data", "scrape", "--out", str(out), "--limit", "2", "--min-free-gb", "0"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    entries = read_manifest(out)
    assert len(entries) == 2
    assert all((out / e.zip_path).stat().st_size == e.bytes for e in entries)
