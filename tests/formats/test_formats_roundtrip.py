"""read -> write -> read must give the same Beatmap (WS1 acceptance criterion)."""

import json
from pathlib import Path

import pytest

from bsmapper.formats import MapFolderReader, MapFolderWriter, read_map, write_map

FIXTURES = Path(__file__).parent.parent / "fixtures" / "maps"
OWN_VERSION = {"v2": "v2", "v3": "v3", "v4": "v4", "v2_modded": "v2"}

# Conversions that are lossless for the given fixture. The rest lose things the
# target version cannot express: v2 has no chains, angle offsets or group lighting,
# and v4 stores rotations as per-object lanes, so late rotations become early ones.
LOSSLESS = [
    ("v2", "v3"),
    ("v3", "v3"),
    ("v4", "v3"),
    ("v4", "v4"),
    ("v2_modded", "v3"),
    ("v2_modded", "v4"),
]


@pytest.mark.parametrize("fixture", sorted(OWN_VERSION))
def test_round_trip_in_own_version(fixture, tmp_path):
    original = read_map(FIXTURES / fixture)
    write_map(original, tmp_path, OWN_VERSION[fixture])
    assert read_map(tmp_path) == original


@pytest.mark.parametrize(("fixture", "version"), LOSSLESS)
def test_lossless_conversions(fixture, version, tmp_path):
    original = read_map(FIXTURES / fixture)
    write_map(original, tmp_path, version)
    assert read_map(tmp_path) == original


@pytest.mark.parametrize("fixture", sorted(OWN_VERSION))
@pytest.mark.parametrize("version", ["v2", "v3", "v4"])
def test_writing_is_idempotent(fixture, version, tmp_path):
    """Whatever survives one conversion must survive the next one unchanged."""
    first, second = tmp_path / "a", tmp_path / "b"
    write_map(read_map(FIXTURES / fixture), first, version)
    write_map(read_map(first), second, version)
    files = sorted(p.name for p in first.iterdir())
    assert files == sorted(p.name for p in second.iterdir())
    for name in files:
        assert json.loads((first / name).read_text("utf-8")) == json.loads(
            (second / name).read_text("utf-8")
        ), name


def test_protocol_classes(tmp_path):
    reader, writer = MapFolderReader(), MapFolderWriter()
    beatmap = reader.read(FIXTURES / "v3")
    assert reader.last_report is not None
    assert writer.write(beatmap, tmp_path) == tmp_path
    assert (
        json.loads((tmp_path / "ExpertPlusStandard.dat").read_text("utf-8"))["version"] == "3.3.0"
    )
    assert reader.read(tmp_path) == beatmap
