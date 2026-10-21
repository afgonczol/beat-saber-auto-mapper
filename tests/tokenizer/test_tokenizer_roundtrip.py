"""DifficultyMap -> tokens -> DifficultyMap must be exact (WS4 acceptance criterion)."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from bsmapper.formats import read_map
from bsmapper.schema import Characteristic, Color, CutDirection, Difficulty, DifficultyMap, Note
from bsmapper.tokenizer import (
    Conditioning,
    LightingTokenizer,
    MapTokenizer,
    OffGridError,
    canonical_order,
    prepare_lightshow,
    snap_map,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "maps"
SEEDS = range(150)


def _stem(dmap: DifficultyMap) -> str:
    return f"{dmap.difficulty}{'' if dmap.characteristic == 'Standard' else dmap.characteristic}"


def _fixture_params():
    out = []
    for name in ("v2", "v2_modded", "v3", "v4"):
        beatmap = read_map(FIXTURES / name)
        for dmap in beatmap.difficulties:
            out.append(pytest.param(dmap, id=f"{name}-{_stem(dmap)}"))
    return out


def _fixture_lightshows():
    out = []
    for name in ("v2", "v2_modded", "v3", "v4"):
        beatmap = read_map(FIXTURES / name)
        out.append(pytest.param(beatmap.lightshow, id=f"{name}-song"))
        for dmap in beatmap.difficulties:
            if dmap.lightshow is not None:
                out.append(pytest.param(dmap.lightshow, id=f"{name}-{_stem(dmap)}"))
    return out


def _gameplay(dmap: DifficultyMap) -> dict[str, Counter]:
    return {
        name: Counter(getattr(dmap, name))
        for name in ("notes", "bombs", "obstacles", "arcs", "chains", "rotations")
    }


# --- fixtures -------------------------------------------------------------------


@pytest.mark.parametrize("dmap", _fixture_params())
def test_fixture_round_trip_is_exact(dmap):
    tok = MapTokenizer()
    decoded = tok.decode(tok.encode(dmap), meta=dmap)
    assert decoded == canonical_order(dmap)
    # Same objects with the very same beat values, nothing snapped or reordered away.
    assert _gameplay(decoded) == _gameplay(dmap)


@pytest.mark.parametrize("dmap", _fixture_params())
def test_fixtures_need_no_snapping(dmap):
    snapped, report = snap_map(dmap)
    assert snapped == dmap
    assert report.off_grid == 0 and not report.dropped and not report.quantized


@pytest.mark.parametrize("dmap", _fixture_params())
def test_fixture_round_trip_without_meta_keeps_mode(dmap):
    tok = MapTokenizer()
    decoded = tok.decode(tok.encode(dmap))
    assert (decoded.characteristic, decoded.difficulty) == (dmap.characteristic, dmap.difficulty)
    assert _gameplay(decoded) == _gameplay(dmap)


@pytest.mark.parametrize("lightshow", _fixture_lightshows())
def test_fixture_lightshow_round_trip(lightshow):
    prepared, _ = prepare_lightshow(lightshow)
    tok = LightingTokenizer()
    assert tok.decode(tok.encode(prepared)) == prepared


def test_vanilla_fixture_lightshow_is_lossless():
    """Without Chroma colours or custom data, prepare_lightshow changes nothing except
    ``uses_chroma``, which is derived from the events (this fixture sets it anyway)."""
    lightshow = read_map(FIXTURES / "v2_modded").lightshow
    prepared, report = prepare_lightshow(lightshow)
    assert replace(prepared, uses_chroma=lightshow.uses_chroma) == lightshow
    assert not prepared.uses_chroma
    assert not report.quantized and not report.dropped


# --- randomized maps ----------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_random_on_grid_maps_round_trip_exactly(seed, random_map):
    dmap = random_map(seed)
    tok = MapTokenizer(context_events=seed % 20)
    cond = Conditioning.from_map(dmap, nps=seed / 10, style=(seed % 11) / 10)
    decoded = tok.decode(tok.encode(dmap, cond), meta=dmap)
    assert decoded == canonical_order(dmap)
    assert _gameplay(decoded) == _gameplay(dmap)


@pytest.mark.parametrize("seed", SEEDS)
def test_random_messy_maps_round_trip_after_snapping(seed, random_map):
    dmap = random_map(seed, messy=True)
    snapped, report = snap_map(dmap)
    assert report.objects == sum(c.total() for c in _gameplay(dmap).values())
    tok = MapTokenizer()
    assert tok.decode(tok.encode(snapped), meta=snapped) == canonical_order(snapped)
    # Snapping is idempotent: a snapped map is already on every grid.
    again, report2 = snap_map(snapped)
    assert again == snapped
    assert report2.off_grid == 0 and not report2.dropped and not report2.quantized


@pytest.mark.parametrize("seed", SEEDS)
def test_decoding_is_a_fixed_point(seed, random_map):
    tok = MapTokenizer()
    once = tok.decode(tok.encode(random_map(seed)))
    windows = tok.encode(once)
    assert tok.decode(windows) == once
    assert tok.encode(tok.decode(windows)) == windows


@pytest.mark.parametrize("seed", SEEDS)
def test_random_lightshows_round_trip(seed, random_lightshow):
    tok = LightingTokenizer(context_events=seed % 5)
    lightshow = random_lightshow(seed)
    assert tok.decode(tok.encode(lightshow)) == lightshow
    messy, _ = prepare_lightshow(random_lightshow(seed, messy=True))
    assert tok.decode(tok.encode(messy)) == messy
    assert prepare_lightshow(messy)[0] == messy


def test_decode_ids_matches_decode(random_map):
    tok = MapTokenizer()
    dmap = random_map(3)
    windows = tok.encode(dmap)
    assert tok.decode_ids([w.input_ids() for w in windows]) == tok.decode(windows)


# --- strictness -----------------------------------------------------------------


def _one_note(**kwargs) -> DifficultyMap:
    fields = {"beat": 1.0, "x": 0, "y": 0, "color": Color.RED, "direction": CutDirection.DOWN}
    note = Note(**(fields | kwargs))
    return DifficultyMap(Characteristic.STANDARD, Difficulty.EXPERT, notes=[note])


@pytest.mark.parametrize(
    "dmap",
    [
        _one_note(beat=1.01),
        _one_note(beat=-1.0),
        _one_note(x=4),
        _one_note(angle_offset=7),
    ],
)
def test_encode_refuses_what_it_cannot_round_trip(dmap):
    with pytest.raises(OffGridError):
        MapTokenizer().encode(dmap)
    snapped, report = snap_map(dmap)
    MapTokenizer().encode(snapped)  # fine after snapping
    assert report.off_grid + sum(report.dropped.values()) + sum(report.quantized.values()) == 1


def test_trailing_empty_windows(random_map):
    tok = MapTokenizer()
    dmap = replace(
        random_map(1), notes=[], bombs=[], arcs=[], chains=[], rotations=[], obstacles=[]
    )
    windows = tok.encode(dmap, n_windows=4)
    assert len(windows) == 4
    assert all(w.body == [] for w in windows)
    assert tok.decode(windows, meta=dmap) == dmap
