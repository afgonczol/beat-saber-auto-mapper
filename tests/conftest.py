import pytest


@pytest.fixture(autouse=True)
def _no_light_effects(monkeypatch, tmp_path_factory):
    """Keep a checkout's human effect library (checkpoints/light_effects, WS8c) out of
    tests: pipelines built by ``make_pipeline`` would otherwise pick environments and
    lights from it. Tests that need a library pass one explicitly."""
    from bsmapper.generation import runtime

    empty = tmp_path_factory.mktemp("no_effects")
    monkeypatch.setattr(runtime, "DEFAULT_LIGHT_EFFECTS", empty / "light_effects")
    monkeypatch.setattr(runtime, "configured_light_effects", lambda path=None: None)
