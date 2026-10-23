import pytest


@pytest.fixture(autouse=True)
def _no_user_config(monkeypatch, tmp_path):
    """Tests never read the developer's ~/.bsmapper/config.toml (tests may set their own)."""
    monkeypatch.setenv("BSMAPPER_CONFIG", str(tmp_path / "no-user-config.toml"))
