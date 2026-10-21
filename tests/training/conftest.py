from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from bsmapper.training.synthetic import write_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    return write_synthetic_dataset(tmp_path_factory.mktemp("synthetic"), n_maps=6, n_windows=3)
