import pytest
from synthetic import BPM, make_analysis

from bsmapper.interfaces import AudioAnalysis
from bsmapper.schema import Beatmap


@pytest.fixture
def analysis() -> AudioAnalysis:
    return make_analysis()


@pytest.fixture
def beatmap() -> Beatmap:
    return Beatmap(song_name="Synthetic", bpm=BPM)
