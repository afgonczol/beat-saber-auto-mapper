"""WS8b phase 2: model lighting with a v2 light prompt (style, scene plan, references)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gen_synthetic import make_analysis  # noqa: E402

from bsmapper.generation import GenerateOptions, song_features, tempo_beatmap  # noqa: E402
from bsmapper.generation.lighting import (  # noqa: E402
    LIGHT_STYLE_CHOICES,
    ModelLightingGenerator,
    configured_light_style,
)
from bsmapper.lighting.styles import STYLE_FEATURE_NAMES, StyleModel  # noqa: E402
from bsmapper.models import MapperModel, ModelConfig  # noqa: E402
from bsmapper.schema import Lightshow  # noqa: E402
from bsmapper.tokenizer import LIGHT_VOCAB_SIZES, LightPrompt  # noqa: E402

TINY = dict(
    d_model=32, n_heads=4, ff_mult=2, dropout=0.0, encoder_layers=1, decoder_layers=1,
    light_layers=1, note_memory_layers=1, max_len=192, light_max_len=192, note_memory_len=128,
)  # fmt: skip


def _styles() -> dict:
    n = len(STYLE_FEATURE_NAMES)
    rows = [[3, 3, 1, 0.7], [2, 2, 2, 0.3]]
    return StyleModel(
        names=["pulse", "calm"],
        mean=[0.0] * n,
        scale=[1.0] * n,
        centers=[[0.0] * n, [1.0] * n],
        scenes={"*": {str(b): rows for b in range(5)}},
    ).to_dict()


@pytest.fixture(scope="module")
def v2_model() -> MapperModel:
    torch.manual_seed(0)
    model = MapperModel(ModelConfig(**TINY)).eval()
    model.light_styles = _styles()
    return model


@pytest.fixture(scope="module")
def analysis():
    return make_analysis(duration=20.0)


def test_style_choices_match_the_prototypes() -> None:
    from bsmapper.lighting.styles import STYLE_PROTOTYPES

    assert set(LIGHT_STYLE_CHOICES) == {"auto", "none", *STYLE_PROTOTYPES}


def test_prompt_plan_names_a_style_and_holds_scenes(v2_model, analysis) -> None:
    beatmap = tempo_beatmap(analysis)
    feats = song_features(analysis, beatmap)
    gen = ModelLightingGenerator(v2_model, light_style="calm", max_body_tokens=24)
    plan = gen.prompt_plan(analysis, beatmap, feats)
    assert gen.last_style == "calm"
    prompts = plan["prompts"]
    assert len(prompts) == feats.shape[0] // 384
    assert {p.style for p in prompts} == {1}
    assert all(p.scene in ((3, 3, 1), (2, 2, 2), (0, 0, 0)) for p in prompts)
    assert isinstance(gen.basic_lightshow(analysis, beatmap), Lightshow)
    auto = ModelLightingGenerator(v2_model, seed=3)
    auto.prompt_plan(analysis, beatmap, feats)
    assert auto.last_style in ("pulse", "calm")
    none = ModelLightingGenerator(v2_model, light_style="none")
    assert {p.style for p in none.prompt_plan(analysis, beatmap, feats)["prompts"]} == {None}
    with pytest.raises(ValueError, match="unknown light style"):
        ModelLightingGenerator(v2_model, light_style="accent")  # not in this checkpoint


def test_without_a_style_model_the_prompt_is_empty(analysis) -> None:
    torch.manual_seed(0)
    model = MapperModel(ModelConfig(**TINY)).eval()  # v2 vocabulary, no light_styles
    beatmap = tempo_beatmap(analysis)
    plan = ModelLightingGenerator(model).prompt_plan(
        analysis, beatmap, song_features(analysis, beatmap)
    )
    assert set(plan["prompts"]) == {LightPrompt()}


def test_v1_lighting_decoder_ignores_the_style(analysis) -> None:
    torch.manual_seed(0)
    model = MapperModel(ModelConfig(**TINY, light_vocab=LIGHT_VOCAB_SIZES[1])).eval()
    gen = ModelLightingGenerator(model, light_style="calm", max_body_tokens=24)
    assert isinstance(gen.basic_lightshow(analysis, tempo_beatmap(analysis)), Lightshow)
    assert gen.last_style is None


def test_configured_light_style(tmp_path) -> None:
    cfg = tmp_path / "config.toml"
    assert configured_light_style(cfg) == "auto"
    cfg.write_text('[generate]\nlight_style = "Scenes"\n')
    assert configured_light_style(cfg) == "scenes"
    cfg.write_text('[generate]\nlight_style = "disco"\n')
    with pytest.raises(ValueError, match="light_style"):
        configured_light_style(cfg)
    assert GenerateOptions().light_style == "auto"


def test_song_vector_is_finite_on_generated_features(analysis) -> None:
    from bsmapper.lighting.styles import song_vector

    feats = song_features(analysis, tempo_beatmap(analysis))
    assert np.isfinite(song_vector(feats, 120.0)).all()
