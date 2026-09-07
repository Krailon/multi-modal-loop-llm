"""Validation of the model's public configuration contract."""

from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from multimodal_loop.model.config import ModelConfig

POSITIVE_INTEGER_FIELDS = (
    "vocab_size",
    "max_seq_len",
    "d_model",
    "n_heads",
    "d_ff",
    "n_recurrent_layers",
    "recurrence_depth",
    "image_size",
    "patch_size",
    "num_channels",
)
OPTIONAL_STACK_FIELDS = ("n_prelude_layers", "n_coda_layers")


def test_default_dimensions() -> None:
    config = ModelConfig()

    assert config.d_model == 64
    assert config.image_size == 32
    assert config.recurrence_depth == 2
    assert config.head_dim == 16
    assert config.num_patches == 16
    assert config.patch_dim == 192


def test_custom_dimensions_and_empty_outer_stacks() -> None:
    config = ModelConfig(
        vocab_size=32,
        max_seq_len=64,
        d_model=24,
        n_heads=6,
        d_ff=48,
        n_prelude_layers=0,
        n_recurrent_layers=2,
        n_coda_layers=0,
        recurrence_depth=5,
        image_size=12,
        patch_size=3,
        num_channels=1,
        dropout=0.25,
        layer_norm_eps=1e-6,
    )

    assert config.head_dim == 4
    assert config.num_patches == 16
    assert config.patch_dim == 9
    assert config.n_prelude_layers == config.n_coda_layers == 0
    assert config.recurrence_depth == 5
    assert ModelConfig(**asdict(config)) == config


def test_configuration_changes_require_a_new_validated_instance() -> None:
    config = ModelConfig()

    with pytest.raises(FrozenInstanceError):
        config.d_model = 63
    with pytest.raises(ValueError, match="d_model must be divisible by n_heads"):
        replace(config, d_model=63)


@pytest.mark.parametrize("field", POSITIVE_INTEGER_FIELDS)
@pytest.mark.parametrize("value", [0, -1])
def test_positive_integer_fields_reject_nonpositive_values(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: value})


@pytest.mark.parametrize("field", OPTIONAL_STACK_FIELDS)
def test_outer_stack_counts_reject_negative_values(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: -1})


@pytest.mark.parametrize("field", POSITIVE_INTEGER_FIELDS + OPTIONAL_STACK_FIELDS)
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_integer_fields_reject_other_types(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=field):
        ModelConfig(**{field: value})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"d_model": 63}, "d_model must be divisible by n_heads"),
        ({"n_heads": 128}, "d_model must be divisible by n_heads"),
        ({"image_size": 30}, "image_size must be divisible by patch_size"),
        ({"patch_size": 64}, "image_size must be divisible by patch_size"),
    ],
)
def test_incompatible_dimensions(overrides: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelConfig(**overrides)


@pytest.mark.parametrize("field", ["dropout", "layer_norm_eps"])
@pytest.mark.parametrize("value", [True, "0.1", None, 1j])
def test_real_fields_reject_other_types(field: str, value: object) -> None:
    with pytest.raises(TypeError, match=field):
        ModelConfig(**{field: value})


@pytest.mark.parametrize("field", ["dropout", "layer_norm_eps"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_real_fields_reject_nonfinite_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: value})


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.1])
def test_dropout_range(value: float) -> None:
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=value)


@pytest.mark.parametrize("value", [0.0, -1e-5])
def test_layer_norm_epsilon_must_be_positive(value: float) -> None:
    with pytest.raises(ValueError, match="layer_norm_eps"):
        ModelConfig(layer_norm_eps=value)


@pytest.mark.parametrize("dropout", [0, 0.999])
def test_valid_real_boundaries(dropout: float) -> None:
    config = ModelConfig(dropout=dropout, layer_norm_eps=1)

    assert config.dropout == dropout
    assert config.layer_norm_eps == 1
