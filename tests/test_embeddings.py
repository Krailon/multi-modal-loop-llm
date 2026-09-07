"""Correctness checks for shared image/text sequence construction."""

from collections.abc import Iterator
from dataclasses import replace

import pytest
import torch

from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.embeddings import MultimodalEmbedding
from multimodal_loop.model.recurrent_core import RecurrentTransformerCore


@pytest.fixture(autouse=True)
def seeded_cpu_rng() -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        vocab_size=8,
        max_seq_len=8,
        d_model=4,
        n_heads=1,
        d_ff=8,
        image_size=4,
        patch_size=2,
        num_channels=1,
    )


@pytest.mark.parametrize("num_channels", [None, 1, 3])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shapes_and_cpu_dtypes(
    config: ModelConfig,
    num_channels: int | None,
    index_dtype: torch.dtype,
    dtype: torch.dtype,
) -> None:
    config = replace(config, num_channels=num_channels or 1)
    embedding = MultimodalEmbedding(config).to(device="cpu", dtype=dtype)
    input_ids = torch.tensor([[0, 1, 7], [2, 3, 4]], dtype=index_dtype)
    images = None if num_channels is None else torch.randn(2, num_channels, 4, 4, dtype=dtype)

    tokens = embedding(input_ids, images)

    assert tokens.shape == (2, 3 if images is None else 7, config.d_model)
    assert tokens.device == input_ids.device
    assert tokens.dtype == dtype
    assert torch.isfinite(tokens).all()


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_exact_content_position_and_modality_sums(
    config: ModelConfig, with_images: bool, training: bool
) -> None:
    # Embedding assembly is a plain sum, even when transformer dropout is enabled.
    embedding = MultimodalEmbedding(replace(config, dropout=0.75)).train(training)
    with torch.no_grad():
        embedding.text_embedding.weight.copy_(torch.arange(32).reshape(8, 4) + 100)
        embedding.image_embedding.projection.weight.copy_(torch.eye(4))
        embedding.image_embedding.projection.bias.zero_()
        embedding.position_embedding.weight.copy_(torch.arange(8).unsqueeze(1) * 10)
        embedding.modality_embedding.weight.copy_(
            torch.tensor([[-20, -30, -40, -50], [1000, 2000, 3000, 4000]])
        )

    input_ids = torch.tensor([[0, 7, 2], [1, 2, 3]])
    text_content = torch.tensor(
        [
            [[100, 101, 102, 103], [128, 129, 130, 131], [108, 109, 110, 111]],
            [[104, 105, 106, 107], [108, 109, 110, 111], [112, 113, 114, 115]],
        ],
        dtype=torch.float32,
    )
    text_modality = torch.tensor([-20, -30, -40, -50])
    if with_images:
        images = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
        first_image_patches = torch.tensor(
            [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
        )
        image_content = torch.stack((first_image_patches, first_image_patches + 16))
        image_positions = torch.tensor([0, 10, 20, 30]).unsqueeze(1)
        text_positions = torch.tensor([40, 50, 60]).unsqueeze(1)
        expected_images = image_content + image_positions + torch.tensor([1000, 2000, 3000, 4000])
        expected_text = text_content + text_positions + text_modality
        expected = torch.cat((expected_images, expected_text), dim=1)
    else:
        images = None
        text_positions = torch.tensor([0, 10, 20]).unsqueeze(1)
        expected = text_content + text_positions + text_modality

    torch.testing.assert_close(embedding(input_ids, images), expected, rtol=0, atol=0)


@pytest.mark.parametrize("with_images", [False, True])
def test_gradients_reach_used_embeddings_and_pixels(config: ModelConfig, with_images: bool) -> None:
    embedding = MultimodalEmbedding(config)
    with torch.no_grad():
        embedding.image_embedding.projection.weight.fill_(0.125)
    input_ids = torch.tensor([[0, 2, 2], [3, 0, 7]])
    images = torch.ones(2, 1, 4, 4, requires_grad=True) if with_images else None

    tokens = embedding(input_ids, images)
    tokens.sum().backward()

    # Repeated IDs accumulate gradients, and token zero has no padding semantics.
    expected_text_grad = torch.tensor([2, 0, 2, 1, 0, 0, 0, 1], dtype=torch.float32)
    torch.testing.assert_close(
        embedding.text_embedding.weight.grad, expected_text_grad.unsqueeze(1).expand(8, 4)
    )
    expected_position_grad = torch.zeros(8, 4)
    expected_position_grad[: tokens.shape[1]] = 2
    torch.testing.assert_close(embedding.position_embedding.weight.grad, expected_position_grad)
    expected_modality_grad = torch.tensor([6, 8 if with_images else 0], dtype=torch.float32)
    torch.testing.assert_close(
        embedding.modality_embedding.weight.grad, expected_modality_grad.unsqueeze(1).expand(2, 4)
    )
    for parameter in embedding.image_embedding.parameters():
        if with_images:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.all(parameter.grad != 0)
        else:
            assert parameter.grad is None
    if images is not None:
        assert images.grad is not None
        assert torch.isfinite(images.grad).all()
        assert torch.all(images.grad != 0)


@pytest.mark.parametrize("with_images", [False, True])
def test_exact_capacity_and_overflow(config: ModelConfig, with_images: bool) -> None:
    embedding = MultimodalEmbedding(config)
    images = torch.randn(2, 1, 4, 4) if with_images else None
    text_length = config.max_seq_len - (config.num_patches if with_images else 0)
    input_ids = torch.zeros(2, text_length, dtype=torch.long)

    assert embedding(input_ids, images).shape == (2, config.max_seq_len, config.d_model)
    with pytest.raises(ValueError, match="combined sequence length 9 exceeds max_seq_len 8"):
        embedding(torch.zeros(2, text_length + 1, dtype=torch.long), images)


def test_images_can_overflow_an_otherwise_valid_text_sequence(config: ModelConfig) -> None:
    embedding = MultimodalEmbedding(config)
    input_ids = torch.zeros(2, 5, dtype=torch.long)

    assert embedding(input_ids).shape == (2, 5, config.d_model)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        embedding(input_ids, torch.randn(2, 1, 4, 4))


def test_text_only_when_patch_count_exceeds_capacity(config: ModelConfig) -> None:
    config = replace(config, max_seq_len=2)
    embedding = MultimodalEmbedding(config)

    assert embedding(torch.tensor([[0, 7]])).shape == (1, 2, config.d_model)


@pytest.mark.parametrize("shape", [(), (3,), (1, 2, 3)])
def test_invalid_token_rank(config: ModelConfig, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="rank 2"):
        MultimodalEmbedding(config)(torch.zeros(shape, dtype=torch.long))


@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.uint8, torch.int16, torch.float32, torch.complex64]
)
def test_invalid_token_dtype(config: ModelConfig, dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="int32 or int64"):
        MultimodalEmbedding(config)(torch.zeros(2, 3, dtype=dtype))


@pytest.mark.parametrize("invalid_id", [-1, 8])
def test_token_ids_outside_vocabulary(config: ModelConfig, invalid_id: int) -> None:
    with pytest.raises(ValueError, match=r"input_ids must be in \[0, 8\)"):
        MultimodalEmbedding(config)(torch.tensor([[0, invalid_id]]))


@pytest.mark.parametrize("with_images", [False, True])
def test_text_must_be_nonempty(config: ModelConfig, with_images: bool) -> None:
    images = torch.zeros(2, 1, 4, 4) if with_images else None
    with pytest.raises(ValueError, match="at least one text token"):
        MultimodalEmbedding(config)(torch.empty(2, 0, dtype=torch.long), images)


def test_image_batch_size_must_match(config: ModelConfig) -> None:
    with pytest.raises(ValueError, match="same batch size"):
        MultimodalEmbedding(config)(torch.zeros(2, 3, dtype=torch.long), torch.zeros(1, 1, 4, 4))


@pytest.mark.parametrize("input_device, parameter_device", [("cpu", "meta"), ("meta", "cpu")])
def test_ids_must_match_parameter_device(
    config: ModelConfig, input_device: str, parameter_device: str
) -> None:
    embedding = MultimodalEmbedding(config).to(parameter_device)
    input_ids = torch.zeros(2, 3, dtype=torch.long, device=input_device)
    with pytest.raises(ValueError, match="same device as the embedding parameters"):
        embedding(input_ids)


def test_image_device_must_match(config: ModelConfig) -> None:
    with pytest.raises(ValueError, match="images and input_ids must be on the same device"):
        MultimodalEmbedding(config)(
            torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 1, 4, 4, device="meta")
        )


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((), "rank 4"),
        ((2, 4, 4), "rank 4"),
        ((2, 3, 4, 4), "1 channels"),
        ((2, 1, 2, 4), "spatial size"),
    ],
)
def test_image_shape_validation(config: ModelConfig, shape: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MultimodalEmbedding(config)(torch.zeros(2, 3, dtype=torch.long), torch.zeros(shape))


def test_images_must_be_floating_point(config: ModelConfig) -> None:
    with pytest.raises(TypeError, match="floating-point dtype"):
        MultimodalEmbedding(config)(
            torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 1, 4, 4, dtype=torch.uint8)
        )


def test_image_dtype_is_not_implicitly_cast(config: ModelConfig) -> None:
    with pytest.raises(RuntimeError, match="same dtype"):
        MultimodalEmbedding(config)(
            torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 1, 4, 4, dtype=torch.float64)
        )


@pytest.mark.parametrize("with_images", [False, True])
def test_noncontiguous_inputs_are_preserved(config: ModelConfig, with_images: bool) -> None:
    embedding = MultimodalEmbedding(config)
    input_ids = torch.tensor([[0, 4, 1, 5, 2, 6], [3, 2, 7, 1, 0, 4]])[:, ::2]
    ids_before = input_ids.clone()
    assert not input_ids.is_contiguous()
    images = torch.randn(2, 1, 4, 4).transpose(2, 3) if with_images else None
    images_before = None if images is None else images.clone()
    if images is not None:
        assert not images.is_contiguous()

    tokens = embedding(input_ids, images)
    expected = embedding(input_ids.contiguous(), None if images is None else images.contiguous())

    torch.testing.assert_close(tokens, expected)
    torch.testing.assert_close(input_ids, ids_before, rtol=0, atol=0)
    if images is not None:
        torch.testing.assert_close(images, images_before, rtol=0, atol=0)


def test_text_output_loss_reaches_images_through_recurrent_core(config: ModelConfig) -> None:
    config = replace(config, d_model=8, n_heads=2, d_ff=16)
    embedding = MultimodalEmbedding(config)
    core = RecurrentTransformerCore(config)
    input_ids = torch.tensor([[0, 2, 7], [1, 3, 5]])
    images = torch.randn(2, 1, 4, 4, requires_grad=True)
    tokens = embedding(input_ids, images)
    # Image patches and the first question token form the bidirectional prefix.
    mask = build_prefix_mask(tokens.shape[1], config.num_patches + 1, device=tokens.device)

    state = core(tokens, mask, recurrence_depth=2)
    assert state.shape == tokens.shape
    state[:, config.num_patches :].square().mean().backward()

    # A loss on text outputs alone must train the visual input path through attention.
    for tensor in (images, *embedding.parameters(), *core.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.any(tensor.grad != 0)
