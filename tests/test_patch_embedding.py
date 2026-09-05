"""Correctness checks for direct image-to-token projection."""

import pytest
import torch

from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.patch_embedding import PatchEmbedding


@pytest.mark.parametrize(
    ("batch_size", "num_channels", "image_size", "patch_size", "num_patches"),
    [
        (2, 3, 32, 8, 16),
        (1, 1, 12, 3, 16),
        (3, 3, 8, 8, 1),
        (2, 1, 4, 1, 16),
    ],
)
def test_patch_embedding_shape(
    batch_size: int, num_channels: int, image_size: int, patch_size: int, num_patches: int
) -> None:
    config = ModelConfig(num_channels=num_channels, image_size=image_size, patch_size=patch_size)
    embedding = PatchEmbedding(config)
    images = torch.zeros(batch_size, num_channels, image_size, image_size)

    tokens = embedding(images)

    assert tokens.shape == (batch_size, num_patches, config.d_model)


def test_patch_order_and_shared_linear_projection() -> None:
    config = ModelConfig(image_size=4, patch_size=2, num_channels=2, d_model=8, n_heads=2)
    embedding = PatchEmbedding(config)
    images = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    # Four patches in row order, each containing channel 0 then channel 1.
    expected_patches = torch.tensor(
        [
            [0, 1, 4, 5, 16, 17, 20, 21],
            [2, 3, 6, 7, 18, 19, 22, 23],
            [8, 9, 12, 13, 24, 25, 28, 29],
            [10, 11, 14, 15, 26, 27, 30, 31],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        embedding.projection.weight.copy_(torch.eye(8))
        embedding.projection.bias.zero_()
    torch.testing.assert_close(embedding(images), expected_patches, rtol=0, atol=0)

    # A non-diagonal projection with a nonzero bias is shared by every patch.
    weight = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    bias = torch.arange(8, dtype=torch.float32)
    with torch.no_grad():
        embedding.projection.weight.copy_(weight)
        embedding.projection.bias.copy_(bias)

    expected_tokens = expected_patches @ weight.T + bias
    torch.testing.assert_close(embedding(images), expected_tokens)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((3, 32, 32), "rank 4"),
        ((1, 2, 3, 32, 32), "rank 4"),
        ((2, 1, 32, 32), "3 channels"),
        ((2, 3, 16, 32), "spatial size"),
        ((2, 3, 32, 16), "spatial size"),
        ((2, 3, 64, 64), "spatial size"),
    ],
)
def test_invalid_image_shapes(shape: tuple[int, ...], message: str) -> None:
    embedding = PatchEmbedding(ModelConfig())

    with pytest.raises(ValueError, match=message):
        embedding(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64, torch.bool, torch.complex64])
def test_images_must_be_floating_point(dtype: torch.dtype) -> None:
    embedding = PatchEmbedding(ModelConfig())

    with pytest.raises(TypeError, match="floating-point dtype"):
        embedding(torch.zeros(1, 3, 32, 32, dtype=dtype))


def test_noncontiguous_images() -> None:
    embedding = PatchEmbedding(ModelConfig(image_size=4, patch_size=2))
    images = torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4).transpose(2, 3)
    assert not images.is_contiguous()

    torch.testing.assert_close(embedding(images), embedding(images.contiguous()))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_dtype_conversion_and_backward(dtype: torch.dtype) -> None:
    config = ModelConfig(image_size=8, patch_size=4, d_model=8, n_heads=2)
    embedding = PatchEmbedding(config).to(device="cpu", dtype=dtype)
    # Positive known weights and inputs ensure a nonzero gradient on every pixel.
    with torch.no_grad():
        embedding.projection.weight.fill_(0.125)
        embedding.projection.bias.fill_(0.25)
    images = torch.ones(2, 3, 8, 8, device="cpu", dtype=dtype, requires_grad=True)

    tokens = embedding(images)
    assert tokens.dtype == dtype
    assert tokens.device == images.device
    assert torch.isfinite(tokens).all()
    tokens.square().mean().backward()

    for tensor in (images, embedding.projection.weight, embedding.projection.bias):
        assert tensor.grad is not None
        assert tensor.grad.dtype == dtype
        assert tensor.grad.device == tensor.device
        assert torch.isfinite(tensor.grad).all()
        assert torch.all(tensor.grad != 0)
