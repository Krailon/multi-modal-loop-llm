"""Numerical correctness, visibility, and gradients for self-attention."""

from collections.abc import Iterator
from math import sqrt

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from multimodal_loop.model.attention import SelfAttention, build_causal_mask, build_prefix_mask
from multimodal_loop.model.config import ModelConfig


@pytest.fixture(autouse=True)
def seeded_cpu_rng() -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


def explicit_attention_reference(attention: SelfAttention, x: Tensor, mask: Tensor) -> Tensor:
    """Compute each projection and head separately, without the attention kernel."""
    width = attention.config.d_model
    head_dim = attention.config.head_dim
    projections = [
        F.linear(
            x,
            attention.qkv_proj.weight[index * width : (index + 1) * width],
            attention.qkv_proj.bias[index * width : (index + 1) * width],
        )
        for index in range(3)
    ]
    heads = []
    for head in range(attention.config.n_heads):
        start, end = head * head_dim, (head + 1) * head_dim
        query, key, value = (projection[..., start:end] for projection in projections)
        scores = query @ key.transpose(-2, -1) / sqrt(head_dim)
        probabilities = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        heads.append(probabilities @ value)
    return F.linear(torch.cat(heads, dim=-1), attention.out_proj.weight, attention.out_proj.bias)


def uniform_value_attention() -> SelfAttention:
    """Make each output the mean of the visible inputs, for exact influence tests."""
    attention = SelfAttention(ModelConfig(d_model=4, n_heads=2)).double()
    with torch.no_grad():
        attention.qkv_proj.weight.zero_()
        attention.qkv_proj.bias.zero_()
        attention.qkv_proj.weight[8:].copy_(torch.eye(4, dtype=torch.float64))
        attention.out_proj.weight.copy_(torch.eye(4, dtype=torch.float64))
        attention.out_proj.bias.zero_()
    return attention


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "d_model", "n_heads"),
    [(1, 1, 8, 1), (2, 5, 8, 2), (3, 7, 8, 4), (1, 3, 8, 8), (2, 4, 16, 4), (2, 6, 64, 4)],
)
def test_self_attention_shape(batch_size: int, seq_len: int, d_model: int, n_heads: int) -> None:
    attention = SelfAttention(ModelConfig(d_model=d_model, n_heads=n_heads))
    x = torch.randn(batch_size, seq_len, d_model)

    output = attention(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("n_heads", [1, 2, 4])
@pytest.mark.parametrize("mask_kind", ["default", "causal", "prefix", "full"])
def test_attention_matches_explicit_math(n_heads: int, mask_kind: str) -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=n_heads)).double().eval()
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    masks = {
        "default": None,
        "causal": build_causal_mask(5),
        "prefix": build_prefix_mask(5, 3),
        "full": build_prefix_mask(5, 5),
    }
    mask = masks[mask_kind]
    reference_mask = build_causal_mask(5) if mask is None else mask

    expected = explicit_attention_reference(attention, x, reference_mask)

    torch.testing.assert_close(attention(x, mask), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_causal_attention_blocks_future_input_changes(explicit_mask: bool) -> None:
    attention = uniform_value_attention()
    x = torch.arange(48, dtype=torch.float64).reshape(2, 6, 4)
    changed = x.clone()
    changed[:, 3:] += 12
    mask = build_causal_mask(6) if explicit_mask else None

    original_output = attention(x, mask)
    changed_output = attention(changed, mask)

    torch.testing.assert_close(changed_output[:, :3], original_output[:, :3], rtol=0, atol=0)
    assert not torch.allclose(changed_output[:, 3:], original_output[:, 3:])


def test_prefix_context_cannot_read_answer_inputs() -> None:
    attention = uniform_value_attention()
    x = torch.arange(48, dtype=torch.float64).reshape(2, 6, 4)
    changed = x.clone()
    changed[:, 3:] += 12
    mask = build_prefix_mask(6, 3)

    original_output = attention(x, mask)
    changed_output = attention(changed, mask)

    torch.testing.assert_close(changed_output[:, :3], original_output[:, :3], rtol=0, atol=0)
    assert not torch.allclose(changed_output[:, 3:], original_output[:, 3:])


def test_prefix_answers_cannot_read_future_answer_inputs() -> None:
    attention = uniform_value_attention()
    x = torch.arange(48, dtype=torch.float64).reshape(2, 6, 4)
    changed = x.clone()
    changed[:, -1] += 12
    mask = build_prefix_mask(6, 3)

    original_output = attention(x, mask)
    changed_output = attention(changed, mask)

    torch.testing.assert_close(changed_output[:, :-1], original_output[:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(changed_output[:, -1], original_output[:, -1] + 2)


def test_prefix_context_is_bidirectional() -> None:
    attention = uniform_value_attention()
    x = torch.arange(48, dtype=torch.float64).reshape(2, 6, 4)
    changed = x.clone()
    changed[:, 2] += 12
    mask = build_prefix_mask(6, 3)

    original_output = attention(x, mask)
    changed_output = attention(changed, mask)

    # All three context positions see the changed third context input.
    torch.testing.assert_close(changed_output[:, :3], original_output[:, :3] + 4)
    # Under causal attention, the first position cannot see that input.
    torch.testing.assert_close(attention(changed)[:, 0], attention(x)[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_dtype_and_backward(dtype: torch.dtype) -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=2)).to(device="cpu", dtype=dtype)
    x = torch.randn(2, 5, 8, device="cpu", dtype=dtype, requires_grad=True)

    output = attention(x, build_prefix_mask(5, 2, device=x.device))
    assert output.dtype == dtype
    assert output.device == x.device
    assert torch.isfinite(output).all()
    output.square().mean().backward()

    for tensor in (x, *attention.parameters()):
        assert tensor.grad is not None
        assert tensor.grad.dtype == dtype
        assert tensor.grad.device == tensor.device
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0
    # All three learned projections must receive a gradient.
    for weight_gradient in attention.qkv_proj.weight.grad.chunk(3):
        assert torch.count_nonzero(weight_gradient) > 0


def test_noncontiguous_inputs() -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=2)).double()
    x = torch.randn(2, 8, 5, dtype=torch.float64).transpose(1, 2)
    assert not x.is_contiguous()

    torch.testing.assert_close(attention(x), attention(x.contiguous()))


@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_training_dropout(dropout: float) -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=2, dropout=dropout)).train()
    x = torch.randn(2, 6, 8)

    torch.manual_seed(10)
    first = attention(x)
    torch.manual_seed(11)
    second = attention(x)
    torch.manual_seed(10)
    repeated = attention(x)

    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    if dropout == 0.0:
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    else:
        assert not torch.allclose(first, second)


def test_evaluation_disables_dropout() -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=2, dropout=0.5)).double().eval()
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    mask = build_prefix_mask(5, 3)

    torch.manual_seed(10)
    first = attention(x, mask)
    torch.manual_seed(11)
    second = attention(x, mask)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(
        first, explicit_attention_reference(attention, x, mask), rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((4, 8), "rank 3"),
        ((1, 2, 3, 8), "rank 3"),
        ((2, 4, 7), "width d_model=8"),
        ((1, 0, 8), "positive sequence length"),
    ],
)
def test_invalid_input_shapes(shape: tuple[int, ...], message: str) -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=2))

    with pytest.raises(ValueError, match=message):
        attention(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64, torch.bool, torch.complex64])
def test_inputs_must_be_floating_point(dtype: torch.dtype) -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=2))

    with pytest.raises(TypeError, match="floating-point dtype"):
        attention(torch.zeros(2, 4, 8, dtype=dtype))


@pytest.mark.parametrize("shape", [(4,), (1, 4, 4), (2, 4, 4), (3, 3), (4, 5)])
def test_invalid_mask_shapes(shape: tuple[int, ...]) -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=2))

    with pytest.raises(ValueError, match="attention_mask must have shape"):
        attention(torch.zeros(2, 4, 8), torch.ones(shape, dtype=torch.bool))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64, torch.complex64])
def test_masks_must_be_boolean(dtype: torch.dtype) -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=2))

    with pytest.raises(TypeError, match="attention_mask must have boolean dtype"):
        attention(torch.zeros(2, 4, 8), torch.ones(4, 4, dtype=dtype))


def test_mask_device_must_match_input_device() -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=2))
    # A meta tensor tests device mismatch without needing accelerator hardware.
    mask = torch.ones(4, 4, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="attention_mask must be on the same device as x"):
        attention(torch.zeros(2, 4, 8, device="cpu"), mask)


@pytest.mark.parametrize("all_rows", [False, True])
def test_fully_masked_queries_are_rejected(all_rows: bool) -> None:
    attention = SelfAttention(ModelConfig(d_model=8, n_heads=2))
    mask = build_causal_mask(4)
    if all_rows:
        mask[:] = False
    else:
        mask[2] = False

    with pytest.raises(ValueError, match="attention_mask must allow at least one key per query"):
        attention(torch.zeros(2, 4, 8), mask)
