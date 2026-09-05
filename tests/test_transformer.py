"""Correctness checks for a single transformer block."""

from collections.abc import Iterator

import pytest
import torch
from torch import Tensor, nn

from multimodal_loop.model.attention import build_causal_mask, build_prefix_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.transformer import TransformerBlock


@pytest.fixture(autouse=True)
def seeded_cpu_rng() -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


def pytorch_reference(block: TransformerBlock) -> nn.TransformerEncoderLayer:
    """Copy parameters into PyTorch's independent block implementation, with dropout off."""
    config = block.config
    weight = block.attn.qkv_proj.weight
    reference = nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.n_heads,
        dim_feedforward=config.d_ff,
        dropout=0.0,
        activation="gelu",
        layer_norm_eps=config.layer_norm_eps,
        batch_first=True,
        norm_first=True,
        device=weight.device,
        dtype=weight.dtype,
    ).eval()
    with torch.no_grad():
        reference.self_attn.in_proj_weight.copy_(block.attn.qkv_proj.weight)
        reference.self_attn.in_proj_bias.copy_(block.attn.qkv_proj.bias)
    reference.self_attn.out_proj.load_state_dict(block.attn.out_proj.state_dict())
    reference.linear1.load_state_dict(block.ffn[0].state_dict())
    reference.linear2.load_state_dict(block.ffn[2].state_dict())
    reference.norm1.load_state_dict(block.attn_norm.state_dict())
    reference.norm2.load_state_dict(block.ffn_norm.state_dict())
    return reference


def zero_branch_outputs(block: TransformerBlock) -> None:
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.attn.out_proj.bias.zero_()
        block.ffn[2].weight.zero_()
        block.ffn[2].bias.zero_()


@pytest.mark.parametrize(
    ("batch_size", "seq_len", "d_model", "n_heads", "d_ff"),
    [(1, 1, 8, 1, 11), (2, 5, 8, 2, 13), (3, 7, 16, 4, 19), (2, 3, 64, 4, 256)],
)
def test_block_shape_and_input_preservation(
    batch_size: int, seq_len: int, d_model: int, n_heads: int, d_ff: int
) -> None:
    block = TransformerBlock(ModelConfig(d_model=d_model, n_heads=n_heads, d_ff=d_ff))
    x = torch.randn(batch_size, seq_len, d_model)
    original = x.clone()

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize(("n_heads", "d_ff", "eps"), [(1, 11, 1e-5), (2, 13, 1e-3), (4, 5, 0.25)])
@pytest.mark.parametrize("mask_kind", ["default", "causal", "prefix", "full"])
def test_block_matches_pytorch_reference(
    n_heads: int, d_ff: int, eps: float, mask_kind: str
) -> None:
    config = ModelConfig(d_model=8, n_heads=n_heads, d_ff=d_ff, layer_norm_eps=eps)
    block = TransformerBlock(config).double().eval()
    # Nontrivial, distinct affine parameters exercise both normalization modules.
    with torch.no_grad():
        block.attn_norm.weight.copy_(torch.linspace(0.5, 1.5, 8, dtype=torch.float64))
        block.attn_norm.bias.copy_(torch.linspace(-0.2, 0.3, 8, dtype=torch.float64))
        block.ffn_norm.weight.copy_(torch.linspace(1.4, 0.6, 8, dtype=torch.float64))
        block.ffn_norm.bias.copy_(torch.linspace(0.1, -0.4, 8, dtype=torch.float64))
    reference = pytorch_reference(block)
    # Low variance makes the configured normalization epsilon observable.
    x = torch.randn(2, 5, 8, dtype=torch.float64) * 0.2 + 2
    masks = {
        "default": None,
        "causal": build_causal_mask(5),
        "prefix": build_prefix_mask(5, 3),
        "full": build_prefix_mask(5, 5),
    }
    mask = masks[mask_kind]
    allowed = build_causal_mask(5) if mask is None else mask

    # TransformerEncoderLayer uses True for blocked positions.
    expected = reference(x, src_mask=~allowed)

    torch.testing.assert_close(block(x, mask), expected, rtol=1e-10, atol=1e-10)


def test_normalizations_have_independent_parameters() -> None:
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13))
    original_weight = block.ffn_norm.weight.detach().clone()
    original_bias = block.ffn_norm.bias.detach().clone()

    with torch.no_grad():
        block.attn_norm.weight.add_(0.5)
        block.attn_norm.bias.add_(0.25)

    torch.testing.assert_close(block.ffn_norm.weight, original_weight, rtol=0, atol=0)
    torch.testing.assert_close(block.ffn_norm.bias, original_bias, rtol=0, atol=0)


@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_zero_branches_preserve_identity_and_input_gradient(dropout: float) -> None:
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13, dropout=dropout))
    zero_branch_outputs(block)
    x = torch.randn(2, 5, 8, requires_grad=True)
    original = x.detach().clone()

    output = block(x, build_prefix_mask(5, 3))
    output.sum().backward()

    torch.testing.assert_close(output, original, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_dtype_and_gradients_through_every_component(dtype: torch.dtype) -> None:
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13)).to(
        device="cpu", dtype=dtype
    )
    x = torch.randn(2, 5, 8, device="cpu", dtype=dtype, requires_grad=True)
    original = x.detach().clone()

    output = block(x, build_prefix_mask(5, 2, device=x.device))
    assert output.shape == x.shape
    assert output.dtype == dtype
    assert output.device == x.device
    assert torch.isfinite(output).all()
    output.square().mean().backward()

    # Covers attention, both feed-forward projections, and both LayerNorms.
    for name, tensor in [("input", x), *block.named_parameters()]:
        assert tensor.grad is not None, name
        assert tensor.grad.dtype == dtype, name
        assert tensor.grad.device == tensor.device, name
        assert torch.isfinite(tensor.grad).all(), name
        assert torch.count_nonzero(tensor.grad) > 0, name
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_causal_block_blocks_future_input_changes(explicit_mask: bool) -> None:
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13)).double().eval()
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    changed = x.clone()
    # Changing one feature survives LayerNorm; shifting all features equally would not.
    changed[:, 3:, 0] += 5
    mask = build_causal_mask(6) if explicit_mask else None

    original_output = block(x, mask)
    changed_output = block(changed, mask)

    torch.testing.assert_close(changed_output[:, :3], original_output[:, :3], rtol=0, atol=0)
    assert not torch.allclose(changed_output[:, 3:], original_output[:, 3:])


@pytest.mark.parametrize("changed_start", [3, 5])
def test_prefix_block_blocks_answer_leakage(changed_start: int) -> None:
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13)).double().eval()
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    changed = x.clone()
    changed[:, changed_start:, 0] += 5
    mask = build_prefix_mask(6, 3)

    original_output = block(x, mask)
    changed_output = block(changed, mask)

    # Covers both context isolation and earlier answers' isolation from later answers.
    torch.testing.assert_close(
        changed_output[:, :changed_start], original_output[:, :changed_start], rtol=0, atol=0
    )
    assert not torch.allclose(changed_output[:, changed_start:], original_output[:, changed_start:])


def test_prefix_block_context_is_bidirectional_and_batches_are_independent() -> None:
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13)).double().eval()
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    changed = x.clone()
    changed[0, 2, 0] += 5
    mask = build_prefix_mask(6, 3)

    original_output = block(x, mask)
    changed_output = block(changed, mask)

    assert not torch.allclose(changed_output[0, :2], original_output[0, :2])
    torch.testing.assert_close(changed_output[1], original_output[1], rtol=0, atol=0)
    torch.testing.assert_close(block(changed)[0, :2], block(x)[0, :2], rtol=0, atol=0)


def test_noncontiguous_inputs() -> None:
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13)).double()
    x = torch.randn(2, 8, 5, dtype=torch.float64).transpose(1, 2)
    original = x.clone()
    assert not x.is_contiguous()

    torch.testing.assert_close(block(x), block(x.contiguous()))
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("branch", ["attention", "feedforward"])
def test_dropout_on_each_residual_branch(branch: str) -> None:
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13, dropout=0.5)).train()
    zero_branch_outputs(block)
    with torch.no_grad():
        if branch == "attention":
            block.attn.out_proj.bias.fill_(1)
        else:
            block.ffn[2].bias.fill_(1)
    x = torch.zeros(2, 4, 8)

    # Only the selected branch contributes a constant one before residual dropout.
    # Attention-probability dropout cannot affect these constant branch outputs.
    torch.manual_seed(10)
    first = block(x)
    torch.manual_seed(11)
    second = block(x)
    torch.manual_seed(10)
    repeated = block(x)

    assert torch.all((first == 0) | (first == 2))
    assert (first == 0).any()
    assert (first == 2).any()
    assert not torch.allclose(first, second)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.testing.assert_close(block.eval()(x), torch.ones_like(x), rtol=0, atol=0)


def test_evaluation_disables_all_dropout() -> None:
    block = (
        TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13, dropout=0.5)).double().eval()
    )
    reference = pytorch_reference(block)
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    mask = build_prefix_mask(5, 3)

    torch.manual_seed(10)
    first = block(x, mask)
    torch.manual_seed(11)
    second = block(x, mask)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, reference(x, src_mask=~mask), rtol=1e-10, atol=1e-10)


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
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13))

    with pytest.raises(ValueError, match=message):
        block(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64, torch.bool, torch.complex64])
def test_inputs_must_be_floating_point(dtype: torch.dtype) -> None:
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13))

    with pytest.raises(TypeError, match="floating-point dtype"):
        block(torch.zeros(2, 4, 8, dtype=dtype))


@pytest.mark.parametrize(
    ("mask", "error", "message"),
    [
        (torch.ones(1, 4, 4, dtype=torch.bool), ValueError, "attention_mask must have shape"),
        (torch.ones(4, 4), TypeError, "attention_mask must have boolean dtype"),
        (torch.zeros(4, 4, dtype=torch.bool), ValueError, "at least one key per query"),
        (
            torch.ones(4, 4, dtype=torch.bool, device="meta"),
            ValueError,
            "attention_mask must be on the same device as x",
        ),
    ],
)
def test_mask_validation_is_delegated(mask: Tensor, error: type[Exception], message: str) -> None:
    block = TransformerBlock(ModelConfig(d_model=8, n_heads=2, d_ff=13))

    with pytest.raises(error, match=message):
        block(torch.zeros(2, 4, 8, device="cpu"), mask)
