"""Parameter sharing, runtime depth, and full backpropagation through recurrence."""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from torch import Tensor

from multimodal_loop.model.attention import build_causal_mask, build_prefix_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.recurrent_core import RecurrentTransformerCore


@pytest.fixture(autouse=True)
def seeded_cpu_rng() -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


@pytest.mark.parametrize("prefix_len", [None, 2])
def test_runtime_depth_reuses_parameters_and_forwards_masks(prefix_len: int | None) -> None:
    config = ModelConfig(d_model=8, n_heads=2, d_ff=13, n_recurrent_layers=2, recurrence_depth=2)
    core = RecurrentTransformerCore(config)
    x = torch.randn(2, 4, 8)
    mask = None if prefix_len is None else build_prefix_mask(4, prefix_len)
    original_config = asdict(config)
    original_state = {name: value.clone() for name, value in core.state_dict().items()}
    parameter_ids = [id(parameter) for parameter in core.parameters()]
    parameter_count = sum(parameter.numel() for parameter in core.parameters())
    assert parameter_count == sum(parameter.numel() for parameter in core.stack.parameters())
    assert len(core.stack.blocks) == config.n_recurrent_layers
    observed = []

    def record_call(module, args, kwargs):
        passed_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        observed.append((module, passed_mask))

    handles = [
        block.register_forward_pre_hook(record_call, with_kwargs=True)
        for block in core.stack.blocks
    ]
    try:
        for override in (None, 1, 5, 2, None):
            observed.clear()
            output = core(x, mask, recurrence_depth=override)
            depth = config.recurrence_depth if override is None else override

            assert output.shape == x.shape
            assert [id(module) for module, _ in observed] == [
                id(block) for block in core.stack.blocks
            ] * depth
            assert all(passed_mask is mask for _, passed_mask in observed)
            assert [id(parameter) for parameter in core.parameters()] == parameter_ids
            assert sum(parameter.numel() for parameter in core.parameters()) == parameter_count
            assert core.state_dict().keys() == original_state.keys()
            for name, value in core.state_dict().items():
                torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
            assert core.config is config
            assert asdict(config) == original_config
    finally:
        for handle in handles:
            handle.remove()


def test_configured_depth_does_not_change_parameter_count_or_state_structure() -> None:
    shallow = RecurrentTransformerCore(
        ModelConfig(d_model=8, n_heads=2, d_ff=13, n_recurrent_layers=2, recurrence_depth=1)
    )
    deeper = RecurrentTransformerCore(
        ModelConfig(d_model=8, n_heads=2, d_ff=13, n_recurrent_layers=2, recurrence_depth=5)
    )

    assert sum(p.numel() for p in shallow.parameters()) == sum(
        p.numel() for p in deeper.parameters()
    )
    assert shallow.state_dict().keys() == deeper.state_dict().keys()


@pytest.mark.parametrize(("num_layers", "depth"), [(1, 1), (2, 3), (2, 5)])
@pytest.mark.parametrize("prefix_len", [None, 2])
def test_shared_gradients_equal_sum_from_independent_unrolled_copies(
    num_layers: int, depth: int, prefix_len: int | None
) -> None:
    config = ModelConfig(
        d_model=8, n_heads=2, d_ff=13, n_recurrent_layers=num_layers, recurrence_depth=2
    )
    core = RecurrentTransformerCore(config).double()
    # These reference steps have equal initial values but entirely separate parameters.
    reference_steps = [deepcopy(core.stack) for _ in range(depth)]
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    mask = None if prefix_len is None else build_prefix_mask(4, prefix_len)

    actual = core(x, mask, recurrence_depth=depth)
    expected = reference_x
    for step in reference_steps:
        expected = step(expected, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    actual.square().mean().backward()
    expected.square().mean().backward()
    assert x.grad is not None
    assert reference_x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-9, atol=1e-10)

    reference_parameters = [dict(step.named_parameters()) for step in reference_steps]
    for name, parameter in core.stack.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
        gradients = [parameters[name].grad for parameters in reference_parameters]
        assert all(gradient is not None for gradient in gradients), name
        assert all(torch.isfinite(gradient).all() for gradient in gradients), name
        # Every occurrence contributes, including the earliest recurrent step.
        assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients), name
        torch.testing.assert_close(
            parameter.grad, torch.stack(gradients).sum(dim=0), rtol=1e-9, atol=1e-10
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_core_cpu_dtype_noncontiguous_inputs_and_gradients(dtype: torch.dtype) -> None:
    config = ModelConfig(d_model=8, n_heads=2, d_ff=13, n_recurrent_layers=2)
    core = RecurrentTransformerCore(config).to(device="cpu", dtype=dtype)
    x = torch.randn(2, 8, 5, device="cpu", dtype=dtype).transpose(1, 2).requires_grad_()
    original = x.detach().clone()
    assert not x.is_contiguous()

    output = core(x, recurrence_depth=3)
    assert output.shape == x.shape
    assert output.dtype == dtype
    assert output.device == x.device
    torch.testing.assert_close(output, core(x.contiguous(), recurrence_depth=3))
    output.square().mean().backward()

    for name, tensor in [("input", x), *core.named_parameters()]:
        assert tensor.grad is not None, name
        assert tensor.grad.dtype == dtype, name
        assert tensor.grad.device == x.device, name
        assert torch.isfinite(tensor.grad).all(), name
        assert torch.count_nonzero(tensor.grad) > 0, name
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mask_kind", "changed_start"),
    [("default", 3), ("causal", 3), ("prefix", 3), ("prefix", 5)],
)
def test_mask_isolation_across_blocks_and_recurrences(mask_kind: str, changed_start: int) -> None:
    core = (
        RecurrentTransformerCore(ModelConfig(d_model=8, n_heads=2, d_ff=13, n_recurrent_layers=2))
        .double()
        .eval()
    )
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    changed = x.clone()
    changed[:, changed_start:, 0] += 5
    masks = {"default": None, "causal": build_causal_mask(6), "prefix": build_prefix_mask(6, 3)}
    mask = masks[mask_kind]

    original_output = core(x, mask, recurrence_depth=3)
    changed_output = core(changed, mask, recurrence_depth=3)

    torch.testing.assert_close(
        changed_output[:, :changed_start], original_output[:, :changed_start], rtol=0, atol=0
    )
    assert not torch.allclose(changed_output[:, changed_start:], original_output[:, changed_start:])


def test_prefix_context_influence_and_batch_isolation_after_recurrence() -> None:
    core = (
        RecurrentTransformerCore(ModelConfig(d_model=8, n_heads=2, d_ff=13, n_recurrent_layers=2))
        .double()
        .eval()
    )
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    changed = x.clone()
    changed[0, 2, 0] += 5
    mask = build_prefix_mask(6, 3)

    original_output = core(x, mask, recurrence_depth=3)
    changed_output = core(changed, mask, recurrence_depth=3)

    assert not torch.allclose(changed_output[0, :2], original_output[0, :2])
    torch.testing.assert_close(changed_output[1], original_output[1], rtol=0, atol=0)


def test_training_recurrence_preserves_ordinary_dropout_rng_progression() -> None:
    core = RecurrentTransformerCore(
        ModelConfig(d_model=8, n_heads=2, d_ff=13, n_recurrent_layers=2, dropout=0.5)
    ).train()
    x = torch.randn(2, 4, 8)
    mask = build_prefix_mask(4, 2)
    initial_rng = torch.get_rng_state().clone()

    expected = x
    for _ in range(3):
        expected = core.stack(expected, mask)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)

    actual = core(x, mask, recurrence_depth=3)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert not torch.equal(initial_rng, expected_rng)


def test_core_evaluation_disables_dropout_and_is_deterministic() -> None:
    core = RecurrentTransformerCore(
        ModelConfig(d_model=8, n_heads=2, d_ff=13, n_recurrent_layers=2, dropout=0.5)
    ).eval()
    x = torch.randn(2, 4, 8)
    mask = build_prefix_mask(4, 2)

    torch.manual_seed(10)
    first = core(x, mask, recurrence_depth=3)
    torch.manual_seed(11)
    initial_rng = torch.get_rng_state().clone()
    second = core(x, mask, recurrence_depth=3)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), initial_rng)
    assert all(not module.training for module in core.modules())


@pytest.mark.parametrize("depth", [True, False, 2.0, 1.5, "2"])
def test_runtime_depth_requires_an_integer(depth: object) -> None:
    core = RecurrentTransformerCore(ModelConfig(d_model=8, n_heads=2, d_ff=13))

    with pytest.raises(TypeError, match="recurrence_depth must be an integer"):
        core(torch.zeros(2, 4, 8), recurrence_depth=depth)


@pytest.mark.parametrize("depth", [0, -1])
def test_runtime_depth_must_be_positive(depth: int) -> None:
    core = RecurrentTransformerCore(ModelConfig(d_model=8, n_heads=2, d_ff=13))

    with pytest.raises(ValueError, match="recurrence_depth must be positive"):
        core(torch.zeros(2, 4, 8), recurrence_depth=depth)


@pytest.mark.parametrize(
    ("x", "mask", "error", "message"),
    [
        (torch.zeros(4, 8), None, ValueError, "rank 3"),
        (torch.zeros(2, 4, 7), None, ValueError, "width d_model=8"),
        (torch.zeros(2, 0, 8), None, ValueError, "positive sequence length"),
        (torch.zeros(2, 4, 8, dtype=torch.int64), None, TypeError, "floating-point dtype"),
        (torch.zeros(2, 4, 8), torch.ones(4, 4), TypeError, "boolean dtype"),
        (
            torch.zeros(2, 4, 8),
            torch.zeros(4, 4, dtype=torch.bool),
            ValueError,
            "at least one key per query",
        ),
    ],
)
def test_core_reuses_stack_validation(
    x: Tensor, mask: Tensor | None, error: type[Exception], message: str
) -> None:
    core = RecurrentTransformerCore(ModelConfig(d_model=8, n_heads=2, d_ff=13))

    with pytest.raises(error, match=message):
        core(x, mask, recurrence_depth=3)
