"""End-to-end language prediction, recurrence, and multimodal leakage checks."""

from collections.abc import Iterator
from dataclasses import replace
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from multimodal_loop.model.attention import build_causal_mask, build_prefix_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.model.recurrent_core import RecurrentTransformerCore
from multimodal_loop.model.transformer import TransformerBlock, TransformerStack


@pytest.fixture(autouse=True)
def seeded_cpu_rng() -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        max_seq_len=12,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_recurrent_layers=2,
        image_size=4,
        patch_size=2,
    )


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("prelude_layers, coda_layers", [(1, 1), (0, 0), (0, 1), (1, 0)])
def test_shifted_language_loss_forward_and_backward(
    config: ModelConfig,
    with_images: bool,
    dtype: torch.dtype,
    prelude_layers: int,
    coda_layers: int,
) -> None:
    config = replace(config, n_prelude_layers=prelude_layers, n_coda_layers=coda_layers)
    model = MultimodalLoopTransformer(config).to(device="cpu", dtype=dtype)
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    images = torch.randn(2, 3, 4, 4, dtype=dtype, requires_grad=True) if with_images else None
    num_patches = config.num_patches if with_images else 0
    seq_len = num_patches + input_ids.shape[1]
    question_length = 2
    mask = build_prefix_mask(seq_len, num_patches + question_length) if with_images else None

    logits = model(input_ids, images, recurrence_depth=3, attention_mask=mask)

    assert logits.shape == (2, seq_len, config.vocab_size)
    assert logits.device == input_ids.device
    assert logits.dtype == dtype
    assert torch.isfinite(logits).all()
    logits.retain_grad()
    if with_images:
        # The final question position predicts answer 1; no visual/question target loss.
        loss_start = num_patches + question_length - 1
        targets = input_ids[:, question_length:]
    else:
        loss_start = 0
        targets = input_ids[:, 1:]
    predictions = logits[:, loss_start:-1]
    assert predictions.shape[:2] == targets.shape
    loss = F.cross_entropy(predictions.reshape(-1, config.vocab_size), targets.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, :loss_start]) == 0
    assert torch.count_nonzero(logits.grad[:, -1]) == 0
    assert torch.count_nonzero(logits.grad[:, loss_start]) > 0
    for name, parameter in model.named_parameters():
        if not with_images and name.startswith("embeddings.image_embedding."):
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
            assert parameter.grad.dtype == dtype, name
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.count_nonzero(parameter.grad) > 0, name
    if images is not None:
        assert images.grad is not None
        assert torch.isfinite(images.grad).all()
        assert torch.count_nonzero(images.grad) > 0


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("explicit_mask", [False, True])
def test_stage_order_runtime_depth_and_shared_mask(
    config: ModelConfig, with_images: bool, explicit_mask: bool
) -> None:
    config = replace(config, n_prelude_layers=2, n_coda_layers=2)
    model = MultimodalLoopTransformer(config).eval()
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    images = torch.randn(1, 3, 4, 4) if with_images else None
    seq_len = input_ids.shape[1] + (config.num_patches if with_images else 0)
    mask = build_prefix_mask(seq_len, seq_len - 3) if explicit_mask else None
    original_parameters = dict(model.named_parameters())
    original_count = sum(parameter.numel() for parameter in model.parameters())
    original_state_keys = tuple(model.state_dict())
    observed_names = []
    observed_masks = []
    head_outputs = []

    def record_call(name, module, args, kwargs):
        observed_names.append(name)
        if isinstance(module, (TransformerStack, TransformerBlock, RecurrentTransformerCore)):
            observed_masks.append(kwargs.get("attention_mask", args[1] if len(args) > 1 else None))

    def record_head_output(module, args, output):
        head_outputs.append(output)

    stage_names = {"embeddings", "prelude", "core", "core.stack", "coda", "final_norm", "lm_head"}
    handles = [
        module.register_forward_pre_hook(partial(record_call, name), with_kwargs=True)
        for name, module in model.named_modules()
        if name in stage_names or isinstance(module, TransformerBlock)
    ]
    handles.append(model.lm_head.register_forward_hook(record_head_output))
    try:
        with torch.no_grad():
            for override in (1, None, 5, None):
                observed_names.clear()
                observed_masks.clear()
                head_outputs.clear()
                logits = model(input_ids, images, override, mask)
                depth = config.recurrence_depth if override is None else override

                assert observed_names == (
                    ["embeddings", "prelude", "prelude.blocks.0", "prelude.blocks.1", "core"]
                    + ["core.stack", "core.stack.blocks.0", "core.stack.blocks.1"] * depth
                    + ["coda", "coda.blocks.0", "coda.blocks.1", "final_norm", "lm_head"]
                )
                # One actual mask object is shared even when the caller omits it.
                used_mask = mask if explicit_mask else observed_masks[0]
                assert all(passed is used_mask for passed in observed_masks)
                if not explicit_mask:
                    torch.testing.assert_close(used_mask, build_causal_mask(seq_len))
                assert logits is head_outputs[0]  # No softmax or position slicing after the head.
                assert model.config is config
                assert model.core.config is config
                assert config.recurrence_depth == 2
                assert tuple(model.state_dict()) == original_state_keys
                assert sum(parameter.numel() for parameter in model.parameters()) == original_count
                assert all(
                    parameter is original_parameters[name]
                    for name, parameter in model.named_parameters()
                )
    finally:
        for handle in handles:
            handle.remove()


def test_outer_stacks_and_normalized_head_have_independent_parameters(config: ModelConfig) -> None:
    config = replace(config, layer_norm_eps=0.25)
    model = MultimodalLoopTransformer(config)
    groups = [
        {id(parameter) for parameter in module.parameters()}
        for module in (
            model.embeddings,
            model.prelude,
            model.core,
            model.coda,
            model.final_norm,
            model.lm_head,
        )
    ]
    assert sum(len(group) for group in groups) == len(set().union(*groups))
    assert model.final_norm.normalized_shape == (config.d_model,)
    assert model.final_norm.eps == config.layer_norm_eps
    assert model.lm_head.weight.shape == (config.vocab_size, config.d_model)
    assert model.lm_head.bias is None


def forward_with_recurrent_states(
    model: MultimodalLoopTransformer,
    input_ids: Tensor,
    images: Tensor | None,
    depth: int,
    mask: Tensor | None,
) -> tuple[Tensor, list[Tensor]]:
    """Inspect every completed recurrence without adding a model diagnostics API."""
    states = []

    def capture_state(module, args, output):
        states.append(output.detach().clone())

    handle = model.core.stack.register_forward_hook(capture_state)
    try:
        with torch.no_grad():
            logits = model(input_ids, images, recurrence_depth=depth, attention_mask=mask)
    finally:
        handle.remove()
    assert len(states) == depth
    return logits, states


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("explicit_mask", [False, True])
def test_causal_future_inputs_cannot_change_earlier_logits(
    config: ModelConfig, with_images: bool, explicit_mask: bool
) -> None:
    model = MultimodalLoopTransformer(config).double().eval()
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    changed_ids = input_ids.clone()
    changed_ids[:, 3:] += 4
    images = torch.randn(2, 3, 4, 4, dtype=torch.float64) if with_images else None
    num_patches = config.num_patches if with_images else 0
    mask = build_causal_mask(num_patches + 5) if explicit_mask else None
    original, _ = forward_with_recurrent_states(model, input_ids, images, 3, mask)
    changed, _ = forward_with_recurrent_states(model, changed_ids, images, 3, mask)

    torch.testing.assert_close(
        changed[:, : num_patches + 3], original[:, : num_patches + 3], rtol=0, atol=0
    )
    assert not torch.allclose(changed[:, num_patches + 3 :], original[:, num_patches + 3 :])


@pytest.mark.parametrize("depth", [1, 3, 5])
@pytest.mark.parametrize("changed_start", [2, 4])
def test_prefix_answer_isolation_after_every_recurrence(
    config: ModelConfig, depth: int, changed_start: int
) -> None:
    config = replace(config, n_prelude_layers=2, n_coda_layers=2)
    model = MultimodalLoopTransformer(config).double().eval()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    changed_ids = input_ids.clone()
    changed_ids[:, changed_start:] += 4
    images = torch.randn(2, 3, 4, 4, dtype=torch.float64)
    prefix_len = config.num_patches + 2
    protected_len = config.num_patches + changed_start
    mask = build_prefix_mask(config.num_patches + input_ids.shape[1], prefix_len)

    original_logits, original_states = forward_with_recurrent_states(
        model, input_ids, images, depth, mask
    )
    changed_logits, changed_states = forward_with_recurrent_states(
        model, changed_ids, images, depth, mask
    )

    # Includes image states, question states, and any earlier answer inputs.
    for original, changed in zip(original_states, changed_states, strict=True):
        torch.testing.assert_close(
            changed[:, :protected_len], original[:, :protected_len], rtol=0, atol=0
        )
        assert not torch.allclose(changed[:, protected_len:], original[:, protected_len:])
    torch.testing.assert_close(
        changed_logits[:, :protected_len], original_logits[:, :protected_len], rtol=0, atol=0
    )
    assert not torch.allclose(changed_logits[:, protected_len:], original_logits[:, protected_len:])


@pytest.mark.parametrize("depth", [1, 3, 5])
@pytest.mark.parametrize("source", ["image", "question"])
def test_prefix_context_influences_answers_without_crossing_batch_items(
    config: ModelConfig, depth: int, source: str
) -> None:
    model = MultimodalLoopTransformer(config).double().eval()
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    images = torch.randn(2, 3, 4, 4, dtype=torch.float64)
    changed_ids = input_ids.clone()
    changed_images = images.clone()
    if source == "question":
        changed_ids[0, 1] = 10  # The last question input may influence earlier visual states.
    else:
        changed_images[0, 0, :2, :2] += 5
    prefix_len = config.num_patches + 2
    mask = build_prefix_mask(config.num_patches + input_ids.shape[1], prefix_len)

    original_logits, original_states = forward_with_recurrent_states(
        model, input_ids, images, depth, mask
    )
    changed_logits, changed_states = forward_with_recurrent_states(
        model, changed_ids, changed_images, depth, mask
    )

    for original, changed in zip(original_states, changed_states, strict=True):
        torch.testing.assert_close(changed[1], original[1], rtol=0, atol=0)
        assert not torch.allclose(changed[0, : prefix_len - 1], original[0, : prefix_len - 1])
    torch.testing.assert_close(changed_logits[1], original_logits[1], rtol=0, atol=0)
    # Start at the final question position, which predicts the first answer token.
    assert not torch.allclose(
        changed_logits[0, prefix_len - 1 :], original_logits[0, prefix_len - 1 :]
    )


@pytest.mark.parametrize("depth", [1, 3, 5])
@pytest.mark.parametrize(
    "mask_kind, prediction_text_position", [("causal", 2), ("prefix", 1), ("prefix", 3)]
)
def test_protected_predictions_have_zero_gradient_to_future_embeddings(
    config: ModelConfig, depth: int, mask_kind: str, prediction_text_position: int
) -> None:
    model = MultimodalLoopTransformer(config).double().eval()
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    images = torch.randn(1, 3, 4, 4, dtype=torch.float64)
    seq_len = config.num_patches + input_ids.shape[1]
    mask = (
        build_prefix_mask(seq_len, config.num_patches + 2)
        if mask_kind == "prefix"
        else build_causal_mask(seq_len)
    )
    embedded_states = []

    def retain_embedding_gradient(module, args, output):
        output.retain_grad()
        embedded_states.append(output)

    handle = model.embeddings.register_forward_hook(retain_embedding_gradient)
    try:
        logits = model(input_ids, images, recurrence_depth=depth, attention_mask=mask)
    finally:
        handle.remove()
    position = config.num_patches + prediction_text_position
    loss = F.cross_entropy(logits[:, position], input_ids[:, prediction_text_position + 1])
    loss.backward()

    # Inspect token positions, not shared embedding-table rows that may have multiple uses.
    gradient = embedded_states[0].grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[:, position + 1 :]) == 0
    assert torch.count_nonzero(gradient[:, : config.num_patches]) > 0
    assert torch.count_nonzero(gradient[:, config.num_patches : position + 1]) > 0


@pytest.mark.parametrize("with_images", [False, True])
def test_evaluation_is_deterministic_and_preserves_noncontiguous_inputs(
    config: ModelConfig, with_images: bool
) -> None:
    model = MultimodalLoopTransformer(replace(config, dropout=0.5)).double().eval()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=torch.int32)[:, ::2]
    images = torch.randn(2, 3, 4, 4, dtype=torch.float64).transpose(2, 3) if with_images else None
    assert not input_ids.is_contiguous()
    if images is not None:
        assert not images.is_contiguous()
    ids_before = input_ids.clone()
    images_before = None if images is None else images.clone()
    seq_len = input_ids.shape[1] + (config.num_patches if with_images else 0)
    mask = build_prefix_mask(seq_len, seq_len - 1)
    mask_before = mask.clone()
    rng_before = torch.get_rng_state().clone()

    with torch.no_grad():
        first = model(input_ids, images, attention_mask=mask)
        second = model(input_ids, images, attention_mask=mask)
        contiguous = model(
            input_ids.contiguous(),
            None if images is None else images.contiguous(),
            attention_mask=mask,
        )

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, contiguous, rtol=0, atol=0)
    torch.testing.assert_close(input_ids, ids_before, rtol=0, atol=0)
    torch.testing.assert_close(mask, mask_before, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng_before)
    if images is not None:
        torch.testing.assert_close(images, images_before, rtol=0, atol=0)


@pytest.mark.parametrize("with_images", [False, True])
def test_combined_capacity_boundary(config: ModelConfig, with_images: bool) -> None:
    model = MultimodalLoopTransformer(config)
    images = torch.zeros(1, 3, 4, 4) if with_images else None
    text_length = config.max_seq_len - (config.num_patches if with_images else 0)
    assert model(torch.zeros(1, text_length, dtype=torch.long), images).shape == (
        1,
        config.max_seq_len,
        config.vocab_size,
    )
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(torch.zeros(1, text_length + 1, dtype=torch.long), images)


@pytest.mark.parametrize(
    "input_ids, images, error, message",
    [
        (torch.zeros(3, dtype=torch.long), None, ValueError, "rank 2"),
        (torch.zeros(1, 3), None, TypeError, "int32 or int64"),
        (torch.tensor([[16]]), None, ValueError, "input_ids must be in"),
        (torch.zeros(1, 0, dtype=torch.long), None, ValueError, "at least one text token"),
        (torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, 4), ValueError, "rank 4"),
        (
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(2, 3, 4, 4),
            ValueError,
            "same batch size",
        ),
        (
            torch.zeros(1, 3, dtype=torch.long),
            torch.zeros(1, 3, 4, 4, device="meta"),
            ValueError,
            "same device",
        ),
    ],
)
def test_embedding_validation_reaches_model_callers(
    config: ModelConfig,
    input_ids: Tensor,
    images: Tensor | None,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        MultimodalLoopTransformer(config)(input_ids, images)


@pytest.mark.parametrize(
    "mask, error, message",
    [
        (torch.ones(3, 3, dtype=torch.bool), ValueError, "must have shape"),
        (torch.ones(7, 7), TypeError, "boolean dtype"),
        (torch.zeros(7, 7, dtype=torch.bool), ValueError, "at least one key per query"),
        (torch.ones(7, 7, dtype=torch.bool, device="meta"), ValueError, "same device"),
    ],
)
def test_masks_cover_the_combined_sequence(
    config: ModelConfig, mask: Tensor, error: type[Exception], message: str
) -> None:
    model = MultimodalLoopTransformer(replace(config, n_prelude_layers=0))
    with pytest.raises(error, match=message):
        model(torch.tensor([[0, 1, 2]]), torch.zeros(1, 3, 4, 4), attention_mask=mask)


@pytest.mark.parametrize(
    "depth, error", [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)]
)
def test_runtime_depth_validation(
    config: ModelConfig, depth: object, error: type[Exception]
) -> None:
    with pytest.raises(error, match="recurrence_depth"):
        MultimodalLoopTransformer(config)(torch.tensor([[0, 1, 2]]), recurrence_depth=depth)
