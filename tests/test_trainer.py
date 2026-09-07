"""Optimizer updates and correctness of the minimal fixed-batch training path."""

import subprocess
import sys
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from math import isfinite
from pathlib import Path

import pytest
import torch

from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.trainer import train_on_batch


@pytest.fixture(autouse=True)
def seeded_cpu_rng() -> Iterator[None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(vocab_size=16, d_model=8, n_heads=2, d_ff=16, image_size=4, patch_size=2)


@pytest.mark.parametrize(
    "with_images, question_length", [(False, 0), (True, 2), (False, 2), (True, 0)]
)
@pytest.mark.parametrize("depth", [1, 3])
def test_training_reduces_loss_and_updates_active_parameters(
    config: ModelConfig, with_images: bool, question_length: int, depth: int
) -> None:
    model = MultimodalLoopTransformer(config).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    images = torch.randn(2, 3, 4, 4) if with_images else None
    originals = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    # Pre-existing invalid gradients must be cleared before the first update.
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    repetitions = []

    def count_recurrence(module, args, output):
        repetitions.append(None)

    handle = model.core.stack.register_forward_hook(count_recurrence)
    try:
        losses = train_on_batch(
            model,
            optimizer,
            input_ids,
            images,
            steps=12,
            question_length=question_length,
            recurrence_depth=depth,
        )
    finally:
        handle.remove()

    assert model.training
    assert len(losses) == 12
    assert all(isinstance(loss, float) and isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0] * 0.7
    assert len(repetitions) == 12 * depth
    assert config.recurrence_depth == 2
    for name, parameter in model.named_parameters():
        if not with_images and name.startswith("embeddings.image_embedding."):
            torch.testing.assert_close(parameter, originals[name], rtol=0, atol=0)
            assert parameter.grad is None
            assert parameter not in optimizer.state
        else:
            assert torch.isfinite(parameter).all(), name
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert not torch.equal(parameter, originals[name]), name
            assert optimizer.state[parameter]["step"].item() == 12


@pytest.mark.parametrize("with_images", [False, True])
def test_training_prefix_mask_and_first_answer_supervision(
    config: ModelConfig, with_images: bool
) -> None:
    model = MultimodalLoopTransformer(config).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32)
    images = (
        torch.randn(1, 3, 4, 4, dtype=torch.float64, requires_grad=True) if with_images else None
    )
    ids_before = input_ids.clone()
    images_before = None if images is None else images.detach().clone()
    observed_masks, observed_logits, observed_embeddings = [], [], []

    def capture_mask(module, args, kwargs):
        observed_masks.append(kwargs["attention_mask"])

    def retain_logits(module, args, output):
        output.retain_grad()
        observed_logits.append(output)

    def retain_embeddings(module, args, output):
        output.retain_grad()
        observed_embeddings.append(output)

    handles = [
        model.register_forward_pre_hook(capture_mask, with_kwargs=True),
        model.register_forward_hook(retain_logits),
        model.embeddings.register_forward_hook(retain_embeddings),
    ]
    try:
        losses = train_on_batch(
            model, optimizer, input_ids, images, steps=1, question_length=2, recurrence_depth=3
        )
    finally:
        for handle in handles:
            handle.remove()

    num_image_tokens = config.num_patches if with_images else 0
    prefix_len = num_image_tokens + 2
    torch.testing.assert_close(
        observed_masks[0], build_prefix_mask(num_image_tokens + 5, prefix_len)
    )
    logit_grad = observed_logits[0].grad
    assert logit_grad is not None
    assert torch.count_nonzero(logit_grad[:, : prefix_len - 1]) == 0
    assert torch.count_nonzero(logit_grad[:, -1]) == 0
    assert torch.all(logit_grad[:, prefix_len - 1 : -1].abs().sum(dim=-1) > 0)
    embedding_grad = observed_embeddings[0].grad
    assert embedding_grad is not None
    # The last answer input has no supervised successor and cannot leak backward.
    assert torch.count_nonzero(embedding_grad[:, -1]) == 0
    assert torch.count_nonzero(embedding_grad[:, :prefix_len]) > 0
    assert isfinite(losses[0])
    torch.testing.assert_close(input_ids, ids_before, rtol=0, atol=0)
    if images is not None:
        assert images.grad is not None
        assert torch.isfinite(images.grad).all()
        assert torch.count_nonzero(images.grad) > 0
        torch.testing.assert_close(images, images_before, rtol=0, atol=0)


def test_split_calls_preserve_optimizer_state_and_dropout_progression(config: ModelConfig) -> None:
    config = replace(config, dropout=0.25)
    uninterrupted = MultimodalLoopTransformer(config)
    split = deepcopy(uninterrupted)
    first_optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=0.01, weight_decay=0.0)
    second_optimizer = torch.optim.AdamW(split.parameters(), lr=0.01, weight_decay=0.0)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    images = torch.randn(1, 3, 4, 4)
    rng = torch.get_rng_state().clone()

    expected = train_on_batch(
        uninterrupted, first_optimizer, input_ids, images, steps=4, question_length=2
    )
    torch.set_rng_state(rng)
    actual = train_on_batch(split, second_optimizer, input_ids, images, steps=2, question_length=2)
    actual += train_on_batch(split, second_optimizer, input_ids, images, steps=2, question_length=2)

    assert actual == expected
    for name, parameter in split.named_parameters():
        torch.testing.assert_close(parameter, uninterrupted.state_dict()[name], rtol=0, atol=0)
        assert second_optimizer.state[parameter]["step"].item() == 4


def test_nonfinite_loss_does_not_update_parameters(config: ModelConfig) -> None:
    model = MultimodalLoopTransformer(config)
    original = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(FloatingPointError, match="loss must be finite"):
        train_on_batch(
            model,
            optimizer,
            torch.tensor([[0, 1, 2]]),
            torch.full((1, 3, 4, 4), float("nan")),
            steps=1,
            question_length=1,
        )
    assert not optimizer.state
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"steps": True}, TypeError, "steps"),
        ({"steps": 0}, ValueError, "steps"),
        ({"question_length": True}, TypeError, "question_length"),
        ({"question_length": -1}, ValueError, "question_length"),
        ({"question_length": 3}, ValueError, "at least one target"),
        ({"recurrence_depth": 0}, ValueError, "recurrence_depth"),
    ],
)
def test_invalid_training_settings(
    config: ModelConfig, kwargs: dict, error: type[Exception], message: str
) -> None:
    model = MultimodalLoopTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(error, match=message):
        train_on_batch(model, optimizer, torch.tensor([[0, 1, 2]]), **kwargs)
    assert not optimizer.state


@pytest.mark.parametrize("shape", [(3,), (0, 3), (1, 1)])
def test_training_requires_predictable_targets(config: ModelConfig, shape: tuple[int, ...]) -> None:
    model = MultimodalLoopTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="rank 2|nonempty batch|at least two"):
        train_on_batch(model, optimizer, torch.zeros(shape, dtype=torch.long))
    assert not optimizer.state


@pytest.mark.parametrize("text_only", [False, True])
def test_training_script_runs_from_installed_package(text_only: bool) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    command = [sys.executable, str(script), "--steps", "2", "--seed", "7"]
    if text_only:
        command.append("--text-only")
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    assert "Fixed synthetic batch:" in result.stdout
    assert "recurrence_depth=2" in result.stdout
    expected_question_length = 0 if text_only else 3
    assert f"question_length={expected_question_length}" in result.stdout
    lines = [line for line in result.stdout.splitlines() if line.startswith("step=")]
    assert len(lines) == 2
    assert all(isfinite(float(line.split("loss=")[1])) for line in lines)
