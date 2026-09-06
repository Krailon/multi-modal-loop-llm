"""Numerical alignment and supervision-mask checks for language-token loss."""

from math import exp, log

import pytest
import torch
from torch import Tensor

from multimodal_loop.train.losses import shifted_cross_entropy


@pytest.mark.parametrize("num_image_tokens", [0, 2])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shifted_loss_matches_individual_target_probabilities(
    num_image_tokens: int, masked: bool, dtype: torch.dtype
) -> None:
    input_ids = torch.tensor([[2, 0, 1, 2], [1, 2, 0, 1]])
    text_scores = torch.tensor(
        [
            [[4, 0, 1], [0, 3, -2], [1, -1, 5], [2, 2, 2]],
            [[-1, 0, 2], [3, 1, -2], [0, 2, 1], [9, 0, -9]],
        ],
        dtype=dtype,
    )
    logits = torch.cat(
        (torch.full((2, num_image_tokens, 3), 10.0, dtype=dtype), text_scores), dim=1
    )
    logits.requires_grad_()
    if masked:
        # Unequal selected counts test averaging over tokens, rather than per-item means.
        target_mask = torch.tensor([[True, False, True, True], [False, True, False, False]])
        supervised = [(0, 1, 1), (0, 2, 2), (1, 0, 2)]
    else:
        target_mask = None
        supervised = [(0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 0, 2), (1, 1, 0), (1, 2, 1)]
    # Independent scalar negative log-likelihoods, including ordinary target ID zero.
    expected_terms = []
    for batch, position, target in supervised:
        scores = text_scores[batch, position].tolist()
        expected_terms.append(log(sum(exp(score) for score in scores)) - scores[target])
    expected = torch.tensor(sum(expected_terms) / len(expected_terms), dtype=dtype)

    loss = shifted_cross_entropy(
        logits, input_ids, num_image_tokens=num_image_tokens, target_mask=target_mask
    )

    torch.testing.assert_close(loss, expected)
    assert loss.dtype == dtype
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None
    active_rows = torch.zeros(logits.shape[:2], dtype=torch.bool)
    for batch, position, _ in supervised:
        active_rows[batch, num_image_tokens + position] = True
    assert torch.all(logits.grad[active_rows].abs().sum(dim=-1) > 0)
    assert torch.count_nonzero(logits.grad[~active_rows]) == 0


def test_ignored_logits_do_not_change_answer_loss() -> None:
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.zeros(1, 6, 4)
    mask = torch.tensor([[False, False, True, True]])
    original = shifted_cross_entropy(logits, input_ids, num_image_tokens=2, target_mask=mask)
    changed = logits.clone()
    changed[:, :3, 0] = 100  # Images and the question logit preceding a question target.
    changed[:, -1, 1] = -100

    torch.testing.assert_close(
        shifted_cross_entropy(changed, input_ids, num_image_tokens=2, target_mask=mask),
        original,
        rtol=0,
        atol=0,
    )
    changed[:, 3, 2] = 5  # The final question logit predicts the first answer (ID 2).
    assert (
        shifted_cross_entropy(changed, input_ids, num_image_tokens=2, target_mask=mask) < original
    )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_noncontiguous_inputs_are_preserved(index_dtype: torch.dtype) -> None:
    input_ids = torch.tensor(
        [[0, 1, 2, 3, 1, 2, 0, 3], [3, 2, 1, 0, 2, 1, 3, 0]], dtype=index_dtype
    )[:, ::2]
    logits = torch.arange(48, dtype=torch.float64).reshape(2, 4, 6).transpose(1, 2).requires_grad_()
    mask = torch.ones(4, 2, dtype=torch.bool).T
    originals = [tensor.detach().clone() for tensor in (input_ids, logits, mask)]
    assert all(not tensor.is_contiguous() for tensor in (input_ids, logits, mask))

    loss = shifted_cross_entropy(logits, input_ids, num_image_tokens=2, target_mask=mask)
    reference = shifted_cross_entropy(
        logits.contiguous(),
        input_ids.contiguous(),
        num_image_tokens=2,
        target_mask=mask.contiguous(),
    )
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    for tensor, original in zip((input_ids, logits, mask), originals, strict=True):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mask", [torch.zeros(1, 3, dtype=torch.bool), torch.tensor([[True, False, False]])]
)
def test_no_predictable_selected_targets_is_an_error(mask: Tensor) -> None:
    with pytest.raises(ValueError, match="at least one next-token target"):
        shifted_cross_entropy(torch.zeros(1, 3, 4), torch.tensor([[0, 1, 2]]), target_mask=mask)


@pytest.mark.parametrize(
    "logits, input_ids, error, message",
    [
        (torch.zeros(1, 3, 4), torch.zeros(3, dtype=torch.long), ValueError, "rank 2"),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.long), ValueError, "at least two"),
        (torch.zeros(0, 3, 4), torch.zeros(0, 3, dtype=torch.long), ValueError, "nonempty batch"),
        (torch.zeros(1, 3, 4), torch.zeros(1, 3), TypeError, "int32 or int64"),
        (torch.zeros(3, 4), torch.zeros(1, 3, dtype=torch.long), ValueError, "rank 3"),
        (torch.zeros(1, 4, 4), torch.zeros(1, 3, dtype=torch.long), ValueError, "must have shape"),
        (
            torch.zeros(1, 3, 0),
            torch.zeros(1, 3, dtype=torch.long),
            ValueError,
            "positive vocab_size",
        ),
        (
            torch.zeros(1, 3, 4, dtype=torch.long),
            torch.zeros(1, 3, dtype=torch.long),
            TypeError,
            "floating-point",
        ),
        (
            torch.zeros(1, 3, 4, device="meta"),
            torch.zeros(1, 3, dtype=torch.long),
            ValueError,
            "same device",
        ),
        (torch.zeros(1, 3, 4), torch.tensor([[0, -1, 2]]), ValueError, "input_ids must be in"),
        (torch.zeros(1, 3, 4), torch.tensor([[0, 1, 4]]), ValueError, "input_ids must be in"),
    ],
)
def test_invalid_loss_inputs(
    logits: Tensor, input_ids: Tensor, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        shifted_cross_entropy(logits, input_ids)


@pytest.mark.parametrize("count, error", [(True, TypeError), (1.5, TypeError), (-1, ValueError)])
def test_invalid_image_token_count(count: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="num_image_tokens"):
        shifted_cross_entropy(
            torch.zeros(1, 3, 4), torch.tensor([[0, 1, 2]]), num_image_tokens=count
        )


@pytest.mark.parametrize(
    "mask, error, message",
    [
        (torch.ones(3, dtype=torch.bool), ValueError, "same shape"),
        (torch.ones(1, 3), TypeError, "boolean dtype"),
        (torch.ones(1, 3, dtype=torch.bool, device="meta"), ValueError, "same device"),
    ],
)
def test_invalid_target_mask(mask: Tensor, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        shifted_cross_entropy(torch.zeros(1, 3, 4), torch.tensor([[0, 1, 2]]), target_mask=mask)
