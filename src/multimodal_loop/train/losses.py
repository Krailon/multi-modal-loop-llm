"""Shifted language-token cross-entropy for image-first model outputs."""

import torch
import torch.nn.functional as F
from torch import Tensor


def shifted_cross_entropy(
    logits: Tensor,
    input_ids: Tensor,
    *,
    num_image_tokens: int = 0,
    target_mask: Tensor | None = None,
) -> Tensor:
    """Average next-token loss over selected text targets across the batch.

    ``logits`` is ``[B, num_image_tokens + T, vocab_size]`` and ``input_ids``
    is ``[B, T]``. Text position ``t`` predicts text token ``t + 1``; image
    logits and the final text logit are unused. The first text token has no
    preceding text position and is never a target, even when images exist.

    Optional boolean ``target_mask[B, T]`` selects *target token positions*,
    before shifting. For a question of length Q >= 1, select positions Q
    onward: the final question logit then predicts the first answer token.
    This mask only selects supervision; it does not control attention or add
    padding support. Token zero is an ordinary vocabulary ID.

    IDs must be int32 or int64 and are converted to int64 for cross-entropy.
    All tensors must share a device. Empty supervision raises an error rather
    than producing a NaN mean. No input is modified in place.
    """
    if isinstance(num_image_tokens, bool) or not isinstance(num_image_tokens, int):
        raise TypeError("num_image_tokens must be an integer")
    if num_image_tokens < 0:
        raise ValueError("num_image_tokens must be nonnegative")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have rank 2 [batch, text_length]")
    if input_ids.shape[0] == 0 or input_ids.shape[1] < 2:
        raise ValueError("loss requires a nonempty batch and at least two text tokens")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must have dtype int32 or int64")
    if logits.ndim != 3:
        raise ValueError("logits must have rank 3 [batch, combined_length, vocab_size]")
    expected_shape = (input_ids.shape[0], num_image_tokens + input_ids.shape[1])
    if logits.shape[:2] != expected_shape or logits.shape[2] == 0:
        raise ValueError(f"logits must have shape {expected_shape} + (positive vocab_size,)")
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")
    if logits.device != input_ids.device:
        raise ValueError("logits and input_ids must be on the same device")
    if (input_ids < 0).any() or (input_ids >= logits.shape[2]).any():
        raise ValueError(f"input_ids must be in [0, {logits.shape[2]})")

    predictions = logits[:, num_image_tokens:-1]
    targets = input_ids[:, 1:].long()
    if target_mask is not None:
        if target_mask.shape != input_ids.shape:
            raise ValueError("target_mask must have the same shape as input_ids")
        if target_mask.dtype != torch.bool:
            raise TypeError("target_mask must have boolean dtype")
        if target_mask.device != input_ids.device:
            raise ValueError("target_mask and input_ids must be on the same device")
        selected = target_mask[:, 1:]
        if not selected.any():
            raise ValueError("target_mask must select at least one next-token target")
        # Keep shapes independent of mask values for XLA. -100 is CE's
        # ignore_index; vocabulary ID zero remains an ordinary target.
        # Clear ignored scores as well, so nonfinite unused logits cannot
        # produce NaN gradients through log_softmax's backward pass.
        predictions = predictions.masked_fill(~selected.unsqueeze(-1), 0)
        targets = targets.masked_fill(~selected, -100)

    return F.cross_entropy(predictions.reshape(-1, logits.shape[2]), targets.reshape(-1))
