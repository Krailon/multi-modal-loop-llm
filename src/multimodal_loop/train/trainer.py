"""A small single-process training loop for one fixed tensor batch."""

import torch
from torch import Tensor
from torch.optim import Optimizer

from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.losses import shifted_cross_entropy
from multimodal_loop.train.runtime import finish_step


def train_on_batch(
    model: MultimodalLoopTransformer,
    optimizer: Optimizer,
    input_ids: Tensor,
    images: Tensor | None = None,
    *,
    steps: int = 20,
    question_length: int = 0,
    recurrence_depth: int | None = None,
) -> list[float]:
    """Repeat optimizer steps on one batch and return each pre-update loss.

    ``question_length=0`` uses causal attention and all predictable text
    targets. A positive question length uses a bidirectional image/question
    prefix and supervises only answer targets, including the first answer.
    Every item shares the same text/question lengths; no padding is handled.

    The caller owns the model, optimizer, and tensor placement. This function
    enables training mode, clears parameter gradients before every step, and
    preserves optimizer state across calls. Depth is fixed for this call;
    ``None`` uses the model configuration. There is no recurrence sampling,
    clipping, scheduling, accumulation, or RNG reset inside the loop.
    """
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if isinstance(question_length, bool) or not isinstance(question_length, int):
        raise TypeError("question_length must be an integer")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have rank 2 [batch, text_length]")
    if input_ids.shape[0] == 0 or input_ids.shape[1] < 2:
        raise ValueError("training requires a nonempty batch and at least two text tokens")
    text_length = input_ids.shape[1]
    if not 0 <= question_length < text_length:
        raise ValueError("question_length must be nonnegative and leave at least one target token")

    num_image_tokens = model.config.num_patches if images is not None else 0
    attention_mask = None
    target_mask = None
    if question_length > 0:
        attention_mask = build_prefix_mask(
            num_image_tokens + text_length,
            num_image_tokens + question_length,
            device=input_ids.device,
        )
        target_mask = (
            (torch.arange(text_length, device=input_ids.device) >= question_length)
            .unsqueeze(0)
            .expand_as(input_ids)
        )

    model.train()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_ids, images, recurrence_depth=recurrence_depth, attention_mask=attention_mask
        )
        loss = shifted_cross_entropy(
            logits, input_ids, num_image_tokens=num_image_tokens, target_mask=target_mask
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss must be finite")
        loss.backward()
        optimizer.step()
        finish_step(input_ids.device)
        losses.append(loss.detach().item())
    return losses
