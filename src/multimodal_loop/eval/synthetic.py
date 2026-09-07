"""Question-only, full-vocabulary answer evaluation for synthetic colors."""

import random
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from torch.nn import functional as F

from multimodal_loop.data.collator import ColorQuestionBatch
from multimodal_loop.data.text import ColorQuestionTokenizer
from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.runtime import finish_step, preserve_device_rng
from multimodal_loop.train.synthetic import _integer


@dataclass(frozen=True)
class EvaluationMetrics:
    total: int
    correct: int
    accuracy: float
    invalid_predictions: int
    loss: float


def evaluate_synthetic(
    model: MultimodalLoopTransformer,
    batches: Iterable[ColorQuestionBatch],
    *,
    recurrence_depth: int,
) -> EvaluationMetrics:
    """Read the last question logit; answers are targets, never forward inputs.

    Loss is averaged over examples, including a partial final batch. Argmax
    covers the entire vocabulary; question/extra token predictions are errors.
    Evaluation restores module modes and CPU/selected-device random streams.
    """
    _integer("recurrence_depth", recurrence_depth, 1)
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    total, correct, invalid, loss_sum = 0, 0, 0, 0.0
    tokenizer = ColorQuestionTokenizer()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), preserve_device_rng(device), torch.inference_mode():
            try:
                model.eval()
                for batch in batches:
                    if (
                        batch.question_length != tokenizer.question_length
                        or batch.input_ids.ndim != 2
                        or batch.input_ids.shape[1] != tokenizer.question_length + 1
                    ):
                        raise ValueError(
                            "color evaluation requires six question IDs and one answer"
                        )
                    batch = batch.to(device)
                    question = batch.input_ids[:, : batch.question_length]
                    length = batch.num_image_tokens + batch.question_length
                    logits = model(
                        question,
                        batch.images,
                        recurrence_depth=recurrence_depth,
                        attention_mask=build_prefix_mask(length, length, device=device),
                    )[:, -1]
                    targets = batch.input_ids[:, -1]
                    loss = F.cross_entropy(logits, targets, reduction="sum")
                    predictions = logits.argmax(dim=-1)
                    matches = (predictions == targets).sum()
                    bad = (
                        (predictions < tokenizer.question_length)
                        | (predictions >= tokenizer.vocab_size)
                    ).sum()
                    finish_step(device)
                    loss_value = loss.item()
                    if not isfinite(loss_value):
                        raise ValueError("evaluation loss must be finite")
                    loss_sum += loss_value
                    correct += matches.item()
                    invalid += bad.item()
                    total += targets.shape[0]
            finally:
                finish_step(device)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for module, mode in modes:
            module.training = mode
    if total == 0:
        raise ValueError("evaluation requires at least one example")
    return EvaluationMetrics(total, correct, correct / total, invalid, loss_sum / total)
