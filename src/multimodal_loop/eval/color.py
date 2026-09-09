"""Shared question-only prediction for one-token color answers (IDs 6 through 9)."""

import random
from collections.abc import Iterable
from math import isfinite

import numpy as np
import torch
from torch.nn import functional as F

from multimodal_loop.data.collator import ColorQuestionBatch
from multimodal_loop.eval.synthetic import EvaluationMetrics
from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.runtime import finish_step, preserve_device_rng
from multimodal_loop.train.synthetic import _integer


def predict_color_answers(
    model: MultimodalLoopTransformer,
    batches: Iterable[ColorQuestionBatch],
    *,
    recurrence_depth: int,
    question_length: int,
    details: list[dict] | None = None,
) -> tuple[EvaluationMetrics, list[int]]:
    """Preserve modes and random streams, including on failed evaluation."""
    _integer("recurrence_depth", recurrence_depth, 1)
    _integer("question_length", question_length, 1)
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    total, correct, invalid, loss_sum = 0, 0, 0, 0.0
    predictions = []
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), preserve_device_rng(device), torch.inference_mode():
            try:
                model.eval()
                for batch in batches:
                    if (
                        batch.question_length != question_length
                        or batch.input_ids.ndim != 2
                        or batch.input_ids.shape[1] != question_length + 1
                    ):
                        raise ValueError(
                            f"color evaluation requires {question_length} question IDs "
                            "and one answer"
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
                    predicted = logits.argmax(dim=-1)
                    matches = (predicted == targets).sum()
                    # Color IDs did not move when relational question tokens were appended.
                    bad = ((predicted < 6) | (predicted >= 10)).sum()
                    if details is not None:
                        individual_loss = F.cross_entropy(logits, targets, reduction="none")
                        probabilities = logits.softmax(dim=-1)
                        confidence = probabilities.gather(1, predicted[:, None]).squeeze(1)
                        target_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
                    finish_step(device)
                    loss_value = loss.item()
                    if not isfinite(loss_value):
                        raise ValueError("evaluation loss must be finite")
                    loss_sum += loss_value
                    correct += matches.item()
                    invalid += bad.item()
                    total += targets.shape[0]
                    predictions.extend(predicted.cpu().tolist())
                    if details is not None:
                        details.extend(
                            dict(
                                zip(
                                    (
                                        "prediction_id",
                                        "target_id",
                                        "loss",
                                        "confidence",
                                        "target_probability",
                                    ),
                                    values,
                                    strict=True,
                                )
                            )
                            for values in zip(
                                predicted.cpu().tolist(),
                                targets.cpu().tolist(),
                                individual_loss.cpu().tolist(),
                                confidence.cpu().tolist(),
                                target_probability.cpu().tolist(),
                                strict=True,
                            )
                        )
            finally:
                finish_step(device)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for module, mode in modes:
            module.training = mode
    if total == 0:
        raise ValueError("evaluation requires at least one example")
    return EvaluationMetrics(
        total, correct, correct / total, invalid, loss_sum / total
    ), predictions
