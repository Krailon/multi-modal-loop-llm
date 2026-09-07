"""Question-only, full-vocabulary answer evaluation for synthetic colors."""

import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from math import isfinite

import numpy as np
import torch
from torch.nn import functional as F

from multimodal_loop.data.collator import ColorQuestionBatch
from multimodal_loop.data.multimodal import SyntheticColorDataset
from multimodal_loop.data.text import ColorQuestionTokenizer
from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.runtime import finish_step, preserve_device_rng
from multimodal_loop.train.synthetic import SyntheticTrainingConfig, _integer, synthetic_loader


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


def _image_control_batches(batches, dataset, *, permutation=None, blank=False):
    """Replace only pixels; donor indices address the full dataset in stored order."""
    offset = 0
    for batch in batches:
        count = batch.input_ids.shape[0]
        if blank:
            batch = replace(batch, images=torch.zeros_like(batch.images))
        elif permutation is not None:
            images = torch.stack(
                [dataset[index].image for index in permutation[offset : offset + count]]
            )
            batch = replace(batch, images=images)
        offset += count
        yield batch


def evaluate_image_controls(
    model: MultimodalLoopTransformer,
    dataset: SyntheticColorDataset,
    *,
    recurrence_depth: int,
    batch_size: int = 32,
    shuffle_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict:
    """Compare correct, permuted and zero pixels without changing supervision.

    Returns JSON-compatible metrics and donor permutations (recipient index ->
    donor index). Permutations include ordinary self/same-color coincidences;
    labels are used only for a diagnostic pairing fraction, never donor selection.
    The input dataset's stored order is retained in every condition. No updates
    are performed and evaluation preserves modes, gradients and random streams.
    """
    settings = SyntheticTrainingConfig(batch_size=batch_size, recurrence_depth=recurrence_depth)
    seeds = tuple(shuffle_seeds)
    if not seeds:
        raise ValueError("shuffle_seeds must not be empty")
    for seed in seeds:
        _integer("shuffle seed", seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("shuffle seeds must be distinct")
    if len(dataset) == 0:
        raise ValueError("image controls require at least one example")

    def score(*, permutation=None, blank=False):
        batches = synthetic_loader(dataset, model.config, settings)
        return asdict(
            evaluate_synthetic(
                model,
                _image_control_batches(batches, dataset, permutation=permutation, blank=blank),
                recurrence_depth=recurrence_depth,
            )
        )

    correct = score()
    shuffled = []
    # Read supervision only for reporting; never inspect scene metadata.
    answers = [dataset[index].answer for index in range(len(dataset))]
    for seed in seeds:
        permutation = list(range(len(dataset)))
        random.Random(seed).shuffle(permutation)
        metrics = score(permutation=permutation)
        shuffled.append(
            {
                "seed": seed,
                "permutation": permutation,
                "same_color_pairing_fraction": sum(
                    answers[index] == answers[donor] for index, donor in enumerate(permutation)
                )
                / len(dataset),
                "metrics": metrics,
                "accuracy_gap": correct["accuracy"] - metrics["accuracy"],
            }
        )
    blank = score(blank=True)
    summary = {}
    for name in ("accuracy", "loss"):
        values = [row["metrics"][name] for row in shuffled]
        summary[name] = {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
    return {
        "correct": correct,
        "shuffled": shuffled,
        "blank": blank,
        "shuffled_summary": summary,
        "accuracy_gaps": {
            "correct_minus_shuffled_mean": correct["accuracy"] - summary["accuracy"]["mean"],
            "correct_minus_blank": correct["accuracy"] - blank["accuracy"],
        },
    }
