"""Question-only relational evaluation and image/question dependence controls."""

import random
from collections.abc import Iterable
from dataclasses import asdict, replace
from itertools import combinations
from math import isfinite

import numpy as np
import torch
from torch.nn import functional as F

from multimodal_loop.data.collator import ColorQuestionBatch
from multimodal_loop.data.relational_dataset import RelationalColorDataset
from multimodal_loop.data.text import RelationalColorTokenizer
from multimodal_loop.eval.synthetic import EvaluationMetrics
from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.relational import relational_loader
from multimodal_loop.train.runtime import finish_step, preserve_device_rng
from multimodal_loop.train.synthetic import SyntheticTrainingConfig, _integer


def _predict_relational(
    model: MultimodalLoopTransformer,
    batches: Iterable[ColorQuestionBatch],
    *,
    recurrence_depth: int,
    details: list[dict] | None = None,
) -> tuple[EvaluationMetrics, list[int]]:
    """Preserve modes and random streams, including on failed evaluation."""
    _integer("recurrence_depth", recurrence_depth, 1)
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    tokenizer = RelationalColorTokenizer()
    total, correct, invalid, loss_sum = 0, 0, 0, 0.0
    predictions = []
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
                            "relational evaluation requires 11 question IDs and one answer"
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


def evaluate_relational(
    model: MultimodalLoopTransformer,
    batches: Iterable[ColorQuestionBatch],
    *,
    recurrence_depth: int,
) -> EvaluationMetrics:
    """Score the final question logit over the full vocabulary; never forward answers."""
    return _predict_relational(model, batches, recurrence_depth=recurrence_depth)[0]


def _grouped_metrics(dataset: RelationalColorDataset, predictions: list[int]) -> dict:
    """Group predictions in stored recipient order, using supervision only to score."""
    if len(predictions) != len(dataset) or not predictions:
        raise ValueError("grouped predictions must cover the entire nonempty dataset")
    tokenizer = RelationalColorTokenizer()
    by_question = {}
    images_correct, pairs_correct, pairs_total = 0, 0, 0
    for index, record in enumerate(dataset.records):
        targets = [tokenizer.encode_answer(qa.answer) for qa in record.questions]
        matches = [predictions[4 * index + q] == target for q, target in enumerate(targets)]
        images_correct += all(matches)
        for qa, match in zip(record.questions, matches, strict=True):
            counts = by_question.setdefault(qa.question, {"total": 0, "correct": 0})
            counts["total"] += 1
            counts["correct"] += match
        for a, b in combinations(range(4), 2):
            if targets[a] != targets[b]:
                pairs_total += 1
                pairs_correct += matches[a] and matches[b]
    for counts in by_question.values():
        counts["accuracy"] = counts["correct"] / counts["total"]
    return {
        "by_question": by_question,
        "all_four": {
            "total": len(dataset.records),
            "correct": images_correct,
            "accuracy": images_correct / len(dataset.records),
        },
        "different_answer_pairs": {
            "total": pairs_total,
            "correct": pairs_correct,
            "accuracy": pairs_correct / pairs_total,
        },
    }


def _control_batches(
    batches: Iterable[ColorQuestionBatch],
    dataset: RelationalColorDataset,
    *,
    image_permutation: list[int] | None = None,
    question_permutations: list[list[int]] | None = None,
    blank: bool = False,
) -> Iterable[ColorQuestionBatch]:
    """Intervene on pixels or question IDs, preserving recipient targets and layout.

    Indices address stored image records, including when a batch divides an image's
    four questions. Question permutations map recipient slot to donor slot.
    """
    if sum((image_permutation is not None, question_permutations is not None, blank)) > 1:
        raise ValueError("apply only one control at a time")
    offset = 0
    tokenizer = RelationalColorTokenizer()
    for batch in batches:
        count = batch.input_ids.shape[0]
        if blank:
            batch = replace(batch, images=torch.zeros_like(batch.images))
        elif image_permutation is not None:
            batch = replace(
                batch,
                images=torch.stack(
                    [
                        dataset[4 * image_permutation[i // 4]].image
                        for i in range(offset, offset + count)
                    ]
                ),
            )
        elif question_permutations is not None:
            ids = batch.input_ids.clone()
            for row, i in enumerate(range(offset, offset + count)):
                image_index, slot = divmod(i, 4)
                donor = question_permutations[image_index][slot]
                question = dataset.records[image_index].questions[donor].question
                ids[row, : batch.question_length] = ids.new_tensor(
                    tokenizer.encode_question(question)
                )
            batch = replace(batch, input_ids=ids)
        offset += count
        yield batch
    if offset != len(dataset):
        raise ValueError("controls require complete batches in stored dataset order")


def evaluate_relational_controls(
    model: MultimodalLoopTransformer,
    dataset: RelationalColorDataset,
    *,
    recurrence_depth: int,
    batch_size: int = 32,
    shuffle_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict:
    """Frozen-model interventions with ordinary, label-independent permutations.

    Grouped/per-question metrics always refer to original recipient supervision.
    Shuffled questions intentionally break question/target alignment. Coincidences
    are retained, so this diagnostic does not have a 25% chance baseline.
    """
    settings = SyntheticTrainingConfig(batch_size=batch_size, recurrence_depth=recurrence_depth)
    seeds = tuple(shuffle_seeds)
    if not seeds:
        raise ValueError("shuffle_seeds must not be empty")
    for seed in seeds:
        _integer("shuffle seed", seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("shuffle seeds must be distinct")
    if not len(dataset):
        raise ValueError("controls require at least one image")

    def score(**control):
        metrics, predictions = _predict_relational(
            model,
            _control_batches(
                relational_loader(dataset, model.config, settings), dataset, **control
            ),
            recurrence_depth=recurrence_depth,
        )
        return {**asdict(metrics), **_grouped_metrics(dataset, predictions)}

    correct = score()
    blank = score(blank=True)
    shuffled_images, shuffled_questions = [], []
    for seed in seeds:
        permutation = list(range(len(dataset.records)))
        random.Random(seed).shuffle(permutation)
        metrics = score(image_permutation=permutation)
        shuffled_images.append(
            {
                "seed": seed,
                "permutation": permutation,
                "metrics": metrics,
                "accuracy_gap": correct["accuracy"] - metrics["accuracy"],
            }
        )
        rng = random.Random(seed)
        question_permutations = []
        for _ in dataset.records:
            slots = list(range(4))
            rng.shuffle(slots)
            question_permutations.append(slots)
        # Diagnostic only: answers do not participate in choosing permutations.
        same_answer = sum(
            record.questions[slot].answer == record.questions[donor].answer
            for record, slots in zip(dataset.records, question_permutations, strict=True)
            for slot, donor in enumerate(slots)
        ) / len(dataset)
        metrics = score(question_permutations=question_permutations)
        shuffled_questions.append(
            {
                "seed": seed,
                "permutations": question_permutations,
                "same_answer_pairing_fraction": same_answer,
                "metrics": metrics,
                "accuracy_gap": correct["accuracy"] - metrics["accuracy"],
            }
        )

    def summary(rows):
        result = {}
        for name in ("accuracy", "loss"):
            values = [row["metrics"][name] for row in rows]
            result[name] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
        return result

    image_summary, question_summary = summary(shuffled_images), summary(shuffled_questions)
    return {
        "correct": correct,
        "blank": blank,
        "shuffled_images": shuffled_images,
        "shuffled_questions": shuffled_questions,
        "shuffled_images_summary": image_summary,
        "shuffled_questions_summary": question_summary,
        "accuracy_gaps": {
            "correct_minus_blank": correct["accuracy"] - blank["accuracy"],
            "correct_minus_shuffled_images_mean": correct["accuracy"]
            - image_summary["accuracy"]["mean"],
            "correct_minus_shuffled_questions_mean": correct["accuracy"]
            - question_summary["accuracy"]["mean"],
        },
    }
