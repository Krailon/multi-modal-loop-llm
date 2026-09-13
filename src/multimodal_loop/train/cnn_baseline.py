"""Deterministic CNN data path and bounded training, independent of sequence models."""

import random
from dataclasses import dataclass
from math import isfinite

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from multimodal_loop.data.shape_grounding import ShapeColorTokenizer
from multimodal_loop.train.runtime import preserve_device_rng


@dataclass(frozen=True)
class CNNTrainingConfig:
    seed: int = 0
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0

    def __post_init__(self):
        if type(self.seed) is not int or type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("seed must be integer and batch size positive integer")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        if not isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be nonnegative and finite")


def collate_cnn(examples):
    """Read only public pixels, question text and external answer supervision."""
    tokenizer = ShapeColorTokenizer()
    return (
        torch.stack([e.image for e in examples]),
        torch.tensor([tokenizer.encode_question(e.question) for e in examples], dtype=torch.int64),
        torch.tensor([tokenizer.encode_answer(e.answer) for e in examples], dtype=torch.int64),
    )


def cnn_loader(dataset, training, *, epoch=None):
    indices = list(range(len(dataset)))
    if epoch is not None:
        random.Random(f"{training.seed}:train:{epoch}").shuffle(indices)
    return DataLoader(
        dataset,
        batch_size=training.batch_size,
        sampler=indices,
        collate_fn=collate_cnn,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )


def predict_cnn(model, dataset, training):
    device = next(model.parameters()).device
    modes = [(m, m.training) for m in model.modules()]
    details = []
    try:
        with torch.random.fork_rng(devices=[]), preserve_device_rng(device), torch.inference_mode():
            model.eval()
            for batch in cnn_loader(dataset, training):
                images, questions, targets = (t.to(device) for t in batch)
                logits = model(images, questions)
                losses = F.cross_entropy(logits, targets, reduction="none")
                if not torch.isfinite(losses).all():
                    raise ValueError("evaluation loss must be finite")
                predictions = logits.argmax(-1)
                probabilities = logits.softmax(-1)
                values = (
                    predictions,
                    targets,
                    losses,
                    probabilities.gather(1, predictions[:, None]).squeeze(1),
                    probabilities.gather(1, targets[:, None]).squeeze(1),
                )
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
                            row,
                            strict=True,
                        )
                    )
                    for row in zip(*(v.cpu().tolist() for v in values), strict=True)
                )
    finally:
        for module, mode in modes:
            module.training = mode
    if not details:
        raise ValueError("evaluation requires examples")
    total = len(details)
    correct = sum(d["prediction_id"] == d["target_id"] for d in details)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "invalid_predictions": sum(not 6 <= d["prediction_id"] < 10 for d in details),
        "loss": sum(d["loss"] for d in details) / total,
    }, details


def train_cnn(model, optimizer, dataset, training, budget, *, publish):
    history = []
    steps = examples = epoch = offset = interval_examples = interval_steps = 0
    interval_loss = 0.0
    device = next(model.parameters()).device

    def record():
        metrics, _ = predict_cnn(model, dataset, training)
        history.append(
            {
                "completed_steps": steps,
                "examples_seen": examples,
                "completed_passes": epoch,
                "batches_in_pass": offset,
                "training_fit": metrics,
                "train": None
                if not interval_steps
                else {
                    "loss": interval_loss / interval_examples,
                    "examples": interval_examples,
                    "steps": interval_steps,
                },
            }
        )
        publish(history)

    record()
    while steps < budget.max_steps:
        loader = cnn_loader(dataset, training, epoch=epoch)
        for index, batch in enumerate(loader):
            model.train()
            images, questions, targets = (t.to(device) for t in batch)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images, questions), targets)
            if not torch.isfinite(loss):
                raise ValueError("training loss must be finite")
            loss.backward()
            optimizer.step()
            count = len(targets)
            examples += count
            steps += 1
            interval_examples += count
            interval_steps += 1
            interval_loss += loss.item() * count
            offset = index + 1
            if offset == len(loader):
                epoch += 1
                offset = 0
            if steps % budget.evaluation_interval == 0 or steps == budget.max_steps:
                record()
                interval_examples = interval_steps = 0
                interval_loss = 0.0
            if steps == budget.max_steps:
                break
    return history
