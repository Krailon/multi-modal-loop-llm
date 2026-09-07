"""Fixed-depth, single-process epochs for manifest-backed color questions."""

import random
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from multimodal_loop.data.collator import ColorQuestionBatch, SyntheticColorCollator
from multimodal_loop.data.multimodal import SyntheticColorDataset
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.trainer import train_on_batch


def _integer(name: str, value: int, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True)
class SyntheticTrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    recurrence_depth: int = 2

    def __post_init__(self) -> None:
        _integer("batch_size", self.batch_size, 1)
        _integer("recurrence_depth", self.recurrence_depth, 1)
        _integer("seed", self.seed)
        for name, positive in (("learning_rate", True), ("weight_decay", False)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(
                    f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
                )


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    examples: int
    steps: int


def synthetic_loader(
    dataset: SyntheticColorDataset,
    model_config: ModelConfig,
    training_config: SyntheticTrainingConfig,
    *,
    epoch: int | None = None,
) -> DataLoader:
    """epoch=None preserves manifest order; zero-based epochs shuffle locally.

    A private generator isolates DataLoader iterator base-seed draws from the
    model's dropout stream. No workers, discarded examples, or global RNG draws.
    """
    if dataset.image_size != model_config.image_size:
        raise ValueError("manifest and model image_size must match")
    indices = list(range(len(dataset)))
    if epoch is not None:
        _integer("epoch", epoch, 0)
        random.Random(f"{training_config.seed}:train:{epoch}").shuffle(indices)
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        sampler=indices,
        collate_fn=SyntheticColorCollator(model_config),
        num_workers=0,
        drop_last=False,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )


def train_synthetic_epoch(
    model: MultimodalLoopTransformer,
    optimizer: AdamW,
    batches: Iterable[ColorQuestionBatch],
    *,
    recurrence_depth: int,
) -> EpochMetrics:
    """Visit each supplied batch once, supervising only its one-token answers."""
    _integer("recurrence_depth", recurrence_depth, 1)
    device = next(model.parameters()).device
    total_loss, examples, steps = 0.0, 0, 0
    for batch in batches:
        batch = batch.to(device)
        loss = train_on_batch(
            model,
            optimizer,
            batch.input_ids,
            batch.images,
            steps=1,
            question_length=batch.question_length,
            recurrence_depth=recurrence_depth,
        )[0]
        count = batch.input_ids.shape[0]
        total_loss += loss * count
        examples += count
        steps += 1
    if examples == 0:
        raise ValueError("training requires at least one example")
    return EpochMetrics(total_loss / examples, examples, steps)
