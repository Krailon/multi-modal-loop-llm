"""Deterministic direct-color loading and an explicit update-limited training budget."""

import random
from dataclasses import asdict, dataclass
from itertools import islice

import torch
from torch.utils.data import DataLoader

from multimodal_loop.data.shape_grounding import ShapeColorCollator, ShapeColorDataset
from multimodal_loop.eval.color import predict_color_answers
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.train.synthetic import SyntheticTrainingConfig, _integer, train_synthetic_epoch


@dataclass(frozen=True)
class ShapeGroundingConfig:
    max_steps: int = 2880
    evaluation_interval: int = 288

    def __post_init__(self):
        _integer("max_steps", self.max_steps, 1)
        _integer("evaluation_interval", self.evaluation_interval, 1)


def shape_color_loader(
    dataset: ShapeColorDataset,
    model_config: ModelConfig,
    training: SyntheticTrainingConfig,
    *,
    epoch: int | None = None,
):
    if dataset.image_size != model_config.image_size:
        raise ValueError("manifest and model image_size must match")
    indices = list(range(len(dataset)))
    if epoch is not None:
        _integer("epoch", epoch, 0)
        random.Random(f"{training.seed}:train:{epoch}").shuffle(indices)
    return DataLoader(
        dataset,
        batch_size=training.batch_size,
        sampler=indices,
        collate_fn=ShapeColorCollator(model_config),
        num_workers=0,
        drop_last=False,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )


def train_shape_grounding(model, optimizer, manifest, training, budget, *, publish=None):
    """Train fresh weights supplied by the caller; stop even in the middle of a pass.

    Evaluate at fixed update intervals, independently of pass boundaries. The final
    partial batch, when present in a small smoke corpus, counts its actual examples.
    No validation metric affects scheduling or checkpoint selection.
    """
    train_data = ShapeColorDataset(manifest, "train")
    validation = ShapeColorDataset(manifest, "validation")
    return train_shape_datasets(
        model, optimizer, train_data, validation, training, budget, publish=publish
    )


def train_shape_datasets(
    model, optimizer, train_data, validation, training, budget, *, publish=None
):
    """Shared update-limited loop; datasets supply pixels and direct-color supervision."""
    history = []
    steps = examples = epoch = offset = 0
    interval_loss = interval_examples = interval_steps = 0

    def record():
        metrics, _ = predict_color_answers(
            model,
            shape_color_loader(validation, model.config, training),
            recurrence_depth=training.recurrence_depth,
            question_length=6,
        )
        row = {
            "completed_steps": steps,
            "examples_seen": examples,
            "completed_passes": epoch,
            "batches_in_pass": offset,
            "train": None
            if not interval_steps
            else {
                "loss": interval_loss / interval_examples,
                "examples": interval_examples,
                "steps": interval_steps,
            },
            "validation": asdict(metrics),
        }
        history.append(row)
        if publish is not None:
            publish(history)

    record()
    while steps < budget.max_steps:
        loader = shape_color_loader(train_data, model.config, training, epoch=epoch)
        iterator = iter(loader)
        offset = 0
        while offset < len(loader) and steps < budget.max_steps:
            count = min(
                len(loader) - offset,
                budget.max_steps - steps,
                budget.evaluation_interval - steps % budget.evaluation_interval,
            )
            metrics = train_synthetic_epoch(
                model,
                optimizer,
                islice(iterator, count),
                recurrence_depth=training.recurrence_depth,
            )
            steps += metrics.steps
            offset += metrics.steps
            examples += metrics.examples
            interval_steps += metrics.steps
            interval_examples += metrics.examples
            interval_loss += metrics.loss * metrics.examples
            if offset == len(loader):
                epoch += 1
                offset = 0
                pass_finished = True
            else:
                pass_finished = False
            if steps % budget.evaluation_interval == 0 or steps == budget.max_steps:
                record()
                interval_loss = interval_examples = interval_steps = 0
            if pass_finished:
                break
    return history
