"""Deterministic question-wise loading for fixed-depth relational training."""

import random

import torch
from torch.utils.data import DataLoader

from multimodal_loop.data.collator import RelationalColorCollator
from multimodal_loop.data.relational_dataset import RelationalColorDataset
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.train.synthetic import SyntheticTrainingConfig, _integer


def relational_loader(
    dataset: RelationalColorDataset,
    model_config: ModelConfig,
    training_config: SyntheticTrainingConfig,
    *,
    epoch: int | None = None,
) -> DataLoader:
    """Preserve manifest order for evaluation; shuffle QA indices per epoch.

    Private random streams keep ordering independent of model dropout. Each
    question is visited once, including a partial final batch, with no workers.
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
        collate_fn=RelationalColorCollator(model_config),
        num_workers=0,
        drop_last=False,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
