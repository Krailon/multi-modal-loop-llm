"""Small full-resolution CNN reference for fixed direct shape/color questions."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CNNConfig:
    image_size: int = 32
    vocab_size: int = 17
    channels: tuple[int, ...] = (32, 64, 64, 64, 64)
    question_dim: int = 32
    hidden_dim: int = 128

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        if self.image_size != 32 or self.vocab_size != 17:
            raise ValueError("CNN baseline requires 32-pixel images and vocabulary size 17")
        for value in (*self.channels, self.question_dim, self.hidden_dim):
            if type(value) is not int or value < 1:
                raise ValueError("CNN dimensions must be positive integers")
        if not self.channels:
            raise ValueError("CNN requires convolutional layers")


class CNNBaseline(nn.Module):
    """Pixels and six question tokens in; 17 answer logits out. No answer input."""

    def __init__(self, config: CNNConfig | None = None):
        super().__init__()
        config = CNNConfig() if config is None else config
        self.config = config
        layers = []
        previous = 3
        for width in config.channels:
            layers.extend([nn.Conv2d(previous, width, 3, padding=1), nn.ReLU()])
            previous = width
        self.visual = nn.Sequential(*layers)
        self.question = nn.Embedding(config.vocab_size, config.question_dim)
        self.head = nn.Sequential(
            nn.Linear(previous + config.question_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.vocab_size),
        )

    def forward(self, images: Tensor, question_ids: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 32, 32) or images.shape[0] == 0:
            raise ValueError("images must be [B,3,32,32] with B > 0")
        if question_ids.shape != (images.shape[0], 6) or question_ids.dtype != torch.int64:
            raise ValueError("question_ids must be int64 [B,6], excluding answers")
        visual = self.visual(images).mean(dim=(-2, -1))
        question = self.question(question_ids).mean(dim=1)
        return self.head(torch.cat((visual, question), dim=1))


def forward_macs(config: CNNConfig) -> int:
    """Convolution/linear multiply-accumulates per QA; excludes pooling and activations."""
    channels = (3, *config.channels)
    return (
        config.image_size**2
        * sum(a * b * 9 for a, b in zip(channels[:-1], channels[1:], strict=True))
        + (channels[-1] + config.question_dim) * config.hidden_dim
        + config.hidden_dim * config.vocab_size
    )
