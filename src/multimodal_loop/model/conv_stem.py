"""Convolutional image tokens with the original transformer sequence layout."""

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer


@dataclass(frozen=True)
class ConvStemConfig:
    channels: tuple[int, ...] = (32, 64, 64, 64, 64)
    pool_size: int = 8
    init_seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "channels", tuple(self.channels))
        if not self.channels:
            raise ValueError("stem requires at least one convolution")
        for value in (*self.channels, self.pool_size):
            if type(value) is not int or value < 1:
                raise ValueError("stem widths and pool size must be positive integers")
        if type(self.init_seed) is not int or not 0 <= self.init_seed < 2**63:
            raise ValueError("stem init_seed must be a nonnegative integer below 2**63")

    def validate_model(self, model: ModelConfig):
        if self.channels[-1] != model.d_model or self.pool_size != model.patch_size:
            raise ValueError("stem output width and pooling grid must match the model")


class ConvStemEmbedding(nn.Module):
    """Local features followed by regional mean pooling, in row-major token order."""

    def __init__(self, config: ModelConfig, stem_config: ConvStemConfig):
        super().__init__()
        stem_config.validate_model(config)
        self.config = config
        self.stem_config = stem_config
        layers = []
        previous = config.num_channels
        for width in stem_config.channels:
            layers.extend([nn.Conv2d(previous, width, 3, padding=1), nn.ReLU()])
            previous = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AvgPool2d(stem_config.pool_size, stride=stem_config.pool_size)

    def forward(self, images: Tensor) -> Tensor:
        expected = (self.config.num_channels, self.config.image_size, self.config.image_size)
        if images.ndim != 4 or images.shape[1:] != expected:
            raise ValueError(
                f"images must have shape [B,{expected[0]},{expected[1]},{expected[2]}]"
            )
        if not images.is_floating_point():
            raise TypeError("images must have a floating-point dtype")
        return self.pool(self.features(images)).flatten(2).transpose(1, 2)


def build_conv_stem_model(
    config: ModelConfig, stem_config: ConvStemConfig | None = None
) -> MultimodalLoopTransformer:
    """Initialize on CPU; preserve the ordinary baseline's non-image draws and RNG tail.

    The caller seeds the ordinary model initialization. Only the new image stem
    uses its explicit independent seed. No reference checkpoint is read.
    """
    stem_config = ConvStemConfig() if stem_config is None else stem_config
    stem_config.validate_model(config)
    with torch.device("cpu"):
        model = MultimodalLoopTransformer(config)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(stem_config.init_seed)
            model.embeddings.image_embedding = ConvStemEmbedding(config, stem_config)
    return model


def initialization_provenance(model) -> dict:
    """Hash initial parameter values for review; call before any training update."""
    hashes = {"non_image": hashlib.sha256(), "image_stem": hashlib.sha256()}
    for name, value in model.state_dict().items():
        key = "image_stem" if name.startswith("embeddings.image_embedding.") else "non_image"
        h = hashes[key]
        h.update(f"{name}:{tuple(value.shape)}:{value.dtype}\n".encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {
        "method": "ordinary baseline first; stem replaced under independent preserved CPU RNG",
        "stem_init_seed": model.embeddings.image_embedding.stem_config.init_seed,
        **{f"{k}_sha256": h.hexdigest() for k, h in hashes.items()},
    }


def forward_macs(config: ModelConfig, stem: ConvStemConfig, *, question_only: bool) -> dict:
    """Dense conv/linear/attention MAC estimate, excluding other operations and backward."""
    stem.validate_model(config)
    length = config.num_patches + (6 if question_only else 7)
    width = config.d_model
    blocks = (
        config.n_prelude_layers
        + config.recurrence_depth * config.n_recurrent_layers
        + config.n_coda_layers
    )
    shared = (
        blocks * (length * (4 * width**2 + 2 * width * config.d_ff) + 2 * length**2 * width)
        + length * width * config.vocab_size
    )
    channels = (config.num_channels, *stem.channels)
    convolutions = config.image_size**2 * sum(
        9 * a * b for a, b in zip(channels[:-1], channels[1:], strict=True)
    )
    return {
        "sequence_length": length,
        "direct_patch_transformer": shared + config.num_patches * config.patch_dim * width,
        "conv_stem_transformer": shared + convolutions,
    }
