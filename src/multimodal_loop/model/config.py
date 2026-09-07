"""Canonical architecture configuration for the multimodal loop transformer."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ModelConfig:
    """Model dimensions with small defaults for CPU correctness tests.

    ``max_seq_len`` limits the combined visual and text sequence length.
    ``recurrence_depth`` is the default number of applications of the shared
    recurrent stack; ``RecurrentTransformerCore`` accepts runtime overrides.
    Recurrence sampling policies belong to training configuration.

    Images have a fixed square size and use non-overlapping square patches.
    Instances can be serialized with ``dataclasses.asdict`` and reconstructed
    with ``ModelConfig(**values)``. Create a new instance to change settings.
    """

    vocab_size: int = 256
    max_seq_len: int = 128

    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256

    n_prelude_layers: int = 1
    n_recurrent_layers: int = 1
    n_coda_layers: int = 1
    recurrence_depth: int = 2

    image_size: int = 32
    patch_size: int = 8
    num_channels: int = 3

    dropout: float = 0.0
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        minimums = {
            "vocab_size": 1,
            "max_seq_len": 1,
            "d_model": 1,
            "n_heads": 1,
            "d_ff": 1,
            "n_prelude_layers": 0,
            "n_recurrent_layers": 1,
            "n_coda_layers": 0,
            "recurrence_depth": 1,
            "image_size": 1,
            "patch_size": 1,
            "num_channels": 1,
        }
        for name, minimum in minimums.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}, got {value}")

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        for name in ("dropout", "layer_norm_eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number, got {value!r}")
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.layer_norm_eps <= 0.0:
            raise ValueError("layer_norm_eps must be positive")

    @property
    def head_dim(self) -> int:
        """Number of features per attention head."""
        return self.d_model // self.n_heads

    @property
    def num_patches(self) -> int:
        """Number of visual tokens produced for one image."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def patch_dim(self) -> int:
        """Number of pixel values in one flattened patch."""
        return self.num_channels * self.patch_size**2
