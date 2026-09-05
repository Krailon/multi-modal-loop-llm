"""Conventional transformer blocks for the multimodal loop model."""

from torch import Tensor, nn

from multimodal_loop.model.attention import SelfAttention
from multimodal_loop.model.config import ModelConfig


class TransformerBlock(nn.Module):
    """Apply attention and a feed-forward network with pre-branch LayerNorm.

    Each branch normalizes its input, computes an update, and applies dropout
    before adding that update to the residual stream. The feed-forward network
    is Linear -> GELU -> Linear, with width ``config.d_ff``. Normalization acts
    only on the feature dimension, independently at every sequence position.

    ``config.dropout`` controls both residual-branch dropout modules as well as
    the existing attention-probability dropout inside ``SelfAttention``.
    Calling ``eval()`` disables all dropout. Linear layers and the two separate
    affine LayerNorm modules use standard PyTorch initialization.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = SelfAttention(config)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.ffn_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Transform ``[B, T, d_model]`` inputs without modifying them in place.

        Inputs must be floating point with a positive sequence length. Mask
        validation and the causal default follow ``SelfAttention``: explicit
        masks are boolean ``[T, T]`` tensors with ``True`` allowing attention.
        Device and dtype follow normal PyTorch module conventions, without
        implicit transfers or casts. The output has the same shape as ``x``.
        """
        if x.ndim != 3:
            raise ValueError(
                f"x must have rank 3 [batch, seq_len, d_model], got shape {tuple(x.shape)}"
            )
        if x.shape[2] != self.config.d_model:
            raise ValueError(f"x must have width d_model={self.config.d_model}, got {x.shape[2]}")
        if x.shape[1] <= 0:
            raise ValueError("x must have a positive sequence length")
        if not x.is_floating_point():
            raise TypeError(f"x must have a floating-point dtype, got {x.dtype}")

        h = x + self.attn_dropout(self.attn(self.attn_norm(x), attention_mask))
        return h + self.ffn_dropout(self.ffn(self.ffn_norm(h)))
