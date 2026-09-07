"""Conventional transformer blocks and stacks for the multimodal loop model."""

from torch import Tensor, nn

from multimodal_loop.model.attention import (
    SelfAttention,
    _validate_attention_mask,
    _validate_hidden_states,
)
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
        _validate_hidden_states(x, self.config.d_model)

        h = x + self.attn_dropout(self.attn(self.attn_norm(x), attention_mask))
        return h + self.ffn_dropout(self.ffn(self.ffn_norm(h)))


class TransformerStack(nn.Module):
    """Apply independently parameterized transformer blocks in order.

    ``num_layers`` is an explicit nonnegative integer, excluding booleans. A
    zero-layer stack is a parameter-free identity for optional prelude/coda
    stacks. Input and mask validation also apply to that identity case.
    """

    def __init__(self, config: ModelConfig, num_layers: int) -> None:
        super().__init__()
        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise TypeError(f"num_layers must be an integer, got {num_layers!r}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be nonnegative, got {num_layers}")

        self.config = config
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(num_layers))

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Preserve ``[B, T, d_model]`` and pass the mask to every block.

        An omitted mask uses each block's causal default. The input is never
        modified in place; an empty stack returns it directly. The stack adds
        no normalization or residual connection outside its blocks.
        """
        _validate_hidden_states(x, self.config.d_model)
        if attention_mask is not None:
            _validate_attention_mask(attention_mask, x.shape[1], x.device)

        for block in self.blocks:
            x = block(x, attention_mask)
        return x
