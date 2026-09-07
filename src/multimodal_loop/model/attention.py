"""Self-attention and mask construction for text and multimodal sequences."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from multimodal_loop.model.config import ModelConfig


def _validate_hidden_states(x: Tensor, d_model: int) -> None:
    """Check the shared hidden-state contract before attention or normalization."""
    if x.ndim != 3:
        raise ValueError(
            f"x must have rank 3 [batch, seq_len, d_model], got shape {tuple(x.shape)}"
        )
    if x.shape[2] != d_model:
        raise ValueError(f"x must have width d_model={d_model}, got {x.shape[2]}")
    if x.shape[1] <= 0:
        raise ValueError("x must have a positive sequence length")
    if not x.is_floating_point():
        raise TypeError(f"x must have a floating-point dtype, got {x.dtype}")


def _validate_attention_mask(mask: Tensor, seq_len: int, device: torch.device) -> None:
    """Check a shared boolean mask, including for stacks with no blocks."""
    if mask.shape != (seq_len, seq_len):
        raise ValueError(
            f"attention_mask must have shape {(seq_len, seq_len)}, got {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError(f"attention_mask must have boolean dtype, got {mask.dtype}")
    if mask.device != device:
        raise ValueError("attention_mask must be on the same device as x")
    if not mask.any(dim=-1).all():
        raise ValueError("attention_mask must allow at least one key per query")


def build_causal_mask(seq_len: int, *, device: torch.device | str | None = None) -> Tensor:
    """Return a boolean ``[seq_len, seq_len]`` causal attention mask.

    Rows are query positions and columns are key positions. ``True`` allows
    attention, matching PyTorch's scaled dot-product attention convention.
    Each input position can attend to itself and earlier positions; training
    targets must be shifted to the next token.

    ``seq_len`` must be a positive integer, excluding booleans. If ``device``
    is omitted, tensor placement follows PyTorch's default device.
    """
    return build_prefix_mask(seq_len, prefix_len=0, device=device)


def build_prefix_mask(
    seq_len: int, prefix_len: int, *, device: torch.device | str | None = None
) -> Tensor:
    """Return a boolean ``[seq_len, seq_len]`` multimodal prefix mask.

    Rows are queries, columns are keys, and ``True`` allows attention. The first
    ``prefix_len`` input positions form a bidirectional image/question context.
    They cannot attend to any answer inputs. Remaining positions can attend to
    the entire prefix, themselves, and earlier answer inputs.

    Lengths must be integers, excluding booleans, with ``seq_len > 0`` and
    ``0 <= prefix_len <= seq_len``. A zero-length prefix is fully causal; a
    full-length prefix permits all attention. One mask can be shared across
    batch items and attention heads. Padding and per-example prefix lengths
    are not handled here. An omitted ``device`` follows PyTorch's default.
    """
    if isinstance(seq_len, bool) or not isinstance(seq_len, int):
        raise TypeError(f"seq_len must be an integer, got {seq_len!r}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if isinstance(prefix_len, bool) or not isinstance(prefix_len, int):
        raise TypeError(f"prefix_len must be an integer, got {prefix_len!r}")
    if not 0 <= prefix_len <= seq_len:
        raise ValueError(f"prefix_len must be in [0, seq_len={seq_len}], got {prefix_len}")

    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    mask[:prefix_len, :prefix_len] = True
    return mask


class SelfAttention(nn.Module):
    """Conventional multi-head self-attention with a causal default.

    Query, key, and value features share one linear projection; an output
    projection maps the concatenated heads back to the model width. Both
    projections use bias and standard PyTorch initialization. Dropout applies
    to attention probabilities during training only.

    Residual connections, normalization, and positional embeddings belong to
    the surrounding model components.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Map floating-point ``[B, T, d_model]`` inputs to the same shape.

        An omitted mask uses causal attention. Explicit masks must be boolean
        ``[T, T]`` tensors on the input device, with ``True`` allowing attention
        and at least one allowed key per query. The mask is shared across batch
        items and heads; padding and per-example masks are not supported here.

        Device and dtype follow normal PyTorch module conventions, without
        implicit transfers or casts. Only transformed representations are
        returned, without attention weights or a residual connection.
        """
        _validate_hidden_states(x, self.config.d_model)
        batch_size, seq_len, width = x.shape

        if attention_mask is None:
            attention_mask = build_causal_mask(seq_len, device=x.device)
        else:
            _validate_attention_mask(attention_mask, seq_len, x.device)

        # [B, T, 3 * d_model] -> [3, B, n_heads, T, head_dim]
        qkv = self.qkv_proj(x).reshape(
            batch_size, seq_len, 3, self.config.n_heads, self.config.head_dim
        )
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
        )
        # [B, n_heads, T, head_dim] -> [B, T, d_model]
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, width)
        return self.out_proj(attended)
