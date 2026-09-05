"""Attention-mask construction for text and multimodal input sequences."""

import torch
from torch import Tensor


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
