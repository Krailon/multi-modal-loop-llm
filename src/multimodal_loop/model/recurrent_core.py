"""Recurrent-depth computation with one weight-shared transformer stack."""

from torch import Tensor, nn

from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.transformer import TransformerStack


class RecurrentTransformerCore(nn.Module):
    """Repeat one transformer stack while preserving the full autograd graph.

    The stack contains ``config.n_recurrent_layers`` independent blocks. Every
    recurrence reuses those same parameters: ``h_next = stack(h)``. Runtime
    depth changes the computation count, not the number of model parameters.
    Dropout follows normal module training/evaluation behavior, with fresh
    random draws across training steps and no resets inside the loop.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.stack = TransformerStack(config, config.n_recurrent_layers)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        *,
        recurrence_depth: int | None = None,
    ) -> Tensor:
        """Return the final ``[B, T, d_model]`` state after all recurrent steps.

        ``None`` selects ``config.recurrence_depth``. An override must be a
        positive integer, excluding booleans, and may exceed the configured
        default. The same mask is supplied on every repetition; input and mask
        validation follow ``TransformerStack``. Gradients flow through every
        repetition to the input and the shared parameters.
        """
        depth = self.config.recurrence_depth if recurrence_depth is None else recurrence_depth
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise TypeError(f"recurrence_depth must be an integer, got {depth!r}")
        if depth <= 0:
            raise ValueError(f"recurrence_depth must be positive, got {depth}")

        for _ in range(depth):
            x = self.stack(x, attention_mask)
        return x
