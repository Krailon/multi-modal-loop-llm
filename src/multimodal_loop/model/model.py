"""Complete multimodal transformer with a weight-shared recurrent core."""

from torch import Tensor, nn

from multimodal_loop.model.attention import build_causal_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.embeddings import MultimodalEmbedding
from multimodal_loop.model.recurrent_core import RecurrentTransformerCore
from multimodal_loop.model.transformer import TransformerStack


class MultimodalLoopTransformer(nn.Module):
    """Process image and text tokens together and predict language tokens.

    Embeddings are added once, followed by the prelude, recurrent core, coda,
    and a final feature-wise LayerNorm. A bias-free vocabulary projection has
    independent weights from the text embedding table. All components use
    standard PyTorch initialization and device/dtype conventions.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = MultimodalEmbedding(config)
        self.prelude = TransformerStack(config, config.n_prelude_layers)
        self.core = RecurrentTransformerCore(config)
        self.coda = TransformerStack(config, config.n_coda_layers)
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        images: Tensor | None = None,
        recurrence_depth: int | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Return raw ``[B, L, vocab_size]`` logits for every input position.

        ``input_ids`` has shape ``[B, T]``; optional images have shape
        ``[B, C, H, W]``. Image patches precede text, so ``L = num_patches + T``
        with images and ``L = T`` otherwise. Embedding validation applies,
        including the combined sequence limit ``config.max_seq_len``.

        An omitted mask uses causal attention over the complete sequence.
        Explicit masks must be boolean ``[L, L]`` tensors on the input device,
        with ``True`` allowing attention. The same mask is used in all stages
        and recurrent steps. Question-answer callers should build a prefix
        mask covering image/question inputs and excluding answer inputs.

        An omitted depth uses ``config.recurrence_depth``; the core validates
        positive integer overrides and reuses its parameters at every step.
        Logits predict the next token. Callers shift targets and select loss
        positions; for question answering, the final question input predicts
        the first answer token. This method does not compute a loss.
        """
        x = self.embeddings(input_ids, images)
        if attention_mask is None:
            attention_mask = build_causal_mask(x.shape[1], device=x.device)

        x = self.prelude(x, attention_mask)
        x = self.core(x, attention_mask, recurrence_depth=recurrence_depth)
        x = self.coda(x, attention_mask)
        return self.lm_head(self.final_norm(x))
