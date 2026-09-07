"""Assemble image and text tokens with learned position and modality embeddings."""

import torch
from torch import Tensor, nn

from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.patch_embedding import PatchEmbedding


class MultimodalEmbedding(nn.Module):
    """Construct the shared sequence once, before the transformer prelude.

    Image patches precede text tokens. Every token is the sum of its content,
    learned 1D position, and learned modality (text 0, image 1) embeddings.
    Positions run continuously from zero across the combined sequence.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.text_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.image_embedding = PatchEmbedding(config)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.modality_embedding = nn.Embedding(2, config.d_model)

    def forward(self, input_ids: Tensor, images: Tensor | None = None) -> Tensor:
        """Map ``[B, T]`` IDs and optional ``[B, C, H, W]`` images to tokens.

        Return ``[B, T, d_model]`` for text only, or
        ``[B, num_patches + T, d_model]`` with one image per batch item. IDs
        must be int32 or int64, with at least one text token per item. All
        vocabulary IDs, including zero, are ordinary trainable tokens.

        The actual combined length must fit ``config.max_seq_len``. Inputs
        and parameters follow normal PyTorch device and dtype conventions;
        no implicit transfers or casts are performed. Image shape and dtype
        validation are delegated to ``PatchEmbedding``.
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have rank 2 [batch, text_length], "
                f"got shape {tuple(input_ids.shape)}"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"input_ids must have dtype int32 or int64, got {input_ids.dtype}")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one text token per item")
        if input_ids.device != self.text_embedding.weight.device:
            raise ValueError("input_ids must be on the same device as the embedding parameters")
        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            raise ValueError(f"input_ids must be in [0, {self.config.vocab_size})")

        num_image_tokens = self.config.num_patches if images is not None else 0
        seq_len = num_image_tokens + input_ids.shape[1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"combined sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        tokens = self.text_embedding(input_ids)
        if images is not None:
            if images.device != input_ids.device:
                raise ValueError("images and input_ids must be on the same device")
            image_tokens = self.image_embedding(images)
            if image_tokens.shape[0] != input_ids.shape[0]:
                raise ValueError("images and input_ids must have the same batch size")
            tokens = torch.cat((image_tokens, tokens), dim=1)

        positions = torch.arange(seq_len, device=input_ids.device)
        modality_ids = torch.zeros(seq_len, dtype=torch.long, device=input_ids.device)
        modality_ids[:num_image_tokens] = 1
        return tokens + self.position_embedding(positions) + self.modality_embedding(modality_ids)
