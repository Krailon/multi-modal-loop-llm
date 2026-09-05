"""Direct image patch embeddings without a pretrained vision encoder."""

from torch import Tensor, nn

from multimodal_loop.model.config import ModelConfig


class PatchEmbedding(nn.Module):
    """Project non-overlapping image patches into the shared model width.

    Patch tokens are ordered left to right, then top to bottom. Within each
    patch, pixel values are flattened in channel, row, column order. Positional
    and modality embeddings are added separately by the model's embeddings.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.patch_dim, config.d_model)

    def forward(self, images: Tensor) -> Tensor:
        """Map ``[B, C, H, W]`` images to ``[B, num_patches, d_model]``.

        Inputs must be floating-point tensors matching the configured channel
        count and square image size. Device and dtype follow normal PyTorch
        module conventions; this method performs no transfers or casts.
        """
        if images.ndim != 4:
            raise ValueError(
                f"images must have rank 4 [batch, channels, height, width], "
                f"got shape {tuple(images.shape)}"
            )
        if images.shape[1] != self.config.num_channels:
            raise ValueError(
                f"images must have {self.config.num_channels} channels, got {images.shape[1]}"
            )
        expected_size = (self.config.image_size, self.config.image_size)
        if images.shape[2:] != expected_size:
            raise ValueError(
                f"images must have spatial size {expected_size}, got {tuple(images.shape[2:])}"
            )
        if not images.is_floating_point():
            raise TypeError(f"images must have a floating-point dtype, got {images.dtype}")

        patch_size = self.config.patch_size
        # [B, C, grid_rows, grid_cols, patch_rows, patch_cols]
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # [B, grid_rows, grid_cols, C, patch_rows, patch_cols] -> [B, N, patch_dim]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            images.shape[0], self.config.num_patches, self.config.patch_dim
        )
        return self.projection(patches)
