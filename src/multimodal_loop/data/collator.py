"""Model-ready batches for the fixed single-answer synthetic color task."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from multimodal_loop.data.synthetic_shapes import SyntheticExample
from multimodal_loop.data.text import ColorQuestionTokenizer
from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.config import ModelConfig


@dataclass(frozen=True)
class ColorQuestionBatch:
    """Explicit model inputs and supervision, with no scene metadata.

    The final question input predicts the answer. The answer input itself has
    no supervised successor and cannot influence the image/question prefix.
    Frozen fields do not make tensors immutable; .to follows Tensor.to storage
    semantics, including possible sharing when no transfer is necessary.
    """

    images: Tensor
    input_ids: Tensor
    target_mask: Tensor
    attention_mask: Tensor
    question_length: int
    num_image_tokens: int

    def to(self, device: str | torch.device) -> "ColorQuestionBatch":
        """Transfer all tensors together; accelerator initialization is caller-owned."""
        return ColorQuestionBatch(
            images=self.images.to(device),
            input_ids=self.input_ids.to(device),
            target_mask=self.target_mask.to(device),
            attention_mask=self.attention_mask.to(device),
            question_length=self.question_length,
            num_image_tokens=self.num_image_tokens,
        )

    def model_inputs(self) -> dict[str, Tensor]:
        """Only arguments accepted by MultimodalLoopTransformer.forward."""
        return {
            "input_ids": self.input_ids,
            "images": self.images,
            "attention_mask": self.attention_mask,
        }


class SyntheticColorCollator:
    """Stack CPU examples using only image, question, and explicit answer fields.

    Scene metadata is deliberately never accessed, including for validation.
    This allows controlled image replacement later without deriving labels from
    replacement images or metadata. Images must match the model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.tokenizer = ColorQuestionTokenizer()
        if config.num_channels != 3:
            raise ValueError("synthetic color batches require three image channels")
        if config.vocab_size < self.tokenizer.vocab_size:
            raise ValueError("model vocab_size must be at least 10")
        if config.max_seq_len < config.num_patches + self.tokenizer.question_length + 1:
            raise ValueError("model max_seq_len must fit image patches, question, and answer")

    def __call__(self, examples: Sequence[SyntheticExample]) -> ColorQuestionBatch:
        if len(examples) == 0:
            raise ValueError("cannot collate an empty batch")
        images, sequences = [], []
        for example in examples:
            image = example.image
            if not isinstance(image, Tensor):
                raise TypeError("image must be a tensor")
            expected_shape = (3, self.config.image_size, self.config.image_size)
            if image.shape != expected_shape:
                raise ValueError(f"image must have shape {expected_shape}")
            if image.device.type != "cpu":
                raise ValueError("collator expects CPU images; transfer the completed batch")
            if image.dtype != torch.float32:
                raise TypeError("image must have dtype float32")
            if not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
                raise ValueError("image pixels must be finite and in [0, 1]")
            question = self.tokenizer.encode_question(example.question)
            answer = self.tokenizer.encode_answer(example.answer)
            sequences.append((*question, answer))
            images.append(image)
        ids = torch.tensor(sequences, dtype=torch.int64, device="cpu")
        target_mask = torch.zeros(ids.shape, dtype=torch.bool, device="cpu")
        target_mask[:, -1] = True
        question_length = self.tokenizer.question_length
        patches = self.config.num_patches
        return ColorQuestionBatch(
            images=torch.stack(images),
            input_ids=ids,
            target_mask=target_mask,
            attention_mask=build_prefix_mask(
                patches + ids.shape[1], patches + question_length, device="cpu"
            ),
            question_length=question_length,
            num_image_tokens=patches,
        )
