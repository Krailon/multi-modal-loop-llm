"""Direct shape-to-color questions over the unchanged relational scene manifest."""

import operator
from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import Dataset

from multimodal_loop.data.collator import _collate_color_examples, _validate_color_config
from multimodal_loop.data.relational_dataset import RelationalManifest
from multimodal_loop.data.relational_shapes import MultiObjectScene, render_multi_object_scene
from multimodal_loop.data.synthetic_shapes import SHAPES
from multimodal_loop.data.text import RelationalColorTokenizer
from multimodal_loop.model.config import ModelConfig


class ShapeColorTokenizer(RelationalColorTokenizer):
    """Keep all 17 baseline IDs; accept only the three direct-color questions."""

    question_length = 6

    def encode_question(self, question: str) -> tuple[int, ...]:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        for shape in SHAPES:
            if question == f"What color is the {shape}?":
                return (0, 1, 2, 3, self.vocabulary.index(shape), 5)
        raise ValueError("unsupported direct shape-color question")


@dataclass(frozen=True)
class ShapeColorExample:
    image: Tensor
    question: str
    answer: str
    scene: MultiObjectScene
    shape: str


class ShapeColorDataset(Dataset[ShapeColorExample]):
    """Image-major, then square/circle/triangle; test examples are deliberately unavailable."""

    def __init__(self, manifest: RelationalManifest, split: str):
        if split not in ("train", "validation"):
            raise ValueError("shape grounding supports only train and validation")
        self.records = manifest.splits[split]
        self.split = split
        self.image_size = manifest.config.image_size

    def __len__(self):
        return 3 * len(self.records)

    def __getitem__(self, index: int) -> ShapeColorExample:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("shape-color dataset index out of range")
        image, slot = divmod(index, 3)
        scene = self.records[image].scene
        shape = SHAPES[slot]
        target = next(obj for obj in scene.objects if obj.shape == shape)
        return ShapeColorExample(
            render_multi_object_scene(scene),
            f"What color is the {shape}?",
            target.color,
            scene,
            shape,
        )


class ShapeColorCollator:
    """Read only image, question and explicit answer; never inspect scene metadata."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = ShapeColorTokenizer()
        _validate_color_config(config, self.tokenizer)

    def __call__(self, examples):
        return _collate_color_examples(examples, self.config, self.tokenizer)
