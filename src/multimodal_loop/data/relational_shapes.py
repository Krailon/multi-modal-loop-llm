"""Three-object horizontal scenes and independently resolved color relations."""

from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor

from multimodal_loop.data.synthetic_shapes import SHAPES, ShapeScene, _integer, render_scene


@dataclass(frozen=True)
class MultiObjectScene:
    """Inspection/supervision metadata for exactly three objects in one row.

    Bounding-box centers share a vertical coordinate. Adjacent boxes have at
    least one blank pixel between them, and all boxes retain a one-pixel canvas
    margin. Tuple order is arbitrary; repeated shapes and colors are allowed.
    """

    objects: tuple[ShapeScene, ...]
    image_size: int = 32

    def __post_init__(self) -> None:
        _integer("image_size", self.image_size, 6)
        if not isinstance(self.objects, tuple):
            raise TypeError("objects must be a tuple")
        if len(self.objects) != 3:
            raise ValueError("scene requires exactly three objects")
        if any(not isinstance(obj, ShapeScene) for obj in self.objects):
            raise TypeError("objects must contain ShapeScene instances")
        for obj in self.objects:
            if obj.left + obj.size >= self.image_size or obj.top + obj.size >= self.image_size:
                raise ValueError("objects must fit with a one-pixel canvas margin")
        if len({2 * obj.top + obj.size for obj in self.objects}) != 1:
            raise ValueError("bounding-box centers must be vertically aligned")
        ordered = sorted(self.objects, key=lambda obj: obj.left)
        for left, right in pairwise(ordered):
            if right.left - (left.left + left.size) < 1:
                raise ValueError("adjacent bounding boxes require at least one blank pixel")


@dataclass(frozen=True)
class RelationalColorQuestion:
    """Select the immediate spatial neighbor of a uniquely identified shape."""

    anchor_shape: str
    direction: str

    def __post_init__(self) -> None:
        if not isinstance(self.anchor_shape, str) or not isinstance(self.direction, str):
            raise TypeError("anchor_shape and direction must be strings")
        if self.anchor_shape not in SHAPES:
            raise ValueError(f"anchor_shape must be one of {SHAPES}")
        if self.direction not in ("left", "right"):
            raise ValueError("direction must be left or right")

    @property
    def text(self) -> str:
        return f"What color is the object immediately {self.direction} of the {self.anchor_shape}?"


def render_multi_object_scene(scene: MultiObjectScene) -> Tensor:
    """Return fresh CPU float32 [3,H,W] pixels without consuming randomness."""
    image = torch.zeros(3, scene.image_size, scene.image_size, dtype=torch.float32, device="cpu")
    for obj in scene.objects:
        image += render_scene(obj, image_size=scene.image_size)
    return image


def resolve_relation(scene: MultiObjectScene, question: RelationalColorQuestion) -> ShapeScene:
    """Resolve from metadata alone; no rendering or pixel-derived labels."""
    ordered = sorted(scene.objects, key=lambda obj: obj.left)
    anchors = [index for index, obj in enumerate(ordered) if obj.shape == question.anchor_shape]
    if not anchors:
        raise ValueError(f"missing anchor shape: {question.anchor_shape}")
    if len(anchors) != 1:
        raise ValueError(f"ambiguous anchor shape: {question.anchor_shape}")
    target = anchors[0] + (-1 if question.direction == "left" else 1)
    if not 0 <= target < len(ordered):
        raise ValueError(f"anchor has no neighbor to its {question.direction}")
    return ordered[target]


def answer_relational_question(scene: MultiObjectScene, question: RelationalColorQuestion) -> str:
    return resolve_relation(scene, question).color


@dataclass(frozen=True)
class RelationalExample:
    """Pixels and explicit supervision; scene/query metadata is never a model input."""

    image: Tensor
    question: str
    answer: str
    scene: MultiObjectScene
    query: RelationalColorQuestion


def make_relational_example(
    scene: MultiObjectScene, question: RelationalColorQuestion
) -> RelationalExample:
    """Validate the answer before rendering; each example owns its pixel storage."""
    answer = answer_relational_question(scene, question)
    return RelationalExample(
        render_multi_object_scene(scene), question.text, answer, scene, question
    )
