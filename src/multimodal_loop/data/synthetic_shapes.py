"""Deterministic, single-object color questions with disjoint layout splits."""

import random
from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor

SHAPES = ("square", "circle", "triangle")
COLORS = ("red", "green", "blue", "yellow")
RGB = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0))
QUESTION = "What color is the object?"
FORMAT_VERSION = 1


def _integer(name: str, value: int, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _size(size: int) -> None:
    _integer("object size", size, 4)
    if size % 2:
        raise ValueError("object size must be even")


@dataclass(frozen=True)
class SyntheticShapesConfig:
    """Finite corpus settings; split sizes count images, not layouts.

    Every selected layout appears in all four colors in exactly one split.
    Generation uses local Python RNGs and is reproducible for the same config
    and generator/software version. Changing split sizes can change membership.
    """

    image_size: int = 32
    object_sizes: tuple[int, ...] = (8, 12, 16)
    seed: int = 0
    train_size: int = 1024
    validation_size: int = 256
    test_size: int = 256

    def __post_init__(self) -> None:
        _integer("image_size", self.image_size, 6)
        _integer("seed", self.seed)
        if not isinstance(self.object_sizes, tuple):
            raise TypeError("object_sizes must be a tuple")
        if not self.object_sizes:
            raise ValueError("object_sizes must not be empty")
        for size in self.object_sizes:
            _size(size)
            if size > self.image_size - 2:
                raise ValueError("object sizes must fit with a one-pixel margin")
        if len(set(self.object_sizes)) != len(self.object_sizes):
            raise ValueError("object_sizes must be distinct")
        for name in ("train_size", "validation_size", "test_size"):
            value = getattr(self, name)
            _integer(name, value, 4)
            if value % len(COLORS):
                raise ValueError(f"{name} must be divisible by four")
        if self.train_size + self.validation_size + self.test_size > self.capacity:
            raise ValueError(f"requested splits exceed catalog capacity of {self.capacity} images")

    @property
    def capacity(self) -> int:
        return (
            len(COLORS)
            * len(SHAPES)
            * sum((self.image_size - size - 1) ** 2 for size in self.object_sizes)
        )


@dataclass(frozen=True)
class ShapeScene:
    """Supervision/inspection metadata, never a model input.

    Left/top locate the bounding box. Squares fill it; circles are inscribed;
    upright triangles have a top-center apex and a bottom-edge base.
    """

    shape: str
    color: str
    left: int
    top: int
    size: int

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        if self.color not in COLORS:
            raise ValueError(f"color must be one of {COLORS}")
        _integer("left", self.left, 1)
        _integer("top", self.top, 1)
        _size(self.size)

    @property
    def layout(self) -> tuple[str, int, int, int]:
        """Color-independent split identity: shape, size, left, top."""
        return self.shape, self.size, self.left, self.top


@dataclass(frozen=True)
class SyntheticExample:
    image: Tensor
    question: str
    answer: str
    scene: ShapeScene


def build_scene_splits(config: SyntheticShapesConfig) -> dict[str, tuple[ShapeScene, ...]]:
    """Return train/validation/test scenes without replacement across layouts.

    Catalog order is shape, configured size order, top, then left. Layouts are
    shuffled with Random(seed). Per-split ordering uses Random(f'{seed}:{split}').
    All categories are eligible for every split; none are reserved as OOD classes.
    Colors are exactly balanced conditional on each selected layout.
    """
    layouts = [
        (shape, size, left, top)
        for shape in SHAPES
        for size in config.object_sizes
        for top, left in product(range(1, config.image_size - size), repeat=2)
    ]
    random.Random(config.seed).shuffle(layouts)
    splits = {}
    start = 0
    for split, count in (
        ("train", config.train_size),
        ("validation", config.validation_size),
        ("test", config.test_size),
    ):
        end = start + count // len(COLORS)
        scenes = [
            ShapeScene(shape=shape, color=color, left=left, top=top, size=size)
            for shape, size, left, top in layouts[start:end]
            for color in COLORS
        ]
        random.Random(f"{config.seed}:{split}").shuffle(scenes)
        splits[split] = tuple(scenes)
        start = end
    return splits


def render_scene(scene: ShapeScene, *, image_size: int = 32) -> Tensor:
    """Return fresh CPU float32 [3,H,W] pixels in [0,1], with hard edges.

    Inclusion is tested at pixel centers using integer arithmetic, with no
    antialiasing. Even sizes place circle centers/triangle apexes between pixels.
    A one-pixel black canvas margin is required on every side.
    """
    _integer("image_size", image_size, 6)
    if scene.left + scene.size >= image_size or scene.top + scene.size >= image_size:
        raise ValueError("scene must fit with a one-pixel margin")
    coordinates = torch.arange(scene.size, device="cpu")
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    if scene.shape == "square":
        mask = torch.ones(scene.size, scene.size, dtype=torch.bool, device="cpu")
    elif scene.shape == "circle":
        mask = (2 * x + 1 - scene.size) ** 2 + (2 * y + 1 - scene.size) ** 2 <= scene.size**2
    else:
        mask = 2 * torch.abs(2 * x + 1 - scene.size) <= 2 * y + 1
    rgb = torch.tensor(RGB[COLORS.index(scene.color)], dtype=torch.float32, device="cpu")
    image = torch.zeros(3, image_size, image_size, dtype=torch.float32, device="cpu")
    image[:, scene.top : scene.top + scene.size, scene.left : scene.left + scene.size] = (
        rgb[:, None, None] * mask
    )
    return image


def make_example(scene: ShapeScene, *, image_size: int = 32) -> SyntheticExample:
    """Render an independent image with the fixed question and one-word answer."""
    return SyntheticExample(
        render_scene(scene, image_size=image_size), QUESTION, scene.color, scene
    )
