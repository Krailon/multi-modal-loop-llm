"""Finite geometry-separated corpus of balanced relational color questions."""

import random
from dataclasses import dataclass
from itertools import permutations, product
from math import comb

from multimodal_loop.data.relational_shapes import (
    MultiObjectScene,
    RelationalColorQuestion,
    answer_relational_question,
)
from multimodal_loop.data.synthetic_shapes import COLORS, SHAPES, ShapeScene, _integer, _size

FORMAT_VERSION = 1
CORPUS_KIND = "relational_color_rows"
Geometry = tuple[tuple[int, int, int], ...]  # Spatially ordered (left, top, size).


@dataclass(frozen=True)
class RelationalCorpusConfig:
    """Counts refer to geometries: each expands to 144 images and 576 QA pairs."""

    image_size: int = 32
    object_sizes: tuple[int, ...] = (6, 8)
    seed: int = 0
    train_geometry_count: int = 16
    validation_geometry_count: int = 4
    test_geometry_count: int = 4

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
                raise ValueError("object sizes must fit with a one-pixel canvas margin")
        if len(set(self.object_sizes)) != len(self.object_sizes):
            raise ValueError("object_sizes must be distinct")
        for name in ("train_geometry_count", "validation_geometry_count", "test_geometry_count"):
            _integer(name, getattr(self, name), 1)
        requested = (
            self.train_geometry_count + self.validation_geometry_count + self.test_geometry_count
        )
        if requested > self.capacity:
            raise ValueError(f"requested geometries exceed catalog capacity of {self.capacity}")

    @property
    def capacity(self) -> int:
        """Count placements without materializing the finite geometry catalog."""
        total = 0
        for sizes in product(self.object_sizes, repeat=3):
            # Four positive gaps: two canvas margins and two internal gaps.
            slack = self.image_size - sum(sizes) - 4
            if slack >= 0:
                total += comb(slack + 3, 3) * (self.image_size - max(sizes) - 1)
        return total


@dataclass(frozen=True)
class RelationalQA:
    query: RelationalColorQuestion
    question: str
    answer: str


@dataclass(frozen=True)
class RelationalSceneRecord:
    """One image's scene and four questions; all metadata, not model inputs."""

    scene: MultiObjectScene
    questions: tuple[RelationalQA, ...]


def _geometry_catalog(config: RelationalCorpusConfig) -> list[Geometry]:
    catalog = []
    for sizes in product(config.object_sizes, repeat=3):
        slack = config.image_size - sum(sizes) - 4
        if slack < 0:
            continue
        half_height = max(sizes) // 2
        for center in range(half_height + 1, config.image_size - half_height):
            for left_margin in range(1, slack + 2):
                for first_gap in range(1, slack + 3 - left_margin):
                    for second_gap in range(1, slack + 4 - left_margin - first_gap):
                        positions = (
                            left_margin,
                            left_margin + sizes[0] + first_gap,
                            left_margin + sizes[0] + first_gap + sizes[1] + second_gap,
                        )
                        catalog.append(
                            tuple(
                                (left, center - size // 2, size)
                                for left, size in zip(positions, sizes, strict=True)
                            )
                        )
    return catalog


def build_relational_splits(
    config: RelationalCorpusConfig,
) -> dict[str, tuple[RelationalSceneRecord, ...]]:
    """Allocate geometries before expanding all shape/color/question variants.

    Enumeration order is size triples, vertical center, left margin, first gap,
    second gap. Every coordinate loop ascends; size order follows the config.
    Only local RNGs are used. Stored records have canonical spatial object order.
    """
    geometries = _geometry_catalog(config)
    random.Random(config.seed).shuffle(geometries)
    splits = {}
    start = 0
    for split, count in (
        ("train", config.train_geometry_count),
        ("validation", config.validation_geometry_count),
        ("test", config.test_geometry_count),
    ):
        records = []
        for geometry in geometries[start : start + count]:
            for shapes in permutations(SHAPES):
                for colors in permutations(COLORS, 3):
                    scene = MultiObjectScene(
                        tuple(
                            ShapeScene(shape, color, left, top, size)
                            for shape, color, (left, top, size) in zip(
                                shapes, colors, geometry, strict=True
                            )
                        ),
                        image_size=config.image_size,
                    )
                    questions = []
                    for index, obj in enumerate(scene.objects):
                        for direction, target in (("left", index - 1), ("right", index + 1)):
                            if 0 <= target < 3:
                                query = RelationalColorQuestion(obj.shape, direction)
                                questions.append(
                                    RelationalQA(
                                        query, query.text, answer_relational_question(scene, query)
                                    )
                                )
                    records.append(RelationalSceneRecord(scene, tuple(questions)))
        random.Random(f"{config.seed}:{split}").shuffle(records)
        splits[split] = tuple(records)
        start += count
    return splits
