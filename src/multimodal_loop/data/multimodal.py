"""Validated, manifest-backed datasets for the single-object color task."""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType

from torch.utils.data import Dataset

from multimodal_loop.data.synthetic_shapes import (
    COLORS,
    FORMAT_VERSION,
    QUESTION,
    ShapeScene,
    SyntheticExample,
    SyntheticShapesConfig,
    make_example,
)


@dataclass(frozen=True)
class SyntheticManifest:
    config: SyntheticShapesConfig
    splits: Mapping[str, tuple[ShapeScene, ...]]
    content: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def parse_synthetic_manifest(content: str) -> SyntheticManifest:
    """Validate stored records without regenerating splits or rendering images."""
    payload = json.loads(content)
    if not isinstance(payload, dict) or type(payload.get("format_version")) is not int:
        raise ValueError("manifest must have an integer format_version")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported manifest format_version")
    settings = payload.get("config")
    if not isinstance(settings, dict) or set(settings) != {
        field.name for field in fields(SyntheticShapesConfig)
    }:
        raise ValueError("manifest config must contain all SyntheticShapesConfig fields")
    if not isinstance(settings["object_sizes"], list):
        raise ValueError("manifest object_sizes must be a list")
    config = SyntheticShapesConfig(**{**settings, "object_sizes": tuple(settings["object_sizes"])})
    records = payload.get("splits")
    if not isinstance(records, dict) or set(records) != {"train", "validation", "test"}:
        raise ValueError("manifest must contain train, validation, and test splits")
    splits = {}
    seen_layouts = set()
    for split, count in (
        ("train", config.train_size),
        ("validation", config.validation_size),
        ("test", config.test_size),
    ):
        if not isinstance(records[split], list) or len(records[split]) != count:
            raise ValueError(f"manifest {split} count does not match config")
        scenes = []
        colors_by_layout = {}
        for record in records[split]:
            if not isinstance(record, dict) or set(record) != {"scene", "question", "answer"}:
                raise ValueError("manifest record requires scene, question, and answer")
            if not isinstance(record["scene"], dict):
                raise ValueError("manifest scene must be a dictionary")
            scene = ShapeScene(**record["scene"])
            if scene.size not in config.object_sizes or (
                scene.left + scene.size >= config.image_size
                or scene.top + scene.size >= config.image_size
            ):
                raise ValueError("manifest scene does not fit configured sizes and canvas")
            if record["question"] != QUESTION or record["answer"] != scene.color:
                raise ValueError("manifest question or answer disagrees with the color task")
            colors_by_layout.setdefault(scene.layout, Counter())[scene.color] += 1
            scenes.append(scene)
        if any(counts != Counter(COLORS) for counts in colors_by_layout.values()):
            raise ValueError("each layout must appear exactly once in all four colors")
        if seen_layouts.intersection(colors_by_layout):
            raise ValueError("manifest layouts must be disjoint across splits")
        seen_layouts.update(colors_by_layout)
        splits[split] = tuple(scenes)
    return SyntheticManifest(config, MappingProxyType(splits), content)


def load_synthetic_manifest(path: str | Path) -> SyntheticManifest:
    """Load the exact UTF-8 file contents for hashing and self-contained resume."""
    return parse_synthetic_manifest(Path(path).read_bytes().decode("utf-8"))


class SyntheticColorDataset(Dataset[SyntheticExample]):
    """Render each stored scene on demand, in manifest order, with fresh pixels."""

    def __init__(self, manifest: SyntheticManifest, split: str) -> None:
        if split not in manifest.splits:
            raise ValueError("split must be train, validation, or test")
        self.scenes = manifest.splits[split]
        self.image_size = manifest.config.image_size

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> SyntheticExample:
        return make_example(self.scenes[index], image_size=self.image_size)
