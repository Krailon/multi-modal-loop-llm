"""Validated relational manifests and question-wise, on-demand datasets."""

import hashlib
import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass, fields
from itertools import permutations, product
from pathlib import Path
from types import MappingProxyType

from torch.utils.data import Dataset

from multimodal_loop.data.relational_corpus import (
    CORPUS_KIND,
    FORMAT_VERSION,
    RelationalCorpusConfig,
    RelationalQA,
    RelationalSceneRecord,
)
from multimodal_loop.data.relational_shapes import (
    MultiObjectScene,
    RelationalColorQuestion,
    RelationalExample,
    answer_relational_question,
    render_multi_object_scene,
)
from multimodal_loop.data.synthetic_shapes import COLORS, SHAPES, ShapeScene


@dataclass(frozen=True)
class RelationalManifest:
    config: RelationalCorpusConfig
    splits: Mapping[str, tuple[RelationalSceneRecord, ...]]
    content: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


def _fields(value, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def _read_record(record: dict, config: RelationalCorpusConfig) -> RelationalSceneRecord:
    _fields(record, {"scene", "questions"}, "record")
    scene_data = record["scene"]
    _fields(scene_data, {"objects", "image_size"}, "scene")
    if not isinstance(scene_data["objects"], list):
        raise ValueError("scene objects must be a list")
    objects = []
    for obj in scene_data["objects"]:
        _fields(obj, {f.name for f in fields(ShapeScene)}, "object")
        objects.append(ShapeScene(**obj))
    scene = MultiObjectScene(tuple(objects), image_size=scene_data["image_size"])
    if scene.image_size != config.image_size:
        raise ValueError("scene image_size must match manifest config")
    if any(obj.size not in config.object_sizes for obj in objects):
        raise ValueError("scene object sizes must match manifest config")
    if objects != sorted(objects, key=lambda obj: obj.left):
        raise ValueError("scene objects must be stored in spatial order")
    if {obj.shape for obj in objects} != set(SHAPES):
        raise ValueError("corpus scenes require one of each shape")
    if len({obj.color for obj in objects}) != 3:
        raise ValueError("corpus scenes require three distinct colors")
    question_data = record["questions"]
    if not isinstance(question_data, list) or len(question_data) != 4:
        raise ValueError("each record must contain four questions")
    expected_queries = {
        (obj.shape, direction)
        for index, obj in enumerate(objects)
        for direction, target in (("left", index - 1), ("right", index + 1))
        if 0 <= target < 3
    }
    seen = set()
    questions = []
    for qa in question_data:
        _fields(qa, {"query", "question", "answer"}, "question record")
        _fields(qa["query"], {"anchor_shape", "direction"}, "query")
        query = RelationalColorQuestion(**qa["query"])
        identity = (query.anchor_shape, query.direction)
        if identity in seen or identity not in expected_queries:
            raise ValueError("record must contain each valid query exactly once")
        seen.add(identity)
        if qa["question"] != query.text:
            raise ValueError("question text must match its query exactly")
        if qa["answer"] != answer_relational_question(scene, query):
            raise ValueError("answer must match the resolved relation")
        questions.append(RelationalQA(query, qa["question"], qa["answer"]))
    return RelationalSceneRecord(scene, tuple(questions))


def parse_relational_manifest(content: str) -> RelationalManifest:
    """Validate stored metadata without rendering or regenerating the corpus.

    All records and question order remain as stored. Completeness is checked
    per geometry, making all six question texts exactly color-balanced.
    """
    payload = json.loads(content)
    _fields(payload, {"kind", "format_version", "config", "software", "splits"}, "manifest")
    if (
        payload["kind"] != CORPUS_KIND
        or type(payload["format_version"]) is not int
        or payload["format_version"] != FORMAT_VERSION
    ):
        raise ValueError("unsupported relational manifest kind or format_version")
    settings = payload["config"]
    _fields(settings, {f.name for f in fields(RelationalCorpusConfig)}, "config")
    if not isinstance(settings["object_sizes"], list):
        raise ValueError("object_sizes must be a list")
    config = RelationalCorpusConfig(**{**settings, "object_sizes": tuple(settings["object_sizes"])})
    software = payload["software"]
    if not isinstance(software, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in software.items()
    ):
        raise ValueError("software metadata must contain strings")
    _fields(payload["splits"], {"train", "validation", "test"}, "splits")
    expected_variants = set(product(permutations(SHAPES), permutations(COLORS, 3)))
    seen_geometries = set()
    splits = {}
    for split in ("train", "validation", "test"):
        count = getattr(config, f"{split}_geometry_count")
        records = payload["splits"][split]
        if not isinstance(records, list) or len(records) != count * 144:
            raise ValueError(f"{split} record count must match config")
        parsed = []
        variants = {}
        for record in records:
            item = _read_record(record, config)
            objects = item.scene.objects
            geometry = tuple((obj.left, obj.top, obj.size) for obj in objects)
            variant = (tuple(obj.shape for obj in objects), tuple(obj.color for obj in objects))
            group = variants.setdefault(geometry, set())
            if variant in group:
                raise ValueError("duplicate shape/color variant within a geometry")
            group.add(variant)
            parsed.append(item)
        if len(variants) != count or any(group != expected_variants for group in variants.values()):
            raise ValueError("each geometry must contain the complete shape/color expansion")
        if seen_geometries.intersection(variants):
            raise ValueError("geometries must be disjoint across splits")
        seen_geometries.update(variants)
        splits[split] = tuple(parsed)
    return RelationalManifest(config, MappingProxyType(splits), content)


def load_relational_manifest(path: str | Path) -> RelationalManifest:
    """Preserve exact UTF-8 bytes, including newlines, in the content and hash."""
    return parse_relational_manifest(Path(path).read_bytes().decode("utf-8"))


class RelationalColorDataset(Dataset[RelationalExample]):
    """Flatten stored images then their questions; render fresh pixels per item."""

    def __init__(self, manifest: RelationalManifest, split: str) -> None:
        if split not in manifest.splits:
            raise ValueError("split must be train, validation, or test")
        self.records = manifest.splits[split]
        self.image_size = manifest.config.image_size

    def __len__(self) -> int:
        return len(self.records) * 4

    def __getitem__(self, index: int) -> RelationalExample:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("relational dataset index out of range")
        record_index, question_index = divmod(index, 4)
        record = self.records[record_index]
        qa = record.questions[question_index]
        return RelationalExample(
            render_multi_object_scene(record.scene), qa.question, qa.answer, record.scene, qa.query
        )
