"""Unique union of original training scenes and complete fixed-origin size families."""

import hashlib
import json
from dataclasses import asdict, dataclass
from types import MappingProxyType

from torch.utils.data import Dataset

from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.data.size_intervention import (
    CONDITIONS,
    DiagnosticScene,
    InterventionExample,
    render_diagnostic_scene,
    variant_scene,
)
from multimodal_loop.data.synthetic_shapes import SHAPES


@dataclass(frozen=True)
class MatchedRecord:
    scene: DiagnosticScene


@dataclass(frozen=True)
class MatchedSizeManifest:
    source: object
    splits: object
    derivation: dict
    content: str

    @property
    def config(self):
        return self.source.config

    @property
    def sha256(self):
        return hashlib.sha256(self.content.encode()).hexdigest()


def scene_key(scene):
    return tuple((o.shape, o.color, o.left, o.top, o.size) for o in scene.objects)


def origin_key(scene):
    return tuple((o.left, o.top) for o in scene.objects)


def derive_matched_size(source):
    """Preserve originals and add canonical, unique variants; never select held-out scenes."""
    reserved = {
        origin_key(r.scene) for split in ("validation", "test") for r in source.splits[split]
    }
    originals = [variant_scene(r.scene) for r in source.splits["train"]]
    if any(origin_key(scene) in reserved for scene in originals):
        raise ValueError("training origins overlap held-out origins")
    original_keys = [scene_key(scene) for scene in originals]
    if len(set(original_keys)) != len(original_keys):
        raise ValueError("duplicate original scenes")
    unique = dict(zip(original_keys, originals, strict=True))
    families, eligibility = [], []
    for index, scene in enumerate(originals):
        variants, failures = [], []
        for name, sizes in CONDITIONS.items():
            if sizes is None:
                continue
            try:
                variant = variant_scene(scene, sizes)
                if any(
                    o.left + o.size >= scene.image_size or o.top + o.size >= scene.image_size
                    for o in variant.objects
                ):
                    raise ValueError("one_pixel_margin")
                variants.append(variant)
            except ValueError as error:
                failures.append({"condition": name, "reason": str(error)})
        eligibility.append({"source_index": index, "eligible": not failures, "failures": failures})
        if not failures:
            families.append((index, [scene_key(s) for s in variants]))
            unique.update((scene_key(s), s) for s in variants)
    keys = original_keys + sorted(set(unique) - set(original_keys))
    indices = {key: i for i, key in enumerate(keys)}
    if any(origin_key(unique[key]) in reserved for key in keys):
        raise ValueError("generated training origins overlap held-out origins")
    derivation = {
        "version": 1,
        "source_manifest_sha256": source.sha256,
        "original_images": len(originals),
        "additional_images": len(keys) - len(originals),
        "total_images": len(keys),
        "eligible_families": len(families),
        "conditions": [name for name in CONDITIONS if name != "original"],
        "eligibility": eligibility,
        "families": [
            {"source_index": i, "variant_indices": [indices[k] for k in group]}
            for i, group in families
        ],
        "policy": {
            "fixed_origins": True,
            "canvas_margin": 1,
            "horizontal_gap": 1,
            "vertical_alignment": "relaxed",
            "order": "originals then sorted full scene keys",
        },
    }
    train = tuple(MatchedRecord(unique[key]) for key in keys)
    payload = {
        "kind": "matched_size_color",
        "format_version": 1,
        "source_manifest_content": source.content,
        "derivation": derivation,
        "train": [asdict(r) for r in train],
    }
    content = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    return MatchedSizeManifest(
        source, MappingProxyType({**source.splits, "train": train}), derivation, content
    )


def parse_matched_size(content):
    data = json.loads(content)
    if (
        not isinstance(data, dict)
        or data.get("kind") != "matched_size_color"
        or type(data.get("format_version")) is not int
        or data["format_version"] != 1
    ):
        raise ValueError("incompatible matched-size manifest")
    source = parse_relational_manifest(data["source_manifest_content"])
    expected = derive_matched_size(source)
    if data != json.loads(expected.content):
        raise ValueError("matched-size corpus disagrees with deterministic derivation")
    # Preserve supplied bytes for checkpoint/hash identity, as with relational manifests.
    return MatchedSizeManifest(source, expected.splits, expected.derivation, content)


class MatchedSizeDataset(Dataset):
    def __init__(self, manifest, split):
        if split not in ("train", "validation"):
            raise ValueError("matched-size training/evaluation excludes test")
        self.records = manifest.splits[split]
        self.image_size = manifest.config.image_size
        self.split = split

    def __len__(self):
        return 3 * len(self.records)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        scene = variant_scene(self.records[index // 3].scene)
        shape = SHAPES[index % 3]
        target = next(o for o in scene.objects if o.shape == shape)
        return InterventionExample(
            render_diagnostic_scene(scene),
            f"What color is the {shape}?",
            target.color,
            scene,
            shape,
        )
