"""One balanced training arrangement, selected without inspecting predictions."""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from types import MappingProxyType

from multimodal_loop.data.matched_size import (
    MatchedSizeDataset,
    origin_key,
    parse_matched_size,
    scene_key,
)
from multimodal_loop.data.size_intervention import CONDITIONS
from multimodal_loop.data.synthetic_shapes import COLORS, SHAPES

SOURCE_SHA256 = "25071dc462e1e8bdadee82ca7b0531cdde871c4f9735613532f41210c9192828"
FIT_CONDITIONS = tuple(k for k in CONDITIONS if k != "original")


@dataclass(frozen=True)
class QuartetFitManifest:
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


def derive_quartet_fit(source):
    groups = defaultdict(dict)
    for family in source.derivation["families"]:
        indices = family["variant_indices"]
        scene = source.splits["train"][indices[0]].scene
        triangle = next(o for o in scene.objects if o.shape == "triangle")
        identity = tuple((o.shape, o.color) for o in scene.objects)
        group = (origin_key(scene), triangle.size)
        groups[group][identity] = indices
    identities = {
        tuple(zip(shapes, colors, strict=True))
        for shapes in permutations(SHAPES)
        for colors in permutations(COLORS, 3)
    }
    eligible = sorted(k for k, families in groups.items() if set(families) == identities)
    if not eligible:
        raise ValueError("no complete balanced arrangement")
    selected = eligible[0]
    reserved = {
        origin_key(r.scene) for split in ("validation", "test") for r in source.splits[split]
    }
    if selected[0] in reserved:
        raise ValueError("training origins overlap held-out origins")
    indices, families = [], []
    for identity in sorted(identities):
        start = len(indices)
        indices.extend(groups[selected][identity])
        families.append(list(range(start, start + 4)))
    records = tuple(source.splits["train"][i] for i in indices)
    if len({scene_key(r.scene) for r in records}) != 576:
        raise ValueError("quartet selection must contain 576 unique images")
    for r in records:
        if any(
            min(o.left, o.top) < 1
            or o.left + o.size >= r.scene.image_size
            or o.top + o.size >= r.scene.image_size
            for o in r.scene.objects
        ):
            raise ValueError("quartet selection requires blank margins")
    balance = {
        "shape_orders": dict(
            sorted(Counter(",".join(o.shape for o in r.scene.objects) for r in records).items())
        ),
        "color_assignments": dict(
            sorted(Counter(",".join(o.color for o in r.scene.objects) for r in records).items())
        ),
        "conditions": dict.fromkeys(FIT_CONDITIONS, len(families)),
        "queried_shapes": dict.fromkeys(SHAPES, len(records)),
        "answer_colors": dict(
            sorted(Counter(o.color for r in records for o in r.scene.objects).items())
        ),
    }
    derivation = {
        "source_manifest_sha256": source.sha256,
        "origins": [list(p) for p in selected[0]],
        "triangle_size": selected[1],
        "selection": "lexicographically first complete origin/triangle-size group",
        "order": "sorted shape/color identities, then fixed size condition order",
        "source_indices": indices,
        "families": families,
        "conditions": list(FIT_CONDITIONS),
        "balance": balance,
        "images": len(records),
        "questions": 3 * len(records),
    }
    content = (
        json.dumps(
            {
                "kind": "quartet_training_fit",
                "format_version": 1,
                "source_manifest_content": source.content,
                "derivation": derivation,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return QuartetFitManifest(
        source, MappingProxyType({"train": records, "training_fit": records}), derivation, content
    )


def parse_quartet_fit(content):
    data = json.loads(content)
    if (
        not isinstance(data, dict)
        or data.get("kind") != "quartet_training_fit"
        or type(data.get("format_version")) is not int
        or data["format_version"] != 1
    ):
        raise ValueError("incompatible quartet-fit manifest")
    source = parse_matched_size(data["source_manifest_content"])
    expected = derive_quartet_fit(source)
    if data != json.loads(expected.content):
        raise ValueError("quartet-fit manifest disagrees with derivation")
    return QuartetFitManifest(source, expected.splits, expected.derivation, content)


class QuartetFitDataset(MatchedSizeDataset):
    def __init__(self, manifest):
        self.records = manifest.splits["training_fit"]
        self.image_size = manifest.config.image_size
        self.split = "training_fit"
