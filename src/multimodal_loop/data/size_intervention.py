"""Fixed-origin size variants for frozen diagnostics, separate from corpus scenes."""

from collections import Counter
from dataclasses import asdict, dataclass, replace

import torch
from torch.utils.data import Dataset

from multimodal_loop.data.geometry_diversity import geometry_of
from multimodal_loop.data.synthetic_shapes import SHAPES, ShapeScene, render_scene

CONDITIONS = {
    "original": None,
    "circle6_square6": (6, 6),
    "circle6_square8": (6, 8),
    "circle8_square6": (8, 6),
    "circle8_square8": (8, 8),
}


@dataclass(frozen=True)
class DiagnosticScene:
    """Fully visible boxes with horizontal gaps; edge contact is allowed."""

    objects: tuple[ShapeScene, ...]
    image_size: int = 32

    def __post_init__(self):
        if len(self.objects) != 3 or {o.shape for o in self.objects} != set(SHAPES):
            raise ValueError("require one of each shape")
        if len({o.color for o in self.objects}) != 3:
            raise ValueError("require distinct colors")
        if any(
            o.left + o.size > self.image_size or o.top + o.size > self.image_size
            for o in self.objects
        ):
            raise ValueError("outside_canvas")
        ordered = sorted(self.objects, key=lambda o: o.left)
        if any(b.left - (a.left + a.size) < 1 for a, b in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("insufficient_horizontal_gap")


def variant_scene(scene, sizes=None):
    objects = tuple(
        replace(o, size=sizes[0] if o.shape == "circle" else sizes[1])
        if sizes is not None and o.shape != "triangle"
        else o
        for o in scene.objects
    )
    return DiagnosticScene(objects, scene.image_size)


def render_diagnostic_scene(scene):
    # The existing primitive requires a margin. Padding permits edge contact;
    # validation above guarantees the crop removes no object pixels.
    padded = torch.zeros(
        3, scene.image_size + 2, scene.image_size + 2, dtype=torch.float32, device="cpu"
    )
    for obj in scene.objects:
        padded += render_scene(obj, image_size=scene.image_size + 2)
    return padded[:, : scene.image_size, : scene.image_size].clone()


@dataclass(frozen=True)
class InterventionCase:
    source_index: int
    geometry_id: str
    condition: str
    scene: DiagnosticScene


def build_size_cases(manifest):
    """Select on validity of all four variants before any inference."""
    records = manifest.splits["validation"]
    ids = {
        g: f"validation:g{i:03d}" for i, g in enumerate(sorted({geometry_of(r) for r in records}))
    }
    cases, eligibility = [], []
    for index, record in enumerate(records):
        variants, failures = {}, []
        for name, sizes in CONDITIONS.items():
            try:
                variants[name] = variant_scene(record.scene, sizes)
            except ValueError as error:
                failures.append({"condition": name, "reason": str(error)})
        geometry = ids[geometry_of(record)]
        eligibility.append(
            {
                "source_index": index,
                "geometry_id": geometry,
                "eligible": not failures,
                "failures": failures,
            }
        )
        if not failures:
            cases.extend(
                InterventionCase(index, geometry, name, scene) for name, scene in variants.items()
            )
    counts = Counter(r["geometry_id"] for r in eligibility if r["eligible"])
    return cases, {
        "source_images": len(records),
        "eligible_images": sum(counts.values()),
        "eligible_by_geometry": dict(sorted(counts.items())),
        "records": eligibility,
    }


@dataclass(frozen=True)
class InterventionExample:
    image: torch.Tensor
    question: str
    answer: str
    scene: DiagnosticScene
    shape: str


class SizeInterventionDataset(Dataset):
    def __init__(self, cases, condition):
        if condition not in CONDITIONS:
            raise ValueError("unknown condition")
        self.cases = [case for case in cases if case.condition == condition]
        if not self.cases:
            raise ValueError("no eligible images")

    def __len__(self):
        return 3 * len(self.cases)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        case = self.cases[index // 3]
        shape = SHAPES[index % 3]
        target = next(o for o in case.scene.objects if o.shape == shape)
        return InterventionExample(
            render_diagnostic_scene(case.scene),
            f"What color is the {shape}?",
            target.color,
            case.scene,
            shape,
        )


def case_metadata(case):
    return {
        **asdict(case),
        "edge_contact": any(
            o.left + o.size == case.scene.image_size or o.top + o.size == case.scene.image_size
            for o in case.scene.objects
        ),
    }
