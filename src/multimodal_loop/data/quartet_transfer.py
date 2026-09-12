"""Complete balanced arrangements unseen by the quartet-fit checkpoint."""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations

from multimodal_loop.data.matched_size import MatchedSizeDataset, origin_key, scene_key
from multimodal_loop.data.quartet_fit import FIT_CONDITIONS
from multimodal_loop.data.synthetic_shapes import COLORS, SHAPES


@dataclass(frozen=True)
class TransferPopulation:
    name: str
    role: str
    records: tuple
    source_indices: tuple
    origins: tuple
    triangle_size: int

    @property
    def families(self):
        return [list(range(i, i + 4)) for i in range(0, len(self.records), 4)]


class QuartetTransferDataset(MatchedSizeDataset):
    def __init__(self, population):
        self.records = population.records
        self.split = population.name
        self.image_size = self.records[0].scene.image_size


def build_transfer_populations(fit):
    source = fit.source
    groups = defaultdict(dict)
    for family in source.derivation["families"]:
        indices = tuple(family["variant_indices"])
        scene = source.splits["train"][indices[0]].scene
        key = (origin_key(scene), next(o.size for o in scene.objects if o.shape == "triangle"))
        groups[key][tuple((o.shape, o.color) for o in scene.objects)] = indices
    identities = sorted(
        tuple(zip(shapes, colors, strict=True))
        for shapes in permutations(SHAPES)
        for colors in permutations(COLORS, 3)
    )
    learned = (tuple(map(tuple, fit.derivation["origins"])), fit.derivation["triangle_size"])
    complete = sorted(k for k, families in groups.items() if set(families) == set(identities))
    if learned not in complete:
        raise ValueError("learned arrangement is not complete")
    others = [k for k in complete if k[0] != learned[0]]
    if not others:
        raise ValueError("no unseen complete arrangements")
    reserved = {
        origin_key(r.scene) for split in ("validation", "test") for r in source.splits[split]
    }
    populations = []
    seen = set()
    for index, key in enumerate([learned, *others]):
        if key[0] in reserved:
            raise ValueError("transfer origins overlap reserved held-out origins")
        indices = tuple(i for identity in identities for i in groups[key][identity])
        records = tuple(source.splits["train"][i] for i in indices)
        keys = {scene_key(r.scene) for r in records}
        if len(keys) != 576 or seen.intersection(keys):
            raise ValueError("duplicate or incomplete population")
        seen.update(keys)
        if index == 0 and indices != tuple(fit.derivation["source_indices"]):
            raise ValueError("learned population order changed")
        populations.append(
            TransferPopulation(
                "learned" if index == 0 else f"transfer_{index:02d}",
                "reproduction_control" if index == 0 else "transfer",
                records,
                indices,
                key[0],
                key[1],
            )
        )
    return tuple(populations)


def transfer_manifest(fit, populations):
    data = {
        "kind": "quartet_transfer",
        "format_version": 1,
        "fit_manifest_sha256": fit.sha256,
        "source_manifest_sha256": fit.source.sha256,
        "selection": "all complete arrangements with origins different from learned arrangement",
        "order": "learned first; sorted origins/triangle size; sorted identities; size conditions",
        "conditions": list(FIT_CONDITIONS),
        "populations": [
            {
                "name": p.name,
                "role": p.role,
                "origins": p.origins,
                "triangle_size": p.triangle_size,
                "source_indices": p.source_indices,
                "families": p.families,
                "images": len(p.records),
                "questions": 3 * len(p.records),
            }
            for p in populations
        ],
    }
    content = json.dumps(data, indent=2, allow_nan=False) + "\n"
    return content, hashlib.sha256(content.encode()).hexdigest()
