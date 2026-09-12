"""Fixed alternating arrangement assignment with complete balanced quartets."""

import hashlib
import json
from dataclasses import dataclass, replace
from types import MappingProxyType

from multimodal_loop.data.matched_size import MatchedSizeDataset
from multimodal_loop.data.quartet_fit import parse_quartet_fit
from multimodal_loop.data.quartet_transfer import build_transfer_populations, transfer_manifest

FIT_SHA256 = "86bdb774cef97e6b76752f016edd2b73473e8c989b77a7e2cedcb4e295894180"
TRAIN_NAMES = ("learned", "transfer_02", "transfer_04", "transfer_06")
TRANSFER_NAMES = ("transfer_01", "transfer_03", "transfer_05", "transfer_07")


@dataclass(frozen=True)
class MultiArrangementManifest:
    source: object
    populations: tuple
    splits: object
    derivation: dict
    content: str

    @property
    def config(self):
        return self.source.config

    @property
    def sha256(self):
        return hashlib.sha256(self.content.encode()).hexdigest()


def derive_multi_arrangement(fit):
    originals = build_transfer_populations(fit)
    if len(originals) % 2:
        raise ValueError("require an even number of complete arrangements")
    populations = tuple(
        replace(p, role="training_fit" if i % 2 == 0 else "transfer")
        for i, p in enumerate(originals)
    )
    splits = {
        role: tuple(r for p in populations if p.role == role for r in p.records)
        for role in ("training_fit", "transfer")
    }
    splits["train"] = splits["training_fit"]
    content, _ = transfer_manifest(fit, populations)
    derivation = {
        "assignment": "alternating coordinate-sorted entries; even indices train, odd transfer",
        "population_metadata": json.loads(content),
        "training_arrangements": [p.name for p in populations if p.role == "training_fit"],
        "transfer_arrangements": [p.name for p in populations if p.role == "transfer"],
        "training_questions": 3 * len(splits["train"]),
        "transfer_questions": 3 * len(splits["transfer"]),
        "images_per_arrangement": 576,
        "families_per_arrangement": 144,
        "balance_per_arrangement": fit.derivation["balance"],
    }
    content = (
        json.dumps(
            {
                "kind": "multi_arrangement_quartet_fit",
                "format_version": 1,
                "source_fit_manifest_content": fit.content,
                "derivation": derivation,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return MultiArrangementManifest(fit, populations, MappingProxyType(splits), derivation, content)


def parse_multi_arrangement(content):
    data = json.loads(content)
    if (
        not isinstance(data, dict)
        or data.get("kind") != "multi_arrangement_quartet_fit"
        or type(data.get("format_version")) is not int
        or data["format_version"] != 1
    ):
        raise ValueError("incompatible multi-arrangement manifest")
    fit = parse_quartet_fit(data["source_fit_manifest_content"])
    expected = derive_multi_arrangement(fit)
    if data != json.loads(expected.content):
        raise ValueError("multi-arrangement manifest disagrees with derivation")
    return replace(expected, content=content)


class MultiArrangementDataset(MatchedSizeDataset):
    """Only training records are exposed through the training dataset."""

    def __init__(self, manifest):
        self.records = manifest.splits["training_fit"]
        self.split = "training_fit"
        self.image_size = manifest.config.image_size
