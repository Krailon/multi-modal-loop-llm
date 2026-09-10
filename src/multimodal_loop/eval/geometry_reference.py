"""Read-only identity checks for the completed direct shape-grounding reference."""

import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

import torch

from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.eval.baseline_artifacts import BASELINE_MANIFEST_SHA256
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig
from multimodal_loop.train.shape_grounding_checkpoint import _validate_progress, tokenizer_metadata
from multimodal_loop.train.synthetic import SyntheticTrainingConfig

REFERENCE_HASHES = {
    "training/last.pt": "529cd69edb893236d23ff0e9b704a15f5a0d8e3db0486d988f942f08f5ccb41d",
    "diagnosis/summary.json": "5d51dde2bfbfb6db984e37a0fddf58afa220cfa792dadfbf5913efc04afa844d",
    "diagnosis/controls.json": "0aa731371ca133a661a4daaaae10309a8107147574ad8f49271e2f2416d62515",
}
REFERENCE_REVISION = "e61402bacb14d2198a0b1a4a4180718457b3e784"


def stage_geometry_reference(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    name = "milestone2_shape_grounding"
    if source.is_dir():
        root = source if (source / "training" / "last.pt").is_file() else source / name
        if not (root / "training" / "last.pt").is_file():
            raise ValueError("select the completed shape-grounding reference directory")
        return root
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError("reference must be the shape-grounding ZIP or extracted directory")
    if destination.exists():
        raise ValueError("reference staging requires a fresh directory")
    with zipfile.ZipFile(source) as archive:
        seen = set()
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or not path.parts
                or path.parts[0] != name
                or stat.S_ISLNK(member.external_attr >> 16)
                or path in seen
            ):
                raise ValueError("unsafe or duplicate reference archive member")
            seen.add(path)
        if not all(PurePosixPath(name) / p in seen for p in REFERENCE_HASHES):
            raise ValueError("incomplete reference archive")
        destination.mkdir(parents=True)
        archive.extractall(destination)
    return destination / name


def audit_geometry_reference(root):
    """No model construction, RNG restoration or inference, including on test."""
    root = Path(root)
    for name, digest in REFERENCE_HASHES.items():
        if file_hash(root / name) != digest:
            raise ValueError(f"reference identity mismatch: {name}")
    payload = torch.load(root / "training" / "last.pt", map_location="cpu", weights_only=True)
    manifest = parse_relational_manifest(payload["manifest_content"])
    if (
        manifest.sha256 != BASELINE_MANIFEST_SHA256
        or (root / "training" / "manifest.json").read_bytes() != manifest.content.encode()
    ):
        raise ValueError("reference source manifest mismatch")
    settings = json.loads((root / "training" / "settings.json").read_text())
    if settings["revision"] != REFERENCE_REVISION or settings["dirty"] or settings["smoke"]:
        raise ValueError("reference revision or run kind mismatch")
    for a, b in (
        ("model", "config"),
        ("training", "training_config"),
        ("budget", "budget"),
        ("tokenizer", "tokenizer"),
    ):
        if settings[a] != payload[b]:
            raise ValueError(f"reference settings mismatch: {a}")
    if payload["tokenizer"] != tokenizer_metadata() or payload["completed_steps"] != 2880:
        raise ValueError("reference task or budget mismatch")
    _validate_progress(
        manifest,
        SyntheticTrainingConfig(**payload["training_config"]),
        ShapeGroundingConfig(**payload["budget"]),
        payload["history"],
    )
    if json.loads((root / "training" / "metrics.json").read_text()) != payload["history"]:
        raise ValueError("reference history mismatch")
    return {
        "settings": settings,
        "manifest": manifest,
        "summary": json.loads((root / "diagnosis" / "summary.json").read_text()),
        "controls": json.loads((root / "diagnosis" / "controls.json").read_text()),
    }
