"""Read-only identity checks for the completed geometry-diversity reference."""

import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

import torch

from multimodal_loop.data.geometry_diversity import validate_geometry_derivation
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig
from multimodal_loop.train.shape_grounding_checkpoint import _validate_progress, tokenizer_metadata
from multimodal_loop.train.synthetic import SyntheticTrainingConfig

REFERENCE_HASHES = {
    "training/last.pt": "49012c1c488478c7eb66e27b979b59c22a47a92d696120ad9279718afbd71456",
    "diagnosis/summary.json": "5351a33268a02aa743a138c4849ba90c12ee4edd3a39e5479dbc4ef851528d88",
    "diagnosis/controls.json": "bd0a025992218418eec1a1ef23ebd8363f38165f53cb20a2aed91de67c4e872e",
    "data/manifest.json": "047ac5a844760a08ceb9d89dc7f1c5ed73ef157c85de86e2b58337e7b1e98975",
    "data/source_manifest.json": "c5a9102eb619bfb134af48177bc1d1c2a083ce9b98d18a116be817b218c78b6c",
    "data/derivation.json": "0b15adab6028909082cd46b010ef1ea4c8af225cc11492a9978c65eb8703cb05",
    "data/coverage.json": "7761419a57f3522cd5d2c72d3d76941f5d556aaa914bcf9e2b1ebb0f9d01b138",
    "training/settings.json": "8ffad0c77a117960116d3d4604fd4a858f56a43691f63eef16aec028d727e72b",
    "training/metrics.json": "81b36d3e5e411851fac5f11533cf502d7b7efaf1c2eb1f88d89f9b6ad100ae67",
}
REFERENCE_REVISION = "54010afe4d636ae85cff3c60117e5cebbc77e74f"
REFERENCE_MANIFEST = "047ac5a844760a08ceb9d89dc7f1c5ed73ef157c85de86e2b58337e7b1e98975"


def stage_budget_reference(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    name = "milestone2_geometry_diversity"
    if source.is_dir():
        root = source if (source / "training" / "last.pt").is_file() else source / name
        if not (root / "training" / "last.pt").is_file():
            raise ValueError("select the completed geometry-diversity reference directory")
        return root
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError("reference must be the geometry-diversity ZIP or extracted directory")
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


def audit_budget_reference(root):
    """No model construction, RNG restoration or inference, including on test."""
    root = Path(root)
    for name, digest in REFERENCE_HASHES.items():
        if file_hash(root / name) != digest:
            raise ValueError(f"reference identity mismatch: {name}")
    payload = torch.load(root / "training" / "last.pt", map_location="cpu", weights_only=True)
    manifest = parse_relational_manifest(payload["manifest_content"])
    if (
        manifest.sha256 != REFERENCE_MANIFEST
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
    derivation = json.loads((root / "data" / "derivation.json").read_text())
    validate_geometry_derivation(manifest, derivation)
    if payload["corpus_derivation"] != derivation or settings["corpus_derivation"] != derivation:
        raise ValueError("reference derivation mismatch")
    if payload["examples_seen"] != 92160 or payload["budget"] != {
        "max_steps": 2880,
        "evaluation_interval": 288,
    }:
        raise ValueError("reference progress mismatch")
    return {
        "settings": settings,
        "manifest": manifest,
        "summary": json.loads((root / "diagnosis" / "summary.json").read_text()),
        "controls": json.loads((root / "diagnosis" / "controls.json").read_text()),
    }
