"""Read-only audit and safe staging of the recorded Milestone 2 baseline."""

import json
import stat
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import yaml

from multimodal_loop.data.relational_corpus import RelationalCorpusConfig
from multimodal_loop.eval.relational_protocol import (
    assess_validation_gates,
    validate_control_report,
)
from multimodal_loop.train.kaggle import NotebookRun, file_hash, inspect_checkpoint
from multimodal_loop.train.synthetic import SyntheticTrainingConfig

BASELINE_CHECKPOINT_SHA256 = "d6e39b3aebc75907bb12f548320f4d77c657c88ee8330c4c2d08940b41cbbffb"
BASELINE_MANIFEST_SHA256 = "c5a9102eb619bfb134af48177bc1d1c2a083ce9b98d18a116be817b218c78b6c"
BASELINE_REVISION = "f0e1334a4ed582b1aba526eec80506fef2660392"


def stage_baseline(source: str | Path, destination: str | Path) -> Path:
    """Read an extracted directory, or extract a ZIP into a fresh staging tree.

    Validate every member before writing any files. Never follow ZIP symlinks or
    paths outside the single expected archive root. Source files remain untouched.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.is_dir():
        root = (
            source if (source / "training" / "last.pt").exists() else source / "milestone2_baseline"
        )
        if not (root / "training" / "last.pt").is_file():
            raise ValueError("Select the baseline archive or extracted baseline directory")
        return root
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError("Baseline source must be a ZIP file or extracted directory")
    if destination.exists():
        raise ValueError("Archive staging requires a fresh directory")
    with zipfile.ZipFile(source) as archive:
        seen = set()
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or not path.parts
                or path.parts[0] != "milestone2_baseline"
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError("Unsafe baseline archive member")
            if path in seen:
                raise ValueError("Duplicate baseline archive member")
            seen.add(path)
        if PurePosixPath("milestone2_baseline/training/last.pt") not in seen:
            raise ValueError("Baseline archive has no checkpoint")
        destination.mkdir(parents=True)
        archive.extractall(destination)
    return destination / "milestone2_baseline"


def audit_baseline(root: str | Path) -> dict:
    """Verify the original artifact identity before any model inference."""
    root = Path(root)
    checkpoint = root / "training" / "last.pt"
    if file_hash(checkpoint) != BASELINE_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint is not the recorded frozen Milestone 2 baseline")

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    protocol = read("provenance/protocol.json")
    settings = read("training/settings.json")
    report = read("validation/controls.json")
    acceptance = read("validation/acceptance.json")
    model = yaml.safe_load((root / "provenance" / "model.yaml").read_text())
    if (
        protocol["revision"] != BASELINE_REVISION
        or protocol["smoke"] is not False
        or protocol["target_epochs"] != 10
        or protocol["model"] != model
    ):
        raise ValueError("Baseline provenance disagrees with the recorded run")
    run = NotebookRun(
        Path.cwd(), root, model, RelationalCorpusConfig(), SyntheticTrainingConfig(), False
    )
    payload = inspect_checkpoint(run, checkpoint)
    expected_corpus = {**asdict(run.corpus), "object_sizes": list(run.corpus.object_sizes)}
    if protocol["corpus"] != expected_corpus or protocol["training"] != payload["training_config"]:
        raise ValueError("Baseline protocol settings disagree with checkpoint")
    if payload["completed_epochs"] != 10 or payload["completed_steps"] != 2880:
        raise ValueError("Baseline must be the epoch-10 checkpoint")
    manifest_hash = payload["manifest_sha256"]
    if manifest_hash != BASELINE_MANIFEST_SHA256:
        raise ValueError("Manifest is not the recorded baseline corpus")
    for name in ("training/manifest.json", "data/manifest.json"):
        if (root / name).read_bytes() != payload["manifest_content"].encode("utf-8"):
            raise ValueError("Baseline manifest copies disagree")
    if read("training/metrics.json") != payload["history"]:
        raise ValueError("Baseline history copies disagree")
    for name, expected in (
        ("model", payload["config"]),
        ("training", payload["training_config"]),
        ("manifest_sha256", manifest_hash),
        ("tokenizer", payload["tokenizer"]),
    ):
        if settings[name] != expected:
            raise ValueError("Baseline settings disagree with checkpoint")
    for name, expected in (
        ("manifest.sha256", manifest_hash),
        ("final_checkpoint.sha256", BASELINE_CHECKPOINT_SHA256),
    ):
        if (root / "provenance" / name).read_text().strip() != expected:
            raise ValueError("Baseline provenance hash disagrees with artifact")
    validate_control_report(
        report,
        checkpoint_hash=BASELINE_CHECKPOINT_SHA256,
        manifest_hash=manifest_hash,
        model=model,
        training=payload["training_config"],
        epochs=10,
        steps=2880,
        images=576,
        seeds=list(range(5)),
    )
    assessment = assess_validation_gates(report["results"])
    if (
        acceptance["checkpoint_sha256"] != BASELINE_CHECKPOINT_SHA256
        or acceptance["controls_sha256"] != file_hash(root / "validation" / "controls.json")
        or acceptance["gates"] != assessment["gates"]
        or acceptance["passed"] != assessment["passed"]
    ):
        raise ValueError("Baseline acceptance report disagrees with audited controls")
    final_validation = payload["history"][-1]["validation"]
    if final_validation != {key: report["results"]["correct"][key] for key in final_validation}:
        raise ValueError("Baseline validation history disagrees with controls")
    return {
        "checkpoint_sha256": BASELINE_CHECKPOINT_SHA256,
        "manifest_sha256": manifest_hash,
        "training_revision": BASELINE_REVISION,
        "controls": report,
        "training_history": payload["history"],
    }
