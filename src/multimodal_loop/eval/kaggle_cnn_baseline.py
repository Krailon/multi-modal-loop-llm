"""One fixed CNN run on Kaggle, with audited source and output artifacts."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from multimodal_loop.data.geometry_diversity import presentation_summary
from multimodal_loop.eval.cnn_reference import read_cnn_reference
from multimodal_loop.eval.kaggle_quartet_fit import archive_quartet_fit
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.model.cnn_baseline import CNNConfig, forward_macs
from multimodal_loop.train.cnn_baseline import CNNTrainingConfig
from multimodal_loop.train.cnn_checkpoint import load_cnn_checkpoint
from multimodal_loop.train.kaggle import (
    cuda_runtime,
    file_hash,
    repository_revision,
    run_logged_command,
)


@dataclass(frozen=True)
class CNNRun:
    repo: Path
    root: Path
    source: Path
    manifest_sha256: str


def prepare_cnn_baseline(repo, root, source):
    repo, root, source = (Path(p).resolve() for p in (repo, root, source))
    if root.exists():
        raise ValueError("use a fresh run directory")
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in (repo, source)):
        raise ValueError("sources, checkout and output must use separate paths")
    revision, runtime = repository_revision(repo), cuda_runtime()
    manifest, reference, _, hashes = read_cnn_reference(source)
    config = CNNConfig(**yaml.safe_load((repo / "configs/cnn_baseline.yaml").read_text()))
    if config != CNNConfig():
        raise ValueError("model differs from fixed CNN baseline")
    (root / "data").mkdir(parents=True)
    (root / "provenance").mkdir()
    (root / "data/manifest.json").write_text(manifest.content)
    write_json(root / "data/derivation.json", manifest.derivation)
    write_json(root / "data/presentations.json", presentation_summary(manifest, max_steps=8640))
    write_json(root / "provenance/reference_summary.json", reference)
    write_json(
        root / "provenance/protocol.json",
        {
            "kind": "milestone2_cnn_baseline_v1",
            "revision": revision,
            "runtime": runtime,
            "reference_hashes": hashes,
            "reference_archive_sha256": file_hash(source) if source.is_file() else None,
            "manifest_sha256": manifest.sha256,
            "model": asdict(config),
            "training": asdict(CNNTrainingConfig()),
            "budget": {"max_steps": 8640, "evaluation_interval": 216},
            "parameters": 145329,
            "forward_macs_per_qa": forward_macs(config),
            "training_arrangements": manifest.derivation["training_arrangements"],
            "transfer_arrangements": manifest.derivation["transfer_arrangements"],
            "qa_presentations": 276480,
            "presentations_per_qa": 40,
            "monitoring_qa_presentations": 283392,
            "final_evaluation_qa_presentations": 13824,
            "training_criteria_per_arrangement": {
                "per_shape_accuracy": 0.99,
                "correct_family_fraction": 0.95,
            },
            "transfer_accuracy_gates": [],
            "recurrence_depth": None,
            "interpretation": (
                "exploratory single-seed model-family control; equal exposure, unequal compute"
            ),
        },
    )
    return CNNRun(repo, root, source, manifest.sha256)


def run_cnn_baseline(run):
    if any((run.root / p).exists() for p in ("training", "diagnosis")):
        raise ValueError("use a fresh run directory; resume is unsupported")
    run_logged_command(
        run.repo,
        run.root,
        "train",
        [
            "scripts/train_cnn_baseline.py",
            "--manifest",
            str(run.root / "data/manifest.json"),
            "--model-config",
            str(run.repo / "configs/cnn_baseline.yaml"),
            "--output-dir",
            str(run.root / "training"),
            "--device",
            "cuda:0",
            "--max-steps",
            "8640",
            "--evaluation-interval",
            "216",
        ],
    )
    checkpoint = run.root / "training/last.pt"
    digest = file_hash(checkpoint)
    _, manifest, _, payload = load_cnn_checkpoint(checkpoint)
    if (
        payload["smoke"]
        or manifest.sha256 != run.manifest_sha256
        or payload["completed_steps"] != 8640
    ):
        raise ValueError("training disagrees with fixed protocol")
    run_logged_command(
        run.repo,
        run.root,
        "evaluate",
        [
            "scripts/evaluate_cnn_baseline.py",
            "--checkpoint",
            str(checkpoint),
            "--reference-source",
            str(run.source),
            "--output-dir",
            str(run.root / "diagnosis"),
            "--device",
            "cuda:0",
        ],
    )
    report = json.loads((run.root / "diagnosis/summary.json").read_text())
    if (
        report["checkpoint_sha256"] != digest
        or file_hash(checkpoint) != digest
        or report["manifest_sha256"] != run.manifest_sha256
        or report["smoke"]
        or report["completed_steps"] != 8640
        or report["examples_seen"] != 276480
    ):
        raise ValueError("evaluation disagrees with fixed protocol")
    if json.loads((run.root / "training/presentations.json").read_text()) != json.loads(
        (run.root / "data/presentations.json").read_text()
    ):
        raise ValueError("actual exposures differ from planned exposures")
    for role in ("training_fit", "transfer"):
        if (
            report["aggregates"][role]["summary"]["total"] != 6912
            or report["aggregates"][role]["families"]["total"] != 576
        ):
            raise ValueError("incorrect aggregate population")
    paths = sorted(
        p
        for directory in ("data", "training", "diagnosis", "provenance", "logs")
        for p in (run.root / directory).rglob("*")
        if p.is_file()
    )
    write_json(
        run.root / "provenance/final.json",
        {p.relative_to(run.root).as_posix(): file_hash(p) for p in paths},
    )
    return report


def archive_cnn_baseline(run):
    return archive_quartet_fit(run)
