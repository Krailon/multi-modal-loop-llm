"""Fixed four-arrangement training budget and final transfer diagnostics on Kaggle."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from multimodal_loop.data.geometry_diversity import presentation_summary
from multimodal_loop.data.multi_arrangement import (
    TRAIN_NAMES,
    TRANSFER_NAMES,
    derive_multi_arrangement,
)
from multimodal_loop.eval.kaggle_quartet_fit import archive_quartet_fit
from multimodal_loop.eval.multi_arrangement_reference import read_multi_reference
from multimodal_loop.eval.quartet_transfer_reference import read_transfer_reference
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.train.kaggle import (
    cuda_runtime,
    file_hash,
    repository_revision,
    run_logged_command,
)
from multimodal_loop.train.synthetic import SyntheticTrainingConfig


@dataclass(frozen=True)
class MultiArrangementRun:
    repo: Path
    root: Path
    transfer_source: Path
    manifest_sha256: str


def prepare_multi_arrangement(repo, root, fit_source, transfer_source):
    repo, root, fit_source, transfer_source = (
        Path(p).resolve() for p in (repo, root, fit_source, transfer_source)
    )
    if root.exists():
        raise ValueError("use a fresh run directory")
    if any(
        root == p or root.is_relative_to(p) or p.is_relative_to(root)
        for p in (repo, fit_source, transfer_source)
    ):
        raise ValueError("sources, checkout and output must use separate paths")
    revision, runtime = repository_revision(repo), cuda_runtime()
    _, fit, _, _, fit_hashes = read_transfer_reference(fit_source)
    reference, _, transfer_hashes = read_multi_reference(transfer_source, fit)
    manifest = derive_multi_arrangement(fit)
    if (
        tuple(manifest.derivation["training_arrangements"]) != TRAIN_NAMES
        or tuple(manifest.derivation["transfer_arrangements"]) != TRANSFER_NAMES
    ):
        raise ValueError("population assignment differs from fixed protocol")
    model = yaml.safe_load((repo / "configs/relational_baseline.yaml").read_text())
    if model != asdict(ModelConfig(vocab_size=17)):
        raise ValueError("model differs from fixed baseline")
    (root / "data").mkdir(parents=True)
    (root / "provenance").mkdir()
    (root / "data/manifest.json").write_text(manifest.content)
    write_json(root / "data/derivation.json", manifest.derivation)
    write_json(root / "data/presentations.json", presentation_summary(manifest, max_steps=8640))
    write_json(root / "provenance/reference_summary.json", reference)
    write_json(
        root / "provenance/protocol.json",
        {
            "kind": "milestone2_multi_arrangement_v1",
            "revision": revision,
            "runtime": runtime,
            "fit_reference_hashes": fit_hashes,
            "transfer_reference_hashes": transfer_hashes,
            "fit_archive_sha256": file_hash(fit_source) if fit_source.is_file() else None,
            "transfer_archive_sha256": file_hash(transfer_source)
            if transfer_source.is_file()
            else None,
            "manifest_sha256": manifest.sha256,
            "model": model,
            "training": asdict(SyntheticTrainingConfig()),
            "training_arrangements": list(TRAIN_NAMES),
            "transfer_arrangements": list(TRANSFER_NAMES),
            "budget": {"max_steps": 8640, "evaluation_interval": 216},
            "initialization": "fresh seed 0; reference weights never initialize training",
            "qa_presentations": 276480,
            "passes": 40,
            "training_questions": 6912,
            "transfer_questions": 6912,
            "training_criteria_per_arrangement": {
                "per_shape_accuracy": 0.99,
                "correct_family_fraction": 0.95,
            },
            "transfer_accuracy_gates": [],
            "interpretation": "exploratory; known scores; fourfold updates versus one arrangement",
        },
    )
    return MultiArrangementRun(repo, root, transfer_source, manifest.sha256)


def run_multi_arrangement(run):
    if any((run.root / name).exists() for name in ("training", "diagnosis")):
        raise ValueError("use a fresh run directory; resume is unsupported")
    run_logged_command(
        run.repo,
        run.root,
        "train",
        [
            "scripts/train_multi_arrangement.py",
            "--manifest",
            str(run.root / "data/manifest.json"),
            "--model-config",
            str(run.repo / "configs/relational_baseline.yaml"),
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
    run_logged_command(
        run.repo,
        run.root,
        "evaluate",
        [
            "scripts/evaluate_multi_arrangement.py",
            "--checkpoint",
            str(checkpoint),
            "--reference-source",
            str(run.transfer_source),
            "--output-dir",
            str(run.root / "diagnosis"),
            "--device",
            "cuda:0",
        ],
    )

    def read(p):
        return json.loads((run.root / p).read_text())

    report = read("diagnosis/summary.json")
    expected = {
        "kind": "multi_arrangement_quartet_diagnostics_v1",
        "smoke": False,
        "completed_steps": 8640,
        "examples_seen": 276480,
        "checkpoint_sha256": digest,
        "manifest_sha256": run.manifest_sha256,
        "model": asdict(ModelConfig(vocab_size=17)),
        "training": asdict(SyntheticTrainingConfig()),
        "budget": {"max_steps": 8640, "evaluation_interval": 216},
        "evaluation": {
            "training_arrangements": list(TRAIN_NAMES),
            "transfer_arrangements": list(TRANSFER_NAMES),
            "batch_size": 32,
            "recurrence_depth": 2,
        },
    }
    if any(report.get(k) != v for k, v in expected.items()) or file_hash(checkpoint) != digest:
        raise ValueError("evaluation disagrees with fixed protocol")
    history = read("training/metrics.json")
    if (
        [h["completed_steps"] for h in history] != list(range(0, 8641, 216))
        or any("training_fit" not in h or "validation" in h for h in history)
        or history[-1]["completed_passes"] != 40
    ):
        raise ValueError("monitoring disagrees with fixed training population or budget")
    for role in ("training_fit", "transfer"):
        if (
            report["aggregates"][role]["summary"]["total"] != 6912
            or report["aggregates"][role]["families"]["total"] != 576
        ):
            raise ValueError("incorrect aggregate population")
    if read("training/presentations.json") != read("data/presentations.json"):
        raise ValueError("actual exposures differ from planned exposures")
    paths = [
        "data/manifest.json",
        "data/derivation.json",
        "data/presentations.json",
        "training/last.pt",
        "training/settings.json",
        "training/manifest.json",
        "training/metrics.json",
        "training/presentations.json",
        "diagnosis/summary.json",
        "diagnosis/comparison.json",
        "diagnosis/reference_summary.json",
        "diagnosis/reference_transfer_aggregate.json",
        "diagnosis/predictions.jsonl",
        "diagnosis/inspection.html",
        "provenance/protocol.json",
        "provenance/reference_summary.json",
    ]
    paths.extend(p.relative_to(run.root).as_posix() for p in (run.root / "logs").glob("*.log"))
    write_json(run.root / "provenance/final.json", {p: file_hash(run.root / p) for p in paths})
    return report


def archive_multi_arrangement(run):
    """Reuse checked data/training/diagnosis/provenance/log packaging."""
    return archive_quartet_fit(run)
