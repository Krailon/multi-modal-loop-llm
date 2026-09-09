"""Kaggle orchestration for the fixed direct shape-grounding sanity experiment."""

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from multimodal_loop.eval.baseline_artifacts import (
    BASELINE_MANIFEST_SHA256,
    audit_baseline,
    stage_baseline,
)
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.train.kaggle import (
    cuda_runtime,
    file_hash,
    repository_revision,
    run_logged_command,
)


@dataclass(frozen=True)
class ShapeGroundingRun:
    repo: Path
    root: Path
    baseline: Path
    model: dict


def prepare_shape_grounding(repo, root, source):
    repo, root, source = Path(repo).resolve(), Path(root).resolve(), Path(source).resolve()
    if root.exists():
        raise ValueError("shape grounding requires a fresh run directory")
    for other in (repo, source):
        if root == other or root.is_relative_to(other) or other.is_relative_to(root):
            raise ValueError("source, checkout and outputs must use separate paths")
    revision = repository_revision(repo)
    runtime = cuda_runtime()
    baseline = stage_baseline(source, root / "staging")
    audit = audit_baseline(baseline)
    original = json.loads((baseline / "training" / "settings.json").read_text())
    model = yaml.safe_load((repo / "configs" / "relational_baseline.yaml").read_text())
    if model != original["model"]:
        raise ValueError("model configuration differs from the original baseline")
    provenance = root / "provenance"
    provenance.mkdir(parents=True, exist_ok=False)
    write_json(
        provenance / "protocol.json",
        {
            "kind": "milestone2_direct_shape_grounding_v1",
            "revision": revision,
            "runtime": runtime,
            "source": str(source),
            "source_archive_sha256": file_hash(source) if source.is_file() else None,
            "source_checkpoint_sha256": audit["checkpoint_sha256"],
            "manifest_sha256": BASELINE_MANIFEST_SHA256,
            "model": model,
            "training": original["training"],
            "max_steps": 2880,
            "evaluation_interval": 288,
            "qa_presentations": 92160,
            "question": "What color is the {shape}?",
            "shape_order": ["square", "circle", "triangle"],
            "initialization": "fresh, seed 0; no baseline weights loaded",
            "shuffle_seeds": list(range(5)),
            "splits": ["train", "validation"],
            "criteria": {
                "train_each_shape": 0.95,
                "validation_each_shape": 0.90,
                "blank_gap": 0.30,
                "shuffled_image_gap": 0.30,
                "shuffled_question_gap": 0.30,
            },
            "compute_note": "Matched updates/presentations; shorter sequences use less compute.",
        },
    )
    return ShapeGroundingRun(repo, root, baseline, model)


def run_shape_grounding(run):
    if (run.root / "training").exists() or (run.root / "diagnosis").exists():
        raise ValueError("use a fresh run directory; training resume is not supported")
    run_logged_command(
        run.repo,
        run.root,
        "train",
        [
            "scripts/train_shape_grounding.py",
            "--manifest",
            str(run.baseline / "data" / "manifest.json"),
            "--model-config",
            str(run.repo / "configs" / "relational_baseline.yaml"),
            "--output-dir",
            str(run.root / "training"),
            "--device",
            "cuda:0",
            "--max-steps",
            "2880",
            "--evaluation-interval",
            "288",
        ],
    )
    checkpoint = run.root / "training" / "last.pt"
    checkpoint_hash = file_hash(checkpoint)
    run_logged_command(
        run.repo,
        run.root,
        "evaluate",
        [
            "scripts/evaluate_shape_grounding.py",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(run.root / "diagnosis"),
            "--device",
            "cuda:0",
            "--shuffle-seeds",
            "0",
            "1",
            "2",
            "3",
            "4",
        ],
    )
    if file_hash(checkpoint) != checkpoint_hash:
        raise ValueError("evaluation changed the checkpoint")
    report = json.loads((run.root / "diagnosis" / "summary.json").read_text())
    expected = {
        "kind": "direct_shape_color_diagnostics",
        "format_version": 1,
        "smoke": False,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": BASELINE_MANIFEST_SHA256,
        "completed_steps": 2880,
        "examples_seen": 92160,
        "model": run.model,
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "seed": 0,
            "recurrence_depth": 2,
        },
        "budget": {"max_steps": 2880, "evaluation_interval": 288},
        "evaluation": {
            "splits": ["train", "validation"],
            "shuffle_seeds": list(range(5)),
            "recurrence_depth": 2,
            "batch_size": 32,
        },
    }
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("report disagrees with the fixed direct-grounding protocol")
    if set(report["splits"]) != {"train", "validation"} or any(
        report["splits"][s]["total"] != n for s, n in (("train", 6912), ("validation", 1728))
    ):
        raise ValueError("report has incomplete split coverage")
    write_json(
        run.root / "provenance" / "final.json",
        {
            "checkpoint_sha256": checkpoint_hash,
            "summary_sha256": file_hash(run.root / "diagnosis" / "summary.json"),
            "controls_sha256": file_hash(run.root / "diagnosis" / "controls.json"),
        },
    )
    return report


def archive_shape_grounding(run):
    if not (run.root / "provenance" / "final.json").is_file():
        raise ValueError("complete training and evaluation before archiving")
    destination = run.root.parent / f"{run.root.name}_artifacts.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ("training", "diagnosis", "provenance", "logs"):
            for path in sorted((run.root / directory).rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=Path(run.root.name) / path.relative_to(run.root))
    return destination
