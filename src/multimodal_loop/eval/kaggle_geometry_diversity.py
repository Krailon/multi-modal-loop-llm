"""Fixed-budget Kaggle follow-up changing only training geometry diversity."""

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from multimodal_loop.data.relational_dataset import load_relational_manifest
from multimodal_loop.eval.geometry_reference import (
    REFERENCE_HASHES,
    audit_geometry_reference,
    stage_geometry_reference,
)
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.train.kaggle import (
    cuda_runtime,
    file_hash,
    repository_revision,
    run_logged_command,
)


@dataclass(frozen=True)
class GeometryDiversityRun:
    repo: Path
    root: Path
    reference: Path
    model: dict
    training: dict


def prepare_geometry_diversity(repo, root, source):
    repo, root, source = Path(repo).resolve(), Path(root).resolve(), Path(source).resolve()
    if root.exists():
        raise ValueError("geometry diversity requires a fresh run directory")
    for other in (repo, source):
        if root == other or root.is_relative_to(other) or other.is_relative_to(root):
            raise ValueError("source, checkout and outputs must use separate paths")
    revision, runtime = repository_revision(repo), cuda_runtime()
    reference = stage_geometry_reference(source, root / "staging")
    audit = audit_geometry_reference(reference)
    settings = audit["settings"]
    model = yaml.safe_load((repo / "configs" / "relational_baseline.yaml").read_text())
    if model != settings["model"]:
        raise ValueError("model differs from the direct-grounding reference")
    (root / "provenance").mkdir(parents=True, exist_ok=False)
    write_json(
        root / "provenance" / "protocol.json",
        {
            "kind": "milestone2_geometry_diversity_v1",
            "revision": revision,
            "runtime": runtime,
            "source": str(source),
            "source_archive_sha256": file_hash(source) if source.is_file() else None,
            "reference_hashes": REFERENCE_HASHES,
            "source_manifest_sha256": audit["manifest"].sha256,
            "model": model,
            "training": settings["training"],
            "budget": settings["budget"],
            "geometry_count": 128,
            "selection_seed": 0,
            "patch_size": 8,
            "qa_presentations": 92160,
            "initialization": "fresh weights, seed 0",
            "splits": ["train", "validation"],
            "shuffle_seeds": list(range(5)),
            "criteria": [
                {"name": g["name"], "threshold": g["threshold"]}
                for g in audit["summary"]["assessment"]["gates"]
            ],
        },
    )
    # Keep the comparison evidence, without bundling another copy of reference weights.
    write_json(root / "provenance" / "reference_summary.json", audit["summary"])
    write_json(root / "provenance" / "reference_controls.json", audit["controls"])
    return GeometryDiversityRun(repo, root, reference, model, settings["training"])


def compare_reference(reference, reference_controls, current, controls):
    """Compare the same held-out records; distinguish the two training corpora."""

    def scalars(m):
        return {
            "accuracy": m["accuracy"],
            "loss": m["loss"],
            "all_three_accuracy": m["all_three"]["accuracy"],
            "circle_square_pair_accuracy": m["circle_square_pair"]["accuracy"],
        }

    old, new = scalars(reference["splits"]["validation"]), scalars(current["splits"]["validation"])
    rows = [
        {"metric": k, "reference": old[k], "current": new[k], "difference": new[k] - old[k]}
        for k in old
    ]
    for name, old_m, new_m in (
        (
            "shape_accuracy",
            reference["splits"]["validation"]["breakdowns"]["shape"],
            current["splits"]["validation"]["breakdowns"]["shape"],
        ),
    ):
        a, b = ({r["shape"]: r["accuracy"] for r in m} for m in (old_m, new_m))
        rows.extend(
            {"metric": f"{name}:{k}", "reference": a[k], "current": b[k], "difference": b[k] - a[k]}
            for k in sorted(a)
        )
    a, b = reference_controls["results"]["accuracy_gaps"], controls["results"]["accuracy_gaps"]
    rows.extend(
        {"metric": k, "reference": a[k], "current": b[k], "difference": b[k] - a[k]} for k in a
    )
    return {
        "validation": rows,
        "training": {
            "reference_original": reference["splits"]["train"],
            "current_original": current["training_geometry_subsets"]["original"],
            "current_added": current["training_geometry_subsets"]["added"],
            "current_full": current["splits"]["train"],
        },
        "note": "Differences are current minus reference; lower loss is better. "
        "Training corpora differ; compare original-layout subsets separately.",
    }


def run_geometry_diversity(run):
    if any((run.root / name).exists() for name in ("data", "training", "diagnosis")):
        raise ValueError("use a fresh run directory; training resume is not supported")
    data = run.root / "data"
    run_logged_command(
        run.repo,
        run.root,
        "generate",
        [
            "scripts/generate_geometry_diversity.py",
            "--source-manifest",
            str(run.reference / "training" / "manifest.json"),
            "--output-dir",
            str(data),
            "--geometry-count",
            "128",
        ],
    )
    manifest = load_relational_manifest(data / "manifest.json")
    coverage = json.loads((data / "coverage.json").read_text())
    if (
        coverage["coordinates"]["selected"] != 204
        or coverage["coordinates"]["missing_features"]
        or coverage["patch_offsets"]["selected"] != 329
    ):
        raise ValueError("coverage does not meet the fixed protocol")
    run_logged_command(
        run.repo,
        run.root,
        "train",
        [
            "scripts/train_shape_grounding.py",
            "--manifest",
            str(data / "manifest.json"),
            "--geometry-source-manifest",
            str(data / "source_manifest.json"),
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
        raise ValueError("frozen evaluation changed the checkpoint")
    report = json.loads((run.root / "diagnosis" / "summary.json").read_text())
    expected = {
        "kind": "direct_shape_color_diagnostics",
        "format_version": 1,
        "smoke": False,
        "completed_steps": 2880,
        "examples_seen": 92160,
        "model": run.model,
        "training": run.training,
        "budget": {"max_steps": 2880, "evaluation_interval": 288},
        "manifest_sha256": manifest.sha256,
        "checkpoint_sha256": checkpoint_hash,
        "evaluation": {
            "splits": ["train", "validation"],
            "shuffle_seeds": list(range(5)),
            "recurrence_depth": 2,
            "batch_size": 32,
        },
    }
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("report disagrees with the fixed geometry-diversity protocol")
    if (
        set(report["splits"]) != {"train", "validation"}
        or report["splits"]["train"]["total"] != 55296
        or report["splits"]["validation"]["total"] != 1728
        or report["training_geometry_subsets"]["original"]["total"] != 6912
        or report["training_geometry_subsets"]["added"]["total"] != 48384
    ):
        raise ValueError("report has incomplete split/subset coverage")
    if report["corpus_derivation"] != json.loads((data / "derivation.json").read_text()):
        raise ValueError("report derivation provenance mismatch")
    presentations = json.loads((run.root / "training" / "presentations.json").read_text())
    if presentations != json.loads((data / "presentations.json").read_text())["derived"]:
        raise ValueError("training presentations disagree with the planned stream")
    controls = json.loads((run.root / "diagnosis" / "controls.json").read_text())
    comparison = compare_reference(
        json.loads((run.root / "provenance" / "reference_summary.json").read_text()),
        json.loads((run.root / "provenance" / "reference_controls.json").read_text()),
        report,
        controls,
    )
    write_json(run.root / "diagnosis" / "comparison.json", comparison)
    write_json(
        run.root / "provenance" / "final.json",
        {
            "manifest_sha256": manifest.sha256,
            "checkpoint_sha256": checkpoint_hash,
            "summary_sha256": file_hash(run.root / "diagnosis" / "summary.json"),
            "controls_sha256": file_hash(run.root / "diagnosis" / "controls.json"),
            "comparison_sha256": file_hash(run.root / "diagnosis" / "comparison.json"),
        },
    )
    return report


def archive_geometry_diversity(run):
    if not (run.root / "provenance" / "final.json").is_file():
        raise ValueError("complete generation, training and evaluation before archiving")
    destination = run.root.parent / f"{run.root.name}_artifacts.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ("data", "training", "diagnosis", "provenance", "logs"):
            for path in sorted((run.root / directory).rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=Path(run.root.name) / path.relative_to(run.root))
    return destination
