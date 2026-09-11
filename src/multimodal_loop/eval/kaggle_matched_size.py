"""Fixed-budget Kaggle training on the unique union of matched size variants."""

import json
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yaml

from multimodal_loop.data.geometry_diversity import presentation_summary
from multimodal_loop.data.matched_size import derive_matched_size
from multimodal_loop.eval.focused_diagnosis import validate_rows
from multimodal_loop.eval.matched_size import compare_matched, size_groups
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.eval.size_reference import (
    REFERENCE_HASHES,
    audit_size_reference,
    stage_size_reference,
)
from multimodal_loop.train.kaggle import (
    cuda_runtime,
    file_hash,
    repository_revision,
    run_logged_command,
)

INTERVENTION_SUMMARY_SHA256 = "f161c9dadac4d12b669ca9eeb2b6e12dd3527182e8deb27c99c5f52790616f04"


def read_intervention_reference(source):
    source = Path(source)
    if source.is_dir():
        root = (
            source
            if (source / "diagnosis" / "summary.json").is_file()
            else source / "milestone2_size_intervention"
        )
        content = (root / "diagnosis" / "summary.json").read_bytes()
    else:
        with zipfile.ZipFile(source) as archive:
            name = "milestone2_size_intervention/diagnosis/summary.json"
            if archive.namelist().count(name) != 1:
                raise ValueError("expected one intervention summary")
            content = archive.read(name)
    if sha256(content).hexdigest() != INTERVENTION_SUMMARY_SHA256:
        raise ValueError("intervention reference identity mismatch")
    return json.loads(content)


def reference_size_groups(reference, audit):
    result = {}
    for split in ("train", "validation"):
        path = reference / "diagnosis" / f"{split}_examples.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        validate_rows(rows, audit["manifest"], split, audit["summary"]["splits"][split])
        result[split] = {
            "predictions_sha256": file_hash(path),
            "groups": size_groups(rows, audit["manifest"].splits[split]),
        }
    return result


@dataclass(frozen=True)
class MatchedSizeRun:
    repo: Path
    root: Path
    reference: Path
    manifest_sha256: str
    model: dict
    training: dict


def prepare_matched_size(repo, root, source, intervention_source):
    repo, root, source, intervention_source = map(
        lambda p: Path(p).resolve(), (repo, root, source, intervention_source)
    )
    if root.exists():
        raise ValueError("use a fresh run directory")
    if any(
        root == p or root.is_relative_to(p) or p.is_relative_to(root)
        for p in (repo, source, intervention_source)
    ):
        raise ValueError("sources, checkout and output must use separate paths")
    revision, runtime = repository_revision(repo), cuda_runtime()
    intervention = read_intervention_reference(intervention_source)
    reference = stage_size_reference(source, root / "staging")
    audit = audit_size_reference(reference)
    manifest = derive_matched_size(audit["manifest"])
    expected = {
        "original_images": 18432,
        "additional_images": 21024,
        "total_images": 39456,
        "eligible_families": 7008,
    }
    if any(manifest.derivation[k] != v for k, v in expected.items()):
        raise ValueError("derived corpus disagrees with protocol counts")
    model = yaml.safe_load((repo / "configs" / "relational_baseline.yaml").read_text())
    if model != audit["settings"]["model"]:
        raise ValueError("model differs from reference")
    (root / "data").mkdir(parents=True)
    (root / "provenance").mkdir()
    (root / "data" / "manifest.json").write_text(manifest.content)
    write_json(root / "data" / "derivation.json", manifest.derivation)
    exposures = presentation_summary(manifest, max_steps=5760, batch_size=32, seed=0)
    write_json(root / "data" / "presentations.json", exposures)
    write_json(root / "provenance" / "reference_summary.json", audit["summary"])
    write_json(root / "provenance" / "reference_controls.json", audit["controls"])
    write_json(root / "provenance" / "reference_intervention.json", intervention)
    write_json(
        root / "provenance" / "reference_relative_size.json",
        reference_size_groups(reference, audit),
    )
    write_json(
        root / "provenance" / "protocol.json",
        {
            "kind": "milestone2_matched_size_training_v1",
            "revision": revision,
            "runtime": runtime,
            "reference_hashes": REFERENCE_HASHES,
            "intervention_summary_sha256": INTERVENTION_SUMMARY_SHA256,
            "source_archive_sha256": file_hash(source) if source.is_file() else None,
            "intervention_archive_sha256": file_hash(intervention_source)
            if intervention_source.is_file()
            else None,
            "source_manifest_sha256": manifest.source.sha256,
            "manifest_sha256": manifest.sha256,
            "model": model,
            "training": audit["settings"]["training"],
            "budget": {"max_steps": 5760, "evaluation_interval": 288},
            "qa_presentations": 184320,
            "initialization": "fresh seed 0",
            "corpus_counts": expected,
            "sampling": "uniform deterministic per-pass shuffle of unique union",
            "splits": ["train", "validation"],
            "shuffle_seeds": list(range(5)),
            "training_criteria_population": "full expanded corpus",
            "size_intervention": "unchanged 384 eligible validation images; edge contact allowed",
            "criteria": [
                {"name": g["name"], "threshold": g["threshold"]}
                for g in audit["summary"]["assessment"]["gates"]
            ],
        },
    )
    return MatchedSizeRun(
        repo, root, reference, manifest.sha256, model, audit["settings"]["training"]
    )


def run_matched_size(run):
    if any((run.root / n).exists() for n in ("training", "diagnosis")):
        raise ValueError("use a fresh run directory; resume is unsupported")
    run_logged_command(
        run.repo,
        run.root,
        "train",
        [
            "scripts/train_matched_size.py",
            "--manifest",
            str(run.root / "data" / "manifest.json"),
            "--model-config",
            str(run.repo / "configs" / "relational_baseline.yaml"),
            "--output-dir",
            str(run.root / "training"),
            "--device",
            "cuda:0",
            "--max-steps",
            "5760",
            "--evaluation-interval",
            "288",
        ],
    )
    checkpoint = run.root / "training" / "last.pt"
    digest = file_hash(checkpoint)
    run_logged_command(
        run.repo,
        run.root,
        "evaluate",
        [
            "scripts/evaluate_matched_size.py",
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

    def read(name):
        return json.loads((run.root / name).read_text())

    report = read("diagnosis/summary.json")
    expected = {
        "kind": "matched_size_diagnostics_v1",
        "smoke": False,
        "completed_steps": 5760,
        "examples_seen": 184320,
        "manifest_sha256": run.manifest_sha256,
        "checkpoint_sha256": digest,
        "model": run.model,
        "training": run.training,
        "budget": {"max_steps": 5760, "evaluation_interval": 288},
        "evaluation": {
            "splits": ["train", "validation"],
            "shuffle_seeds": list(range(5)),
            "recurrence_depth": 2,
            "batch_size": 32,
        },
    }
    if file_hash(checkpoint) != digest or any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("evaluation disagrees with fixed protocol")
    if (
        set(report["splits"]) != {"train", "validation"}
        or report["splits"]["train"]["total"] != 118368
        or report["splits"]["validation"]["total"] != 1728
        or report["training_subsets"]["original"]["total"] != 55296
        or report["training_subsets"]["added"]["total"] != 63072
    ):
        raise ValueError("incomplete split/subset coverage")
    if read("training/presentations.json") != read("data/presentations.json"):
        raise ValueError("actual exposures disagree with plan")
    controls = read("diagnosis/controls.json")
    intervention = read("diagnosis/size_intervention.json")
    if controls["checkpoint_sha256"] != digest or intervention["checkpoint_sha256"] != digest:
        raise ValueError("diagnostic checkpoint identity mismatch")
    comparison = compare_matched(
        read("provenance/reference_summary.json"),
        read("provenance/reference_controls.json"),
        report,
        controls,
        read("provenance/reference_intervention.json"),
        intervention,
    )
    comparison["relative_size"] = {
        "reference": read("provenance/reference_relative_size.json"),
        "current_original_training": report["relative_size_training_subsets"]["original"],
        "current_validation": report["relative_size"]["validation"],
    }
    write_json(run.root / "diagnosis" / "comparison.json", comparison)
    paths = [
        "data/manifest.json",
        "data/derivation.json",
        "data/presentations.json",
        "training/last.pt",
        "diagnosis/summary.json",
        "diagnosis/controls.json",
        "diagnosis/comparison.json",
        "diagnosis/train_examples.jsonl",
        "diagnosis/validation_examples.jsonl",
        "diagnosis/size_intervention.json",
        "diagnosis/size_intervention_predictions.jsonl",
    ]
    write_json(run.root / "provenance" / "final.json", {p: file_hash(run.root / p) for p in paths})
    return report


def archive_matched_size(run):
    if not (run.root / "provenance" / "final.json").exists():
        raise ValueError("complete training and evaluation before archiving")
    destination = run.root.parent / f"{run.root.name}_artifacts.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ("data", "training", "diagnosis", "provenance", "logs"):
            for path in sorted((run.root / directory).rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=Path(run.root.name) / path.relative_to(run.root))
    return destination
