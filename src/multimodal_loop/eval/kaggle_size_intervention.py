"""Audited, evaluation-only Kaggle workflow for the fixed size intervention."""

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from multimodal_loop.data.size_intervention import CONDITIONS, build_size_cases, case_metadata
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.eval.size_intervention import (
    compare_original,
    evaluate_cases,
    intervention_preview,
)
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
from multimodal_loop.train.shape_grounding_checkpoint import load_shape_checkpoint

EXPECTED_GEOMETRIES = {
    "validation:g000": 48,
    "validation:g001": 144,
    "validation:g002": 48,
    "validation:g003": 144,
}


def checked_cases(manifest):
    cases, eligibility = build_size_cases(manifest)
    if (
        eligibility["eligible_by_geometry"] != EXPECTED_GEOMETRIES
        or eligibility["eligible_images"] != 384
    ):
        raise ValueError("eligibility disagrees with the fixed size protocol")
    return cases, eligibility


@dataclass(frozen=True)
class SizeInterventionRun:
    repo: Path
    root: Path
    reference: Path


def prepare_size_intervention(repo, root, source):
    repo, root, source = Path(repo).resolve(), Path(root).resolve(), Path(source).resolve()
    if root.exists():
        raise ValueError("use a fresh run directory")
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in (repo, source)):
        raise ValueError("source, checkout and output paths must be separate")
    revision, runtime = repository_revision(repo), cuda_runtime()
    reference = stage_size_reference(source, root / "staging")
    audit = audit_size_reference(reference)
    cases, eligibility = checked_cases(audit["manifest"])
    (root / "provenance").mkdir(parents=True)
    (root / "data").mkdir()
    shutil.copyfile(reference / "data" / "manifest.json", root / "data" / "manifest.json")
    write_json(root / "data" / "eligibility.json", eligibility)
    write_json(root / "data" / "cases.json", [case_metadata(c) for c in cases])
    write_json(root / "provenance" / "reference_summary.json", audit["summary"])
    write_json(
        root / "provenance" / "protocol.json",
        {
            "kind": "milestone2_size_intervention_v1",
            "revision": revision,
            "runtime": runtime,
            "source": str(source),
            "source_archive_sha256": file_hash(source) if source.is_file() else None,
            "reference_hashes": REFERENCE_HASHES,
            "conditions": CONDITIONS,
            "split": "validation",
            "eligible_images": 384,
            "qa_evaluations": 5760,
            "recurrence_depth": 2,
            "batch_size": 32,
            "dtype": "float32",
            "edge_contact": "allowed; all pixels visible",
            "origin": "fixed",
            "alignment": "vertical center alignment relaxed",
            "horizontal_gap_pixels": 1,
            "training": "none",
            "model": audit["settings"]["model"],
        },
    )
    return SizeInterventionRun(repo, root, reference)


def evaluate_reference(reference, destination, *, device="cpu"):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("use a fresh evaluation directory")
    reference = Path(reference)
    audit = audit_size_reference(reference)
    checkpoint = reference / "training" / "last.pt"
    digest = file_hash(checkpoint)
    cases, eligibility = checked_cases(audit["manifest"])
    model, manifest, training, payload = load_shape_checkpoint(checkpoint, device=device)
    if manifest.content != audit["manifest"].content or training.recurrence_depth != 2:
        raise ValueError("checkpoint differs from the protocol")
    metrics, rows = evaluate_cases(model, cases, batch_size=32, recurrence_depth=2)
    agreement = compare_original(rows, audit["validation_rows"])
    if file_hash(checkpoint) != digest:
        raise ValueError("frozen evaluation changed checkpoint")
    report = {
        "kind": "size_intervention_v1",
        "checkpoint_sha256": digest,
        "manifest_sha256": manifest.sha256,
        "source_completed_steps": payload["completed_steps"],
        "split": "validation",
        "recurrence_depth": 2,
        "batch_size": 32,
        "qa_evaluations": len(rows),
        "eligibility": eligibility,
        "metrics": metrics,
        "original_reference_agreement": agreement,
    }
    destination.mkdir(parents=True)
    write_json(destination / "summary.json", report)
    with (destination / "predictions.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    (destination / "inspection.html").write_text(
        intervention_preview(cases, rows), encoding="utf-8"
    )
    return report


def run_size_intervention(run):
    if (run.root / "diagnosis").exists():
        raise ValueError("use a fresh run directory")
    checkpoint = run.reference / "training" / "last.pt"
    digest = file_hash(checkpoint)
    run_logged_command(
        run.repo,
        run.root,
        "evaluate",
        [
            "scripts/evaluate_size_intervention.py",
            "--reference-dir",
            str(run.reference),
            "--output-dir",
            str(run.root / "diagnosis"),
            "--device",
            "cuda:0",
        ],
    )
    report = json.loads((run.root / "diagnosis" / "summary.json").read_text())
    if (
        report["checkpoint_sha256"] != digest
        or file_hash(checkpoint) != digest
        or report["qa_evaluations"] != 5760
        or report["split"] != "validation"
        or report["source_completed_steps"] != 5760
        or report["recurrence_depth"] != 2
        or report["batch_size"] != 32
        or report["eligibility"] != json.loads((run.root / "data" / "eligibility.json").read_text())
        or report["manifest_sha256"] != file_hash(run.root / "data" / "manifest.json")
    ):
        raise ValueError("evaluation disagrees with protocol")
    write_json(
        run.root / "provenance" / "final.json",
        {
            "checkpoint_sha256": digest,
            **{
                name: file_hash(run.root / name)
                for name in (
                    "data/manifest.json",
                    "data/eligibility.json",
                    "data/cases.json",
                    "diagnosis/summary.json",
                    "diagnosis/predictions.jsonl",
                    "diagnosis/inspection.html",
                )
            },
        },
    )
    return report


def archive_size_intervention(run):
    if not (run.root / "provenance" / "final.json").exists():
        raise ValueError("complete evaluation before archiving")
    destination = run.root.parent / f"{run.root.name}_artifacts.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ("data", "diagnosis", "provenance", "logs"):
            for path in sorted((run.root / directory).rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=Path(run.root.name) / path.relative_to(run.root))
    return destination
