"""Kaggle orchestration for the original frozen baseline's diagnostic report."""

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from multimodal_loop.eval.baseline_artifacts import audit_baseline, stage_baseline
from multimodal_loop.train.kaggle import (
    cuda_runtime,
    file_hash,
    repository_revision,
    run_logged_command,
)


@dataclass(frozen=True)
class DiagnosticRun:
    repo: Path
    root: Path
    baseline: Path
    checkpoint_hash: str


def prepare_diagnostics(repo: str | Path, root: str | Path, source: str | Path) -> DiagnosticRun:
    repo, root, source = Path(repo).resolve(), Path(root).resolve(), Path(source).resolve()
    if root.exists():
        raise ValueError("Diagnostic run directory already exists; choose a fresh directory")
    for other in (repo, source):
        if root == other or root.is_relative_to(other) or other.is_relative_to(root):
            raise ValueError("Diagnostics, source artifacts and checkout must use separate paths")
    if not source.exists():
        raise ValueError("Set BASELINE_SOURCE to the attached baseline ZIP or directory")
    revision = repository_revision(repo)
    runtime = cuda_runtime()
    baseline = stage_baseline(source, root / "staging")
    audit = audit_baseline(baseline)
    provenance = {
        "source": str(source),
        "source_archive_sha256": file_hash(source) if source.is_file() else None,
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "manifest_sha256": audit["manifest_sha256"],
        "training_revision": audit["training_revision"],
        "diagnostic_revision": revision,
        "runtime": runtime,
    }
    output = root / "provenance"
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    return DiagnosticRun(repo, root, baseline, audit["checkpoint_sha256"])


def run_diagnostics(run: DiagnosticRun) -> dict:
    destination = run.root / "diagnosis"
    if destination.exists():
        raise ValueError("Diagnostic outputs already exist; choose a fresh run directory")
    run_logged_command(
        run.repo,
        run.root,
        "diagnose",
        [
            "scripts/diagnose_relational.py",
            "--baseline-dir",
            str(run.baseline),
            "--output-dir",
            str(destination),
            "--device",
            "cuda:0",
        ],
    )
    if file_hash(run.baseline / "training" / "last.pt") != run.checkpoint_hash:
        raise ValueError("Source checkpoint changed during diagnosis")
    summary = json.loads((destination / "summary.json").read_text())
    if (
        summary["checkpoint_sha256"] != run.checkpoint_hash
        or summary["evaluation"]
        != {"splits": ["train", "validation"], "recurrence_depth": 2, "batch_size": 32}
        or set(summary["splits"]) != {"train", "validation"}
    ):
        raise ValueError("Diagnostic report disagrees with requested frozen run")
    for split, total in (("train", 9216), ("validation", 2304)):
        if summary["splits"][split]["total"] != total:
            raise ValueError("Diagnostic report has incomplete split coverage")
    return summary


def diagnostic_archive(run: DiagnosticRun) -> Path:
    """Archive new reports/logs/provenance, leaving staged baseline files separate."""
    if not (run.root / "diagnosis" / "summary.json").is_file():
        raise ValueError("Complete the diagnosis before packaging artifacts")
    destination = run.root.parent / f"{run.root.name}_artifacts.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ("diagnosis", "provenance", "logs"):
            for path in sorted((run.root / directory).rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=Path(run.root.name) / path.relative_to(run.root))
    return destination
