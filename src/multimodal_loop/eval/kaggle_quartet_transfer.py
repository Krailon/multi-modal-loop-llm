"""Single frozen-evaluation Kaggle workflow; no training entry point."""

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from multimodal_loop.data.quartet_transfer import build_transfer_populations, transfer_manifest
from multimodal_loop.eval.quartet_transfer_reference import (
    REFERENCE_HASHES,
    read_transfer_reference,
)
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.train.kaggle import (
    cuda_runtime,
    file_hash,
    repository_revision,
    run_logged_command,
)


@dataclass(frozen=True)
class QuartetTransferRun:
    repo: Path
    root: Path
    source: Path
    manifest_sha256: str


def prepare_quartet_transfer(repo, root, source):
    repo, root, source = (Path(p).resolve() for p in (repo, root, source))
    if root.exists():
        raise ValueError("use a fresh run directory")
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in (repo, source)):
        raise ValueError("checkout, source and output must use separate paths")
    revision, runtime = repository_revision(repo), cuda_runtime()
    _, fit, _, _, hashes = read_transfer_reference(source)
    populations = build_transfer_populations(fit)
    if len(populations) != 8:
        raise ValueError("research requires eight complete arrangements")
    content, digest = transfer_manifest(fit, populations)
    (root / "data").mkdir(parents=True)
    (root / "provenance").mkdir()
    (root / "data/transfer_manifest.json").write_text(content)
    write_json(
        root / "provenance/protocol.json",
        {
            "kind": "milestone2_quartet_transfer_v1",
            "revision": revision,
            "runtime": runtime,
            "reference_hashes": hashes,
            "transfer_manifest_sha256": digest,
            "source_archive_sha256": file_hash(source) if source.is_file() else None,
            "model_updates": 0,
            "recurrence_depth": 2,
            "batch_size": 32,
            "dtype": "float32",
            "total_questions": 13824,
            "transfer_questions": 12096,
            "reproduction_questions": 1728,
            "transfer_arrangements": 7,
            "accuracy_gates": [],
            "reproduction_check": "all 1728 prediction IDs match reference",
            "reserved_splits_evaluated": [],
        },
    )
    return QuartetTransferRun(repo, root, source, digest)


def run_quartet_transfer(run):
    if (run.root / "diagnosis").exists():
        raise ValueError("use a fresh evaluation directory")
    run_logged_command(
        run.repo,
        run.root,
        "evaluate",
        [
            "scripts/evaluate_quartet_transfer.py",
            "--source",
            str(run.source),
            "--output-dir",
            str(run.root / "diagnosis"),
            "--device",
            "cuda:0",
        ],
    )
    report = json.loads((run.root / "diagnosis/summary.json").read_text())
    if (
        report["kind"] != "quartet_transfer_diagnostics_v1"
        or report["smoke"]
        or report["checkpoint_sha256"] != REFERENCE_HASHES["training/last.pt"]
        or report["transfer_manifest_sha256"] != run.manifest_sha256
        or report["evaluation"]["total_questions"] != 13824
        or report["evaluation"]["transfer_questions"] != 12096
        or report["transfer_aggregate"]["summary"]["total"] != 12096
        or report["transfer_aggregate"]["families"]["total"] != 1008
    ):
        raise ValueError("evaluation disagrees with fixed protocol")
    if (run.root / "data/transfer_manifest.json").read_bytes() != (
        run.root / "diagnosis/transfer_manifest.json"
    ).read_bytes():
        raise ValueError("evaluation population changed")
    paths = [
        "data/transfer_manifest.json",
        "provenance/protocol.json",
        "diagnosis/transfer_manifest.json",
        "diagnosis/summary.json",
        "diagnosis/reference_summary.json",
        "diagnosis/reference_hashes.json",
        "diagnosis/matched_comparisons.json",
        "diagnosis/predictions.jsonl",
        "diagnosis/inspection.html",
    ]
    paths.extend(p.relative_to(run.root).as_posix() for p in (run.root / "logs").glob("*.log"))
    write_json(run.root / "provenance/final.json", {p: file_hash(run.root / p) for p in paths})
    # A reproduction failure is a recorded diagnostic result, not a reason to discard the archive.
    return report


def archive_quartet_transfer(run):
    final = run.root / "provenance/final.json"
    if not final.is_file():
        raise ValueError("complete evaluation before archiving")
    hashes = json.loads(final.read_text())
    for name, digest in hashes.items():
        if file_hash(run.root / name) != digest:
            raise ValueError("artifact changed since final audit")
    destination = run.root.parent / f"{run.root.name}_artifacts.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [*sorted(hashes), "provenance/final.json"]:
            archive.write(run.root / name, arcname=Path(run.root.name) / name)
    return destination
