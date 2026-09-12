"""Kaggle orchestration for the fixed, training-only quartet fitting check."""

import json
import zipfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import yaml

from multimodal_loop.data.geometry_diversity import presentation_summary
from multimodal_loop.data.matched_size import parse_matched_size
from multimodal_loop.data.quartet_fit import SOURCE_SHA256, derive_quartet_fit
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
class QuartetFitRun:
    repo: Path
    root: Path
    manifest_sha256: str


def read_fit_source(source):
    """Read only the pinned corpus, without extracting or loading reference weights."""
    source = Path(source)
    if source.is_dir():
        root = (
            source
            if (source / "data/manifest.json").is_file()
            else (source / "milestone2_matched_size_training")
        )
        content = (root / "data/manifest.json").read_bytes()
    else:
        with zipfile.ZipFile(source) as archive:
            name = "milestone2_matched_size_training/data/manifest.json"
            if archive.namelist().count(name) != 1:
                raise ValueError("expected exactly one matched-size manifest")
            content = archive.read(name)
    if sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("source manifest identity mismatch")
    return parse_matched_size(content.decode())


def prepare_quartet_fit(repo, root, source):
    repo, root, source = (Path(p).resolve() for p in (repo, root, source))
    if root.exists():
        raise ValueError("use a fresh run directory")
    if any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in (repo, source)):
        raise ValueError("checkout, source and output must use separate paths")
    revision, runtime = repository_revision(repo), cuda_runtime()
    manifest = derive_quartet_fit(read_fit_source(source))
    model = yaml.safe_load((repo / "configs/relational_baseline.yaml").read_text())
    if model != asdict(ModelConfig(vocab_size=17)):
        raise ValueError("model differs from fixed baseline")
    if (
        manifest.derivation["origins"] != [[1, 4], [10, 4], [20, 4]]
        or manifest.derivation["triangle_size"] != 8
    ):
        raise ValueError("selected arrangement differs from protocol")
    (root / "data").mkdir(parents=True)
    (root / "provenance").mkdir()
    (root / "data/manifest.json").write_text(manifest.content)
    write_json(root / "data/derivation.json", manifest.derivation)
    write_json(root / "data/presentations.json", presentation_summary(manifest, max_steps=2160))
    write_json(
        root / "provenance/protocol.json",
        {
            "kind": "milestone2_quartet_training_fit_v1",
            "revision": revision,
            "runtime": runtime,
            "source_manifest_sha256": SOURCE_SHA256,
            "source_archive_sha256": file_hash(source) if source.is_file() else None,
            "manifest_sha256": manifest.sha256,
            "model": model,
            "training": asdict(SyntheticTrainingConfig()),
            "budget": {"max_steps": 2160, "evaluation_interval": 54},
            "initialization": "fresh seed 0",
            "qa_presentations": 69120,
            "passes": 40,
            "images": 576,
            "questions": 1728,
            "families": 144,
            "evaluation": "training_fit only; no validation or test inference",
            "criteria": {"per_shape_accuracy": 0.99, "both_correct_all_four": 0.95},
        },
    )
    return QuartetFitRun(repo, root, manifest.sha256)


def run_quartet_fit(run):
    if any((run.root / n).exists() for n in ("training", "diagnosis")):
        raise ValueError("use a fresh run directory; resume is unsupported")
    run_logged_command(
        run.repo,
        run.root,
        "train",
        [
            "scripts/train_quartet_fit.py",
            "--manifest",
            str(run.root / "data/manifest.json"),
            "--model-config",
            str(run.repo / "configs/relational_baseline.yaml"),
            "--output-dir",
            str(run.root / "training"),
            "--device",
            "cuda:0",
            "--max-steps",
            "2160",
            "--evaluation-interval",
            "54",
        ],
    )
    checkpoint = run.root / "training/last.pt"
    digest = file_hash(checkpoint)
    run_logged_command(
        run.repo,
        run.root,
        "evaluate",
        [
            "scripts/evaluate_quartet_fit.py",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(run.root / "diagnosis"),
            "--device",
            "cuda:0",
        ],
    )

    def read(name):
        return json.loads((run.root / name).read_text())

    report = read("diagnosis/summary.json")
    expected = {
        "kind": "quartet_training_fit_diagnostics_v1",
        "smoke": False,
        "completed_steps": 2160,
        "examples_seen": 69120,
        "manifest_sha256": run.manifest_sha256,
        "source_manifest_sha256": SOURCE_SHA256,
        "checkpoint_sha256": digest,
        "model": asdict(ModelConfig(vocab_size=17)),
        "training": asdict(SyntheticTrainingConfig()),
        "budget": {"max_steps": 2160, "evaluation_interval": 54},
        "evaluation": {"splits": ["training_fit"], "recurrence_depth": 2, "batch_size": 32},
    }
    if any(report.get(k) != v for k, v in expected.items()) or file_hash(checkpoint) != digest:
        raise ValueError("evaluation disagrees with fixed protocol")
    history = read("training/metrics.json")
    if (
        report["training_fit"]["total"] != 1728
        or report["families"]["total"] != 144
        or [h["completed_steps"] for h in history] != list(range(0, 2161, 54))
        or any("validation" in h or "training_fit" not in h for h in history)
        or history[-1]["completed_passes"] != 40
    ):
        raise ValueError("incomplete training-fit coverage or history")
    if read("training/presentations.json") != read("data/presentations.json"):
        raise ValueError("actual exposures disagree with plan")
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


def archive_quartet_fit(run):
    final = run.root / "provenance/final.json"
    if not final.is_file():
        raise ValueError("complete training and evaluation before archiving")
    for name, digest in json.loads(final.read_text()).items():
        if file_hash(run.root / name) != digest:
            raise ValueError("artifact changed since final audit")
    destination = run.root.parent / f"{run.root.name}_artifacts.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ("data", "training", "diagnosis", "provenance", "logs"):
            for path in sorted((run.root / directory).rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=Path(run.root.name) / path.relative_to(run.root))
    return destination
