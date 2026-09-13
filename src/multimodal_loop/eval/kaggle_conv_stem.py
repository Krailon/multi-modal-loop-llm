"""One fixed convolutional-stem run on Kaggle, with audited source and output artifacts."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from multimodal_loop.data.geometry_diversity import presentation_summary
from multimodal_loop.eval.conv_stem_reference import read_conv_stem_references
from multimodal_loop.eval.kaggle_quartet_fit import archive_quartet_fit
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.conv_stem import ConvStemConfig, forward_macs
from multimodal_loop.train.conv_stem_checkpoint import load_conv_stem_checkpoint
from multimodal_loop.train.kaggle import (
    cuda_runtime,
    file_hash,
    repository_revision,
    run_logged_command,
)
from multimodal_loop.train.synthetic import SyntheticTrainingConfig


@dataclass(frozen=True)
class ConvStemRun:
    repo: Path
    root: Path
    transformer_source: Path
    cnn_source: Path
    manifest_sha256: str


def prepare_conv_stem(repo, root, transformer_source, cnn_source):
    repo, root, transformer_source, cnn_source = (
        Path(p).resolve() for p in (repo, root, transformer_source, cnn_source)
    )
    if root.exists():
        raise ValueError("use a fresh run directory")
    if any(
        root == p or root.is_relative_to(p) or p.is_relative_to(root)
        for p in (repo, transformer_source, cnn_source)
    ):
        raise ValueError("sources, checkout and output must use separate paths")
    revision, runtime = repository_revision(repo), cuda_runtime()
    references = read_conv_stem_references(transformer_source, cnn_source)
    manifest = references["transformer"][0]
    config = ModelConfig(**yaml.safe_load((repo / "configs/relational_baseline.yaml").read_text()))
    stem = ConvStemConfig(**yaml.safe_load((repo / "configs/conv_stem.yaml").read_text()))
    if config != ModelConfig(vocab_size=17) or stem != ConvStemConfig():
        raise ValueError("model differs from fixed convolutional-stem baseline")
    (root / "data").mkdir(parents=True)
    (root / "provenance").mkdir()
    (root / "data/manifest.json").write_text(manifest.content)
    write_json(root / "data/derivation.json", manifest.derivation)
    write_json(root / "data/presentations.json", presentation_summary(manifest, max_steps=8640))
    write_json(root / "provenance/reference_summary.json", {n: r[1] for n, r in references.items()})
    write_json(
        root / "provenance/protocol.json",
        {
            "kind": "milestone2_conv_stem_v1",
            "revision": revision,
            "runtime": runtime,
            "reference_hashes": {n: r[3] for n, r in references.items()},
            "reference_archive_sha256": {
                n: file_hash(p) if p.is_file() else None
                for n, p in (("transformer", transformer_source), ("cnn", cnn_source))
            },
            "manifest_sha256": manifest.sha256,
            "model": asdict(config),
            "stem": asdict(stem),
            "training": asdict(SyntheticTrainingConfig()),
            "budget": {"max_steps": 8640, "evaluation_interval": 216},
            "parameters": 290752,
            "forward_macs": {
                "training": forward_macs(config, stem, question_only=False),
                "evaluation": forward_macs(config, stem, question_only=True),
            },
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
            "recurrence_depth": 2,
            "interpretation": (
                "exploratory single-seed image-token intervention; equal exposure, unequal compute"
            ),
        },
    )
    return ConvStemRun(repo, root, transformer_source, cnn_source, manifest.sha256)


def run_conv_stem(run):
    if any((run.root / p).exists() for p in ("training", "diagnosis")):
        raise ValueError("use a fresh run directory; resume is unsupported")
    run_logged_command(
        run.repo,
        run.root,
        "train",
        [
            "scripts/train_conv_stem.py",
            "--manifest",
            str(run.root / "data/manifest.json"),
            "--model-config",
            str(run.repo / "configs/relational_baseline.yaml"),
            "--stem-config",
            str(run.repo / "configs/conv_stem.yaml"),
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
    _, manifest, _, payload = load_conv_stem_checkpoint(checkpoint)
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
            "scripts/evaluate_conv_stem.py",
            "--checkpoint",
            str(checkpoint),
            "--transformer-source",
            str(run.transformer_source),
            "--cnn-source",
            str(run.cnn_source),
            "--output-dir",
            str(run.root / "diagnosis"),
            "--device",
            "cuda:0",
        ],
    )
    report = json.loads((run.root / "diagnosis/summary.json").read_text())
    if (
        report["kind"] != "conv_stem_direct_shape_color_diagnostics_v1"
        or report["checkpoint_sha256"] != digest
        or file_hash(checkpoint) != digest
        or report["manifest_sha256"] != run.manifest_sha256
        or report["smoke"]
        or report["completed_steps"] != 8640
        or report["examples_seen"] != 276480
    ):
        raise ValueError("evaluation disagrees with fixed protocol")
    expected_metadata = {
        "model": payload["config"],
        "stem": payload["stem_config"],
        "training": payload["training_config"],
        "budget": payload["budget"],
    }
    if any(report.get(k) != json.loads(json.dumps(v)) for k, v in expected_metadata.items()):
        raise ValueError("evaluation configuration differs from checkpoint")
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
    required = (
        "data/manifest.json",
        "data/presentations.json",
        "training/last.pt",
        "training/settings.json",
        "training/manifest.json",
        "training/metrics.json",
        "training/runtime.json",
        "training/presentations.json",
        "diagnosis/summary.json",
        "diagnosis/comparison.json",
        "diagnosis/reference_summary.json",
        "diagnosis/predictions.jsonl",
        "diagnosis/inspection.html",
        "diagnosis/runtime.json",
        "provenance/protocol.json",
        "provenance/reference_summary.json",
    )
    if any(not (run.root / name).is_file() for name in required):
        raise ValueError("incomplete experiment artifacts")
    if json.loads((run.root / "training/metrics.json").read_text()) != payload["history"]:
        raise ValueError("training history differs from checkpoint")
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


def archive_conv_stem(run):
    return archive_quartet_fit(run)
