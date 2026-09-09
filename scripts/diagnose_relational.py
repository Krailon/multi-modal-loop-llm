"""Diagnose the frozen Milestone 2 baseline on train and validation only."""

import argparse
import json
from pathlib import Path

import torch

from multimodal_loop.data.relational_dataset import RelationalColorDataset
from multimodal_loop.eval.baseline_artifacts import audit_baseline
from multimodal_loop.eval.relational_diagnostics import diagnose_relational, inspection_html
from multimodal_loop.train.kaggle import file_hash, repository_revision
from multimodal_loop.train.relational_checkpoint import load_relational_checkpoint
from multimodal_loop.train.runtime import runtime_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        source, output = args.baseline_dir.resolve(), args.output_dir.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Diagnostic output and baseline must be separate directory trees")
        if output.exists():
            raise ValueError("Diagnostic output directory already exists; choose a fresh directory")
        if torch.device(args.device).type != "cuda":
            raise ValueError("The frozen baseline requires its saved CUDA backend")
        audit = audit_baseline(source)
        revision = repository_revision(Path(__file__).resolve().parents[1])
        checkpoint = source / "training" / "last.pt"
        saved = load_relational_checkpoint(checkpoint, device=args.device)
        datasets = {
            split: RelationalColorDataset(saved.manifest, split)
            for split in ("train", "validation")
        }
        diagnoses = {}
        for split, dataset in datasets.items():
            print(f"Diagnosing {split}: {len(dataset)} QA examples, R=2, batch_size=32", flush=True)
            diagnoses[split] = diagnose_relational(
                saved.model, dataset, split=split, recurrence_depth=2, batch_size=32
            )
        if file_hash(checkpoint) != audit["checkpoint_sha256"]:
            raise ValueError("Source checkpoint changed during diagnosis")
        old = audit["controls"]["results"]["correct"]
        new = diagnoses["validation"].summary
        keys = ("total", "correct", "accuracy", "invalid_predictions", "loss")
        comparison = {
            "original": {key: old[key] for key in keys},
            "diagnostic": {key: new[key] for key in keys},
            "differences": {key: new[key] - old[key] for key in keys},
            "exact_match": all(new[key] == old[key] for key in keys),
        }
        report = {
            "kind": "relational_frozen_diagnostics",
            "format_version": 1,
            "checkpoint_sha256": audit["checkpoint_sha256"],
            "manifest_sha256": audit["manifest_sha256"],
            "original_controls_sha256": file_hash(source / "validation" / "controls.json"),
            "training_revision": audit["training_revision"],
            "diagnostic_revision": revision,
            "completed_epochs": saved.completed_epochs,
            "completed_steps": saved.completed_steps,
            "evaluation": {
                "splits": ["train", "validation"],
                "recurrence_depth": 2,
                "batch_size": 32,
            },
            "runtime": runtime_metadata(next(saved.model.parameters()).device),
            "torch_num_threads": torch.get_num_threads(),
            "splits": {split: diagnosis.summary for split, diagnosis in diagnoses.items()},
            "validation_comparison": comparison,
        }
        # Compute and validate all outputs before publishing any of them.
        summary = json.dumps(report, indent=2, allow_nan=False) + "\n"
        rows = {
            split: "".join(json.dumps(row, allow_nan=False) + "\n" for row in diagnosis.examples)
            for split, diagnosis in diagnoses.items()
        }
        preview = inspection_html(diagnoses, datasets)
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(summary, encoding="utf-8")
        for split, content in rows.items():
            (output / f"{split}_examples.jsonl").write_text(content, encoding="utf-8")
        (output / "inspection.html").write_text(preview, encoding="utf-8")
        for split, diagnosis in diagnoses.items():
            print(
                f"{split}: accuracy={diagnosis.summary['accuracy']:.4%}, "
                f"loss={diagnosis.summary['loss']:.6f}, "
                f"all-four={diagnosis.summary['all_four']['accuracy']:.4%}"
            )
        print("Validation comparison:", json.dumps(comparison, allow_nan=False))
        print(f"Wrote diagnostics to {output}")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
