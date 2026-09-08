"""Evaluate held-out image/question controls using a relational checkpoint."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from multimodal_loop.data.relational_dataset import RelationalColorDataset
from multimodal_loop.eval.relational import evaluate_relational_controls
from multimodal_loop.train.relational_checkpoint import load_relational_checkpoint
from multimodal_loop.train.runtime import runtime_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cpu", help="Use the checkpoint's backend.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/relational_controls"))
    parser.add_argument("--batch-size", type=int, help="Defaults to the saved training batch size.")
    parser.add_argument("--shuffle-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    destination = args.output_dir / "controls.json"
    if destination.exists():
        parser.error("controls.json already exists; choose a new output directory")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if len(set(args.shuffle_seeds)) != len(args.shuffle_seeds):
        parser.error("shuffle seeds must be distinct")
    try:
        with args.checkpoint.open("rb") as handle:
            checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        saved = load_relational_checkpoint(args.checkpoint, device=args.device)
        batch_size = (
            saved.training_config.batch_size if args.batch_size is None else args.batch_size
        )
        depth = saved.training_config.recurrence_depth
        results = evaluate_relational_controls(
            saved.model,
            RelationalColorDataset(saved.manifest, args.split),
            recurrence_depth=depth,
            batch_size=batch_size,
            shuffle_seeds=tuple(args.shuffle_seeds),
        )
        report = {
            "kind": "relational_color_controls",
            "format_version": 1,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": saved.manifest.sha256,
            "completed_epochs": saved.completed_epochs,
            "completed_steps": saved.completed_steps,
            "model": asdict(saved.model.config),
            "training": asdict(saved.training_config),
            "evaluation": {
                "split": args.split,
                "batch_size": batch_size,
                "recurrence_depth": depth,
                "shuffle_seeds": args.shuffle_seeds,
            },
            "runtime": runtime_metadata(next(saved.model.parameters()).device),
            "torch_num_threads": torch.get_num_threads(),
            "results": results,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"{args.split.capitalize()} relational controls: R={depth}, batch_size={batch_size}")
    print(f"{'Condition':<18} {'Correct/total':>14} {'Accuracy':>10} {'Loss':>12} {'Invalid':>8}")
    rows = [("correct", results["correct"])]
    for kind in ("images", "questions"):
        rows.extend(
            (f"{kind} seed {row['seed']}", row["metrics"]) for row in results[f"shuffled_{kind}"]
        )
    rows.append(("blank", results["blank"]))
    for name, metrics in rows:
        counts = f"{metrics['correct']}/{metrics['total']}"
        print(
            f"{name:<18} {counts:>14} {metrics['accuracy']:>10.4%} {metrics['loss']:>12.6f} "
            f"{metrics['invalid_predictions']:>8}"
        )
    for kind in ("images", "questions"):
        summary = results[f"shuffled_{kind}_summary"]
        print(f"Shuffled {kind} mean accuracy={summary['accuracy']['mean']:.4%}")
    print(f"All-four accuracy: {results['correct']['all_four']['accuracy']:.4%}")
    pair_accuracy = results["correct"]["different_answer_pairs"]["accuracy"]
    print(f"Different-answer pair accuracy: {pair_accuracy:.4%}")
    print(f"Accuracy gaps: {results['accuracy_gaps']}")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
