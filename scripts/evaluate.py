"""Evaluate validation image controls using a saved synthetic training checkpoint."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from multimodal_loop.data.multimodal import SyntheticColorDataset
from multimodal_loop.eval.synthetic import evaluate_image_controls
from multimodal_loop.train.runtime import runtime_metadata
from multimodal_loop.train.synthetic_checkpoint import load_synthetic_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu", help="Use the checkpoint's backend.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/color_controls"))
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
        saved = load_synthetic_checkpoint(args.checkpoint, device=args.device)
        batch_size = (
            saved.training_config.batch_size if args.batch_size is None else args.batch_size
        )
        depth = saved.training_config.recurrence_depth
        results = evaluate_image_controls(
            saved.model,
            SyntheticColorDataset(saved.manifest, "validation"),
            recurrence_depth=depth,
            batch_size=batch_size,
            shuffle_seeds=tuple(args.shuffle_seeds),
        )
        report = {
            "format_version": 1,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": saved.manifest.sha256,
            "completed_epochs": saved.completed_epochs,
            "completed_steps": saved.completed_steps,
            "model": asdict(saved.model.config),
            "training": asdict(saved.training_config),
            "evaluation": {
                "split": "validation",
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
    print(f"Validation image controls: R={depth}, batch_size={batch_size}")
    print(f"{'Condition':<18} {'Correct/total':>14} {'Accuracy':>10} {'Loss':>12} {'Invalid':>8}")
    rows = [("correct", results["correct"])]
    rows.extend((f"shuffled seed {row['seed']}", row["metrics"]) for row in results["shuffled"])
    rows.append(("blank", results["blank"]))
    for name, metrics in rows:
        counts = f"{metrics['correct']}/{metrics['total']}"
        print(
            f"{name:<18} {counts:>14} {metrics['accuracy']:>10.4%} {metrics['loss']:>12.6f} "
            f"{metrics['invalid_predictions']:>8}"
        )
    summary = results["shuffled_summary"]
    print(
        f"Shuffled mean accuracy={summary['accuracy']['mean']:.4%}, "
        f"mean loss={summary['loss']['mean']:.6f}"
    )
    print(f"Accuracy gaps: {results['accuracy_gaps']}")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
