"""Evaluate a complete matched-size checkpoint on training and held-out validation."""

import argparse
from pathlib import Path

from multimodal_loop.eval.matched_size import evaluate_matched_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    report = evaluate_matched_checkpoint(
        args.checkpoint, args.output_dir, device=args.device, seeds=tuple(args.shuffle_seeds)
    )
    print(report["assessment"])


if __name__ == "__main__":
    main()
