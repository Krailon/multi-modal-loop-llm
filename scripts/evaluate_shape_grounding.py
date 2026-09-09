"""Evaluate a frozen direct-color checkpoint on training/validation and validation controls."""

import argparse
import json
from pathlib import Path

from multimodal_loop.eval.shape_grounding_run import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    try:
        report = evaluate_checkpoint(
            args.checkpoint, args.output_dir, device=args.device, seeds=tuple(args.shuffle_seeds)
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report["assessment"], indent=2))


if __name__ == "__main__":
    main()
