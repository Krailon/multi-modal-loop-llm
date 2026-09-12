"""Evaluate the final quartet-fit checkpoint on its training population only."""

import argparse
from pathlib import Path

from multimodal_loop.eval.quartet_fit import evaluate_quartet_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate_quartet_checkpoint(args.checkpoint, args.output_dir, device=args.device)
    print(report["assessment"], flush=True)


if __name__ == "__main__":
    main()
