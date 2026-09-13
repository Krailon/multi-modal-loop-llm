"""Evaluate the final CNN checkpoint and compare saved transformer predictions."""

import argparse
from pathlib import Path

from multimodal_loop.eval.cnn_baseline import evaluate_cnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate_cnn(args.checkpoint, args.output_dir, args.reference_source, device=args.device)


if __name__ == "__main__":
    main()
