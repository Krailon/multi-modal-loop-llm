"""Evaluate final multi-arrangement weights and compare saved reference predictions."""

import argparse
from pathlib import Path

from multimodal_loop.eval.multi_arrangement import evaluate_multi_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate_multi_checkpoint(
        args.checkpoint, args.output_dir, args.reference_source, device=args.device
    )
    print("Training fit criteria:", report["training_assessment"], flush=True)


if __name__ == "__main__":
    main()
