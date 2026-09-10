"""Evaluate the fixed size intervention on the audited 5,760-update checkpoint."""

import argparse
from pathlib import Path

from multimodal_loop.eval.kaggle_size_intervention import evaluate_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate_reference(args.reference_dir, args.output_dir, device=args.device)
    print("QA evaluations:", report["qa_evaluations"])
    print("Original prediction agreement:", report["original_reference_agreement"])


if __name__ == "__main__":
    main()
