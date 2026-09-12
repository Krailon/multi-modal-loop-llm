"""Evaluate the frozen quartet-fit model on other complete training-source arrangements."""

import argparse
from pathlib import Path

from multimodal_loop.eval.quartet_transfer import evaluate_transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--smoke", action="store_true", help="Allow explicitly marked test fixtures"
    )
    args = parser.parse_args()
    evaluate_transfer(args.source, args.output_dir, device=args.device, smoke=args.smoke)


if __name__ == "__main__":
    main()
