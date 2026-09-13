"""Evaluate the final convolutional-stem checkpoint against both saved references."""

import argparse
from pathlib import Path

from multimodal_loop.eval.conv_stem import evaluate_conv_stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-source", type=Path, required=True)
    parser.add_argument("--cnn-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate_conv_stem(
        args.checkpoint,
        args.output_dir,
        args.transformer_source,
        args.cnn_source,
        device=args.device,
    )


if __name__ == "__main__":
    main()
