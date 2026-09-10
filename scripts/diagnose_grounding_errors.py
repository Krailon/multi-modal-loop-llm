"""Diagnose saved size/layout errors without model inference."""

import argparse
from pathlib import Path

from multimodal_loop.eval.focused_diagnosis import run_diagnosis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_diagnosis(args.reference, args.current, args.output_dir)
    print(f"Saved report and preview to {args.output_dir}")


if __name__ == "__main__":
    main()
