"""Run a reproducible training smoke test on a fixed synthetic tensor batch."""

import argparse
from math import isfinite

import torch

from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.trainer import train_on_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--text-length", type=int, default=8)
    parser.add_argument(
        "--question-length",
        type=int,
        default=None,
        help="Prefix question length; defaults to 3 with images and 0 for --text-only.",
    )
    parser.add_argument("--text-only", action="store_true", help="Omit synthetic images.")
    parser.add_argument("--recurrence-depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.steps <= 0 or args.batch_size <= 0 or args.recurrence_depth <= 0:
        parser.error("steps, batch-size, and recurrence-depth must be positive")
    if args.text_length < 2:
        parser.error("text-length must be at least 2")
    question_length = args.question_length
    if question_length is None:
        question_length = 0 if args.text_only else 3
    if not 0 <= question_length < args.text_length:
        parser.error("question-length must be nonnegative and less than text-length")
    if not isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    if not isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")

    config = ModelConfig(recurrence_depth=args.recurrence_depth)
    num_image_tokens = 0 if args.text_only else config.num_patches
    if num_image_tokens + args.text_length > config.max_seq_len:
        parser.error(f"combined image/text length must not exceed {config.max_seq_len}")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = MultimodalLoopTransformer(config).to(device)
    input_ids = torch.randint(config.vocab_size, (args.batch_size, args.text_length), device=device)
    images = None
    if not args.text_only:
        images = torch.randn(
            args.batch_size,
            config.num_channels,
            config.image_size,
            config.image_size,
            device=device,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    mode = "text-only" if args.text_only else "multimodal"
    print(
        f"Fixed synthetic batch: mode={mode}, device={device}, seed={args.seed}, "
        f"recurrence_depth={config.recurrence_depth}, question_length={question_length}"
    )
    losses = train_on_batch(
        model,
        optimizer,
        input_ids,
        images,
        steps=args.steps,
        question_length=question_length,
        recurrence_depth=args.recurrence_depth,
    )
    for step, loss in enumerate(losses, start=1):
        print(f"step={step:03d} loss={loss:.6f}")


if __name__ == "__main__":
    main()
