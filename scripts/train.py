"""Run a reproducible training smoke test on a fixed synthetic tensor batch."""

import argparse
from math import isfinite
from pathlib import Path

import torch

from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.checkpoint import load_checkpoint, save_checkpoint
from multimodal_loop.train.runtime import resolve_device, runtime_metadata, seed_everything
from multimodal_loop.train.trainer import train_on_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=20, help="Optimizer updates for this invocation."
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or xla (TPU).")
    parser.add_argument(
        "--save-checkpoint", type=Path, help="Save after all requested updates finish."
    )
    parser.add_argument("--resume", type=Path, help="Continue a run on its saved backend.")
    fresh = parser.add_argument_group("new run settings (cannot be supplied with --resume)")
    fresh.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    fresh.add_argument("--text-length", type=int, default=argparse.SUPPRESS)
    fresh.add_argument(
        "--question-length",
        type=int,
        default=argparse.SUPPRESS,
        help="Prefix question length; defaults to 3 with images and 0 for --text-only.",
    )
    fresh.add_argument(
        "--text-only", action="store_true", default=argparse.SUPPRESS, help="Omit synthetic images."
    )
    fresh.add_argument("--recurrence-depth", type=int, default=argparse.SUPPRESS)
    fresh.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    fresh.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    fresh.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("steps must be positive")
    try:
        device = resolve_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    defaults = {
        "batch_size": 2,
        "text_length": 8,
        "question_length": None,
        "text_only": False,
        "recurrence_depth": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "seed": 0,
    }
    if args.resume is not None:
        overrides = sorted(defaults.keys() & vars(args).keys())
        if overrides:
            parser.error(
                "--resume cannot be combined with new run settings: "
                + ", ".join("--" + name.replace("_", "-") for name in overrides)
            )
        checkpoint = load_checkpoint(args.resume, device=device)
        model, optimizer = checkpoint.model, checkpoint.optimizer
        input_ids, images = checkpoint.input_ids, checkpoint.images
        question_length = checkpoint.question_length
        recurrence_depth = checkpoint.recurrence_depth
        seed = checkpoint.seed
        completed_steps = checkpoint.completed_steps
        print(f"Resumed {args.resume} after {completed_steps} completed steps")
    else:
        for name, default in defaults.items():
            if not hasattr(args, name):
                setattr(args, name, default)
        if args.batch_size <= 0 or args.recurrence_depth <= 0:
            parser.error("batch-size and recurrence-depth must be positive")
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
        seed = args.seed
        seed_everything(seed, device)
        model = MultimodalLoopTransformer(config)
        input_ids = torch.randint(config.vocab_size, (args.batch_size, args.text_length))
        images = None
        if not args.text_only:
            images = torch.randn(
                args.batch_size,
                config.num_channels,
                config.image_size,
                config.image_size,
            )
        model.to(device)
        input_ids = input_ids.to(device)
        images = None if images is None else images.to(device)
        optimizer_options = {} if device.type == "cpu" else {"foreach": False, "fused": False}
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            **optimizer_options,
        )
        recurrence_depth = args.recurrence_depth
        completed_steps = 0
    mode = "text-only" if images is None else "multimodal"
    print(f"Runtime: {runtime_metadata(device)}")
    print(
        f"Fixed synthetic batch: mode={mode}, device={device}, seed={seed}, "
        f"recurrence_depth={recurrence_depth}, question_length={question_length}"
    )
    losses = train_on_batch(
        model,
        optimizer,
        input_ids,
        images,
        steps=args.steps,
        question_length=question_length,
        recurrence_depth=recurrence_depth,
    )
    for step, loss in enumerate(losses, start=completed_steps + 1):
        print(f"step={step:03d} loss={loss:.6f}")
    if args.save_checkpoint is not None:
        completed_steps += len(losses)
        save_checkpoint(
            args.save_checkpoint,
            model,
            optimizer,
            completed_steps=completed_steps,
            input_ids=input_ids,
            images=images,
            question_length=question_length,
            recurrence_depth=recurrence_depth,
            seed=seed,
        )
        print(f"Saved {args.save_checkpoint} after {completed_steps} completed steps")


if __name__ == "__main__":
    main()
