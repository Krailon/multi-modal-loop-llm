"""Train fixed-depth color answers and evaluate held-out validation layouts."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from multimodal_loop.data.multimodal import SyntheticColorDataset, load_synthetic_manifest
from multimodal_loop.eval.synthetic import evaluate_synthetic
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.runtime import resolve_device, runtime_metadata, seed_everything
from multimodal_loop.train.synthetic import (
    SyntheticTrainingConfig,
    synthetic_loader,
    train_synthetic_epoch,
)
from multimodal_loop.train.synthetic_checkpoint import (
    load_synthetic_checkpoint,
    save_synthetic_checkpoint,
    tokenizer_metadata,
    validate_synthetic_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10, help="Additional epochs this invocation.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or xla (TPU).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_training"))
    parser.add_argument("--resume", type=Path)
    fresh = parser.add_argument_group("new run settings (cannot be supplied with --resume)")
    fresh.add_argument("--manifest", type=Path, default=argparse.SUPPRESS)
    fresh.add_argument(
        "--model-config",
        type=Path,
        default=argparse.SUPPRESS,
        help="YAML mapping of ModelConfig overrides.",
    )
    for name, kind in (
        ("batch-size", int),
        ("learning-rate", float),
        ("weight-decay", float),
        ("seed", int),
        ("recurrence-depth", int),
    ):
        fresh.add_argument(f"--{name}", type=kind, default=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    fresh_keys = {"manifest", "model_config", *asdict(SyntheticTrainingConfig())}
    overrides = fresh_keys & vars(args).keys()
    if args.resume is not None and overrides:
        parser.error(
            "--resume cannot be combined with new run settings: " + ", ".join(sorted(overrides))
        )
    artifact_names = ("settings.json", "manifest.json", "metrics.json", "last.pt")
    if args.resume is None and any((args.output_dir / name).exists() for name in artifact_names):
        parser.error(
            "output directory already contains run artifacts; use --resume or a new directory"
        )
    try:
        device = resolve_device(args.device)
        if args.resume is not None:
            saved = load_synthetic_checkpoint(args.resume, device=device)
            model, optimizer, manifest = saved.model, saved.optimizer, saved.manifest
            training = saved.training_config
            completed_epochs, completed_steps = saved.completed_epochs, saved.completed_steps
            history = saved.history
        else:
            manifest = load_synthetic_manifest(
                getattr(args, "manifest", Path("outputs/synthetic_shapes/manifest.json"))
            )
            model_overrides = {}
            if hasattr(args, "model_config"):
                model_overrides = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
                if not isinstance(model_overrides, dict):
                    raise ValueError("model-config must contain a YAML mapping")
            config = ModelConfig(
                **{"vocab_size": 10, "image_size": manifest.config.image_size, **model_overrides}
            )
            validate_synthetic_config(config, manifest)
            training_overrides = {
                key: vars(args)[key]
                for key in asdict(SyntheticTrainingConfig())
                if key in vars(args)
            }
            training = SyntheticTrainingConfig(
                **{"recurrence_depth": config.recurrence_depth, **training_overrides}
            )
            seed_everything(training.seed, device)
            with torch.device("cpu"):
                model = MultimodalLoopTransformer(config).to(dtype=torch.float32)
            model.to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training.learning_rate,
                weight_decay=training.weight_decay,
                foreach=False,
                fused=False,
            )
            completed_epochs, completed_steps, history = 0, 0, []
    except (OSError, TypeError, ValueError, RuntimeError, yaml.YAMLError) as error:
        parser.error(str(error))

    train_data = SyntheticColorDataset(manifest, "train")
    validation_data = SyntheticColorDataset(manifest, "validation")
    validation_loader = synthetic_loader(validation_data, model.config, training)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_bytes(manifest.content.encode("utf-8"))
    settings = {
        "model": asdict(model.config),
        "training": asdict(training),
        "manifest_sha256": manifest.sha256,
        "tokenizer": tokenizer_metadata(),
        "runtime": runtime_metadata(device),
        "torch_num_threads": torch.get_num_threads(),
    }
    (args.output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Runtime: {settings['runtime']}; manifest SHA256={manifest.sha256}", flush=True)

    def publish() -> None:
        # Commit the authoritative checkpoint first; a stale JSON file is rebuilt
        # from checkpoint history on resume. Epoch metrics never select a model.
        save_synthetic_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            manifest=manifest,
            training_config=training,
            completed_epochs=completed_epochs,
            completed_steps=completed_steps,
            history=history,
        )
        (args.output_dir / "metrics.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        row = history[-1]
        train_loss = "n/a" if row["train"] is None else f"{row['train']['loss']:.6f}"
        print(
            f"epoch={completed_epochs} R={training.recurrence_depth} steps={completed_steps} "
            f"train_loss={train_loss} validation_loss={row['validation']['loss']:.6f} "
            f"validation_accuracy={row['validation']['accuracy']:.4f}",
            flush=True,
        )

    if not history:
        metrics = evaluate_synthetic(
            model, validation_loader, recurrence_depth=training.recurrence_depth
        )
        history.append(
            {"epoch": 0, "completed_steps": 0, "train": None, "validation": asdict(metrics)}
        )
    publish()
    for epoch in range(completed_epochs, completed_epochs + args.epochs):
        metrics = train_synthetic_epoch(
            model,
            optimizer,
            synthetic_loader(train_data, model.config, training, epoch=epoch),
            recurrence_depth=training.recurrence_depth,
        )
        validation = evaluate_synthetic(
            model, validation_loader, recurrence_depth=training.recurrence_depth
        )
        completed_epochs = epoch + 1
        completed_steps += metrics.steps
        history.append(
            {
                "epoch": completed_epochs,
                "completed_steps": completed_steps,
                "train": asdict(metrics),
                "validation": asdict(validation),
            }
        )
        publish()


if __name__ == "__main__":
    main()
