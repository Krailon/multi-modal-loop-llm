"""Train a fresh direct shape-color model for an explicit number of updates."""

import argparse
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from multimodal_loop.data.geometry_diversity import presentation_summary
from multimodal_loop.data.matched_size import MatchedSizeDataset, parse_matched_size
from multimodal_loop.data.shape_grounding import ShapeColorCollator
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.eval.size_reference import REFERENCE_MANIFEST
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.matched_size_checkpoint import save_matched_checkpoint
from multimodal_loop.train.runtime import resolve_device, runtime_metadata, seed_everything
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig, train_shape_datasets
from multimodal_loop.train.shape_grounding_checkpoint import tokenizer_metadata
from multimodal_loop.train.synthetic import SyntheticTrainingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/relational_baseline.yaml")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=5760)
    parser.add_argument("--evaluation-interval", type=int, default=288)
    parser.add_argument(
        "--smoke", action="store_true", help="Allow a non-baseline manifest; label run as smoke."
    )
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise ValueError("training requires a fresh output directory; resume is not supported")
        manifest = parse_matched_size(args.manifest.read_text())
        if not args.smoke and manifest.source.sha256 != REFERENCE_MANIFEST:
            raise ValueError("research requires the fixed 128-layout source")
        config = ModelConfig(**yaml.safe_load(args.model_config.read_text()))
        ShapeColorCollator(config)
        if config.vocab_size != 17 or config.image_size != manifest.config.image_size:
            raise ValueError("model does not match the direct-color task")
        device = resolve_device(args.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("this experiment supports CPU and CUDA")
        training = SyntheticTrainingConfig(recurrence_depth=config.recurrence_depth)
        budget = ShapeGroundingConfig(args.max_steps, args.evaluation_interval)
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
    except (OSError, ValueError, TypeError, RuntimeError, yaml.YAMLError) as error:
        parser.error(str(error))
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "manifest.json").write_bytes(manifest.content.encode("utf-8"))
    write_json(
        args.output_dir / "settings.json",
        {
            "kind": "matched_size_color",
            "corpus_derivation": manifest.derivation,
            "smoke": args.smoke,
            "revision": revision,
            "dirty": dirty,
            "model": asdict(config),
            "training": asdict(training),
            "budget": asdict(budget),
            "tokenizer": tokenizer_metadata(),
            "manifest_sha256": manifest.sha256,
            "runtime": runtime_metadata(device),
            "torch_num_threads": torch.get_num_threads(),
        },
    )

    write_json(
        args.output_dir / "presentations.json",
        presentation_summary(
            manifest, max_steps=budget.max_steps, batch_size=training.batch_size, seed=training.seed
        ),
    )

    def publish(history):
        save_matched_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            manifest=manifest,
            training=training,
            budget=budget,
            history=history,
            smoke=args.smoke,
        )
        write_json(args.output_dir / "metrics.json", history)
        row = history[-1]
        print(
            f"step={row['completed_steps']} examples={row['examples_seen']} "
            f"validation_accuracy={row['validation']['accuracy']:.4f} "
            f"validation_loss={row['validation']['loss']:.6f}",
            flush=True,
        )

    train_shape_datasets(
        model,
        optimizer,
        MatchedSizeDataset(manifest, "train"),
        MatchedSizeDataset(manifest, "validation"),
        training,
        budget,
        publish=publish,
    )


if __name__ == "__main__":
    main()
