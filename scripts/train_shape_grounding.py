"""Train a fresh direct shape-color model for an explicit number of updates."""

import argparse
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from multimodal_loop.data.geometry_diversity import derive_geometry_corpus, presentation_summary
from multimodal_loop.data.relational_dataset import load_relational_manifest
from multimodal_loop.data.shape_grounding import ShapeColorCollator
from multimodal_loop.eval.baseline_artifacts import BASELINE_MANIFEST_SHA256
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.runtime import resolve_device, runtime_metadata, seed_everything
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig, train_shape_grounding
from multimodal_loop.train.shape_grounding_checkpoint import (
    save_shape_checkpoint,
    tokenizer_metadata,
)
from multimodal_loop.train.synthetic import SyntheticTrainingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--geometry-source-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/relational_baseline.yaml")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=2880)
    parser.add_argument("--evaluation-interval", type=int, default=288)
    parser.add_argument(
        "--smoke", action="store_true", help="Allow a non-baseline manifest; label run as smoke."
    )
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise ValueError("training requires a fresh output directory; resume is not supported")
        manifest = load_relational_manifest(args.manifest)
        if (
            not args.smoke
            and args.geometry_source_manifest is None
            and manifest.sha256 != BASELINE_MANIFEST_SHA256
        ):
            raise ValueError("research run requires the original baseline manifest SHA256")
        config = ModelConfig(**yaml.safe_load(args.model_config.read_text()))
        ShapeColorCollator(config)
        if config.vocab_size != 17 or config.image_size != manifest.config.image_size:
            raise ValueError("model must have vocab_size=17 and match manifest image_size")
        derivation = None
        if args.geometry_source_manifest is not None:
            source = load_relational_manifest(args.geometry_source_manifest)
            if not args.smoke and (
                source.sha256 != BASELINE_MANIFEST_SHA256
                or manifest.config.train_geometry_count != 128
                or config.patch_size != 8
            ):
                raise ValueError("derived research requires the original source and 128 geometries")
            expected, derivation = derive_geometry_corpus(
                source,
                geometry_count=manifest.config.train_geometry_count,
                patch_size=config.patch_size,
            )
            if expected.content != manifest.content:
                raise ValueError("manifest disagrees with deterministic geometry derivation")
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
            "kind": "direct_shape_color",
            "corpus_derivation": derivation,
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

    if derivation is not None:
        write_json(
            args.output_dir / "presentations.json",
            presentation_summary(
                manifest,
                max_steps=budget.max_steps,
                batch_size=training.batch_size,
                seed=training.seed,
            ),
        )

    def publish(history):
        save_shape_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            manifest=manifest,
            training=training,
            budget=budget,
            history=history,
            smoke=args.smoke,
            corpus_derivation=derivation,
        )
        write_json(args.output_dir / "metrics.json", history)
        row = history[-1]
        print(
            f"step={row['completed_steps']} examples={row['examples_seen']} "
            f"validation_accuracy={row['validation']['accuracy']:.4f} "
            f"validation_loss={row['validation']['loss']:.6f}",
            flush=True,
        )

    train_shape_grounding(model, optimizer, manifest, training, budget, publish=publish)


if __name__ == "__main__":
    main()
