"""Train the fixed full-resolution CNN baseline on four arrangements."""

import argparse
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from multimodal_loop.data.geometry_diversity import presentation_summary
from multimodal_loop.data.multi_arrangement import MultiArrangementDataset, parse_multi_arrangement
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.model.cnn_baseline import CNNBaseline, CNNConfig, forward_macs
from multimodal_loop.train.cnn_baseline import CNNTrainingConfig, train_cnn
from multimodal_loop.train.cnn_checkpoint import save_cnn_checkpoint, validate_cnn_protocol
from multimodal_loop.train.runtime import resolve_device, runtime_metadata, seed_everything
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/cnn_baseline.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=8640)
    parser.add_argument("--evaluation-interval", type=int, default=216)
    parser.add_argument(
        "--smoke", action="store_true", help="Label reduced fixtures/budgets as smoke"
    )
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise ValueError("training requires a fresh output directory; resume is unsupported")
        manifest = parse_multi_arrangement(args.manifest.read_text())
        config = CNNConfig(**yaml.safe_load(args.model_config.read_text()))
        training = CNNTrainingConfig()
        budget = ShapeGroundingConfig(args.max_steps, args.evaluation_interval)
        validate_cnn_protocol(manifest, config, training, budget, smoke=args.smoke)
        device = resolve_device(args.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("CNN experiment supports CPU and CUDA")
        seed_everything(training.seed, device)
        with torch.device("cpu"):
            model = CNNBaseline(config).to(dtype=torch.float32)
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
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
    (args.output_dir / "manifest.json").write_text(manifest.content)
    write_json(
        args.output_dir / "settings.json",
        {
            "kind": "cnn_direct_shape_color",
            "smoke": args.smoke,
            "revision": revision,
            "dirty": dirty,
            "model": asdict(config),
            "training": asdict(training),
            "budget": asdict(budget),
            "manifest_sha256": manifest.sha256,
            "runtime": runtime_metadata(device),
            "torch_num_threads": torch.get_num_threads(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "forward_macs_per_qa": forward_macs(config),
            "optimizer": {
                "name": "AdamW",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
            },
        },
    )
    write_json(
        args.output_dir / "presentations.json",
        presentation_summary(
            manifest, max_steps=budget.max_steps, batch_size=training.batch_size, seed=training.seed
        ),
    )

    def publish(history):
        save_cnn_checkpoint(
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
            f"step={row['completed_steps']} training_fit_accuracy="
            f"{row['training_fit']['accuracy']:.4f} loss={row['training_fit']['loss']:.6f}",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    train_cnn(
        model, optimizer, MultiArrangementDataset(manifest), training, budget, publish=publish
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    write_json(
        args.output_dir / "runtime.json",
        {
            "training_and_monitoring_seconds": time.perf_counter() - start,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else None,
        },
    )


if __name__ == "__main__":
    main()
