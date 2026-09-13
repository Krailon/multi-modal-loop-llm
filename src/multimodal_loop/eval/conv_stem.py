"""Final convolutional-stem diagnostics against both frozen reference populations."""

import json
import time
from pathlib import Path

import torch

from multimodal_loop.data.quartet_transfer import QuartetTransferDataset
from multimodal_loop.eval.conv_stem_reference import read_conv_stem_references
from multimodal_loop.eval.multi_arrangement import (
    aggregate_role,
    assess_training,
    metric_comparison,
)
from multimodal_loop.eval.quartet_transfer import matched_changes, population_metrics
from multimodal_loop.eval.shape_grounding import diagnose_shape_grounding, inspection_html
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.train.conv_stem_checkpoint import load_conv_stem_checkpoint
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.runtime import runtime_metadata


def evaluate_conv_stem(checkpoint, destination, transformer_source, cnn_source, *, device="cpu"):
    checkpoint, destination = Path(checkpoint), Path(destination)
    if destination.exists():
        raise ValueError("use a fresh evaluation directory")
    digest = file_hash(checkpoint)
    model, manifest, training, payload = load_conv_stem_checkpoint(checkpoint, device=device)
    if payload["completed_steps"] != payload["budget"]["max_steps"]:
        raise ValueError("complete the fixed training budget before final evaluation")
    references = read_conv_stem_references(transformer_source, cnn_source, smoke=payload["smoke"])
    if any(ref[0].sha256 != manifest.sha256 for ref in references.values()):
        raise ValueError("reference corpus differs from checkpoint")
    actual_device = next(model.parameters()).device
    if actual_device.type == "cuda":
        torch.cuda.synchronize(actual_device)
        torch.cuda.reset_peak_memory_stats(actual_device)
    start = time.perf_counter()
    rows, metrics, datasets = {}, {}, {}
    for p in manifest.populations:
        print(f"Diagnosing {p.name}: {p.role}", flush=True)
        datasets[p.name] = QuartetTransferDataset(p)
        _, rows[p.name] = diagnose_shape_grounding(model, datasets[p.name], training)
        metrics[p.name] = population_metrics(rows[p.name], p.records)
    aggregates = {
        role: aggregate_role(manifest.populations, rows, role)
        for role in ("training_fit", "transfer")
    }
    comparisons = {}
    for name, (_, reference, old_rows, _) in references.items():
        comparisons[name] = {
            "reference_checkpoint_sha256": reference["checkpoint_sha256"],
            "current_checkpoint_sha256": digest,
            "transfer_arrangements": manifest.derivation["transfer_arrangements"],
            "transfer_aggregate": metric_comparison(
                reference["aggregates"]["transfer"], aggregates["transfer"]
            ),
            "arrangements": {
                p.name: metric_comparison(reference["arrangements"][p.name], metrics[p.name])
                for p in manifest.populations
                if p.role == "transfer"
            },
            "matched_changes": {
                p.name: matched_changes(old_rows[p.name], rows[p.name])
                for p in manifest.populations
                if p.role == "transfer"
            },
            "budget_note": (
                "equal updates and QA exposure; image encoding, parameters and compute differ"
            ),
        }
    if actual_device.type == "cuda":
        torch.cuda.synchronize(actual_device)
    performance = {
        "final_evaluation_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(actual_device)
        if actual_device.type == "cuda"
        else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(actual_device)
        if actual_device.type == "cuda"
        else None,
    }
    report = {
        "kind": "conv_stem_direct_shape_color_diagnostics_v1",
        "smoke": payload["smoke"],
        "checkpoint_sha256": digest,
        "manifest_sha256": manifest.sha256,
        "completed_steps": payload["completed_steps"],
        "examples_seen": payload["examples_seen"],
        "model": payload["config"],
        "stem": payload["stem_config"],
        "training": payload["training_config"],
        "budget": payload["budget"],
        "runtime": runtime_metadata(next(model.parameters()).device),
        "arrangements": metrics,
        "aggregates": aggregates,
        "assessment": assess_training(manifest.populations, metrics),
        "reference_hashes": {name: ref[3] for name, ref in references.items()},
        "transfer_accuracy_gates": [],
    }
    if file_hash(checkpoint) != digest:
        raise ValueError("checkpoint changed during evaluation")
    destination.mkdir(parents=True)
    write_json(destination / "summary.json", report)
    write_json(destination / "comparison.json", comparisons)
    write_json(destination / "runtime.json", performance)
    write_json(
        destination / "reference_summary.json", {name: ref[1] for name, ref in references.items()}
    )
    (destination / "predictions.jsonl").write_text(
        "".join(
            json.dumps({**row, "arrangement": p.name, "role": p.role}, allow_nan=False) + "\n"
            for p in manifest.populations
            for row in rows[p.name]
        )
    )
    (destination / "inspection.html").write_text(inspection_html(rows, datasets))
    return report
