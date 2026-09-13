"""Final CNN assessment against the identical four-arrangement transformer population."""

import json
from pathlib import Path

from multimodal_loop.data.quartet_transfer import QuartetTransferDataset
from multimodal_loop.eval.cnn_reference import read_cnn_reference
from multimodal_loop.eval.multi_arrangement import (
    aggregate_role,
    assess_training,
    metric_comparison,
)
from multimodal_loop.eval.quartet_transfer import matched_changes, population_metrics
from multimodal_loop.eval.shape_grounding import inspection_html, shape_prediction_rows
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.train.cnn_baseline import predict_cnn
from multimodal_loop.train.cnn_checkpoint import load_cnn_checkpoint
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.runtime import runtime_metadata


def evaluate_cnn(checkpoint, destination, reference_source, *, device="cpu"):
    checkpoint, destination = Path(checkpoint), Path(destination)
    if destination.exists():
        raise ValueError("use a fresh evaluation directory")
    digest = file_hash(checkpoint)
    model, manifest, training, payload = load_cnn_checkpoint(checkpoint, device=device)
    if payload["completed_steps"] != payload["budget"]["max_steps"]:
        raise ValueError("complete the fixed training budget before final evaluation")
    reference_manifest, reference, old_rows, hashes = read_cnn_reference(
        reference_source, smoke=payload["smoke"]
    )
    if reference_manifest.sha256 != manifest.sha256:
        raise ValueError("reference corpus differs from checkpoint")
    rows, metrics, datasets = {}, {}, {}
    for p in manifest.populations:
        print(f"Diagnosing {p.name}: {p.role}", flush=True)
        datasets[p.name] = QuartetTransferDataset(p)
        _, details = predict_cnn(model, datasets[p.name], training)
        rows[p.name] = shape_prediction_rows(datasets[p.name], details)
        metrics[p.name] = population_metrics(rows[p.name], p.records)
    aggregates = {
        role: aggregate_role(manifest.populations, rows, role)
        for role in ("training_fit", "transfer")
    }
    comparison = {
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
        "budget_note": "equal updates and QA exposure; model families and compute differ",
    }
    report = {
        "kind": "cnn_direct_shape_color_diagnostics_v1",
        "smoke": payload["smoke"],
        "checkpoint_sha256": digest,
        "manifest_sha256": manifest.sha256,
        "completed_steps": payload["completed_steps"],
        "examples_seen": payload["examples_seen"],
        "model": payload["config"],
        "training": payload["training_config"],
        "budget": payload["budget"],
        "runtime": runtime_metadata(next(model.parameters()).device),
        "arrangements": metrics,
        "aggregates": aggregates,
        "assessment": assess_training(manifest.populations, metrics),
        "reference_hashes": hashes,
        "transfer_accuracy_gates": [],
    }
    if file_hash(checkpoint) != digest:
        raise ValueError("checkpoint changed during evaluation")
    destination.mkdir(parents=True)
    write_json(destination / "summary.json", report)
    write_json(destination / "comparison.json", comparison)
    write_json(destination / "reference_summary.json", reference)
    (destination / "predictions.jsonl").write_text(
        "".join(
            json.dumps({**row, "arrangement": p.name, "role": p.role}, allow_nan=False) + "\n"
            for p in manifest.populations
            for row in rows[p.name]
        )
    )
    (destination / "inspection.html").write_text(inspection_html(rows, datasets))
    return report
