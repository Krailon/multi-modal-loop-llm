"""Publish complete frozen train/validation reports from a direct-color checkpoint."""

import json
from pathlib import Path

from multimodal_loop.data.shape_grounding import ShapeColorDataset
from multimodal_loop.eval.shape_grounding import (
    assess_shape_grounding,
    diagnose_shape_grounding,
    evaluate_shape_controls,
    inspection_html,
)
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.runtime import runtime_metadata
from multimodal_loop.train.shape_grounding_checkpoint import load_shape_checkpoint


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_checkpoint(checkpoint, destination, *, device="cpu", seeds=(0, 1, 2, 3, 4)):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("evaluation requires a fresh output directory")
    checkpoint = Path(checkpoint)
    checkpoint_hash = file_hash(checkpoint)
    model, manifest, training, payload = load_shape_checkpoint(checkpoint, device=device)
    if payload["completed_steps"] != payload["budget"]["max_steps"]:
        raise ValueError("complete the recorded update budget before final evaluation")
    datasets = {s: ShapeColorDataset(manifest, s) for s in ("train", "validation")}
    summaries, rows = {}, {}
    for split, dataset in datasets.items():
        print(f"Diagnosing {split}: {len(dataset)} direct-color questions", flush=True)
        summaries[split], rows[split] = diagnose_shape_grounding(model, dataset, training)
    print("Evaluating validation image/question controls", flush=True)
    controls = evaluate_shape_controls(
        model, datasets["validation"], training, seeds=seeds, correct=summaries["validation"]
    )
    report = {
        "kind": "direct_shape_color_diagnostics",
        "smoke": payload["smoke"],
        "format_version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest.sha256,
        "completed_steps": payload["completed_steps"],
        "examples_seen": payload["examples_seen"],
        "model": payload["config"],
        "training": payload["training_config"],
        "budget": payload["budget"],
        "tokenizer": payload["tokenizer"],
        "evaluation": {
            "splits": ["train", "validation"],
            "shuffle_seeds": list(seeds),
            "recurrence_depth": training.recurrence_depth,
            "batch_size": training.batch_size,
        },
        "runtime": runtime_metadata(next(model.parameters()).device),
        "splits": summaries,
        "assessment": assess_shape_grounding(summaries["train"], summaries["validation"], controls),
    }
    if file_hash(checkpoint) != checkpoint_hash:
        raise ValueError("checkpoint changed during frozen evaluation")
    destination.mkdir(parents=True)
    write_json(destination / "summary.json", report)
    write_json(
        destination / "controls.json",
        {
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": manifest.sha256,
            "results": controls,
        },
    )
    for split, examples in rows.items():
        with (destination / f"{split}_examples.jsonl").open("w", encoding="utf-8") as handle:
            for row in examples:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    (destination / "inspection.html").write_text(inspection_html(rows, datasets), encoding="utf-8")
    return report
