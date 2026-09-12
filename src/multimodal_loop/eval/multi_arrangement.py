"""Final four-arrangement training fit and identical-population transfer comparisons."""

import json
from dataclasses import replace
from pathlib import Path

from multimodal_loop.data.quartet_transfer import QuartetTransferDataset
from multimodal_loop.eval.multi_arrangement_reference import read_multi_reference
from multimodal_loop.eval.quartet_fit import assess_fit
from multimodal_loop.eval.quartet_transfer import (
    aggregate_transfer,
    matched_changes,
    population_metrics,
)
from multimodal_loop.eval.shape_grounding import diagnose_shape_grounding, inspection_html
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.multi_arrangement_checkpoint import load_multi_checkpoint
from multimodal_loop.train.runtime import runtime_metadata


def aggregate_role(populations, rows, role):
    selected = tuple(replace(p, role="transfer") for p in populations if p.role == role)
    return aggregate_transfer(selected, rows)


def assess_training(populations, metrics):
    per_arrangement = {
        p.name: assess_fit(metrics[p.name]["summary"], metrics[p.name]["families"])
        for p in populations
        if p.role == "training_fit"
    }
    if not per_arrangement:
        raise ValueError("training assessment requires training arrangements")
    return {
        "arrangements": per_arrangement,
        "passed": all(a["passed"] for a in per_arrangement.values()),
    }


def metric_comparison(reference, current):
    if (
        reference["summary"]["total"] != current["summary"]["total"]
        or reference["families"]["total"] != current["families"]["total"]
    ):
        raise ValueError("comparison requires identical denominators")

    def scalars(m):
        return {
            "accuracy": m["summary"]["accuracy"],
            "loss": m["summary"]["loss"],
            "pair_accuracy": m["summary"]["circle_square_pair"]["accuracy"],
            "correct_family_fraction": m["families"]["fraction"],
            **{r["shape"]: r["accuracy"] for r in m["summary"]["breakdowns"]["shape"]},
        }

    old, new = scalars(reference), scalars(current)
    return {k: {"reference": old[k], "current": new[k], "difference": new[k] - old[k]} for k in old}


def evaluate_multi_checkpoint(checkpoint, destination, reference_source, *, device="cpu"):
    checkpoint, destination = Path(checkpoint), Path(destination)
    if destination.exists():
        raise ValueError("use a fresh evaluation directory")
    digest = file_hash(checkpoint)
    model, manifest, training, payload = load_multi_checkpoint(checkpoint, device=device)
    if payload["completed_steps"] != payload["budget"]["max_steps"]:
        raise ValueError("complete the fixed budget before final evaluation")
    reference, old_rows, hashes = read_multi_reference(
        reference_source, manifest.source, smoke=payload["smoke"]
    )
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
    old_aggregate = aggregate_role(manifest.populations, old_rows, "transfer")
    comparisons = {
        p.name: matched_changes(old_rows[p.name], rows[p.name])
        for p in manifest.populations
        if p.role == "transfer"
    }
    comparison = {
        "reference_checkpoint_sha256": reference["checkpoint_sha256"],
        "current_checkpoint_sha256": digest,
        "transfer_arrangements": manifest.derivation["transfer_arrangements"],
        "transfer_aggregate": metric_comparison(old_aggregate, aggregates["transfer"]),
        "arrangements": {
            p.name: metric_comparison(
                population_metrics(old_rows[p.name], p.records), metrics[p.name]
            )
            for p in manifest.populations
            if p.role == "transfer"
        },
        "matched_changes": comparisons,
        "budget_note": "40 presentations per QA; fourfold updates versus single-arrangement fit",
    }
    report = {
        "kind": "multi_arrangement_quartet_diagnostics_v1",
        "smoke": payload["smoke"],
        "checkpoint_sha256": digest,
        "manifest_sha256": manifest.sha256,
        "fit_manifest_sha256": manifest.source.sha256,
        "completed_steps": payload["completed_steps"],
        "examples_seen": payload["examples_seen"],
        "model": payload["config"],
        "training": payload["training_config"],
        "budget": payload["budget"],
        "runtime": runtime_metadata(next(model.parameters()).device),
        "evaluation": {
            "training_arrangements": manifest.derivation["training_arrangements"],
            "transfer_arrangements": manifest.derivation["transfer_arrangements"],
            "batch_size": training.batch_size,
            "recurrence_depth": training.recurrence_depth,
        },
        "arrangements": metrics,
        "aggregates": aggregates,
        "training_assessment": assess_training(manifest.populations, metrics),
        "reference_hashes": hashes,
        "transfer_assessment": "descriptive; no accuracy gates",
    }
    if file_hash(checkpoint) != digest:
        raise ValueError("frozen evaluation changed checkpoint")
    destination.mkdir(parents=True)
    write_json(destination / "summary.json", report)
    write_json(destination / "comparison.json", comparison)
    write_json(destination / "reference_summary.json", reference)
    write_json(destination / "reference_transfer_aggregate.json", old_aggregate)
    with (destination / "predictions.jsonl").open("w") as handle:
        for p in manifest.populations:
            for row in rows[p.name]:
                handle.write(json.dumps({**row, "arrangement": p.name, "role": p.role}) + "\n")
    (destination / "inspection.html").write_text(inspection_html(rows, datasets))
    return report
