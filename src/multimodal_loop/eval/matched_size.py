"""Final matched-size training diagnostics and the unchanged size intervention."""

import json
from collections import Counter
from math import fsum
from pathlib import Path

from multimodal_loop.data.matched_size import MatchedSizeDataset
from multimodal_loop.data.size_intervention import build_size_cases
from multimodal_loop.eval.focused_diagnosis import prediction_category, relative_size
from multimodal_loop.eval.shape_grounding import (
    assess_shape_grounding,
    diagnose_shape_grounding,
    evaluate_shape_controls,
    inspection_html,
    summarize_examples,
)
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.eval.size_intervention import evaluate_cases, intervention_preview
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.matched_size_checkpoint import load_matched_checkpoint
from multimodal_loop.train.runtime import runtime_metadata


def summarize_rows(rows):
    if not rows:
        raise ValueError("empty subset")
    return {
        "total": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "invalid_predictions": sum(r["predicted_position"] == "invalid_token" for r in rows),
        "loss": fsum(r["loss"] for r in rows) / len(rows),
        **summarize_examples(rows),
    }


def size_groups(rows, records):
    groups = {}
    for relation in ("square_smaller", "equal_size", "square_larger"):
        selected = [r for r in rows if relative_size(records[r["image_index"]]) == relation]
        if not selected:
            continue
        shapes = {}
        for shape in ("square", "circle"):
            queries = [r for r in selected if r["shape"] == shape]
            counts = Counter(prediction_category(r, records[r["image_index"]]) for r in queries)
            shapes[shape] = {"questions": len(queries), "predictions": dict(sorted(counts.items()))}
        groups[relation] = {"metrics": summarize_rows(selected), "shape_predictions": shapes}
    return groups


def evaluate_matched_checkpoint(checkpoint, destination, *, device="cpu", seeds=(0, 1, 2, 3, 4)):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("use a fresh evaluation directory")
    digest = file_hash(checkpoint)
    model, manifest, training, payload = load_matched_checkpoint(checkpoint, device=device)
    if payload["completed_steps"] != payload["budget"]["max_steps"]:
        raise ValueError("complete the update budget before final evaluation")
    datasets = {s: MatchedSizeDataset(manifest, s) for s in ("train", "validation")}
    summaries, rows = {}, {}
    for split, data in datasets.items():
        print(f"Diagnosing {split}: {len(data)} questions", flush=True)
        summaries[split], rows[split] = diagnose_shape_grounding(model, data, training)
    controls = evaluate_shape_controls(
        model, datasets["validation"], training, seeds=seeds, correct=summaries["validation"]
    )
    count = 3 * manifest.derivation["original_images"]
    subsets = {
        "original": summarize_rows(rows["train"][:count]),
        "added": summarize_rows(rows["train"][count:]),
    }
    cases, eligibility = build_size_cases(manifest.source)
    intervention, intervention_rows = evaluate_cases(
        model, cases, batch_size=training.batch_size, recurrence_depth=training.recurrence_depth
    )
    report = {
        "kind": "matched_size_diagnostics_v1",
        "checkpoint_sha256": digest,
        "manifest_sha256": manifest.sha256,
        "source_manifest_sha256": manifest.source.sha256,
        "completed_steps": payload["completed_steps"],
        "examples_seen": payload["examples_seen"],
        "smoke": payload["smoke"],
        "model": payload["config"],
        "training": payload["training_config"],
        "budget": payload["budget"],
        "runtime": runtime_metadata(next(model.parameters()).device),
        "evaluation": {
            "splits": ["train", "validation"],
            "shuffle_seeds": list(seeds),
            "recurrence_depth": training.recurrence_depth,
            "batch_size": training.batch_size,
        },
        "splits": summaries,
        "training_subsets": subsets,
        "relative_size": {s: size_groups(rows[s], datasets[s].records) for s in datasets},
        "relative_size_training_subsets": {
            "original": size_groups(rows["train"][:count], datasets["train"].records),
            "added": size_groups(rows["train"][count:], datasets["train"].records),
        },
        "assessment": assess_shape_grounding(summaries["train"], summaries["validation"], controls),
    }
    if file_hash(checkpoint) != digest:
        raise ValueError("frozen evaluation changed checkpoint")
    destination.mkdir(parents=True)
    write_json(destination / "summary.json", report)
    write_json(
        destination / "controls.json",
        {"checkpoint_sha256": digest, "manifest_sha256": manifest.sha256, "results": controls},
    )
    write_json(
        destination / "size_intervention.json",
        {"checkpoint_sha256": digest, "eligibility": eligibility, "metrics": intervention},
    )
    for name, examples in {
        **{s + "_examples": v for s, v in rows.items()},
        "size_intervention_predictions": intervention_rows,
    }.items():
        with (destination / f"{name}.jsonl").open("w") as handle:
            for row in examples:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    (destination / "inspection.html").write_text(inspection_html(rows, datasets))
    (destination / "size_intervention.html").write_text(
        intervention_preview(cases, intervention_rows)
    )
    return report


def compare_matched(
    reference, reference_controls, current, controls, reference_intervention, intervention
):
    if reference["manifest_sha256"] != current["source_manifest_sha256"]:
        raise ValueError("comparison requires the original source identity")

    def metrics(a, b):
        if a["total"] != b["total"]:
            raise ValueError("comparison requires matching populations")

        def scalars(m):
            return {
                "accuracy": m["accuracy"],
                "loss": m["loss"],
                "pair_accuracy": m["circle_square_pair"]["accuracy"],
                "same_prediction": m["circle_square_same_prediction"]["fraction"],
                "all_three": m["all_three"]["accuracy"],
                **{r["shape"]: r["accuracy"] for r in m["breakdowns"]["shape"]},
            }

        x, y = scalars(a), scalars(b)
        return {k: {"reference": x[k], "current": y[k], "difference": y[k] - x[k]} for k in x}

    if reference_intervention["eligibility"] != intervention["eligibility"]:
        raise ValueError("size-intervention populations differ")
    gaps_a, gaps_b = (
        reference_controls["results"]["accuracy_gaps"],
        controls["results"]["accuracy_gaps"],
    )
    size_comparison = {}
    for condition, old in reference_intervention["metrics"]["conditions"].items():
        new = intervention["metrics"]["conditions"][condition]
        if old["images"] != new["images"]:
            raise ValueError("size condition counts differ")
        size_comparison[condition] = {
            "reference": old,
            "current": new,
            "accuracy_difference": new["accuracy"] - old["accuracy"],
            "pair_accuracy_difference": new["circle_square_pair_accuracy"]
            - old["circle_square_pair_accuracy"],
        }
    return {
        "validation": metrics(reference["splits"]["validation"], current["splits"]["validation"]),
        "original_training": metrics(
            reference["splits"]["train"], current["training_subsets"]["original"]
        ),
        "expanded_training": current["splits"]["train"],
        "added_training": current["training_subsets"]["added"],
        "control_gaps": {
            k: {"reference": gaps_a[k], "current": gaps_b[k], "difference": gaps_b[k] - gaps_a[k]}
            for k in gaps_a
        },
        "size_conditions": size_comparison,
        "size_invariance": {
            "reference": reference_intervention["metrics"]["invariance"],
            "current": intervention["metrics"]["invariance"],
        },
        "size_reversal": {
            "reference": reference_intervention["metrics"]["size_reversal"],
            "current": intervention["metrics"]["size_reversal"],
        },
        "both_correct_all_four": {
            "reference": reference_intervention["metrics"]["both_circle_square_correct_all_four"],
            "current": intervention["metrics"]["both_circle_square_correct_all_four"],
        },
    }
