"""Frozen training-fit diagnostics with independently grouped quartet correctness."""

import json
from pathlib import Path

from multimodal_loop.data.quartet_fit import FIT_CONDITIONS, QuartetFitDataset
from multimodal_loop.data.synthetic_shapes import SHAPES
from multimodal_loop.eval.matched_size import size_groups, summarize_rows
from multimodal_loop.eval.shape_grounding import diagnose_shape_grounding, inspection_html
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.quartet_fit_checkpoint import load_quartet_checkpoint
from multimodal_loop.train.runtime import runtime_metadata


def family_metrics(rows, families):
    """Require complete ordered image/shape coverage; never confuse invariance with correctness."""
    images = [i for family in families for i in family]
    if (
        not families
        or any(len(f) != 4 for f in families)
        or sorted(images) != list(range(len(images)))
        or len(rows) != 3 * len(images)
    ):
        raise ValueError("incomplete or duplicate family coverage")
    for index, row in enumerate(rows):
        image, slot = divmod(index, 3)
        if (
            row["image_index"] != image
            or row["shape"] != SHAPES[slot]
            or type(row["correct"]) is not bool
        ):
            raise ValueError("prediction order disagrees with image/shape coverage")
    details = []
    for family_id, family in enumerate(families):
        pair = [all(rows[3 * i + slot]["correct"] for slot in (0, 1)) for i in family]
        details.append(
            {
                "family_id": family_id,
                "image_indices": family,
                "pair_correct_by_condition": dict(zip(FIT_CONDITIONS, pair, strict=True)),
                "both_correct_all_four": all(pair),
            }
        )
    correct = sum(d["both_correct_all_four"] for d in details)
    return {
        "total": len(families),
        "correct": correct,
        "fraction": correct / len(families),
        "families": details,
    }


def assess_fit(summary, families):
    by_shape = {r["shape"]: r for r in summary["breakdowns"]["shape"]}
    if set(by_shape) != set(SHAPES):
        raise ValueError("assessment requires every queried shape")
    gates = [
        {
            "name": "training_fit_" + shape,
            "value": by_shape[shape]["accuracy"],
            "threshold": 0.99,
            "passed": by_shape[shape]["accuracy"] >= 0.99,
        }
        for shape in SHAPES
    ]
    gates.append(
        {
            "name": "both_correct_all_four",
            "value": families["fraction"],
            "threshold": 0.95,
            "passed": families["fraction"] >= 0.95,
        }
    )
    return {"gates": gates, "passed": all(g["passed"] for g in gates)}


def evaluate_quartet_checkpoint(checkpoint, destination, *, device="cpu"):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("use a fresh evaluation directory")
    digest = file_hash(Path(checkpoint))
    model, manifest, training, payload = load_quartet_checkpoint(checkpoint, device=device)
    if payload["completed_steps"] != payload["budget"]["max_steps"]:
        raise ValueError("complete the fixed budget before final evaluation")
    dataset = QuartetFitDataset(manifest)
    summary, rows = diagnose_shape_grounding(model, dataset, training)
    families = family_metrics(rows, manifest.derivation["families"])
    conditions = {
        condition: summarize_rows([row for row in rows if row["image_index"] % 4 == slot])
        for slot, condition in enumerate(FIT_CONDITIONS)
    }
    report = {
        "kind": "quartet_training_fit_diagnostics_v1",
        "smoke": payload["smoke"],
        "checkpoint_sha256": digest,
        "manifest_sha256": manifest.sha256,
        "source_manifest_sha256": manifest.source.sha256,
        "completed_steps": payload["completed_steps"],
        "examples_seen": payload["examples_seen"],
        "model": payload["config"],
        "training": payload["training_config"],
        "budget": payload["budget"],
        "runtime": runtime_metadata(next(model.parameters()).device),
        "evaluation": {
            "splits": ["training_fit"],
            "recurrence_depth": training.recurrence_depth,
            "batch_size": training.batch_size,
        },
        "training_fit": summary,
        "conditions": conditions,
        "relative_size": size_groups(rows, dataset.records),
        "families": families,
        "assessment": assess_fit(summary, families),
    }
    if file_hash(Path(checkpoint)) != digest:
        raise ValueError("frozen evaluation changed checkpoint")
    destination.mkdir(parents=True)
    write_json(destination / "summary.json", report)
    with (destination / "training_fit_examples.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    (destination / "inspection.html").write_text(
        inspection_html({"training_fit": rows}, {"training_fit": dataset})
    )
    return report
