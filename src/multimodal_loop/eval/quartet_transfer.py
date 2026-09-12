"""Descriptive frozen transfer evaluation, with a separate reproduction control."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from multimodal_loop.data.quartet_fit import FIT_CONDITIONS
from multimodal_loop.data.quartet_transfer import (
    QuartetTransferDataset,
    build_transfer_populations,
    transfer_manifest,
)
from multimodal_loop.eval.matched_size import size_groups, summarize_rows
from multimodal_loop.eval.quartet_fit import family_metrics
from multimodal_loop.eval.quartet_transfer_reference import read_transfer_reference
from multimodal_loop.eval.shape_grounding import diagnose_shape_grounding, inspection_html
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.quartet_fit_checkpoint import load_quartet_checkpoint
from multimodal_loop.train.runtime import runtime_metadata


def matched_changes(reference, current):
    if len(reference) != len(current) or not reference:
        raise ValueError("matched comparison requires equal nonempty populations")
    details = []
    for index, (a, b) in enumerate(zip(reference, current, strict=True)):
        if any(
            a[k] != b[k]
            for k in ("image_index", "question_index", "shape", "question", "answer", "target_id")
        ):
            raise ValueError("matched question identity mismatch")
        details.append(
            {
                "example_index": index,
                "reference_prediction_id": a["prediction_id"],
                "prediction_id": b["prediction_id"],
                "prediction_changed": a["prediction_id"] != b["prediction_id"],
                "correct_before": a["correct"],
                "correct_after": b["correct"],
            }
        )
    return {
        "questions": len(details),
        "prediction_changes": sum(d["prediction_changed"] for d in details),
        "correct_to_wrong": sum(d["correct_before"] and not d["correct_after"] for d in details),
        "wrong_to_correct": sum(not d["correct_before"] and d["correct_after"] for d in details),
        "both_correct": sum(d["correct_before"] and d["correct_after"] for d in details),
        "both_wrong": sum(not d["correct_before"] and not d["correct_after"] for d in details),
        "details": details,
    }


def population_metrics(rows, records):
    return {
        "summary": summarize_rows(rows),
        "families": family_metrics(
            rows, [list(range(i, i + 4)) for i in range(0, len(records), 4)]
        ),
        "conditions": {
            condition: summarize_rows([row for row in rows if row["image_index"] % 4 == slot])
            for slot, condition in enumerate(FIT_CONDITIONS)
        },
        "relative_size": size_groups(rows, records),
    }


def aggregate_transfer(populations, rows):
    combined, records = [], []
    for p in populations:
        if p.role != "transfer":
            continue
        offset = len(records)
        combined.extend(
            {
                **row,
                "arrangement": p.name,
                "image_index": row["image_index"] + offset,
                "example_index": row["example_index"] + 3 * offset,
            }
            for row in rows[p.name]
        )
        records.extend(p.records)
    if not combined:
        raise ValueError("transfer aggregate cannot include only the control")
    return population_metrics(combined, records)


def evaluate_transfer(source, destination, *, device="cpu", smoke=False):
    destination = Path(destination)
    if destination.exists():
        raise ValueError("use a fresh evaluation directory")
    raw, fit, reference_summary, reference_rows, hashes = read_transfer_reference(
        source, smoke=smoke
    )
    populations = build_transfer_populations(fit)
    if not smoke and (len(populations) != 8 or any(p.triangle_size != 8 for p in populations)):
        raise ValueError("research requires the fixed eight-arrangement population")
    content, digest = transfer_manifest(fit, populations)
    rows, metrics, datasets = {}, {}, {}
    with TemporaryDirectory(prefix="quartet-transfer-") as temporary:
        checkpoint = Path(temporary) / "last.pt"
        checkpoint.write_bytes(raw["training/last.pt"])
        model, restored, training, payload = load_quartet_checkpoint(checkpoint, device=device)
        if (
            restored.sha256 != fit.sha256
            or payload["completed_steps"] != payload["budget"]["max_steps"]
        ):
            raise ValueError("reference checkpoint is incomplete or uses a different manifest")
        if not smoke and payload["smoke"]:
            raise ValueError("research cannot use a smoke checkpoint")
        for p in populations:
            print(f"Evaluating {p.name} ({p.role}): 1728 questions", flush=True)
            datasets[p.name] = QuartetTransferDataset(p)
            _, rows[p.name] = diagnose_shape_grounding(model, datasets[p.name], training)
            metrics[p.name] = population_metrics(rows[p.name], p.records)
        if file_hash(checkpoint) != hashes["training/last.pt"]:
            raise ValueError("frozen evaluation changed checkpoint bytes")
        runtime = runtime_metadata(next(model.parameters()).device)
    reproduction = matched_changes(reference_rows, rows["learned"])
    reproduction["passed"] = reproduction["prediction_changes"] == 0
    comparisons = {
        p.name: matched_changes(rows["learned"], rows[p.name])
        for p in populations
        if p.role == "transfer"
    }
    report = {
        "kind": "quartet_transfer_diagnostics_v1",
        "smoke": smoke,
        "checkpoint_sha256": hashes["training/last.pt"],
        "fit_manifest_sha256": fit.sha256,
        "transfer_manifest_sha256": digest,
        "source_manifest_sha256": fit.source.sha256,
        "reference_completed_steps": payload["completed_steps"],
        "model": payload["config"],
        "runtime": runtime,
        "evaluation": {
            "populations": [p.name for p in populations],
            "transfer_questions": 1728 * (len(populations) - 1),
            "total_questions": 1728 * len(populations),
            "batch_size": training.batch_size,
            "recurrence_depth": training.recurrence_depth,
        },
        "interpretation": "descriptive transfer; no accuracy gates or milestone completion claim",
        "reproduction": {k: v for k, v in reproduction.items() if k != "details"},
        "arrangements": metrics,
        "transfer_aggregate": aggregate_transfer(populations, rows),
        "matched_changes": {
            name: {k: v for k, v in v.items() if k != "details"} for name, v in comparisons.items()
        },
    }
    destination.mkdir(parents=True)
    (destination / "transfer_manifest.json").write_text(content)
    write_json(destination / "summary.json", report)
    write_json(destination / "reference_summary.json", reference_summary)
    write_json(destination / "reference_hashes.json", hashes)
    write_json(
        destination / "matched_comparisons.json", {"reproduction": reproduction, **comparisons}
    )
    with (destination / "predictions.jsonl").open("w") as handle:
        for p in populations:
            for row in rows[p.name]:
                handle.write(json.dumps({**row, "arrangement": p.name, "role": p.role}) + "\n")
    banner = (
        "<p>Reproduction check passed: all learned-arrangement predictions match.</p>"
        if reproduction["passed"]
        else "<h1>Reproduction check FAILED</h1><p>Reference predictions differ; "
        "interpret transfer results only after investigating this discrepancy.</p>"
    )
    (destination / "inspection.html").write_text(banner + inspection_html(rows, datasets))
    print("Reproduction check:", "PASS" if reproduction["passed"] else "FAILED", flush=True)
    return report
