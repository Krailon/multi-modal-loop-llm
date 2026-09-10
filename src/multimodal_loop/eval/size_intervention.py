"""Frozen size intervention: matched scenes and question-only model inputs."""

import html
from collections import defaultdict
from dataclasses import asdict
from math import fsum

import torch
from torch.utils.data import DataLoader

from multimodal_loop.data.preview import raster_svg
from multimodal_loop.data.shape_grounding import ShapeColorCollator, ShapeColorTokenizer
from multimodal_loop.data.size_intervention import (
    CONDITIONS,
    SizeInterventionDataset,
    render_diagnostic_scene,
)
from multimodal_loop.data.synthetic_shapes import SHAPES
from multimodal_loop.eval.color import predict_color_answers


def _accuracy(rows):
    correct = sum(r["correct"] for r in rows)
    return {
        "questions": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "loss": fsum(r["loss"] for r in rows) / len(rows),
    }


def _summary(rows):
    triples = [rows[i : i + 3] for i in range(0, len(rows), 3)]
    return {
        **_accuracy(rows),
        "by_shape": {
            shape: _accuracy([r for r in rows if r["shape"] == shape]) for shape in SHAPES
        },
        "images": len(triples),
        "circle_square_pair_correct": sum(t[0]["correct"] and t[1]["correct"] for t in triples),
        "circle_square_pair_accuracy": sum(t[0]["correct"] and t[1]["correct"] for t in triples)
        / len(triples),
        "other_shape_color_selections": {
            shape: sum(
                r["prediction_kind"] == "other_circle_square" for r in rows if r["shape"] == shape
            )
            for shape in SHAPES[:2]
        },
        "edge_contact_images": sum(any(r["scene_edge_contact"] for r in t) for t in triples),
    }


def summarize_intervention(rows):
    """Invariance is distinct from accuracy; every condition uses the same images."""
    by_condition = {name: [r for r in rows if r["condition"] == name] for name in CONDITIONS}
    original = by_condition["original"]
    if not original or len(original) % 3:
        raise ValueError("incomplete original predictions")
    keys = [(r["source_index"], r["shape"], r["target_id"]) for r in original]
    if len(set(keys)) != len(keys) or any(
        [r["shape"] for r in original[i : i + 3]] != list(SHAPES)
        for i in range(0, len(original), 3)
    ):
        raise ValueError("predictions require unique, complete shape triples")
    if any(
        [(r["source_index"], r["shape"], r["target_id"]) for r in group] != keys
        for group in by_condition.values()
    ):
        raise ValueError("conditions must have identical ordered populations and targets")
    if len(rows) != len(original) * len(CONDITIONS):
        raise ValueError("unexpected condition")
    summaries = {}
    for name, group in by_condition.items():
        layouts = defaultdict(list)
        for row in group:
            layouts[row["geometry_id"]].append(row)
        summaries[name] = {
            **_summary(group),
            "by_geometry": {g: _summary(rs) for g, rs in sorted(layouts.items())},
        }
    variants = [by_condition[name] for name in CONDITIONS if name != "original"]
    stable = {}
    reversal = {}
    for slot, shape in enumerate(SHAPES):
        indices = list(range(slot, len(original), 3))
        stable_correct = sum(all(v[i]["correct"] for v in variants) for i in indices)
        invariant = sum(len({v[i]["prediction_id"] for v in variants}) == 1 for i in indices)
        stable[shape] = {
            "images": len(indices),
            "always_correct": stable_correct,
            "invariant_predictions": invariant,
            "invariant_incorrect": invariant - stable_correct,
            "changed_prediction": len(indices) - invariant,
        }
        a, b = by_condition["circle6_square8"], by_condition["circle8_square6"]
        changes = defaultdict(int)
        for i in indices:
            changes[f"{a[i]['prediction_kind']} -> {b[i]['prediction_kind']}"] += 1
        reversal[shape] = {
            "images": len(indices),
            "prediction_changes": sum(
                a[i]["prediction_id"] != b[i]["prediction_id"] for i in indices
            ),
            "category_transitions": dict(sorted(changes.items())),
        }
    both = sum(
        all(v[i]["correct"] and v[i + 1]["correct"] for v in variants)
        for i in range(0, len(original), 3)
    )
    return {
        "conditions": summaries,
        "invariance": stable,
        "size_reversal": reversal,
        "both_circle_square_correct_all_four": {
            "images": len(original) // 3,
            "correct": both,
            "fraction": both / (len(original) // 3),
        },
    }


def evaluate_cases(model, cases, *, batch_size=32, recurrence_depth=2):
    rows = []
    tokenizer = ShapeColorTokenizer()
    for condition in CONDITIONS:
        dataset = SizeInterventionDataset(cases, condition)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=ShapeColorCollator(model.config),
            generator=torch.Generator(device="cpu").manual_seed(0),
        )
        details = []
        predict_color_answers(
            model, loader, recurrence_depth=recurrence_depth, question_length=6, details=details
        )
        for index, detail in enumerate(details):
            case = dataset.cases[index // 3]
            shape = SHAPES[index % 3]
            target = next(o for o in case.scene.objects if o.shape == shape)
            if detail["target_id"] != tokenizer.encode_answer(target.color):
                raise ValueError("prediction target alignment mismatch")
            token = detail["prediction_id"]
            color = tokenizer.decode_answer(token) if 6 <= token < 10 else None
            predicted_shape = next((o.shape for o in case.scene.objects if o.color == color), None)
            correct = color == target.color
            kind = (
                "correct"
                if correct
                else "invalid_token"
                if color is None
                else "absent_color"
                if predicted_shape is None
                else "other_circle_square"
                if shape in SHAPES[:2] and predicted_shape in SHAPES[:2]
                else predicted_shape
            )
            rows.append(
                {
                    **detail,
                    "condition": condition,
                    "source_index": case.source_index,
                    "geometry_id": case.geometry_id,
                    "shape": shape,
                    "answer": target.color,
                    "prediction": color,
                    "prediction_kind": kind,
                    "predicted_shape": predicted_shape,
                    "correct": correct,
                    "target": asdict(target),
                    "scene_edge_contact": any(
                        o.left + o.size == case.scene.image_size
                        or o.top + o.size == case.scene.image_size
                        for o in case.scene.objects
                    ),
                }
            )
    return summarize_intervention(rows), rows


def compare_original(rows, reference_rows):
    old = {(r["image_index"], r["shape"]): r for r in reference_rows}
    differences = []
    originals = [r for r in rows if r["condition"] == "original"]
    for row in originals:
        ref = old[row["source_index"], row["shape"]]
        if ref["target_id"] != row["target_id"]:
            raise ValueError("reference target mismatch")
        if ref["prediction_id"] != row["prediction_id"]:
            differences.append(
                {
                    "source_index": row["source_index"],
                    "shape": row["shape"],
                    "reference": ref["prediction_id"],
                    "current": row["prediction_id"],
                }
            )
    return {"questions": len(originals), "disagreements": len(differences), "details": differences}


def intervention_preview(cases, rows):
    # First eligible image of each original geometry, chosen before looking at outcomes.
    selected = {}
    for case in cases:
        selected.setdefault(case.geometry_id, case.source_index)
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Size intervention</title>',
        "<style>body{font:16px system-ui;max-width:1100px;"
        "margin:auto}svg{width:160px}article{border:1px solid;"
        "padding:1rem}</style>",
        "<h1>Matched size variants</h1><p>First eligible image per source geometry. "
        "All five conditions; "
        "examples selected before outcomes.</p>",
    ]
    for geometry, index in sorted(selected.items()):
        parts.append(f"<h2>{geometry}: source image {index}</h2>")
        for case in cases:
            if case.source_index != index:
                continue
            parts.extend(
                [
                    f"<article><h3>{case.condition}</h3>",
                    raster_svg(render_diagnostic_scene(case.scene)),
                ]
            )
            for row in rows:
                if row["source_index"] == index and row["condition"] == case.condition:
                    parts.append(
                        "<p>"
                        + html.escape(
                            f"{row['shape']}: target {row['answer']}; "
                            f"prediction {row['prediction']}; "
                            f"confidence {row['confidence']:.2%}"
                        )
                        + "</p>"
                    )
            parts.append("</article>")
    return "\n".join(parts + ["</html>"])
