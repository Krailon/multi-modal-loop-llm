"""Direct shape-grounding diagnostics and frozen image/question controls."""

import html
import random
from collections import defaultdict
from dataclasses import asdict, replace
from math import fsum

import torch

from multimodal_loop.data.preview import raster_svg
from multimodal_loop.data.shape_grounding import ShapeColorTokenizer
from multimodal_loop.data.synthetic_shapes import SHAPES
from multimodal_loop.eval.color import predict_color_answers
from multimodal_loop.train.shape_grounding import shape_color_loader

POSITIONS = ("left", "middle", "right")


def _counts(matches):
    return {"total": len(matches), "correct": sum(matches), "accuracy": sum(matches) / len(matches)}


def summarize_examples(rows):
    """Recipient labels define every group, including under interventions."""
    if not rows or len(rows) % 3:
        raise ValueError("diagnostics require complete images with three questions")
    breakdowns = {}
    for axis in ("shape", "geometry_id", "target_position", "object_size"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[axis]].append(row)
        breakdowns[axis] = [
            {
                axis: key,
                **_counts([r["correct"] for r in group]),
                "loss": fsum(r["loss"] for r in group) / len(group),
            }
            for key, group in sorted(groups.items())
        ]
    triples = [rows[i : i + 3] for i in range(0, len(rows), 3)]
    same = sum(t[0]["prediction_id"] == t[1]["prediction_id"] for t in triples)
    return {
        "breakdowns": breakdowns,
        "all_three": _counts([all(r["correct"] for r in t) for t in triples]),
        "circle_square_pair": _counts([t[0]["correct"] and t[1]["correct"] for t in triples]),
        "circle_square_same_prediction": {
            "total": len(triples),
            "count": same,
            "fraction": same / len(triples),
        },
        "target_position_confusion": {
            target: {
                pred: sum(
                    r["target_position"] == target and r["predicted_position"] == pred for r in rows
                )
                for pred in (*POSITIONS, "absent_color", "invalid_token")
            }
            for target in POSITIONS
        },
    }


def control_batches(batches, dataset, *, blank=False, images=None, questions=None):
    """Substitute only pixels or question IDs; all supervision remains at its recipient."""
    if sum((blank, images is not None, questions is not None)) > 1:
        raise ValueError("apply only one control at a time")
    tokenizer = ShapeColorTokenizer()
    offset = 0
    for batch in batches:
        count = len(batch.input_ids)
        if blank:
            batch = replace(batch, images=torch.zeros_like(batch.images))
        elif images is not None:
            batch = replace(
                batch,
                images=torch.stack(
                    [dataset[3 * images[i // 3]].image for i in range(offset, offset + count)]
                ),
            )
        elif questions is not None:
            ids = batch.input_ids.clone()
            for row, i in enumerate(range(offset, offset + count)):
                image, slot = divmod(i, 3)
                shape = SHAPES[questions[image][slot]]
                ids[row, :6] = ids.new_tensor(
                    tokenizer.encode_question(f"What color is the {shape}?")
                )
            batch = replace(batch, input_ids=ids)
        offset += count
        yield batch
    if offset != len(dataset):
        raise ValueError("controls require complete stored-order coverage")


def diagnose_shape_grounding(model, dataset, training, **control):
    details = []
    metrics, _ = predict_color_answers(
        model,
        control_batches(shape_color_loader(dataset, model.config, training), dataset, **control),
        recurrence_depth=training.recurrence_depth,
        question_length=6,
        details=details,
    )
    geometries = sorted(
        {tuple((o.left, o.top, o.size) for o in r.scene.objects) for r in dataset.records}
    )
    geometry_ids = {g: f"{dataset.split}:g{i:03d}" for i, g in enumerate(geometries)}
    rows = []
    tokenizer = ShapeColorTokenizer()
    for index, detail in enumerate(details):
        image, slot = divmod(index, 3)
        objects = dataset.records[image].scene.objects
        target_position = next(i for i, o in enumerate(objects) if o.shape == SHAPES[slot])
        target = objects[target_position]
        if detail["target_id"] != tokenizer.encode_answer(target.color):
            raise ValueError("prediction order disagrees with supervision")
        predicted = detail["prediction_id"]
        color = tokenizer.decode_answer(predicted) if 6 <= predicted < 10 else None
        position = next(
            (POSITIONS[i] for i, o in enumerate(objects) if o.color == color),
            "invalid_token" if color is None else "absent_color",
        )
        geometry = tuple((o.left, o.top, o.size) for o in objects)
        rows.append(
            {
                **detail,
                "split": dataset.split,
                "example_index": index,
                "image_index": image,
                "question_index": slot,
                "question": f"What color is the {SHAPES[slot]}?",
                "shape": SHAPES[slot],
                "answer": target.color,
                "prediction": color,
                "correct": color == target.color,
                "target_position": POSITIONS[target_position],
                "predicted_position": position,
                "object_size": target.size,
                "geometry_id": geometry_ids[geometry],
                "geometry": [dict(left=x, top=y, size=s) for x, y, s in geometry],
            }
        )
    return {**asdict(metrics), **summarize_examples(rows)}, rows


def evaluate_shape_controls(model, dataset, training, *, seeds=(0, 1, 2, 3, 4), correct=None):
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("shuffle seeds must be distinct integers and nonempty")
    if correct is None:
        correct = diagnose_shape_grounding(model, dataset, training)[0]
    result = {
        "correct": correct,
        "blank": diagnose_shape_grounding(model, dataset, training, blank=True)[0],
    }
    for kind in ("images", "questions"):
        runs = []
        for seed in seeds:
            rng = random.Random(seed)
            if kind == "images":
                permutation = list(range(len(dataset.records)))
                rng.shuffle(permutation)
                same = sum(
                    next(o.color for o in dataset.records[i].scene.objects if o.shape == shape)
                    == next(
                        o.color for o in dataset.records[donor].scene.objects if o.shape == shape
                    )
                    for i, donor in enumerate(permutation)
                    for shape in SHAPES
                ) / len(dataset)
            else:
                permutation = []
                for _ in dataset.records:
                    slots = list(range(3))
                    rng.shuffle(slots)
                    permutation.append(slots)
                same = sum(
                    slot == donor for p in permutation for slot, donor in enumerate(p)
                ) / len(dataset)
            metrics = diagnose_shape_grounding(model, dataset, training, **{kind: permutation})[0]
            runs.append(
                {
                    "seed": seed,
                    "permutation": permutation,
                    "metrics": metrics,
                    "same_answer_pairing_fraction": same,
                    "accuracy_gap": correct["accuracy"] - metrics["accuracy"],
                }
            )
        result[f"shuffled_{kind}"] = runs
        result[f"shuffled_{kind}_summary"] = {
            metric: {
                "mean": fsum(r["metrics"][metric] for r in runs) / len(runs),
                "min": min(r["metrics"][metric] for r in runs),
                "max": max(r["metrics"][metric] for r in runs),
            }
            for metric in ("accuracy", "loss")
        }
    result["accuracy_gaps"] = {
        "correct_minus_blank": correct["accuracy"] - result["blank"]["accuracy"],
        **{
            f"correct_minus_shuffled_{kind}_mean": correct["accuracy"]
            - result[f"shuffled_{kind}_summary"]["accuracy"]["mean"]
            for kind in ("images", "questions")
        },
    }
    return result


def assess_shape_grounding(train, validation, controls):
    expected_gaps = {
        "correct_minus_blank",
        "correct_minus_shuffled_images_mean",
        "correct_minus_shuffled_questions_mean",
    }
    if set(controls["accuracy_gaps"]) != expected_gaps:
        raise ValueError("assessment requires all three control gaps")
    for metrics in (train, validation):
        groups = metrics["breakdowns"]["shape"]
        if len(groups) != 3 or {g["shape"] for g in groups} != set(SHAPES):
            raise ValueError("assessment requires each shape exactly once")
        for group in groups:
            if (
                type(group["total"]) is not int
                or group["total"] <= 0
                or type(group["correct"]) is not int
                or not 0 <= group["correct"] <= group["total"]
                or group["accuracy"] != group["correct"] / group["total"]
            ):
                raise ValueError("shape accuracy disagrees with counts")
    gates = []
    for split, metrics, threshold in (("train", train, 0.95), ("validation", validation, 0.90)):
        for row in metrics["breakdowns"]["shape"]:
            gates.append(
                {
                    "name": f"{split}_{row['shape']}",
                    "value": row["accuracy"],
                    "threshold": threshold,
                    "passed": row["accuracy"] >= threshold,
                }
            )
    for name, gap in controls["accuracy_gaps"].items():
        gates.append({"name": name, "value": gap, "threshold": 0.30, "passed": gap >= 0.30})
    return {"gates": gates, "passed": all(g["passed"] for g in gates)}


def inspection_html(rows_by_split, datasets):
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        "<title>Direct shape grounding errors</title>",
        "<style>body{font:16px system-ui;max-width:900px;margin:auto}"
        "svg{width:192px}td,th{padding:.5rem}article{border:1px solid;padding:1rem}</style>",
        "<h1>Direct shape grounding: selected errors</h1>",
        "<p>Eight highest-confidence incorrect questions per split, ties by example index. "
        "These selected errors are not representative; each shows all three questions.</p>",
    ]
    for split, rows in rows_by_split.items():
        parts.append(f"<h2>{html.escape(split)}</h2>")
        errors = sorted(
            (r for r in rows if not r["correct"]),
            key=lambda r: (-r["confidence"], r["example_index"]),
        )[:8]
        if not errors:
            parts.append("<p>No incorrect examples.</p>")
        for r in errors:
            index = 3 * r["image_index"]
            parts.append(f"<article><h3>{r['geometry_id']} / image {r['image_index']}</h3>")
            parts.append(raster_svg(datasets[split][index].image))
            parts.append(
                "<table><tr><th>Question</th><th>Target</th><th>Prediction</th>"
                "<th>Confidence</th><th>Target probability</th></tr>"
            )
            for q in rows[index : index + 3]:
                prediction = q["prediction"] or f"invalid token {q['prediction_id']}"
                parts.append(
                    f"<tr><td>{html.escape(q['question'])}</td><td>{q['answer']}</td>"
                    f"<td>{html.escape(prediction)}</td><td>{q['confidence']:.2%}</td>"
                    f"<td>{q['target_probability']:.2%}</td></tr>"
                )
            parts.append("</table></article>")
    return "\n".join([*parts, "</html>"])
