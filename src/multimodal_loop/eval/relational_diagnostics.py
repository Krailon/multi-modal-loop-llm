"""Frozen-model train/validation error analysis; metadata is used only after inference."""

import html
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import fsum

from multimodal_loop.data.preview import raster_svg
from multimodal_loop.data.relational_dataset import RelationalColorDataset
from multimodal_loop.data.relational_shapes import resolve_relation
from multimodal_loop.data.text import RelationalColorTokenizer
from multimodal_loop.eval.relational import _grouped_metrics, _predict_relational
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.relational import relational_loader
from multimodal_loop.train.synthetic import SyntheticTrainingConfig

POSITIONS = ("left", "middle", "right")
PREDICTED_POSITIONS = (*POSITIONS, "absent_color", "invalid_token")


@dataclass
class RelationalDiagnosis:
    summary: dict
    examples: list[dict]


def _geometry(record):
    return tuple((obj.left, obj.top, obj.size) for obj in record.scene.objects)


def aggregate_examples(examples: list[dict]) -> dict:
    """Count all examples equally within each named group; no scene resampling."""
    axes = {
        "geometry": ("geometry_id",),
        "anchor_shape": ("anchor_shape",),
        "direction": ("direction",),
        "target_position": ("target_position",),
        "anchor_shape_direction": ("anchor_shape", "direction"),
        "geometry_target_position": ("geometry_id", "target_position"),
    }
    result = {}
    for name, fields in axes.items():
        groups = defaultdict(list)
        for example in examples:
            groups[tuple(example[field] for field in fields)].append(example)
        result[name] = []
        for key, rows in sorted(groups.items()):
            correct = sum(row["correct"] for row in rows)
            group = {
                **dict(zip(fields, key, strict=True)),
                "total": len(rows),
                "correct": correct,
                "accuracy": correct / len(rows),
                "invalid_predictions": sum(
                    row["predicted_position"] == "invalid_token" for row in rows
                ),
                "loss": fsum(row["loss"] for row in rows) / len(rows),
            }
            if "geometry_id" in fields:
                group["geometry"] = rows[0]["geometry"]
            result[name].append(group)
    result["target_position_confusion"] = {
        target: {
            predicted: sum(
                row["target_position"] == target and row["predicted_position"] == predicted
                for row in examples
            )
            for predicted in PREDICTED_POSITIONS
        }
        for target in POSITIONS
    }
    return result


def diagnose_relational(
    model: MultimodalLoopTransformer,
    dataset: RelationalColorDataset,
    *,
    split: str,
    recurrence_depth: int,
    batch_size: int = 32,
) -> RelationalDiagnosis:
    """Infer in manifest order, then attach supervision/inspection metadata.

    Loss/confidence come from the full vocabulary. Invalid predictions are kept
    as IDs with a null color, and remain errors in every aggregation.
    """
    if split not in ("train", "validation"):
        raise ValueError("diagnostics support only train and validation")
    settings = SyntheticTrainingConfig(batch_size=batch_size, recurrence_depth=recurrence_depth)
    details = []
    metrics, predictions = _predict_relational(
        model,
        relational_loader(dataset, model.config, settings),
        recurrence_depth=recurrence_depth,
        details=details,
    )
    if len(details) != len(dataset):
        raise ValueError("diagnostic predictions must cover the entire split")
    # No metadata has been used to construct model inputs or alter predictions.
    geometries = sorted({_geometry(record) for record in dataset.records})
    geometry_ids = {geometry: f"{split}:g{index:03d}" for index, geometry in enumerate(geometries)}
    tokenizer = RelationalColorTokenizer()
    examples = []
    for index, detail in enumerate(details):
        image_index, question_index = divmod(index, 4)
        record = dataset.records[image_index]
        qa = record.questions[question_index]
        target = resolve_relation(record.scene, qa.query)
        target_index = record.scene.objects.index(target)
        if detail["target_id"] != tokenizer.encode_answer(qa.answer):
            raise ValueError("observed target disagrees with stored supervision")
        prediction_id = detail["prediction_id"]
        color = tokenizer.decode_answer(prediction_id) if 6 <= prediction_id < 10 else None
        predicted_position = "invalid_token" if color is None else "absent_color"
        for position, obj in zip(POSITIONS, record.scene.objects, strict=True):
            if obj.color == color:
                predicted_position = position
        geometry = _geometry(record)
        examples.append(
            {
                "split": split,
                "example_index": index,
                "image_index": image_index,
                "question_index": question_index,
                "geometry_id": geometry_ids[geometry],
                "geometry": [dict(left=left, top=top, size=size) for left, top, size in geometry],
                "question": qa.question,
                "answer": qa.answer,
                "prediction": color,
                **detail,
                "correct": detail["target_id"] == prediction_id,
                "anchor_shape": qa.query.anchor_shape,
                "direction": qa.query.direction,
                "target_position": POSITIONS[target_index],
                "predicted_position": predicted_position,
            }
        )
    return RelationalDiagnosis(
        {
            "split": split,
            **asdict(metrics),
            **_grouped_metrics(dataset, predictions),
            "breakdowns": aggregate_examples(examples),
        },
        examples,
    )


def inspection_html(
    diagnoses: dict[str, RelationalDiagnosis], datasets: dict[str, RelationalColorDataset]
) -> str:
    """Show eight highest-confidence incorrect QA examples per split, tie by index.

    Each selected example includes its image and all four question predictions.
    Selection is deliberately diagnostic, not representative of error frequency.
    The same image may appear more than once when multiple selected QAs are wrong.
    """
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>Frozen relational model: errors</title>",
        "<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem}"
        "article{border:1px solid #aaa;padding:1rem;margin:1rem 0}svg{width:200px}"
        "td,th{padding:.4rem;text-align:left}table{border-collapse:collapse}</style>",
        "<h1>Most confident incorrect examples</h1>",
        "<p>Eight incorrect QA examples per split, sorted by predicted-token confidence "
        "descending, then stored example index. These are selected errors, not a representative "
        "sample. All four questions for each selected image are shown.</p>",
    ]
    for split in ("train", "validation"):
        diagnosis = diagnoses[split]
        errors = sorted(
            (row for row in diagnosis.examples if not row["correct"]),
            key=lambda row: (-row["confidence"], row["example_index"]),
        )[:8]
        parts.append(f"<h2>{split}</h2>")
        if not errors:
            parts.append("<p>No incorrect examples.</p>")
        for row in errors:
            index = row["image_index"]
            parts.append(
                f'<article data-example="{row["example_index"]}"><h3>'
                f"{html.escape(row['geometry_id'])} / image {index} / "
                f"question {row['question_index']}</h3>"
            )
            parts.append(raster_svg(datasets[split][4 * index].image))
            parts.append(
                f"<p>Selected error confidence: {row['confidence']:.2%}; "
                f"correct-answer probability: {row['target_probability']:.2%}</p>"
            )
            parts.append(
                "<table><tr><th>Question</th><th>Target</th><th>Prediction</th><th>Confidence</th></tr>"
            )
            for qa in diagnosis.examples[4 * index : 4 * index + 4]:
                prediction = qa["prediction"] or f"invalid token {qa['prediction_id']}"
                parts.append(
                    f"<tr><td>{html.escape(qa['question'])}</td><td>{qa['answer']}</td>"
                    f"<td>{html.escape(prediction)}</td><td>{qa['confidence']:.2%}</td></tr>"
                )
            parts.append("</table></article>")
    parts.append("</html>")
    return "\n".join(parts)
