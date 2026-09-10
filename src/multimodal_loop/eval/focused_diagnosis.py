"""Compare saved direct-grounding predictions without loading model checkpoints."""

import html
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isclose
from pathlib import Path

from multimodal_loop.data.geometry_diversity import geometry_of
from multimodal_loop.data.preview import raster_svg
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.data.shape_grounding import ShapeColorDataset, ShapeColorTokenizer
from multimodal_loop.data.synthetic_shapes import SHAPES
from multimodal_loop.eval.shape_grounding import POSITIONS, summarize_examples
from multimodal_loop.eval.shape_grounding_run import write_json
from multimodal_loop.train.kaggle import file_hash

CATEGORIES = ("correct", "other_circle_square", "triangle", "absent_color", "invalid_token")


@dataclass
class SavedRun:
    manifest: object
    rows: dict
    summary: dict
    hashes: dict


def validate_rows(rows, manifest, split, summary):
    """Check every saved label, index, geometry and aggregate against its manifest."""
    records = manifest.splits[split]
    if len(rows) != 3 * len(records):
        raise ValueError("incomplete prediction coverage")
    tokenizer = ShapeColorTokenizer()
    geometries = {
        g: f"{split}:g{i:03d}" for i, g in enumerate(sorted({geometry_of(r) for r in records}))
    }
    for index, row in enumerate(rows):
        image, slot = divmod(index, 3)
        record = records[image]
        objects = record.scene.objects
        target_pos = next(i for i, o in enumerate(objects) if o.shape == SHAPES[slot])
        target = objects[target_pos]
        token = row["prediction_id"]
        if type(token) is not int:
            raise ValueError("invalid prediction ID type")
        color = tokenizer.decode_answer(token) if 6 <= token < 10 else None
        predicted_pos = next(
            (POSITIONS[i] for i, o in enumerate(objects) if o.color == color),
            "invalid_token" if color is None else "absent_color",
        )
        expected = {
            "split": split,
            "example_index": index,
            "image_index": image,
            "question_index": slot,
            "shape": target.shape,
            "question": f"What color is the {target.shape}?",
            "answer": target.color,
            "target_id": tokenizer.encode_answer(target.color),
            "prediction": color,
            "correct": color == target.color,
            "target_position": POSITIONS[target_pos],
            "predicted_position": predicted_pos,
            "object_size": target.size,
            "geometry_id": geometries[geometry_of(record)],
            "geometry": [dict(left=o.left, top=o.top, size=o.size) for o in objects],
        }
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError(f"prediction alignment mismatch: {split}:{index}")
    derived = {
        "total": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "loss": fsum(r["loss"] for r in rows) / len(rows),
        **summarize_examples(rows),
    }
    # Stored overall loss averages float32 batch reductions; row losses are scalar sums.
    if not isclose(summary["loss"], derived.pop("loss"), rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError("saved loss disagrees with predictions")
    if any(summary[k] != v for k, v in derived.items()):
        raise ValueError("saved summary disagrees with predictions")


def load_saved_run(path):
    """Read manifests/reports only; never deserialize weights or extract ZIP members."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive members")
        roots = [
            n[: -len("provenance/final.json")]
            for n in names
            if n.endswith("/provenance/final.json")
        ]
        if len(roots) != 1:
            raise ValueError("expected one completed run")
        prefix = roots[0]

        def raw(name):
            return archive.read(prefix + name)

        final = json.loads(raw("provenance/final.json"))
        hashes = {}
        for kind, name in (
            ("manifest", "data/manifest.json"),
            ("summary", "diagnosis/summary.json"),
        ):
            hashes[kind] = sha256(raw(name)).hexdigest()
            if hashes[kind] != final[kind + "_sha256"]:
                raise ValueError(f"archive hash mismatch: {kind}")
        manifest = parse_relational_manifest(raw("data/manifest.json").decode())
        summary = json.loads(raw("diagnosis/summary.json"))
        if summary["manifest_sha256"] != manifest.sha256 or set(summary["splits"]) != {
            "train",
            "validation",
        }:
            raise ValueError("unexpected manifest or splits")
        rows = {}
        for split in ("train", "validation"):
            content = raw(f"diagnosis/{split}_examples.jsonl")
            hashes[split + "_predictions"] = sha256(content).hexdigest()
            rows[split] = [json.loads(line) for line in content.splitlines()]
            validate_rows(rows[split], manifest, split, summary["splits"][split])
    hashes["archive"] = file_hash(path)
    return SavedRun(manifest, rows, summary, hashes)


def relative_size(record):
    sizes = {o.shape: o.size for o in record.scene.objects}
    return (
        "square_smaller"
        if sizes["square"] < sizes["circle"]
        else "square_larger"
        if sizes["square"] > sizes["circle"]
        else "equal_size"
    )


def prediction_category(row, record):
    if row["correct"]:
        return "correct"
    if row["predicted_position"] in ("absent_color", "invalid_token"):
        return row["predicted_position"]
    shape = next(o.shape for o in record.scene.objects if o.color == row["prediction"])
    return "triangle" if shape == "triangle" else "other_circle_square"


def transitions(before, after):
    counts = Counter((a["correct"], b["correct"]) for a, b in zip(before, after, strict=True))
    return {
        "wrong_to_correct": counts[False, True],
        "correct_to_wrong": counts[True, False],
        "persistent_error": counts[False, False],
        "persistent_correct": counts[True, True],
    }


def group_counts(rows):
    correct = sum(r["correct"] for r in rows)
    return {"total": len(rows), "correct": correct, "accuracy": correct / len(rows)}


def analyze(before, after):
    if before.manifest.content != after.manifest.content:
        raise ValueError("comparison requires identical manifests")
    result = {"sources": {"reference": before.hashes, "current": after.hashes}, "splits": {}}
    for split in ("train", "validation"):
        old, new = before.rows[split], after.rows[split]
        records = after.manifest.splits[split]
        groups = {}
        for relation in ("square_smaller", "equal_size", "square_larger"):
            indices = [i for i, r in enumerate(records) if relative_size(r) == relation]
            entry = {"images": len(indices), "runs": {}}
            if not indices:
                continue
            for name, rows in (("reference", old), ("current", new)):
                shapes = {}
                for slot, shape in enumerate(SHAPES[:2]):
                    selected = [rows[3 * i + slot] for i in indices]
                    counts = Counter(
                        prediction_category(r, records[r["image_index"]]) for r in selected
                    )
                    errors = len(selected) - counts["correct"]
                    shapes[shape] = {
                        **group_counts(selected),
                        "predictions": {k: counts[k] for k in CATEGORIES},
                        "swap_rate_all": counts["other_circle_square"] / len(selected),
                        "swap_rate_errors": counts["other_circle_square"] / errors
                        if errors
                        else None,
                    }
                entry["runs"][name] = {
                    "shapes": shapes,
                    "pair_accuracy": sum(
                        rows[3 * i]["correct"] and rows[3 * i + 1]["correct"] for i in indices
                    )
                    / len(indices),
                    "identical_prediction_fraction": sum(
                        rows[3 * i]["prediction_id"] == rows[3 * i + 1]["prediction_id"]
                        for i in indices
                    )
                    / len(indices),
                }
            entry["transitions"] = {
                shape: transitions(
                    [old[3 * i + slot] for i in indices], [new[3 * i + slot] for i in indices]
                )
                for slot, shape in enumerate(SHAPES[:2])
            }
            groups[relation] = entry
        layout_rows = []
        for name, rows in (("reference", old), ("current", new)):
            grouped = defaultdict(list)
            for row in rows:
                record = records[row["image_index"]]
                obj = next(o for o in record.scene.objects if o.shape == row["shape"])
                key = (
                    row["geometry_id"],
                    row["shape"],
                    row["target_position"],
                    obj.size,
                    obj.left % 8,
                    obj.top % 8,
                )
                grouped[key].append(row)
            for key, selected in sorted(grouped.items()):
                predicted_shapes = Counter()
                for row in selected:
                    objects = records[row["image_index"]].scene.objects
                    predicted_shape = next(
                        (o.shape for o in objects if o.color == row["prediction"]),
                        row["predicted_position"],
                    )
                    predicted_shapes[predicted_shape] += 1
                layout_rows.append(
                    dict(
                        zip(
                            ("geometry_id", "shape", "position", "size", "patch_x", "patch_y"),
                            key,
                            strict=True,
                        )
                    )
                    | {
                        "run": name,
                        **group_counts(selected),
                        "predicted_shapes": dict(sorted(predicted_shapes.items())),
                    }
                )
        result["splits"][split] = {
            "relative_size": groups,
            "transitions": transitions(old, new),
            "layout_groups": layout_rows,
        }
    return result


def select_pairs(run, limit=8):
    """Deterministic pairs; alternate validation/train to include both splits."""
    selected = []
    queues = {}
    for split, rows in run.rows.items():
        records = run.manifest.splits[split]

        def key(row, records=records):
            return (
                row["geometry_id"],
                tuple(o.shape for o in records[row["image_index"]].scene.objects),
                row["shape"],
            )

        successes = {}
        for row in rows:
            if row["correct"]:
                successes.setdefault(key(row), row["example_index"])
        errors = sorted((r for r in rows if not r["correct"]), key=lambda r: r["example_index"])
        # Avoid spending the preview on color variants of the same matched group.
        seen = set()
        queue = []
        for row in errors:
            k = key(row)
            if k not in seen:
                seen.add(k)
                queue.append(
                    {
                        "split": split,
                        "error_index": row["example_index"],
                        "success_index": successes.get(k),
                    }
                )
        queues[split] = queue
    for i in range(limit):
        split = ("validation", "train")[i % 2]
        if queues.get(split):
            selected.append(queues[split].pop(0))
    return selected


def preview(run, pairs):
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Focused '
        "grounding diagnosis</title>",
        "<style>body{font:16px "
        "system-ui;max-width:1000px;margin:auto}svg{width:192px}article{border:1px "
        "solid;padding:1rem;margin:1rem}</style>",
        "<h1>Matched saved errors and successes</h1><p>Selected by example index, "
        "alternating validation/train; one error per geometry/shape-assignment/query "
        "group. Matches differ in colors, not geometry or shape assignment. These "
        "examples are illustrative, not representative.</p>",
    ]
    datasets = {s: ShapeColorDataset(run.manifest, s) for s in ("train", "validation")}
    for pair in pairs:
        parts.append("<article>")
        for label, field in (("Error", "error_index"), ("Matched success", "success_index")):
            index = pair[field]
            if index is None:
                parts.append("<p>No matching success in this group.</p>")
                continue
            row = run.rows[pair["split"]][index]
            text = (
                f"{label}: {pair['split']}:{index}, {row['geometry_id']}. "
                f"{row['question']} Target: {row['answer']}; "
                f"prediction: {row['prediction']}; confidence: {row['confidence']:.2%}"
            )
            parts.extend(
                [f"<p>{html.escape(text)}</p>", raster_svg(datasets[pair["split"]][index].image)]
            )
        parts.append("</article>")
    return "\n".join(parts + ["</html>"])


def run_diagnosis(reference, current, output):
    output = Path(output)
    if output.exists():
        raise ValueError("use a fresh output directory")
    before, after = load_saved_run(reference), load_saved_run(current)
    if (before.summary["completed_steps"], after.summary["completed_steps"]) != (2880, 5760):
        raise ValueError("expected the 2880 and 5760 update runs")
    result = analyze(before, after)
    pairs = select_pairs(after)
    result["preview_pairs"] = pairs
    output.mkdir(parents=True)
    write_json(output / "report.json", result)
    (output / "inspection.html").write_text(preview(after, pairs), encoding="utf-8")
    return result
