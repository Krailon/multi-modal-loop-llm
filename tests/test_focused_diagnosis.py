"""Saved-prediction classification and alignment, without model inference."""

import copy
from types import SimpleNamespace

import pytest

from multimodal_loop.eval.focused_diagnosis import (
    SavedRun,
    analyze,
    prediction_category,
    relative_size,
    select_pairs,
    transitions,
    validate_rows,
)


def record(square=6, circle=8):
    return SimpleNamespace(
        scene=SimpleNamespace(
            objects=[
                SimpleNamespace(shape="square", color="red", size=square, left=9, top=10),
                SimpleNamespace(shape="circle", color="blue", size=circle, left=18, top=11),
                SimpleNamespace(shape="triangle", color="green", size=6, left=27, top=12),
            ]
        )
    )


@pytest.mark.parametrize(
    "square,circle,expected",
    [(6, 8, "square_smaller"), (8, 6, "square_larger"), (6, 6, "equal_size")],
)
def test_relative_size(square, circle, expected):
    assert relative_size(record(square, circle)) == expected


@pytest.mark.parametrize(
    "color,position,correct,expected",
    [
        ("red", "left", True, "correct"),
        ("blue", "middle", False, "other_circle_square"),
        ("green", "right", False, "triangle"),
        ("yellow", "absent_color", False, "absent_color"),
        (None, "invalid_token", False, "invalid_token"),
    ],
)
def test_prediction_categories(color, position, correct, expected):
    row = {"correct": correct, "prediction": color, "predicted_position": position}
    assert prediction_category(row, record()) == expected


def make_run():
    rows = []
    for image in range(2):
        for slot, shape in enumerate(("square", "circle", "triangle")):
            rows.append(
                {
                    "image_index": image,
                    "example_index": 3 * image + slot,
                    "geometry_id": "validation:g003",
                    "shape": shape,
                    "target_position": ("left", "middle", "right")[slot],
                    "correct": image == 1,
                    "prediction_id": 6,
                    "prediction": "red",
                    "predicted_position": "left",
                }
            )
    manifest = SimpleNamespace(
        content="same", splits={"train": [record(), record()], "validation": [record(), record()]}
    )
    return SavedRun(manifest, {"train": copy.deepcopy(rows), "validation": rows}, {}, {})


def test_transitions_and_group_denominators():
    old, new = make_run(), make_run()
    new.rows["validation"][0]["correct"] = True
    new.rows["validation"][3]["correct"] = False
    result = analyze(old, new)["splits"]["validation"]
    assert result["transitions"] == {
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
        "persistent_error": 2,
        "persistent_correct": 2,
    }
    group = result["relative_size"]["square_smaller"]
    assert group["images"] == 2
    assert group["runs"]["current"]["shapes"]["square"]["total"] == 2
    row = next(r for r in result["layout_groups"] if r["shape"] == "square")
    assert (row["patch_x"], row["patch_y"], row["total"]) == (1, 2, 2)
    with pytest.raises(ValueError):
        transitions([], [{"correct": True}])
    new.manifest = SimpleNamespace(content="different")
    with pytest.raises(ValueError, match="identical"):
        analyze(old, new)


def test_deterministic_matching_and_missing_success():
    run = make_run()
    pairs = select_pairs(run)
    assert pairs == select_pairs(run)
    assert pairs[0] == {"split": "validation", "error_index": 0, "success_index": 3}
    run.rows["validation"][3]["correct"] = False
    assert select_pairs(run)[0]["success_index"] is None
    assert len(select_pairs(run, limit=2)) == 2


def test_incomplete_and_misaligned_predictions():
    run = make_run()
    with pytest.raises(ValueError, match="incomplete"):
        validate_rows([], run.manifest, "train", {})
    # Geometry lookup uses the full scene; this fixture intentionally has all required fields.
    run.rows["train"][0]["example_index"] = 99
    with pytest.raises(ValueError, match="alignment"):
        validate_rows(run.rows["train"], run.manifest, "train", {})


def test_valid_rows_loss_rounding_and_tampering():
    from multimodal_loop.data.shape_grounding import ShapeColorTokenizer
    from multimodal_loop.eval.shape_grounding import summarize_examples

    run = make_run()
    rows = run.rows["train"]
    tokenizer = ShapeColorTokenizer()
    for index, row in enumerate(rows):
        obj = run.manifest.splits["train"][index // 3].scene.objects[index % 3]
        row.update(
            split="train",
            question_index=index % 3,
            geometry_id="train:g000",
            question=f"What color is the {obj.shape}?",
            answer=obj.color,
            target_id=tokenizer.encode_answer(obj.color),
            prediction_id=tokenizer.encode_answer(obj.color),
            prediction=obj.color,
            predicted_position=row["target_position"],
            correct=True,
            object_size=obj.size,
            loss=0.5,
            geometry=[
                dict(left=o.left, top=o.top, size=o.size)
                for o in run.manifest.splits["train"][index // 3].scene.objects
            ],
        )
    summary = {
        "total": 6,
        "correct": 6,
        "accuracy": 1.0,
        "loss": 0.5 + 1e-9,
        **summarize_examples(rows),
    }
    validate_rows(rows, run.manifest, "train", summary)
    bad_summary = dict(summary, loss=0.6)
    with pytest.raises(ValueError, match="loss"):
        validate_rows(rows, run.manifest, "train", bad_summary)
    with pytest.raises(ValueError, match="summary"):
        validate_rows(rows, run.manifest, "train", dict(summary, correct=5))
    rows[0]["answer"] = "wrong"
    with pytest.raises(ValueError, match="alignment"):
        validate_rows(rows, run.manifest, "train", summary)
