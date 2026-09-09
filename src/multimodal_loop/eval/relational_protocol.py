"""Assess the recorded Milestone 2 validation gates without running a model."""

from math import isclose, isfinite

from multimodal_loop.data.relational_shapes import RelationalColorQuestion
from multimodal_loop.data.synthetic_shapes import SHAPES


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _counts(metrics: dict, total: int, name: str) -> float:
    if not isinstance(metrics, dict):
        raise ValueError(f"{name} must contain metrics")
    for key in ("total", "correct"):
        if type(metrics[key]) is not int:
            raise ValueError(f"{name}.{key} must be an integer")
    if metrics["total"] != total or not 0 <= metrics["correct"] <= total:
        raise ValueError(f"{name} counts disagree with protocol")
    accuracy = _number(metrics["accuracy"], f"{name}.accuracy")
    if accuracy != metrics["correct"] / total:
        raise ValueError(f"{name} accuracy disagrees with counts")
    return accuracy


def validate_control_report(
    report: dict,
    *,
    checkpoint_hash: str,
    manifest_hash: str,
    model: dict,
    training: dict,
    epochs: int,
    steps: int,
    images: int,
    seeds: list[int],
) -> None:
    """Reject stale, incomplete or incompatible reports before reuse/assessment.

    Expected values come from the validated checkpoint and fixed notebook protocol.
    No checkpoint path comparison: a byte-identical checkpoint may have moved.
    """
    try:
        expected = {
            "kind": "relational_color_controls",
            "format_version": 1,
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": manifest_hash,
            "completed_epochs": epochs,
            "completed_steps": steps,
            "model": model,
            "training": training,
            "evaluation": {
                "split": "validation",
                "batch_size": 32,
                "recurrence_depth": 2,
                "shuffle_seeds": seeds,
            },
        }
        for key, value in expected.items():
            if report[key] != value:
                raise ValueError(f"controls report {key} disagrees with the selected run")
        results = report["results"]
        questions = {
            RelationalColorQuestion(shape, direction).text
            for shape in SHAPES
            for direction in ("left", "right")
        }

        def condition(metrics, name):
            accuracy = _counts(metrics, 4 * images, name)
            if _number(metrics["loss"], f"{name}.loss") < 0:
                raise ValueError("loss must be nonnegative")
            invalid = metrics["invalid_predictions"]
            if type(invalid) is not int or not 0 <= invalid <= 4 * images - metrics["correct"]:
                raise ValueError("invalid prediction counts disagree with totals")
            by_question = metrics["by_question"]
            if set(by_question) != questions:
                raise ValueError("report must contain all six question texts")
            for question, counts in by_question.items():
                _counts(counts, 4 * images // 6, question)
            if sum(c["correct"] for c in by_question.values()) != metrics["correct"]:
                raise ValueError("question counts disagree with overall counts")
            _counts(metrics["all_four"], images, "all_four")
            _counts(metrics["different_answer_pairs"], 5 * images, "different_answer_pairs")
            return accuracy

        correct = condition(results["correct"], "correct")
        blank = condition(results["blank"], "blank")
        gaps = {"correct_minus_blank": correct - blank}
        for kind in ("images", "questions"):
            rows = results[f"shuffled_{kind}"]
            if [row["seed"] for row in rows] != seeds:
                raise ValueError("report must contain every prescribed shuffle seed in order")
            for row in rows:
                accuracy = condition(row["metrics"], f"shuffled_{kind}")
                if not isclose(
                    _number(row["accuracy_gap"], "accuracy_gap"),
                    correct - accuracy,
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("shuffle accuracy gap disagrees with metrics")
                if kind == "images":
                    permutation = row["permutation"]
                    if any(type(i) is not int for i in permutation) or sorted(permutation) != list(
                        range(images)
                    ):
                        raise ValueError("invalid image permutation")
                else:
                    permutations = row["permutations"]
                    if len(permutations) != images or any(
                        any(type(i) is not int for i in p) or sorted(p) != [0, 1, 2, 3]
                        for p in permutations
                    ):
                        raise ValueError("invalid question permutations")
                    fraction = _number(row["same_answer_pairing_fraction"], "answer match fraction")
                    if not 0 <= fraction <= 1:
                        raise ValueError("answer match fraction must be in [0, 1]")
            summary = results[f"shuffled_{kind}_summary"]
            for metric in ("accuracy", "loss"):
                values = [row["metrics"][metric] for row in rows]
                for key, value in {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }.items():
                    if not isclose(
                        _number(summary[metric][key], "shuffle summary"),
                        value,
                        rel_tol=0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError("shuffle summary disagrees with individual results")
            gaps[f"correct_minus_shuffled_{kind}_mean"] = correct - summary["accuracy"]["mean"]
        if set(results["accuracy_gaps"]) != set(gaps):
            raise ValueError("report must contain all three control gaps")
        for name, value in gaps.items():
            if not isclose(
                _number(results["accuracy_gaps"][name], name), value, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("control gap disagrees with metrics")
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete or malformed controls report") from error


def assess_validation_gates(results: dict) -> dict:
    """Six inclusive gates on unrounded values; provenance is checked separately.

    This function is also usable for displaying hypothetical boundary cases.
    Call validate_control_report first when assessing an actual experiment.
    """
    try:
        correct = results["correct"]
        expected = {
            RelationalColorQuestion(shape, direction).text
            for shape in SHAPES
            for direction in ("left", "right")
        }
        if set(correct["by_question"]) != expected:
            raise ValueError("all six question types are required")
        per_question = [
            _number(row["accuracy"], "question accuracy") for row in correct["by_question"].values()
        ]
        values = [
            ("overall_accuracy", correct["accuracy"], 0.90),
            ("minimum_question_accuracy", min(per_question), 0.80),
            ("all_four_accuracy", correct["all_four"]["accuracy"], 0.80),
        ]
        for name in (
            "correct_minus_blank",
            "correct_minus_shuffled_images_mean",
            "correct_minus_shuffled_questions_mean",
        ):
            values.append((name, results["accuracy_gaps"][name], 0.30))
        gates = []
        for name, value, threshold in values:
            value = _number(value, name)
            if not -1 <= value <= 1 or ("accuracy" in name and value < 0):
                raise ValueError(f"{name} outside its valid range")
            gates.append(
                {"name": name, "value": value, "threshold": threshold, "passed": value >= threshold}
            )
        if any(not 0 <= value <= 1 for value in per_question):
            raise ValueError("question accuracy outside [0, 1]")
        return {"passed": all(gate["passed"] for gate in gates), "gates": gates}
    except (KeyError, TypeError) as error:
        raise ValueError("missing or malformed acceptance metrics") from error
