"""Direct-grounding checkpoints for frozen evaluation, not training resume."""

from dataclasses import asdict
from math import isfinite
from pathlib import Path

import torch

from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.data.shape_grounding import ShapeColorCollator, ShapeColorTokenizer
from multimodal_loop.train.checkpoint import (
    _atomic_save,
    _capture_rng,
    _require_cpu,
    _restore_model,
    _snapshot_model_optimizer,
)
from multimodal_loop.train.runtime import finish_step, preserve_device_rng, resolve_device
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig
from multimodal_loop.train.synthetic import SyntheticTrainingConfig


def tokenizer_metadata():
    return {
        "task": "direct_shape_color",
        "version": 1,
        "vocabulary": list(ShapeColorTokenizer.vocabulary),
        "question_length": 6,
        "shape_order": ["square", "circle", "triangle"],
    }


def _validate_progress(manifest, training, budget, history):
    if not isinstance(history, list) or not history:
        raise ValueError("checkpoint requires nonempty training history")
    steps = history[-1]["completed_steps"]
    if type(steps) is not int or not 0 <= steps <= budget.max_steps:
        raise ValueError("checkpoint progress is outside its budget")
    expected_steps = list(range(0, steps + 1, budget.evaluation_interval))
    if expected_steps[-1] != steps:
        if steps != budget.max_steps:
            raise ValueError("checkpoint must be at an evaluation boundary or final step")
        expected_steps.append(steps)
    if [r["completed_steps"] for r in history] != expected_steps:
        raise ValueError("checkpoint history disagrees with evaluation schedule")
    size = 3 * len(manifest.splits["train"])
    per_pass = (size + training.batch_size - 1) // training.batch_size
    previous_examples = previous_steps = 0
    for row in history:
        passes, offset = divmod(row["completed_steps"], per_pass)
        examples = passes * size + offset * training.batch_size
        expected = {
            "completed_passes": passes,
            "batches_in_pass": offset,
            "examples_seen": examples,
        }
        if any(type(row[k]) is not int or row[k] != v for k, v in expected.items()):
            raise ValueError("checkpoint pass/example progress disagrees with steps")
        validation = row["validation"]
        total = 3 * len(manifest.splits["validation"])
        if (
            validation["total"] != total
            or not 0 <= validation["correct"] <= total
            or not 0 <= validation["invalid_predictions"] <= total - validation["correct"]
            or validation["accuracy"] != validation["correct"] / total
            or not isfinite(validation["loss"])
            or validation["loss"] < 0
        ):
            raise ValueError("checkpoint validation metrics disagree with corpus")
        train = row["train"]
        if row["completed_steps"] == 0:
            if train is not None:
                raise ValueError("step zero must not contain training metrics")
        elif (
            train["examples"] != examples - previous_examples
            or train["steps"] != row["completed_steps"] - previous_steps
            or not isfinite(train["loss"])
            or train["loss"] < 0
        ):
            raise ValueError("checkpoint training metrics disagree with progress")
        previous_steps, previous_examples = row["completed_steps"], examples


def save_shape_checkpoint(
    path, model, optimizer, *, manifest, training, budget, history, smoke=False
):
    _validate_progress(manifest, training, budget, history)
    if type(smoke) is not bool or next(model.parameters()).dtype != torch.float32:
        raise ValueError("shape checkpoints require a boolean smoke flag and float32 model")
    device = next(model.parameters()).device
    with preserve_device_rng(device) as device_rng:
        finish_step(device)
        payload = {
            "kind": "direct_shape_color",
            "format_version": 1,
            "smoke": smoke,
            **_snapshot_model_optimizer(model, optimizer),
            "tokenizer": tokenizer_metadata(),
            "manifest_content": manifest.content,
            "manifest_sha256": manifest.sha256,
            "training_config": asdict(training),
            "budget": asdict(budget),
            "history": history,
            **{
                key: history[-1][key]
                for key in (
                    "completed_steps",
                    "examples_seen",
                    "completed_passes",
                    "batches_in_pass",
                )
            },
            "rng_state": _capture_rng(),
            "device_rng_state": device_rng,
        }
    _atomic_save(payload, Path(path))


def load_shape_checkpoint(path, *, device="cpu"):
    """Load frozen weights on CPU/CUDA without restoring training randomness."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "direct_shape_color"
        or type(payload.get("format_version")) is not int
        or payload["format_version"] != 1
        or payload.get("tokenizer") != tokenizer_metadata()
        or type(payload.get("smoke")) is not bool
    ):
        raise ValueError("incompatible direct shape-color checkpoint")
    _require_cpu(payload)
    manifest = parse_relational_manifest(payload["manifest_content"])
    if manifest.sha256 != payload["manifest_sha256"]:
        raise ValueError("checkpoint manifest SHA256 mismatch")
    training = SyntheticTrainingConfig(**payload["training_config"])
    budget = ShapeGroundingConfig(**payload["budget"])
    _validate_progress(manifest, training, budget, payload["history"])
    if any(
        payload[key] != payload["history"][-1][key]
        for key in ("completed_steps", "examples_seen", "completed_passes", "batches_in_pass")
    ):
        raise ValueError("checkpoint history disagrees with progress")
    device = resolve_device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("shape grounding supports CPU and CUDA")
    # Frozen evaluation may change backend; accelerator training RNG is retained
    # in the payload but is neither installed nor needed to reconstruct CPU weights.
    if payload["model_dtype"] != "float32":
        raise ValueError("shape grounding requires float32 weights")
    model = _restore_model({**payload, "device_rng_state": None}, torch.device("cpu"))
    ShapeColorCollator(model.config)
    if model.config.vocab_size != 17 or model.config.image_size != manifest.config.image_size:
        raise ValueError("checkpoint model disagrees with task")
    with preserve_device_rng(device):
        model.to(device)
        model.eval()
    return model, manifest, training, payload
