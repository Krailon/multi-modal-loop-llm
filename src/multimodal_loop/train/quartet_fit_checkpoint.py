"""Training-fit checkpoints; frozen loading only, with explicit population labels."""

from dataclasses import asdict
from pathlib import Path

import torch

from multimodal_loop.data.quartet_fit import SOURCE_SHA256, parse_quartet_fit
from multimodal_loop.data.shape_grounding import ShapeColorCollator
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.train.checkpoint import (
    _atomic_save,
    _capture_rng,
    _require_cpu,
    _restore_model,
    _snapshot_model_optimizer,
)
from multimodal_loop.train.runtime import finish_step, preserve_device_rng, resolve_device
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig
from multimodal_loop.train.shape_grounding_checkpoint import _validate_progress, tokenizer_metadata
from multimodal_loop.train.synthetic import SyntheticTrainingConfig


def save_quartet_checkpoint(
    path,
    model,
    optimizer,
    *,
    manifest,
    training,
    budget,
    history,
    smoke=False,
):
    _validate_progress(manifest, training, budget, history, evaluation_name="training_fit")
    validate_fit_protocol(manifest, model.config, training, budget, smoke=smoke)
    if type(smoke) is not bool or next(model.parameters()).dtype != torch.float32:
        raise ValueError("shape checkpoints require a boolean smoke flag and float32 model")
    device = next(model.parameters()).device
    with preserve_device_rng(device) as device_rng:
        finish_step(device)
        payload = {
            "kind": "quartet_training_fit",
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


def load_quartet_checkpoint(path, *, device="cpu"):
    """Load frozen weights on CPU/CUDA without restoring training randomness."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "quartet_training_fit"
        or type(payload.get("format_version")) is not int
        or payload["format_version"] != 1
        or payload.get("tokenizer") != tokenizer_metadata()
        or type(payload.get("smoke")) is not bool
    ):
        raise ValueError("incompatible quartet-fit checkpoint")
    _require_cpu(payload)
    manifest = parse_quartet_fit(payload["manifest_content"])
    if manifest.sha256 != payload["manifest_sha256"]:
        raise ValueError("checkpoint manifest SHA256 mismatch")
    training = SyntheticTrainingConfig(**payload["training_config"])
    budget = ShapeGroundingConfig(**payload["budget"])
    _validate_progress(
        manifest, training, budget, payload["history"], evaluation_name="training_fit"
    )
    validate_fit_protocol(
        manifest, ModelConfig(**payload["config"]), training, budget, smoke=payload["smoke"]
    )
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


def validate_fit_protocol(manifest, model, training, budget, *, smoke=False):
    if type(smoke) is not bool:
        raise ValueError("smoke must be boolean")
    if model.vocab_size != 17 or model.image_size != manifest.config.image_size:
        raise ValueError("model disagrees with training-fit task")
    if not smoke and (
        manifest.source.sha256 != SOURCE_SHA256
        or manifest.derivation["origins"] != [[1, 4], [10, 4], [20, 4]]
        or manifest.derivation["triangle_size"] != 8
        or model != ModelConfig(vocab_size=17)
        or training != SyntheticTrainingConfig(recurrence_depth=2)
        or budget != ShapeGroundingConfig(2160, 54)
    ):
        raise ValueError("research settings disagree with fixed quartet-fit protocol")
