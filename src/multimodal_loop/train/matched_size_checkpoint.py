"""Distinct matched-size checkpoints with self-contained derived-corpus provenance."""

from dataclasses import asdict
from pathlib import Path

import torch

from multimodal_loop.data.matched_size import parse_matched_size
from multimodal_loop.data.shape_grounding import ShapeColorCollator
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


def save_matched_checkpoint(
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
    _validate_progress(manifest, training, budget, history)
    if type(smoke) is not bool or next(model.parameters()).dtype != torch.float32:
        raise ValueError("shape checkpoints require a boolean smoke flag and float32 model")
    device = next(model.parameters()).device
    with preserve_device_rng(device) as device_rng:
        finish_step(device)
        payload = {
            "kind": "matched_size_color",
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


def load_matched_checkpoint(path, *, device="cpu"):
    """Load frozen weights on CPU/CUDA without restoring training randomness."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "matched_size_color"
        or type(payload.get("format_version")) is not int
        or payload["format_version"] != 1
        or payload.get("tokenizer") != tokenizer_metadata()
        or type(payload.get("smoke")) is not bool
    ):
        raise ValueError("incompatible matched-size checkpoint")
    _require_cpu(payload)
    manifest = parse_matched_size(payload["manifest_content"])
    if manifest.sha256 != payload["manifest_sha256"]:
        raise ValueError("checkpoint manifest SHA256 mismatch")
    if not payload["smoke"]:
        from multimodal_loop.eval.size_reference import REFERENCE_MANIFEST

        if manifest.source.sha256 != REFERENCE_MANIFEST:
            raise ValueError("research checkpoint requires the fixed source corpus")
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
