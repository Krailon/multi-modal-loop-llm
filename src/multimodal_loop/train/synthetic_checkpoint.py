"""Self-contained epoch-boundary checkpoints for synthetic dataset training."""

from dataclasses import asdict, dataclass, fields
from math import ceil, isfinite
from pathlib import Path

import torch
from torch.optim import AdamW

from multimodal_loop.data.collator import SyntheticColorCollator
from multimodal_loop.data.multimodal import SyntheticManifest, parse_synthetic_manifest
from multimodal_loop.data.text import ColorQuestionTokenizer
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.checkpoint import (
    _atomic_save,
    _capture_rng,
    _require_cpu,
    _restore_model,
    _restore_rng,
    _snapshot_model_optimizer,
)
from multimodal_loop.train.runtime import finish_step, preserve_device_rng, resolve_device
from multimodal_loop.train.synthetic import SyntheticTrainingConfig, _integer


@dataclass
class LoadedSyntheticCheckpoint:
    model: MultimodalLoopTransformer
    optimizer: AdamW
    manifest: SyntheticManifest
    training_config: SyntheticTrainingConfig
    completed_epochs: int
    completed_steps: int
    history: list[dict]


def tokenizer_metadata() -> dict:
    return {
        "version": ColorQuestionTokenizer.version,
        "vocabulary": list(ColorQuestionTokenizer.vocabulary),
    }


def validate_synthetic_config(model_config: ModelConfig, manifest: SyntheticManifest) -> None:
    SyntheticColorCollator(model_config)
    if model_config.vocab_size != ColorQuestionTokenizer.vocab_size:
        raise ValueError("synthetic training requires vocab_size=10")
    if model_config.image_size != manifest.config.image_size:
        raise ValueError("manifest and model image_size must match")


def _validate_progress(
    manifest: SyntheticManifest,
    config: SyntheticTrainingConfig,
    completed_epochs: int,
    completed_steps: int,
    history: list[dict],
) -> None:
    _integer("completed_epochs", completed_epochs, 0)
    _integer("completed_steps", completed_steps, 0)
    steps_per_epoch = ceil(manifest.config.train_size / config.batch_size)
    if completed_steps != completed_epochs * steps_per_epoch:
        raise ValueError("completed_steps must describe complete training epochs")
    if not isinstance(history, list) or len(history) != completed_epochs + 1:
        raise ValueError("history must contain epoch zero and every completed epoch")
    for epoch, record in enumerate(history):
        if not isinstance(record, dict) or set(record) != {
            "epoch",
            "completed_steps",
            "train",
            "validation",
        }:
            raise ValueError("invalid history record fields")
        _integer("history epoch", record["epoch"], 0)
        _integer("history steps", record["completed_steps"], 0)
        if record["epoch"] != epoch or record["completed_steps"] != epoch * steps_per_epoch:
            raise ValueError("history epoch or step count disagrees with progress")
        validation = record["validation"]
        if not isinstance(validation, dict) or set(validation) != {
            "total",
            "correct",
            "accuracy",
            "invalid_predictions",
            "loss",
        }:
            raise ValueError("invalid validation metrics")
        for name in ("total", "correct", "invalid_predictions"):
            _integer(name, validation[name], 0)
        total = validation["total"]
        if (
            total != manifest.config.validation_size
            or validation["correct"] + validation["invalid_predictions"] > total
        ):
            raise ValueError("validation counts disagree with corpus")
        if validation["accuracy"] != validation["correct"] / total:
            raise ValueError("validation accuracy disagrees with counts")
        losses = [validation["loss"]]
        training = record["train"]
        if epoch == 0:
            if training is not None:
                raise ValueError("epoch zero must not contain training metrics")
        else:
            if not isinstance(training, dict) or set(training) != {"loss", "examples", "steps"}:
                raise ValueError("invalid training metrics")
            _integer("training examples", training["examples"], 1)
            _integer("training steps", training["steps"], 1)
            if (
                training["examples"] != manifest.config.train_size
                or training["steps"] != steps_per_epoch
            ):
                raise ValueError("training metrics disagree with corpus")
            losses.append(training["loss"])
        if any(
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not isfinite(loss)
            or loss < 0
            for loss in losses
        ):
            raise ValueError("metric losses must be finite and nonnegative")


def save_synthetic_checkpoint(
    path: str | Path,
    model: MultimodalLoopTransformer,
    optimizer: AdamW,
    *,
    manifest: SyntheticManifest,
    training_config: SyntheticTrainingConfig,
    completed_epochs: int,
    completed_steps: int,
    history: list[dict],
) -> None:
    """Atomically save last epoch, including corpus, metrics and random streams."""
    manifest = parse_synthetic_manifest(manifest.content)
    validate_synthetic_config(model.config, manifest)
    _validate_progress(manifest, training_config, completed_epochs, completed_steps, history)
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("synthetic checkpoints require float32")
    device = resolve_device(next(model.parameters()).device)
    with preserve_device_rng(device) as device_rng:
        finish_step(device)
        payload = {
            "kind": "synthetic_color",
            "format_version": 1,
            **_snapshot_model_optimizer(model, optimizer),
            "training_config": asdict(training_config),
            "tokenizer": tokenizer_metadata(),
            "manifest_content": manifest.content,
            "manifest_sha256": manifest.sha256,
            "completed_epochs": completed_epochs,
            "completed_steps": completed_steps,
            "history": history,
            "rng_state": _capture_rng(),
            "device_rng_state": device_rng,
        }
        _validate_optimizer_config(payload["optimizer_state"], training_config)
        finish_step(device)
    _atomic_save(payload, path)


def _validate_optimizer_config(state: dict, config: SyntheticTrainingConfig) -> None:
    group = state["param_groups"][0]
    if group["lr"] != config.learning_rate or group["weight_decay"] != config.weight_decay:
        raise ValueError("optimizer settings disagree with training_config")


def load_synthetic_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> LoadedSyntheticCheckpoint:
    """Resume on the saved backend; restore RNGs only after complete validation.

    The embedded manifest and history are authoritative, even when external
    manifest/metrics files have disappeared or lag behind the last checkpoint.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "synthetic_color"
        or type(payload.get("format_version")) is not int
        or payload["format_version"] != 1
    ):
        raise ValueError("unsupported synthetic checkpoint format")
    required = {
        "backend",
        "runtime",
        "config",
        "model_dtype",
        "model_training",
        "model_state",
        "optimizer_type",
        "optimizer_state",
        "training_config",
        "tokenizer",
        "manifest_content",
        "manifest_sha256",
        "completed_epochs",
        "completed_steps",
        "history",
        "rng_state",
        "device_rng_state",
    }
    if not required <= payload.keys():
        raise ValueError(
            f"synthetic checkpoint is missing fields: {sorted(required - payload.keys())}"
        )
    _require_cpu(payload)
    if (
        payload["backend"] not in ("cpu", "cuda", "xla")
        or torch.device(device).type != payload["backend"]
    ):
        raise ValueError("checkpoint backend does not match requested device")
    metadata = payload["runtime"]
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and (value is None or isinstance(value, str))
        for key, value in metadata.items()
    ):
        raise ValueError("checkpoint runtime metadata must contain strings or None")
    if payload["tokenizer"] != tokenizer_metadata():
        raise ValueError("checkpoint tokenizer is incompatible")
    manifest = parse_synthetic_manifest(payload["manifest_content"])
    if manifest.sha256 != payload["manifest_sha256"]:
        raise ValueError("checkpoint manifest SHA256 mismatch")
    settings = payload["training_config"]
    if not isinstance(settings, dict) or set(settings) != {
        field.name for field in fields(SyntheticTrainingConfig)
    }:
        raise ValueError("checkpoint requires all training config fields")
    config = SyntheticTrainingConfig(**settings)
    _validate_progress(
        manifest,
        config,
        payload["completed_epochs"],
        payload["completed_steps"],
        payload["history"],
    )
    if payload["model_dtype"] != "float32":
        raise ValueError("synthetic checkpoints require float32")
    device = resolve_device(device)
    model = _restore_model(payload, device)
    validate_synthetic_config(model.config, manifest)
    _validate_optimizer_config(payload["optimizer_state"], config)
    with preserve_device_rng(device):
        model.to(device)
        optimizer = AdamW(model.parameters())
        optimizer.load_state_dict(payload["optimizer_state"])
        finish_step(device)
    result = LoadedSyntheticCheckpoint(
        model,
        optimizer,
        manifest,
        config,
        payload["completed_epochs"],
        payload["completed_steps"],
        payload["history"],
    )
    _restore_rng(payload["rng_state"], device, payload["device_rng_state"])
    return result
