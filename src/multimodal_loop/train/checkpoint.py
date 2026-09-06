"""Self-contained checkpoints for deterministic fixed-batch CPU training."""

import os
import random
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW

from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer

_FORMAT_VERSION = 1
_DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass
class LoadedCheckpoint:
    """Restored training objects and the settings needed for the next update."""

    model: MultimodalLoopTransformer
    optimizer: AdamW
    completed_steps: int
    input_ids: Tensor
    images: Tensor | None
    question_length: int
    recurrence_depth: int
    seed: int | None


def _require_cpu(value: Any) -> None:
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise ValueError("checkpoint version 1 supports CPU tensors only")
    elif isinstance(value, dict):
        for item in value.values():
            _require_cpu(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _require_cpu(item)


def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _validate_training_state(
    model: MultimodalLoopTransformer,
    input_ids: Tensor,
    images: Tensor | None,
    completed_steps: int,
    question_length: int,
    recurrence_depth: int,
    seed: int | None,
) -> None:
    _integer("completed_steps", completed_steps, 0)
    _integer("question_length", question_length, 0)
    _integer("recurrence_depth", recurrence_depth, 1)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
    _require_cpu((input_ids, images))
    if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] < 2:
        raise ValueError("checkpoint batch must be [B, T] with B > 0 and T >= 2")
    if question_length >= input_ids.shape[1]:
        raise ValueError("question_length must leave at least one answer target")
    if (
        images is not None
        and images.dtype != model.embeddings.image_embedding.projection.weight.dtype
    ):
        raise ValueError("checkpoint images must have the model dtype")
    # Embeddings validate IDs, image shapes, and the combined capacity without
    # consuming randomness or running any transformer/dropout computation.
    with torch.no_grad():
        model.embeddings(input_ids, images)


def _validate_optimizer_state(state: dict, parameters: list[Tensor]) -> None:
    groups = state["param_groups"]
    if len(groups) != 1 or groups[0]["params"] != list(range(len(parameters))):
        raise ValueError("checkpoint requires one AdamW group in model parameter order")
    _require_cpu(state)
    for index, values in state["state"].items():
        if not isinstance(index, int) or not 0 <= index < len(parameters):
            raise ValueError("optimizer state references an unknown parameter")
        if not values:
            continue
        names = {"step", "exp_avg", "exp_avg_sq"}
        if groups[0]["amsgrad"]:
            names.add("max_exp_avg_sq")
        if set(values) != names:
            raise ValueError("checkpoint has incompatible AdamW state fields")
        step = values["step"]
        if not isinstance(step, Tensor) or step.ndim != 0 or not torch.isfinite(step) or step < 0:
            raise ValueError("AdamW step must be a finite nonnegative scalar tensor")
        for name in names - {"step"}:
            moment = values[name]
            if (
                not isinstance(moment, Tensor)
                or moment.shape != parameters[index].shape
                or moment.dtype != parameters[index].dtype
            ):
                raise ValueError(f"AdamW {name} must match its parameter shape and dtype")


def _capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
    }


def _restore_rng(state: dict) -> None:
    numpy_state = state["numpy"]
    numpy_native = (
        numpy_state["algorithm"],
        np.array(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    )
    # Validate all three states on local generators before changing global RNGs.
    random.Random(0).setstate(state["python"])
    np.random.RandomState(0).set_state(numpy_native)
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    random.setstate(state["python"])
    np.random.set_state(numpy_native)
    torch.set_rng_state(state["torch_cpu"])


def save_checkpoint(
    path: str | Path,
    model: MultimodalLoopTransformer,
    optimizer: AdamW,
    *,
    completed_steps: int,
    input_ids: Tensor,
    images: Tensor | None = None,
    question_length: int = 0,
    recurrence_depth: int | None = None,
    seed: int | None = None,
) -> None:
    """Atomically save a completed optimizer-step boundary, including its batch.

    Version 1 supports CPU float32/float64 models with all parameters trainable,
    a uniform train/eval mode, and one AdamW group in model parameter order.
    The caller supplies the total completed step count; zero supports an initial
    checkpoint. Gradients are omitted because the next training step clears them.

    RNG snapshots cover Python, NumPy's global generator, and PyTorch CPU. Saving
    does not advance them. Only tensors and ordinary Python values are serialized.
    The parent directory is created if needed; a failed write preserves an
    existing destination. Call synchronously between optimizer updates.
    """
    parameters = list(model.parameters())
    model_state = model.state_dict()
    _require_cpu(model_state)
    dtype = parameters[0].dtype
    if dtype not in _DTYPES.values() or any(p.dtype != dtype for p in parameters):
        raise ValueError("checkpoint model must have uniform float32 or float64 dtype")
    if any(not p.requires_grad for p in parameters):
        raise ValueError("checkpoint version 1 requires all model parameters to be trainable")
    if any(module.training != model.training for module in model.modules()):
        raise ValueError("checkpoint version 1 requires a uniform train/eval mode")
    if type(optimizer) is not AdamW:
        raise TypeError("checkpoint version 1 supports AdamW only")
    if len(optimizer.param_groups) != 1 or [id(p) for p in optimizer.param_groups[0]["params"]] != [
        id(p) for p in parameters
    ]:
        raise ValueError("checkpoint requires one AdamW group in model parameter order")
    optimizer_state = optimizer.state_dict()
    _validate_optimizer_state(optimizer_state, parameters)
    depth = model.config.recurrence_depth if recurrence_depth is None else recurrence_depth
    _validate_training_state(
        model, input_ids, images, completed_steps, question_length, depth, seed
    )
    payload = {
        "format_version": _FORMAT_VERSION,
        "config": asdict(model.config),
        "model_dtype": str(dtype).removeprefix("torch."),
        "model_training": model.training,
        "model_state": model_state,
        "optimizer_type": "AdamW",
        "optimizer_state": optimizer_state,
        "completed_steps": completed_steps,
        "input_ids": input_ids.detach().clone(),
        "images": None if images is None else images.detach().clone(),
        "question_length": question_length,
        "recurrence_depth": depth,
        "seed": seed,
        "rng_state": _capture_rng(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Rebuild a CPU training run and restore its RNG states as the final step.

    Uses restricted ``weights_only=True`` loading, strict model-state loading,
    and the saved dtype before copying weights. Exact continuation requires the
    same CPU software/hardware execution environment. CUDA/XLA RNG, independent
    generators, schedulers, and data-loader state are outside version 1.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if (
        type(payload.get("format_version")) is not int
        or payload["format_version"] != _FORMAT_VERSION
    ):
        raise ValueError("unsupported checkpoint format_version")
    required = {
        "config",
        "model_dtype",
        "model_training",
        "model_state",
        "optimizer_type",
        "optimizer_state",
        "completed_steps",
        "input_ids",
        "images",
        "question_length",
        "recurrence_depth",
        "seed",
        "rng_state",
    }
    if not required <= payload.keys():
        raise ValueError(f"checkpoint is missing fields: {sorted(required - payload.keys())}")
    if payload["model_dtype"] not in _DTYPES:
        raise ValueError("checkpoint model_dtype must be float32 or float64")
    if type(payload["model_training"]) is not bool:
        raise ValueError("checkpoint model_training must be boolean")
    if payload["optimizer_type"] != "AdamW":
        raise ValueError("checkpoint version 1 supports AdamW only")
    if not isinstance(payload["config"], dict) or set(payload["config"]) != {
        field.name for field in fields(ModelConfig)
    }:
        raise ValueError("checkpoint config must contain all ModelConfig fields")
    config = ModelConfig(**payload["config"])
    dtype = _DTYPES[payload["model_dtype"]]
    # Construction must neither consume the restored stream nor disturb the
    # caller's torch stream if validation fails partway through reconstruction.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = MultimodalLoopTransformer(config).to(dtype=dtype)
        if any(
            not isinstance(value, Tensor) or value.dtype != dtype
            for value in payload["model_state"].values()
        ):
            raise ValueError("model state tensors must match model_dtype")
        model.load_state_dict(payload["model_state"], strict=True)
        model.train(payload["model_training"])
        _validate_optimizer_state(payload["optimizer_state"], list(model.parameters()))
        optimizer = AdamW(model.parameters())
        optimizer.load_state_dict(payload["optimizer_state"])
        _validate_training_state(
            model,
            payload["input_ids"],
            payload["images"],
            payload["completed_steps"],
            payload["question_length"],
            payload["recurrence_depth"],
            payload["seed"],
        )
    result = LoadedCheckpoint(
        model=model,
        optimizer=optimizer,
        completed_steps=payload["completed_steps"],
        input_ids=payload["input_ids"],
        images=payload["images"],
        question_length=payload["question_length"],
        recurrence_depth=payload["recurrence_depth"],
        seed=payload["seed"],
    )
    _restore_rng(payload["rng_state"])
    return result
