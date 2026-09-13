"""Distinct frozen convolutional-stem checkpoints and enforced research protocol; no resume."""

from dataclasses import asdict
from pathlib import Path

import torch

from multimodal_loop.data.multi_arrangement import parse_multi_arrangement
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.conv_stem import ConvStemConfig, ConvStemEmbedding, build_conv_stem_model
from multimodal_loop.train.checkpoint import _atomic_save, _capture_rng, _cpu_snapshot
from multimodal_loop.train.runtime import preserve_device_rng, resolve_device
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig
from multimodal_loop.train.shape_grounding_checkpoint import _validate_progress, tokenizer_metadata
from multimodal_loop.train.synthetic import SyntheticTrainingConfig

MANIFEST_SHA256 = "ce16b936ce4ffcecb231794d9541a3c9d39461fef861c9e02349f9e4d68de8c7"
PROGRESS = ("completed_steps", "examples_seen", "completed_passes", "batches_in_pass")


def validate_conv_stem_protocol(manifest, model, stem, training, budget, *, smoke=False):
    if type(smoke) is not bool:
        raise ValueError("smoke must be boolean")
    stem.validate_model(model)
    if model.vocab_size != 17 or model.image_size != manifest.config.image_size:
        raise ValueError("model disagrees with direct-grounding corpus")
    if not smoke and (
        manifest.sha256 != MANIFEST_SHA256
        or model != ModelConfig(vocab_size=17)
        or stem != ConvStemConfig()
        or training != SyntheticTrainingConfig()
        or budget != ShapeGroundingConfig(8640, 216)
    ):
        raise ValueError("research settings disagree with fixed convolutional-stem protocol")


def save_conv_stem_checkpoint(
    path, model, optimizer, *, manifest, training, budget, history, smoke=False
):
    image = model.embeddings.image_embedding
    if not isinstance(image, ConvStemEmbedding):
        raise ValueError("checkpoint requires a convolutional image stem")
    validate_conv_stem_protocol(
        manifest, model.config, image.stem_config, training, budget, smoke=smoke
    )
    _validate_progress(manifest, training, budget, history, evaluation_name="training_fit")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("convolutional-stem requires float32 weights")
    with preserve_device_rng(next(model.parameters()).device) as device_rng:
        payload = {
            "kind": "conv_stem_direct_shape_color",
            "format_version": 1,
            "smoke": smoke,
            "config": asdict(model.config),
            "stem_config": asdict(image.stem_config),
            "model_dtype": "float32",
            "model_state": _cpu_snapshot(model.state_dict()),
            "optimizer_state": _cpu_snapshot(optimizer.state_dict()),
            "training_config": asdict(training),
            "budget": asdict(budget),
            "manifest_content": manifest.content,
            "manifest_sha256": manifest.sha256,
            "tokenizer": tokenizer_metadata(),
            "history": history,
            **{k: history[-1][k] for k in PROGRESS},
            "rng_state": _capture_rng(),
            "device_rng_state": device_rng,
        }
    _atomic_save(payload, Path(path))


def load_conv_stem_checkpoint(path, *, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != "conv_stem_direct_shape_color"
        or type(payload.get("format_version")) is not int
        or payload["format_version"] != 1
        or payload.get("model_dtype") != "float32"
        or payload.get("tokenizer") != tokenizer_metadata()
    ):
        raise ValueError("incompatible convolutional-stem checkpoint")
    manifest = parse_multi_arrangement(payload["manifest_content"])
    if manifest.sha256 != payload["manifest_sha256"]:
        raise ValueError("checkpoint manifest identity mismatch")
    config = ModelConfig(**payload["config"])
    stem = ConvStemConfig(**payload["stem_config"])
    training = SyntheticTrainingConfig(**payload["training_config"])
    budget = ShapeGroundingConfig(**payload["budget"])
    validate_conv_stem_protocol(manifest, config, stem, training, budget, smoke=payload["smoke"])
    _validate_progress(
        manifest, training, budget, payload["history"], evaluation_name="training_fit"
    )
    if any(payload[k] != payload["history"][-1][k] for k in PROGRESS):
        raise ValueError("checkpoint progress disagrees with history")
    device = resolve_device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("convolutional-stem experiment supports CPU and CUDA")
    with torch.random.fork_rng(devices=[]), preserve_device_rng(device), torch.device("cpu"):
        model = build_conv_stem_model(config, stem).to(dtype=torch.float32)
        if any(v.dtype != torch.float32 for v in payload["model_state"].values()):
            raise ValueError("convolutional-stem requires float32 state")
        model.load_state_dict(payload["model_state"], strict=True)
        model.to(device).eval()
    return model, manifest, training, payload
