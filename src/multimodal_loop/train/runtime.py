"""Small single-device runtime helpers; XLA is an optional, lazy import."""

import importlib
import os
import random
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch


def _xla():
    try:
        xla = importlib.import_module("torch_xla")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TPU execution requires compatible torch, torch_xla, and libtpu installations; "
            "use the Kaggle TPU runtime and its preinstalled packages."
        ) from error
    torch_version = str(torch.__version__).split(".")[:2]
    if xla.__version__.split(".")[:2] != torch_version:
        raise RuntimeError("torch and torch_xla must have matching major/minor versions")
    return xla


def resolve_device(requested: str | torch.device = "cpu") -> torch.device:
    """Resolve an explicit CPU, CUDA, or single TPU device without fallback."""
    device = torch.device(requested)
    if device.type == "cpu":
        if device.index not in (None, 0):
            raise ValueError("CPU device index must be zero")
        return torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        return torch.device("cuda", index)
    if device.type == "xla":
        if device.index not in (None, 0):
            raise ValueError("single-device TPU execution uses xla:0")
        if os.environ.get("PJRT_DEVICE", "TPU") != "TPU":
            raise ValueError("XLA execution requires PJRT_DEVICE=TPU")
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        xla = _xla()
        runtime = importlib.import_module("torch_xla.runtime")
        if runtime.is_spmd() or runtime.process_count() != 1:
            raise ValueError("only single-process TPU execution without SPMD is supported")
        if runtime.device_type() != "TPU":
            raise ValueError("XLA execution requires a TPU runtime")
        return xla.device(0)
    raise ValueError("supported devices are CPU, CUDA, and XLA TPU")


def seed_everything(seed: int, device: torch.device) -> None:
    """Seed global host generators and only the selected accelerator."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    # torch.manual_seed also seeds other backends; leave unused devices alone.
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)
    elif device.type == "xla":
        _xla()
        importlib.import_module("torch_xla.core.xla_model").set_rng_state(seed, str(device))


def finish_step(device: torch.device) -> None:
    """Commit an XLA optimizer update before host logging or checkpointing."""
    if device.type == "xla":
        _xla().sync(wait=True)


def capture_device_rng(device: torch.device) -> torch.Tensor | int | None:
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device).clone()
    if device.type == "xla":
        _xla()
        return importlib.import_module("torch_xla.core.xla_model").get_rng_state(str(device))
    return None


def validate_device_rng(state: torch.Tensor | int | None, device: torch.device) -> None:
    if device.type == "cpu":
        if state is not None:
            raise ValueError("CPU checkpoints must not contain accelerator RNG state")
    elif device.type == "cuda":
        # An independent generator validates the payload without changing globals.
        torch.Generator(device=device).set_state(state)
    elif type(state) is not int or not 0 <= state < 2**64:
        raise ValueError("XLA RNG state must be an unsigned 64-bit integer")


def restore_device_rng(state: torch.Tensor | int | None, device: torch.device) -> None:
    validate_device_rng(state, device)
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "xla":
        _xla()
        importlib.import_module("torch_xla.core.xla_model").set_rng_state(state, str(device))


@contextmanager
def preserve_device_rng(device: torch.device):
    """Keep checkpoint materialization from advancing the next random stream."""
    state = capture_device_rng(device)
    try:
        yield state
    finally:
        restore_device_rng(state, device)


def runtime_metadata(device: torch.device) -> dict[str, str | None]:
    result = {"torch": str(torch.__version__), "device": str(device)}
    if device.type == "cuda":
        result.update(cuda=torch.version.cuda, hardware=torch.cuda.get_device_name(device))
    elif device.type == "xla":
        result["torch_xla"] = str(_xla().__version__)
        try:
            result["libtpu"] = version("libtpu")
        except PackageNotFoundError:
            result["libtpu"] = None
    return result
