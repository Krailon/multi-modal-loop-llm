"""Runtime selection and optional dependencies, testable on CPU-only hosts."""

import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from multimodal_loop.train import runtime


def test_cpu_imports_and_execution_do_not_import_xla() -> None:
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith("torch_xla"):
        raise AssertionError("unexpected XLA import")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from multimodal_loop.train import checkpoint, trainer, runtime
assert str(runtime.resolve_device()) == "cpu"
runtime.finish_step(runtime.resolve_device())
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_cpu_seed_replays_all_host_generators() -> None:
    original = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        device = runtime.resolve_device("cpu:0")
        runtime.seed_everything(42, device)
        expected = random.random(), np.random.rand(), torch.rand(5)
        runtime.seed_everything(42, device)
        assert random.random() == expected[0]
        assert np.random.rand() == expected[1]
        torch.testing.assert_close(torch.rand(5), expected[2], rtol=0, atol=0)
    finally:
        random.setstate(original[0])
        np.random.set_state(original[1])
        torch.set_rng_state(original[2])


def test_unavailable_cuda_fails_without_fallback(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        runtime.resolve_device("cuda")


def test_cuda_index_selection(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert runtime.resolve_device("cuda") == torch.device("cuda:1")
    assert runtime.resolve_device("cuda:0") == torch.device("cuda:0")
    with pytest.raises(ValueError, match="index"):
        runtime.resolve_device("cuda:2")


@pytest.mark.parametrize("device", ["meta", "cpu:2", "xla:1"])
def test_unsupported_device_selection(device) -> None:
    with pytest.raises(ValueError):
        runtime.resolve_device(device)


def test_missing_xla_has_actionable_error(monkeypatch) -> None:
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(runtime.importlib, "import_module", missing)
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    with pytest.raises(RuntimeError, match="preinstalled packages"):
        runtime.resolve_device("xla")


@pytest.mark.parametrize(
    "spmd, processes, kind", [(True, 1, "TPU"), (False, 2, "TPU"), (False, 1, "CPU")]
)
def test_xla_rejects_other_execution_modes(monkeypatch, spmd, processes, kind) -> None:
    xla = SimpleNamespace(__version__=str(torch.__version__))
    xr = SimpleNamespace(
        is_spmd=lambda: spmd, process_count=lambda: processes, device_type=lambda: kind
    )
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda name: xla if name == "torch_xla" else xr
    )
    with pytest.raises(ValueError, match="TPU"):
        runtime.resolve_device("xla")


def test_xla_step_and_checkpoint_materialization_preserve_rng(monkeypatch) -> None:
    state = [19]
    waits = []

    def sync(*, wait):
        waits.append(wait)
        state[0] += 1

    xla = SimpleNamespace(__version__=str(torch.__version__), sync=sync)
    xm = SimpleNamespace(
        get_rng_state=lambda device: state[0],
        set_rng_state=lambda seed, device: state.__setitem__(0, seed),
    )
    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda name: xla if name == "torch_xla" else xm
    )
    device = torch.device("xla:0")
    runtime.finish_step(device)
    assert state[0] == 20
    with runtime.preserve_device_rng(device) as saved:
        runtime.finish_step(device)
        assert state[0] == 21
    assert saved == state[0] == 20
    assert waits == [True, True]


def test_mismatched_xla_version_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda name: SimpleNamespace(__version__="1.0.0")
    )
    with pytest.raises(RuntimeError, match="matching major/minor"):
        runtime._xla()
