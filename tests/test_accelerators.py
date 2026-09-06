"""Opt-in real-device checks; unavailable explicitly requested hardware fails."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

DEVICE = os.environ.get("MULTIMODAL_LOOP_TEST_DEVICE")
pytestmark = pytest.mark.skipif(
    DEVICE is None, reason="set MULTIMODAL_LOOP_TEST_DEVICE=cuda or xla"
)


def assert_tree_close(actual, expected, *, rtol, atol):
    if isinstance(expected, torch.Tensor):
        assert actual.device.type == expected.device.type == "cpu"
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_close(actual[key], expected[key], rtol=rtol, atol=atol)
    elif isinstance(expected, (tuple, list)):
        for left, right in zip(actual, expected, strict=True):
            assert_tree_close(left, right, rtol=rtol, atol=atol)
    else:
        assert actual == expected


@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("depth", [1, 3])
def test_hardware_training_and_fresh_process_resume(tmp_path, images, depth):
    assert torch.device(DEVICE).type in ("cpu", "cuda", "xla")
    worker = Path(__file__).with_name("accelerator_worker.py")

    def run(output, *arguments):
        result = subprocess.run(
            [sys.executable, str(worker), "--device", DEVICE, "--output", str(output), *arguments],
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    initial, whole, middle, split = [
        tmp_path / f"{name}.pt" for name in ("initial", "whole", "middle", "split")
    ]
    run(initial, "--depth", str(depth), *(["--images"] if images else []))
    run(whole, "--resume", str(initial), "--steps", "4")
    run(middle, "--resume", str(initial), "--steps", "2")
    run(split, "--resume", str(middle), "--steps", "2")
    complete = torch.load(whole, weights_only=True)
    resumed = torch.load(split, weights_only=True)
    rtol, atol = (1e-4, 1e-5) if torch.device(DEVICE).type == "xla" else (1e-5, 1e-6)
    assert complete["completed_steps"] == resumed["completed_steps"] == 4
    for key in ("model_state", "optimizer_state"):
        assert_tree_close(resumed[key], complete[key], rtol=rtol, atol=atol)
    for key in (
        "rng_state",
        "device_rng_state",
        "config",
        "input_ids",
        "images",
        "recurrence_depth",
        "question_length",
        "backend",
        "model_training",
    ):
        assert_tree_close(resumed[key], complete[key], rtol=0, atol=0)
    outcomes = [
        torch.load(path.with_suffix(".results.pt"), weights_only=True)
        for path in (whole, middle, split)
    ]
    torch.testing.assert_close(
        torch.tensor(outcomes[0]["losses"]),
        torch.tensor(outcomes[1]["losses"] + outcomes[2]["losses"]),
        rtol=rtol,
        atol=atol,
    )
    assert_tree_close(outcomes[2]["draws"], outcomes[0]["draws"], rtol=0, atol=0)
