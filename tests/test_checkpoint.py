"""Disk checkpoint round trips and exact CPU training continuation."""

import random
import subprocess
import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.checkpoint import load_checkpoint, save_checkpoint
from multimodal_loop.train.trainer import train_on_batch


def rng_state() -> tuple:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def restore_rng(state: tuple) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def assert_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_equal(left, right)
    else:
        assert actual == expected


def random_draws() -> tuple:
    return (
        random.random(),
        random.gauss(0, 1),
        np.random.rand(3),
        np.random.normal(size=3),
        torch.rand(5),
    )


@pytest.fixture(autouse=True)
def seeded_rngs() -> Iterator[None]:
    original = rng_state()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    try:
        yield
    finally:
        restore_rng(original)


@pytest.fixture
def model() -> MultimodalLoopTransformer:
    return MultimodalLoopTransformer(
        ModelConfig(
            vocab_size=16,
            d_model=8,
            n_heads=2,
            d_ff=16,
            image_size=4,
            patch_size=2,
            n_recurrent_layers=2,
            dropout=0.25,
        )
    )


def adamw(model: MultimodalLoopTransformer) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=0.01, betas=(0.8, 0.95), eps=1e-6, weight_decay=0.03, amsgrad=True
    )


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("training", [False, True])
def test_round_trip_preserves_training_objects_and_logits(
    tmp_path: Path,
    model: MultimodalLoopTransformer,
    with_images: bool,
    dtype: torch.dtype,
    training: bool,
) -> None:
    model.to(dtype=dtype)
    optimizer = adamw(model)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32)
    images = torch.randn(1, 3, 4, 4, dtype=dtype) if with_images else None
    train_on_batch(
        model, optimizer, input_ids, images, steps=2, question_length=2, recurrence_depth=3
    )
    model.train(training)
    path = tmp_path / "nested" / "state.pt"
    original_rng = rng_state()
    save_checkpoint(
        path,
        model,
        optimizer,
        completed_steps=2,
        input_ids=input_ids,
        images=images,
        question_length=2,
        recurrence_depth=3,
        seed=7,
    )
    assert_equal(rng_state(), original_rng)
    expected_model = deepcopy(model.state_dict())
    expected_optimizer = deepcopy(optimizer.state_dict())
    expected_ids = input_ids.clone()
    expected_images = None if images is None else images.clone()
    with torch.no_grad():
        expected_logits = model.eval()(input_ids, images, recurrence_depth=3)
        model.lm_head.weight.zero_()
        input_ids.fill_(10)
        if images is not None:
            images.add_(100)
    random_draws()

    restored = load_checkpoint(path)

    assert restored.model.config == model.config
    assert restored.model.training is training
    assert all(module.training is training for module in restored.model.modules())
    assert restored.completed_steps == 2
    assert restored.question_length == 2
    assert restored.recurrence_depth == 3
    assert restored.model.config.recurrence_depth == 2
    assert restored.seed == 7
    assert_equal(restored.input_ids, expected_ids)
    assert_equal(restored.images, expected_images)
    assert_equal(restored.model.state_dict(), expected_model)
    assert_equal(restored.optimizer.state_dict(), expected_optimizer)
    assert_equal(rng_state(), original_rng)
    assert all(p.device.type == "cpu" and p.dtype == dtype for p in restored.model.parameters())
    assert all(p.grad is None for p in restored.model.parameters())
    assert all(
        left is right
        for left, right in zip(
            restored.model.parameters(), restored.optimizer.param_groups[0]["params"], strict=True
        )
    )
    with torch.no_grad():
        actual_logits = restored.model.eval()(
            restored.input_ids, restored.images, recurrence_depth=3
        )
    assert_equal(actual_logits, expected_logits)


@pytest.mark.parametrize("with_images", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("depth", [1, 3])
def test_disk_resume_exactly_matches_uninterrupted_dropout_training(
    tmp_path: Path,
    model: MultimodalLoopTransformer,
    with_images: bool,
    dtype: torch.dtype,
    depth: int,
) -> None:
    model.to(dtype=dtype)
    reference = deepcopy(model)
    optimizer, reference_optimizer = adamw(model), adamw(reference)
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    images = torch.randn(2, 3, 4, 4, dtype=dtype) if with_images else None
    question_length = 2 if with_images else 0
    initial_rng = rng_state()
    expected_losses = train_on_batch(
        reference,
        reference_optimizer,
        input_ids,
        images,
        steps=6,
        question_length=question_length,
        recurrence_depth=depth,
    )
    expected_draws = random_draws()

    restore_rng(initial_rng)
    first_losses = train_on_batch(
        model,
        optimizer,
        input_ids,
        images,
        steps=2,
        question_length=question_length,
        recurrence_depth=depth,
    )
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        completed_steps=2,
        input_ids=input_ids,
        images=images,
        question_length=question_length,
        recurrence_depth=depth,
    )
    # Simulate unrelated work and fresh object initialization between runs.
    random_draws()
    MultimodalLoopTransformer(model.config)
    restored = load_checkpoint(path)
    later_losses = train_on_batch(
        restored.model,
        restored.optimizer,
        restored.input_ids,
        restored.images,
        steps=4,
        question_length=restored.question_length,
        recurrence_depth=restored.recurrence_depth,
    )

    assert first_losses + later_losses == expected_losses
    assert_equal(restored.model.state_dict(), reference.state_dict())
    assert_equal(restored.optimizer.state_dict(), reference_optimizer.state_dict())
    assert_equal(random_draws(), expected_draws)


def test_initial_checkpoint_restores_cached_gaussians_and_default_depth(
    tmp_path: Path, model: MultimodalLoopTransformer
) -> None:
    random.gauss(0, 1)
    np.random.normal()
    path = tmp_path / "initial.pt"
    save_checkpoint(
        path, model, adamw(model), completed_steps=0, input_ids=torch.tensor([[0, 1, 2]])
    )
    expected = random_draws()
    random_draws()

    restored = load_checkpoint(path)

    assert restored.completed_steps == 0
    assert not restored.optimizer.state
    assert restored.recurrence_depth == model.config.recurrence_depth
    assert restored.seed is None
    assert_equal(random_draws(), expected)


@pytest.mark.parametrize("stage", ["write", "replace"])
def test_failed_save_preserves_destination_and_cleans_temporary_file(
    tmp_path: Path, model: MultimodalLoopTransformer, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    path = tmp_path / "checkpoint.pt"
    optimizer = adamw(model)
    input_ids = torch.tensor([[0, 1, 2]])
    save_checkpoint(path, model, optimizer, completed_steps=0, input_ids=input_ids)
    original = path.read_bytes()
    before_rng = rng_state()

    def fail_write(payload, stream):
        stream.write(b"partial checkpoint")
        raise OSError("simulated write failure")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    if stage == "write":
        monkeypatch.setattr("multimodal_loop.train.checkpoint.torch.save", fail_write)
    else:
        monkeypatch.setattr("multimodal_loop.train.checkpoint.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        save_checkpoint(path, model, optimizer, completed_steps=0, input_ids=input_ids)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    assert_equal(rng_state(), before_rng)


@pytest.mark.parametrize(
    "change, error, message",
    [
        (lambda p: p.update(format_version=99), ValueError, "format_version"),
        (lambda p: p.pop("rng_state"), ValueError, "missing fields"),
        (lambda p: p["config"].pop("dropout"), ValueError, "all ModelConfig fields"),
        (lambda p: p["config"].update(d_model=7), ValueError, "divisible"),
        (lambda p: p.update(model_dtype="float16"), ValueError, "model_dtype"),
        (lambda p: p.update(model_training=1), ValueError, "boolean"),
        (lambda p: p.update(optimizer_type="SGD"), ValueError, "AdamW only"),
        (lambda p: p.update(completed_steps=-1), ValueError, "completed_steps"),
        (lambda p: p.update(question_length=3), ValueError, "answer target"),
        (lambda p: p.update(recurrence_depth=0), ValueError, "recurrence_depth"),
        (lambda p: p["model_state"].pop("lm_head.weight"), RuntimeError, "Missing key"),
        (
            lambda p: p["model_state"].update({"lm_head.weight": torch.zeros(1)}),
            RuntimeError,
            "size mismatch",
        ),
        (
            lambda p: p["model_state"].update(
                {"lm_head.weight": p["model_state"]["lm_head.weight"].double()}
            ),
            ValueError,
            "model_dtype",
        ),
        (
            lambda p: p["optimizer_state"]["state"][0].update(exp_avg=torch.zeros(1)),
            ValueError,
            "shape and dtype",
        ),
        (
            lambda p: p["rng_state"].update(torch_cpu=torch.zeros(2, dtype=torch.int64)),
            TypeError,
            "ByteTensor",
        ),
    ],
)
def test_incompatible_checkpoint_is_rejected(
    tmp_path: Path, model: MultimodalLoopTransformer, change, error: type[Exception], message: str
) -> None:
    path = tmp_path / "invalid.pt"
    optimizer = adamw(model)
    input_ids = torch.tensor([[0, 1, 2]])
    train_on_batch(model, optimizer, input_ids, steps=1)
    save_checkpoint(path, model, optimizer, completed_steps=1, input_ids=input_ids)
    payload = torch.load(path, weights_only=True)
    change(payload)
    torch.save(payload, path)
    original_rng = rng_state()
    with pytest.raises(error, match=message):
        load_checkpoint(path)
    assert_equal(rng_state(), original_rng)


@pytest.mark.parametrize(
    "unsupported", ["device", "dtype", "optimizer", "order", "groups", "batch_device"]
)
def test_unsupported_save_configuration_is_rejected(
    tmp_path: Path, model: MultimodalLoopTransformer, unsupported: str
) -> None:
    if unsupported == "device":
        model.to("meta")
    if unsupported == "dtype":
        model.half()
    optimizer = adamw(model)
    if unsupported == "optimizer":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if unsupported == "order":
        optimizer = torch.optim.AdamW(list(model.parameters())[::-1])
    if unsupported == "groups":
        parameters = list(model.parameters())
        optimizer = torch.optim.AdamW([{"params": parameters[:1]}, {"params": parameters[1:]}])
    input_ids = torch.zeros(
        1, 3, dtype=torch.long, device="meta" if unsupported == "batch_device" else "cpu"
    )
    path = tmp_path / "unsupported.pt"
    with pytest.raises((TypeError, ValueError), match="CPU|dtype|AdamW"):
        save_checkpoint(path, model, optimizer, completed_steps=0, input_ids=input_ids)
    assert not path.exists()


def run_script(*arguments: str) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    return subprocess.run(
        [sys.executable, str(script), *arguments], capture_output=True, text=True, timeout=60
    )


@pytest.mark.parametrize("with_images", [False, True])
def test_separate_process_resume_matches_uninterrupted_run(
    tmp_path: Path, model: MultimodalLoopTransformer, with_images: bool
) -> None:
    initial, middle, whole, split = [
        tmp_path / f"{name}.pt" for name in ("initial", "middle", "whole", "split")
    ]
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])
    images = torch.randn(1, 3, 4, 4) if with_images else None
    save_checkpoint(
        initial,
        model,
        adamw(model),
        completed_steps=0,
        input_ids=input_ids,
        images=images,
        question_length=2 if with_images else 0,
        recurrence_depth=3,
        seed=7,
    )
    original_file = initial.read_bytes()
    complete = run_script("--resume", str(initial), "--steps", "6", "--save-checkpoint", str(whole))
    first = run_script("--resume", str(initial), "--steps", "2", "--save-checkpoint", str(middle))
    second = run_script("--resume", str(middle), "--steps", "4", "--save-checkpoint", str(split))
    for result in (complete, first, second):
        assert result.returncode == 0, result.stderr
    expected_lines = [line for line in complete.stdout.splitlines() if line.startswith("step=")]
    actual_lines = [
        line for line in (first.stdout + second.stdout).splitlines() if line.startswith("step=")
    ]
    assert expected_lines == actual_lines
    assert actual_lines[0].startswith("step=001")
    assert actual_lines[-1].startswith("step=006")
    assert "recurrence_depth=3" in second.stdout
    assert initial.read_bytes() == original_file
    assert_equal(torch.load(split, weights_only=True), torch.load(whole, weights_only=True))
    assert load_checkpoint(split).completed_steps == 6


def test_script_can_save_a_new_run_and_resume_without_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "fresh.pt"
    first = run_script("--text-only", "--steps", "1", "--save-checkpoint", str(path))
    assert first.returncode == 0, first.stderr
    original_file = path.read_bytes()
    resumed = run_script("--resume", str(path), "--steps", "1")
    assert resumed.returncode == 0, resumed.stderr
    assert "step=002" in resumed.stdout
    assert path.read_bytes() == original_file


@pytest.mark.parametrize(
    "arguments, message",
    [
        (
            (
                "--resume",
                "missing.pt",
                "--seed",
                "7",
                "--learning-rate",
                "0.1",
                "--text-only",
                "--recurrence-depth",
                "3",
            ),
            "new run settings",
        ),
        (("--resume", "missing.pt", "--device", "meta"), "supported devices"),
        (("--save-checkpoint", "unused.pt", "--device", "meta"), "supported devices"),
    ],
)
def test_script_rejects_resume_overrides_and_unsupported_devices(
    arguments: tuple[str, ...], message: str
) -> None:
    result = run_script(*arguments)
    assert result.returncode == 2
    assert message in result.stderr


@pytest.mark.parametrize("version", [1, 2])
def test_checkpoint_versions_load_on_cpu(tmp_path, model, version):
    path = tmp_path / "version.pt"
    save_checkpoint(
        path, model, adamw(model), completed_steps=0, input_ids=torch.tensor([[0, 1, 2]])
    )
    payload = torch.load(path, weights_only=True)
    assert payload["format_version"] == 2
    assert payload["backend"] == "cpu"
    assert payload["device_rng_state"] is None
    assert isinstance(payload["runtime"]["torch"], str)
    if version == 1:
        payload["format_version"] = 1
        for key in ("backend", "device_rng_state", "runtime"):
            del payload[key]
        torch.save(payload, path)
    restored = load_checkpoint(path, device="cpu")
    assert_equal(restored.model.state_dict(), model.state_dict())
    with pytest.raises(ValueError, match="backend"):
        load_checkpoint(path, device="cuda")


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda p: p.pop("backend"), "missing fields"),
        (lambda p: p.update(backend="meta"), "backend"),
        (lambda p: p.update(device_rng_state=3), "accelerator RNG"),
        (lambda p: p.update(runtime={"torch": 2}), "runtime metadata"),
    ],
)
def test_invalid_runtime_metadata_preserves_rng(tmp_path, model, change, message):
    path = tmp_path / "invalid_runtime.pt"
    save_checkpoint(
        path, model, adamw(model), completed_steps=0, input_ids=torch.tensor([[0, 1, 2]])
    )
    payload = torch.load(path, weights_only=True)
    change(payload)
    torch.save(payload, path)
    original = rng_state()
    with pytest.raises(ValueError, match=message):
        load_checkpoint(path)
    assert_equal(rng_state(), original)
