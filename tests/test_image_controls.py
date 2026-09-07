"""Pixel-only interventions, known-answer scoring and checkpoint evaluation CLI."""

import copy
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from multimodal_loop.data.multimodal import SyntheticColorDataset, parse_synthetic_manifest
from multimodal_loop.data.synthetic_shapes import (
    QUESTION,
    SyntheticShapesConfig,
    build_scene_splits,
)
from multimodal_loop.eval.synthetic import (
    _image_control_batches,
    evaluate_image_controls,
    evaluate_synthetic,
)
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.checkpoint import _capture_rng, _restore_rng
from multimodal_loop.train.synthetic import (
    SyntheticTrainingConfig,
    synthetic_loader,
    train_synthetic_epoch,
)
from multimodal_loop.train.synthetic_checkpoint import save_synthetic_checkpoint


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


@pytest.fixture(autouse=True)
def preserve_rng():
    state = _capture_rng()
    yield
    _restore_rng(state, torch.device("cpu"))


@pytest.fixture
def manifest():
    config = SyntheticShapesConfig(
        image_size=8, object_sizes=(4,), train_size=12, validation_size=8, test_size=4
    )
    return parse_synthetic_manifest(
        json.dumps(
            {
                "format_version": 1,
                "config": asdict(config),
                "splits": {
                    split: [
                        {"scene": asdict(scene), "question": QUESTION, "answer": scene.color}
                        for scene in scenes
                    ]
                    for split, scenes in build_scene_splits(config).items()
                },
            }
        )
    )


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=10,
        image_size=8,
        patch_size=4,
        d_model=8,
        n_heads=2,
        d_ff=16,
        max_seq_len=16,
        dropout=0.2,
    )


class PixelReader(nn.Module):
    """Perfect RGB oracle; black pixels produce the constant valid answer red."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, ids, images, *, recurrence_depth, attention_mask):
        assert not self.training and torch.is_inference_mode_enabled()
        assert torch.equal(ids, torch.arange(6).expand(ids.shape[0], -1))
        assert attention_mask.shape == (10, 10) and attention_mask.all()
        self.calls.append((recurrence_depth, ids.clone(), attention_mask.clone()))
        channels = images.sum(dim=(2, 3)) > 0
        predictions = torch.full((len(ids),), 6)
        predictions[channels[:, 1]] = 7
        predictions[channels[:, 2]] = 8
        predictions[channels[:, 0] & channels[:, 1]] = 9
        logits = torch.zeros(len(ids), 10)
        logits[torch.arange(len(ids)), predictions] = 5
        return logits[:, None, :].expand(-1, 10, -1)


@pytest.mark.parametrize("blank", [False, True])
def test_only_pixels_change_and_original_storage_is_preserved(manifest, config, blank):
    dataset = SyntheticColorDataset(manifest, "validation")
    batches = list(synthetic_loader(dataset, config, SyntheticTrainingConfig(batch_size=3)))
    original = [batch.images.clone() for batch in batches]
    permutation = list(reversed(range(8)))
    changed = list(_image_control_batches(batches, dataset, permutation=permutation, blank=blank))
    assert [len(batch.images) for batch in changed] == [3, 3, 2]
    for before, after, pixels in zip(batches, changed, original, strict=True):
        for name in ("input_ids", "target_mask", "attention_mask"):
            assert getattr(before, name) is getattr(after, name)
        assert before.question_length == after.question_length == 6
        assert before.num_image_tokens == after.num_image_tokens == 4
        assert before.images.shape == after.images.shape
        assert before.images.dtype == after.images.dtype == torch.float32
        assert torch.equal(before.images, pixels)
        assert before.images.data_ptr() != after.images.data_ptr()
    actual = torch.cat([batch.images for batch in changed])
    expected = (
        torch.zeros_like(actual) if blank else torch.stack([dataset[i].image for i in permutation])
    )
    assert torch.equal(actual, expected)


def test_oracle_scores_permutations_and_blank_without_metadata(manifest, config):
    class NoMetadataDataset(SyntheticColorDataset):
        def __getitem__(self, index):
            example = super().__getitem__(index)
            return SimpleNamespace(
                image=example.image, question=example.question, answer=example.answer
            )

    dataset = NoMetadataDataset(manifest, "validation")
    model = PixelReader(config)
    state = _capture_rng()
    result = evaluate_image_controls(model, dataset, recurrence_depth=3, batch_size=3)
    assert_equal(state, _capture_rng())
    assert model.training
    assert result["correct"]["correct"] == 8
    assert result["blank"]["correct"] == 2
    assert result["blank"]["accuracy"] == 0.25
    assert result["blank"]["invalid_predictions"] == 0
    assert len(model.calls) == 7 * 3
    assert all(depth == 3 for depth, _, _ in model.calls)
    for condition in range(1, 7):
        assert_equal(model.calls[:3], model.calls[condition * 3 : (condition + 1) * 3])
    for seed, row in enumerate(result["shuffled"]):
        expected = list(range(8))
        random.Random(seed).shuffle(expected)
        assert row["seed"] == seed and row["permutation"] == expected
        assert sorted(expected) == list(range(8))
        assert row["metrics"]["accuracy"] == row["same_color_pairing_fraction"]
        assert row["accuracy_gap"] == 1 - row["metrics"]["accuracy"]
    for key in ("accuracy", "loss"):
        values = [row["metrics"][key] for row in result["shuffled"]]
        assert result["shuffled_summary"][key] == {
            "mean": sum(values) / 5,
            "min": min(values),
            "max": max(values),
        }
    assert result["accuracy_gaps"]["correct_minus_blank"] == 0.75
    assert (
        result["accuracy_gaps"]["correct_minus_shuffled_mean"]
        == 1 - result["shuffled_summary"]["accuracy"]["mean"]
    )
    repeat = evaluate_image_controls(model, dataset, recurrence_depth=3, batch_size=3)
    assert_equal(result, repeat)
    different_batching = evaluate_image_controls(model, dataset, recurrence_depth=3, batch_size=5)
    for a, b in zip(result["shuffled"], different_batching["shuffled"], strict=True):
        assert a["permutation"] == b["permutation"]
        assert a["metrics"]["accuracy"] == b["metrics"]["accuracy"]
        assert a["metrics"]["loss"] == pytest.approx(b["metrics"]["loss"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"recurrence_depth": 0},
        {"shuffle_seeds": ()},
        {"shuffle_seeds": (1, 1)},
        {"shuffle_seeds": (True,)},
        {"shuffle_seeds": (1.2,)},
    ],
)
def test_invalid_settings_fail_before_evaluation(manifest, config, kwargs):
    model = PixelReader(config)
    with pytest.raises((TypeError, ValueError)):
        evaluate_image_controls(
            model,
            SyntheticColorDataset(manifest, "validation"),
            **{
                "recurrence_depth": 2,
                **kwargs,
            },
        )
    assert not model.calls


def test_empty_dataset_rejected(manifest, config):
    dataset = SyntheticColorDataset(manifest, "validation")
    dataset.scenes = ()
    with pytest.raises(ValueError, match="at least one"):
        evaluate_image_controls(PixelReader(config), dataset, recurrence_depth=2)


def test_real_model_state_rng_optimizer_and_baseline_unchanged(manifest, config):
    model = MultimodalLoopTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters())
    dataset = SyntheticColorDataset(manifest, "validation")
    loader = synthetic_loader(dataset, config, SyntheticTrainingConfig(batch_size=3))
    train_synthetic_epoch(model, optimizer, loader, recurrence_depth=2)
    model.prelude.eval()  # Restore even mixed incoming module modes.
    modes = [module.training for module in model.modules()]
    params = copy.deepcopy(model.state_dict())
    opt = copy.deepcopy(optimizer.state_dict())
    grads = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    state = _capture_rng()
    baseline = evaluate_synthetic(model, loader, recurrence_depth=2)
    result = evaluate_image_controls(model, dataset, recurrence_depth=2, batch_size=3)
    assert result["correct"] == asdict(baseline)
    assert_equal(params, model.state_dict())
    assert_equal(opt, optimizer.state_dict())
    assert_equal(grads, [p.grad for p in model.parameters()])
    assert_equal(state, _capture_rng())
    assert modes == [module.training for module in model.modules()]


def make_checkpoint(path, manifest, config):
    model = MultimodalLoopTransformer(config)
    training = SyntheticTrainingConfig(batch_size=3, recurrence_depth=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=0)
    validation = evaluate_synthetic(
        model,
        synthetic_loader(SyntheticColorDataset(manifest, "validation"), config, training),
        recurrence_depth=3,
    )
    save_synthetic_checkpoint(
        path,
        model,
        optimizer,
        manifest=manifest,
        training_config=training,
        completed_epochs=0,
        completed_steps=0,
        history=[
            {"epoch": 0, "completed_steps": 0, "train": None, "validation": asdict(validation)}
        ],
    )
    return validation


def cli(*args):
    return subprocess.run(
        [sys.executable, "scripts/evaluate.py", *map(str, args)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )


def test_cli_embedded_manifest_report_and_preserved_files(manifest, config, tmp_path):
    path = tmp_path / "last.pt"
    validation = make_checkpoint(path, manifest, config)
    before = path.read_bytes()
    (tmp_path / "metrics.json").write_text("existing training metrics")
    output = tmp_path / "controls"
    result = cli("--checkpoint", path, "--output-dir", output)
    assert result.returncode == 0, result.stderr
    assert "R=3, batch_size=3" in result.stdout
    report_path = output / "controls.json"
    report = json.loads(report_path.read_text())
    assert report["evaluation"] == {
        "split": "validation",
        "batch_size": 3,
        "recurrence_depth": 3,
        "shuffle_seeds": [0, 1, 2, 3, 4],
    }
    assert report["manifest_sha256"] == manifest.sha256
    assert report["results"]["correct"] == asdict(validation)
    assert report["completed_epochs"] == report["completed_steps"] == 0
    import hashlib

    assert report["checkpoint_sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
    assert (tmp_path / "metrics.json").read_text() == "existing training metrics"
    saved_report = report_path.read_bytes()
    failed = cli("--checkpoint", path, "--output-dir", output)
    assert failed.returncode != 0 and "already exists" in failed.stderr
    assert report_path.read_bytes() == saved_report
    other_output = tmp_path / "overrides"
    result = cli(
        "--checkpoint",
        path,
        "--output-dir",
        other_output,
        "--batch-size",
        5,
        "--shuffle-seeds",
        9,
        8,
    )
    assert result.returncode == 0, result.stderr
    other = json.loads((other_output / "controls.json").read_text())
    assert other["evaluation"]["batch_size"] == 5
    assert [row["seed"] for row in other["results"]["shuffled"]] == [9, 8]


@pytest.mark.parametrize(
    "arguments",
    [
        ("--batch-size", "0"),
        ("--shuffle-seeds", "1", "1"),
        ("--shuffle-seeds",),
        ("--split", "test"),
        ("--recurrence-depth", "1"),
    ],
)
def test_cli_rejects_invalid_or_unsupported_settings(tmp_path, arguments):
    output = tmp_path / "controls"
    result = cli("--checkpoint", tmp_path / "missing.pt", "--output-dir", output, *arguments)
    assert result.returncode != 0
    assert not output.exists()
