"""Dataset integrity, epoch accounting, question-only scoring and exact resume."""

import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn import functional as F

from multimodal_loop.data.collator import SyntheticColorCollator
from multimodal_loop.data.multimodal import (
    SyntheticColorDataset,
    load_synthetic_manifest,
    parse_synthetic_manifest,
)
from multimodal_loop.data.synthetic_shapes import (
    QUESTION,
    SyntheticShapesConfig,
    build_scene_splits,
    make_example,
)
from multimodal_loop.eval.synthetic import evaluate_synthetic
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.checkpoint import _capture_rng, _restore_rng, load_checkpoint
from multimodal_loop.train.runtime import seed_everything
from multimodal_loop.train.synthetic import (
    SyntheticTrainingConfig,
    synthetic_loader,
    train_synthetic_epoch,
)
from multimodal_loop.train.synthetic_checkpoint import (
    load_synthetic_checkpoint,
    save_synthetic_checkpoint,
)


@pytest.fixture(autouse=True)
def preserve_rng():
    state = _capture_rng()
    yield
    _restore_rng(state, torch.device("cpu"))


@pytest.fixture
def corpus():
    config = SyntheticShapesConfig(
        image_size=8, object_sizes=(4,), train_size=12, validation_size=8, test_size=4
    )
    return {
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


@pytest.fixture
def manifest(corpus):
    return parse_synthetic_manifest(json.dumps(corpus))


@pytest.fixture
def model_config():
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


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_equal(left, right)
    else:
        assert a == b


def test_manifest_honors_stored_order_and_exact_file_bytes(corpus, tmp_path):
    corpus["splits"]["train"].reverse()
    content = json.dumps(corpus, indent=2).replace("\n", "\r\n")
    path = tmp_path / "manifest.json"
    path.write_bytes(content.encode())
    manifest = load_synthetic_manifest(path)
    assert manifest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = SyntheticColorDataset(manifest, "train")
    assert len(dataset) == 12
    for example, record in zip(dataset, corpus["splits"]["train"], strict=True):
        assert asdict(example.scene) == record["scene"]
        assert example.question == record["question"] and example.answer == record["answer"]
        assert torch.equal(example.image, make_example(example.scene, image_size=8).image)
    dataset[0].image.zero_()
    assert dataset[0].image.sum() > 0
    with pytest.raises(ValueError, match="split"):
        SyntheticColorDataset(manifest, "typo")


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "boolean_version",
        "config",
        "count",
        "question",
        "answer",
        "bounds",
        "size",
        "duplicate",
        "overlap",
        "split",
    ],
)
def test_manifest_rejects_corruption(corpus, damage):
    first = corpus["splits"]["train"][0]
    if damage == "version":
        corpus["format_version"] = 2
    elif damage == "boolean_version":
        corpus["format_version"] = True
    elif damage == "config":
        del corpus["config"]["seed"]
    elif damage == "count":
        corpus["splits"]["train"].pop()
    elif damage == "question":
        first["question"] = "What shape?"
    elif damage == "answer":
        first["answer"] = "green" if first["answer"] != "green" else "red"
    elif damage == "bounds":
        first["scene"]["left"] = 4
    elif damage == "size":
        first["scene"]["size"] = 6
    elif damage == "duplicate":
        corpus["splits"]["train"][1] = copy.deepcopy(first)
    elif damage == "overlap":
        layout = {k: v for k, v in first["scene"].items() if k != "color"}
        corpus["splits"]["test"] = [
            {"scene": {**layout, "color": color}, "question": QUESTION, "answer": color}
            for color in ("red", "green", "blue", "yellow")
        ]
    else:
        corpus["splits"]["extra"] = []
    with pytest.raises((TypeError, ValueError)):
        parse_synthetic_manifest(json.dumps(corpus))


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("recurrence_depth", 0),
        ("recurrence_depth", 1.5),
        ("seed", False),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("learning_rate", True),
        ("weight_decay", -1),
        ("weight_decay", float("inf")),
    ],
)
def test_invalid_training_config(field, value):
    with pytest.raises((TypeError, ValueError)):
        SyntheticTrainingConfig(**{field: value})


def test_loader_epoch_coverage_order_partial_batch_and_rng(manifest, model_config):
    config = SyntheticTrainingConfig(batch_size=5, seed=7)
    dataset = SyntheticColorDataset(manifest, "train")
    state = _capture_rng()
    for epoch in (0, 1, 0):
        loader = synthetic_loader(dataset, model_config, config, epoch=epoch)
        expected = list(range(12))
        random.Random(f"7:train:{epoch}").shuffle(expected)
        assert list(loader.sampler) == expected
        batches = list(loader)
        assert [batch.input_ids.shape[0] for batch in batches] == [5, 5, 2]
        expected_batch = SyntheticColorCollator(model_config)([dataset[i] for i in expected])
        assert torch.equal(torch.cat([batch.images for batch in batches]), expected_batch.images)
        assert torch.equal(
            torch.cat([batch.input_ids for batch in batches]), expected_batch.input_ids
        )
    assert_equal(state, _capture_rng())
    validation = synthetic_loader(
        SyntheticColorDataset(manifest, "validation"), model_config, config
    )
    assert list(validation.sampler) == list(range(8))


def test_epoch_counts_weighted_loss_and_fixed_depth(monkeypatch, manifest, model_config):
    config = SyntheticTrainingConfig(batch_size=5)
    loader = synthetic_loader(
        SyntheticColorDataset(manifest, "train"), model_config, config, epoch=0
    )
    model = MultimodalLoopTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters())
    calls = []

    def fake_train(model_arg, optimizer_arg, ids, images, **kwargs):
        assert model_arg is model and optimizer_arg is optimizer
        assert kwargs == {"steps": 1, "question_length": 6, "recurrence_depth": 3}
        assert ids.device == next(model.parameters()).device == images.device
        calls.append(ids.shape[0])
        return [float(len(calls))]

    monkeypatch.setattr("multimodal_loop.train.synthetic.train_on_batch", fake_train)
    result = train_synthetic_epoch(model, optimizer, loader, recurrence_depth=3)
    assert calls == [5, 5, 2]
    assert result.examples == 12 and result.steps == 3
    assert result.loss == (5 + 10 + 6) / 12
    with pytest.raises(ValueError, match="at least one"):
        train_synthetic_epoch(model, optimizer, [], recurrence_depth=3)


def test_evaluation_question_only_full_vocab_and_weighting(monkeypatch, manifest, model_config):
    dataset = SyntheticColorDataset(manifest, "validation")
    batches = list(synthetic_loader(dataset, model_config, SyntheticTrainingConfig(batch_size=5)))
    model = MultimodalLoopTransformer(model_config)
    original = copy.deepcopy(model.state_dict())
    targets = torch.cat([batch.input_ids[:, -1] for batch in batches])
    logits = torch.zeros(8, 10)
    predicted = targets.clone()
    predicted[1] = 0  # invalid question token
    predicted[6] = 1  # invalid prediction in the short batch
    predicted[7] = 6 + (targets[7] - 5) % 4  # valid but wrong color
    logits[torch.arange(8), predicted] = torch.arange(1, 9, dtype=torch.float32)
    cursor = 0

    def forward(ids, images, *, recurrence_depth, attention_mask):
        nonlocal cursor
        assert not model.training and torch.is_inference_mode_enabled()
        assert ids.shape[1] == 6 and torch.equal(ids[0], torch.arange(6))
        assert recurrence_depth == 3
        assert attention_mask.shape == (10, 10) and attention_mask.all()
        count = ids.shape[0]
        result = logits[cursor : cursor + count, None, :].expand(-1, 10, -1)
        cursor += count
        return result

    monkeypatch.setattr(model, "forward", forward)
    state = _capture_rng()
    metrics = evaluate_synthetic(model, batches, recurrence_depth=3)
    assert metrics.total == 8 and metrics.correct == 5 and metrics.invalid_predictions == 2
    assert metrics.accuracy == 5 / 8
    assert metrics.loss == pytest.approx(F.cross_entropy(logits, targets).item())
    assert model.training
    assert_equal(model.state_dict(), original)
    assert_equal(state, _capture_rng())


@pytest.mark.parametrize("training", [True, False])
def test_real_evaluation_preserves_state_and_gradient(manifest, model_config, training):
    model = MultimodalLoopTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters())
    loader = synthetic_loader(
        SyntheticColorDataset(manifest, "train"),
        model_config,
        SyntheticTrainingConfig(batch_size=5),
    )
    train_synthetic_epoch(model, optimizer, loader, recurrence_depth=2)
    model.train(training)
    params = copy.deepcopy(model.state_dict())
    opt = copy.deepcopy(optimizer.state_dict())
    grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    state = _capture_rng()
    evaluate_synthetic(model, loader, recurrence_depth=2)
    assert model.training is training
    assert_equal(state, _capture_rng())
    assert_equal(params, model.state_dict())
    assert_equal(opt, optimizer.state_dict())
    assert_equal(grads, [p.grad for p in model.parameters()])
    with pytest.raises(ValueError, match="at least one"):
        evaluate_synthetic(model, [], recurrence_depth=2)
    assert model.training is training


def run_epoch(model, optimizer, manifest, training, history):
    epoch = len(history) - 1
    train = train_synthetic_epoch(
        model,
        optimizer,
        synthetic_loader(
            SyntheticColorDataset(manifest, "train"), model.config, training, epoch=epoch
        ),
        recurrence_depth=training.recurrence_depth,
    )
    validation = evaluate_synthetic(
        model,
        synthetic_loader(SyntheticColorDataset(manifest, "validation"), model.config, training),
        recurrence_depth=training.recurrence_depth,
    )
    history.append(
        {
            "epoch": epoch + 1,
            "completed_steps": history[-1]["completed_steps"] + train.steps,
            "train": asdict(train),
            "validation": asdict(validation),
        }
    )


def initial_history(model, manifest, training):
    metrics = evaluate_synthetic(
        model,
        synthetic_loader(SyntheticColorDataset(manifest, "validation"), model.config, training),
        recurrence_depth=training.recurrence_depth,
    )
    return [{"epoch": 0, "completed_steps": 0, "train": None, "validation": asdict(metrics)}]


def save(path, model, optimizer, manifest, training, history):
    save_synthetic_checkpoint(
        path,
        model,
        optimizer,
        manifest=manifest,
        training_config=training,
        completed_epochs=len(history) - 1,
        completed_steps=history[-1]["completed_steps"],
        history=history,
    )


@pytest.mark.parametrize("depth", [1, 3])
def test_exact_epoch_resume_with_dropout(manifest, model_config, tmp_path, depth):
    seed_everything(10, torch.device("cpu"))
    model = MultimodalLoopTransformer(model_config)
    training = SyntheticTrainingConfig(batch_size=5, recurrence_depth=depth)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=0, foreach=False, fused=False
    )
    history = initial_history(model, manifest, training)
    run_epoch(model, optimizer, manifest, training, history)
    state = _capture_rng()
    path = tmp_path / "last.pt"
    save(path, model, optimizer, manifest, training, history)
    assert_equal(state, _capture_rng())
    run_epoch(model, optimizer, manifest, training, history)
    expected_rng = _capture_rng()
    restored = load_synthetic_checkpoint(path)
    assert_equal(state, _capture_rng())
    assert restored.completed_epochs == 1 and restored.completed_steps == 3
    run_epoch(
        restored.model,
        restored.optimizer,
        restored.manifest,
        restored.training_config,
        restored.history,
    )
    assert_equal(model.state_dict(), restored.model.state_dict())
    assert_equal(optimizer.state_dict(), restored.optimizer.state_dict())
    assert_equal(history, restored.history)
    assert_equal(expected_rng, _capture_rng())
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "damage",
    [
        "tokenizer",
        "digest",
        "manifest",
        "progress",
        "history",
        "backend",
        "optimizer",
        "dtype",
        "version",
        "missing",
    ],
)
def test_checkpoint_corruption_does_not_change_rng(manifest, model_config, tmp_path, damage):
    model = MultimodalLoopTransformer(model_config)
    training = SyntheticTrainingConfig()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=0)
    path = tmp_path / "last.pt"
    save(path, model, optimizer, manifest, training, initial_history(model, manifest, training))
    payload = torch.load(path, weights_only=True)
    if damage == "tokenizer":
        payload["tokenizer"]["vocabulary"].reverse()
    elif damage == "digest":
        payload["manifest_sha256"] = "wrong"
    elif damage == "manifest":
        payload["manifest_content"] = "{}"
    elif damage == "progress":
        payload["completed_steps"] = 1
    elif damage == "history":
        payload["history"] = []
    elif damage == "backend":
        payload["backend"] = "cuda"
    elif damage == "optimizer":
        payload["optimizer_state"]["param_groups"][0]["lr"] = 2
    elif damage == "dtype":
        payload["model_dtype"] = "float64"
    elif damage == "version":
        payload["format_version"] = True
    else:
        del payload["config"]
    torch.save(payload, path)
    state = _capture_rng()
    with pytest.raises((TypeError, ValueError)):
        load_synthetic_checkpoint(path)
    assert_equal(state, _capture_rng())


def test_atomic_synthetic_checkpoint_failure(monkeypatch, manifest, model_config, tmp_path):
    model = MultimodalLoopTransformer(model_config)
    training = SyntheticTrainingConfig()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=0)
    path = tmp_path / "last.pt"
    history = initial_history(model, manifest, training)
    save(path, model, optimizer, manifest, training, history)
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated failure")

    monkeypatch.setattr("multimodal_loop.train.checkpoint.os.replace", fail)
    with pytest.raises(OSError, match="simulated"):
        save(path, model, optimizer, manifest, training, history)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def cli(*args, success=True):
    result = subprocess.run(
        [sys.executable, "scripts/train_synthetic.py", *map(str, args)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        text=True,
        capture_output=True,
    )
    if success:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result


def test_cli_fresh_process_exact_resume_and_embedded_manifest(corpus, model_config, tmp_path):
    manifest_path = tmp_path / "corpus.json"
    manifest_path.write_text(json.dumps(corpus))
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(asdict(model_config)))
    common = [
        "--manifest",
        manifest_path,
        "--model-config",
        config_path,
        "--batch-size",
        5,
        "--recurrence-depth",
        3,
    ]
    continuous, split = tmp_path / "continuous", tmp_path / "split"
    cli(*common, "--epochs", 2, "--output-dir", continuous)
    cli(*common, "--epochs", 1, "--output-dir", split)
    manifest_path.unlink()
    (split / "metrics.json").write_text("stale")
    result = cli("--resume", split / "last.pt", "--epochs", 1, "--output-dir", split)
    assert "epoch=2 R=3 steps=6" in result.stdout
    a = torch.load(continuous / "last.pt", weights_only=True)
    b = torch.load(split / "last.pt", weights_only=True)
    assert_equal(a, b)
    assert json.loads((split / "metrics.json").read_text()) == b["history"]
    assert (split / "manifest.json").read_text() == b["manifest_content"]
    assert json.loads((split / "settings.json").read_text())["training"]["recurrence_depth"] == 3
    assert (
        "cannot be combined"
        in cli("--resume", split / "last.pt", "--seed", 0, success=False).stderr
    )
    assert "already contains" in cli("--output-dir", split, success=False).stderr


@pytest.mark.parametrize("override", [{"image_size": 16}, {"vocab_size": 11}, {"num_channels": 1}])
def test_cli_rejects_model_corpus_mismatch(corpus, model_config, tmp_path, override):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(corpus))
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(asdict(replace(model_config, **override))))
    cli(
        "--manifest",
        path,
        "--model-config",
        config_path,
        "--output-dir",
        tmp_path / "run",
        success=False,
    )
    assert not (tmp_path / "run").exists()


def test_evaluation_exception_restores_mixed_modes_and_all_rngs(model_config):
    import numpy as np

    model = MultimodalLoopTransformer(model_config)
    model.prelude.eval()
    modes = [module.training for module in model.modules()]
    state = _capture_rng()

    def broken_batches():
        random.random()
        np.random.random()
        torch.rand(1)
        raise RuntimeError("broken iterator")
        yield

    with pytest.raises(RuntimeError, match="broken iterator"):
        evaluate_synthetic(model, broken_batches(), recurrence_depth=2)
    assert_equal(state, _capture_rng())
    assert [module.training for module in model.modules()] == modes


def test_evaluation_rejects_answer_inside_declared_question(manifest, model_config):
    model = MultimodalLoopTransformer(model_config)
    batch = next(
        iter(
            synthetic_loader(
                SyntheticColorDataset(manifest, "validation"),
                model_config,
                SyntheticTrainingConfig(),
            )
        )
    )
    with pytest.raises(ValueError, match="six question IDs"):
        evaluate_synthetic(model, [replace(batch, question_length=7)], recurrence_depth=2)
