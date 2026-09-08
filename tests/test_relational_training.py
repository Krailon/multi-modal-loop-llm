"""Relational training, grouped controls, and deterministic epoch-boundary resume."""

import copy
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from multimodal_loop.data.collator import RelationalColorCollator
from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import (
    RelationalColorDataset,
    parse_relational_manifest,
)
from multimodal_loop.data.text import RelationalColorTokenizer
from multimodal_loop.eval.relational import (
    _control_batches,
    _grouped_metrics,
    evaluate_relational,
    evaluate_relational_controls,
)
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.checkpoint import _capture_rng, _restore_rng
from multimodal_loop.train.relational import relational_loader
from multimodal_loop.train.relational_checkpoint import (
    load_relational_checkpoint,
    save_relational_checkpoint,
    validate_relational_config,
)
from multimodal_loop.train.runtime import seed_everything
from multimodal_loop.train.synthetic import SyntheticTrainingConfig, train_synthetic_epoch
from multimodal_loop.train.synthetic_checkpoint import load_synthetic_checkpoint


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_equal(x, y)
    else:
        assert a == b


@pytest.fixture(autouse=True)
def preserve_rng():
    state = _capture_rng()
    yield
    _restore_rng(state, torch.device("cpu"))


@pytest.fixture(scope="module")
def manifest():
    config = RelationalCorpusConfig(
        image_size=16,
        object_sizes=(4,),
        train_geometry_count=1,
        validation_geometry_count=1,
        test_geometry_count=1,
    )
    splits = build_relational_splits(config)
    # Stored question order must not be interpreted as a target-position code.
    splits = {
        split: [replace(record, questions=tuple(reversed(record.questions))) for record in records]
        for split, records in splits.items()
    }
    return parse_relational_manifest(
        json.dumps(
            {
                "kind": "relational_color_rows",
                "format_version": 1,
                "software": {},
                "config": asdict(config),
                "splits": {
                    split: [asdict(r) for r in records] for split, records in splits.items()
                },
            }
        )
    )


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=17,
        image_size=16,
        patch_size=8,
        max_seq_len=16,
        d_model=8,
        n_heads=2,
        d_ff=16,
        dropout=0.2,
    )


@pytest.fixture
def dataset(manifest):
    return RelationalColorDataset(manifest, "validation")


def test_loader_coverage_order_partial_batches_and_rng(dataset, config):
    settings = SyntheticTrainingConfig(batch_size=127, seed=7)
    state = _capture_rng()
    for epoch in (0, 1, 0, None):
        loader = relational_loader(dataset, config, settings, epoch=epoch)
        expected = list(range(576))
        if epoch is not None:
            random.Random(f"7:train:{epoch}").shuffle(expected)
        assert list(loader.sampler) == expected
        batches = list(loader)
        assert [len(b.input_ids) for b in batches] == [127, 127, 127, 127, 68]
        reference = RelationalColorCollator(config)([dataset[i] for i in expected])
        for field in ("images", "input_ids", "target_mask"):
            assert torch.equal(
                torch.cat([getattr(b, field) for b in batches]), getattr(reference, field)
            )
    assert_equal(state, _capture_rng())
    with pytest.raises(ValueError, match="image_size"):
        relational_loader(dataset, replace(config, image_size=32), settings)
    with pytest.raises(ValueError, match="epoch"):
        relational_loader(dataset, config, settings, epoch=-1)


def test_epoch_counts_loss_alignment_updates_and_depth(dataset, config, monkeypatch):
    settings = SyntheticTrainingConfig(batch_size=127)
    batches = list(relational_loader(dataset, config, settings))
    model = MultimodalLoopTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = copy.deepcopy(model.state_dict())
    calls = []
    original = model.forward

    def forward(ids, images, **kwargs):
        assert ids.shape[1] == 12
        assert kwargs["recurrence_depth"] == 3
        mask = kwargs["attention_mask"]
        assert mask[:15, :15].all() and not mask[:15, 15].any()
        calls.append(len(ids))
        return original(ids, images, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    metrics = train_synthetic_epoch(model, optimizer, batches, recurrence_depth=3)
    assert calls == [127, 127, 127, 127, 68]
    assert metrics.examples == 576 and metrics.steps == 5 and metrics.loss > 0
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert model.embeddings.image_embedding.projection.weight.grad is not None

    calls.clear()

    def fake_train(model, optimizer, ids, images, **kwargs):
        assert kwargs == {"steps": 1, "question_length": 11, "recurrence_depth": 3}
        calls.append(len(ids))
        return [float(len(calls))]

    monkeypatch.setattr("multimodal_loop.train.synthetic.train_on_batch", fake_train)
    metrics = train_synthetic_epoch(model, optimizer, batches, recurrence_depth=3)
    assert metrics.loss == (127 * (1 + 2 + 3 + 4) + 68 * 5) / 576


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_question_only_weighting_invalid_ids_and_state(
    dataset, config, monkeypatch, fail
):
    batches = list(relational_loader(dataset, config, SyntheticTrainingConfig(batch_size=127)))
    targets = torch.cat([b.input_ids[:, -1] for b in batches])
    predictions = targets.clone()
    predictions[1], predictions[2], predictions[3], predictions[-1] = 0, 10, 16, 11
    logits = torch.zeros(576, 17)
    logits[torch.arange(576), predictions] = 5
    model = MultimodalLoopTransformer(config)
    model.coda.eval()  # Restore mixed module modes, not just the root flag.
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    weights, gradients = (
        copy.deepcopy(model.state_dict()),
        [p.grad.clone() for p in model.parameters()],
    )
    modes = [m.training for m in model.modules()]
    state = _capture_rng()
    offset = 0

    def forward(ids, images, **kwargs):
        nonlocal offset
        assert not model.training and torch.is_inference_mode_enabled()
        assert ids.shape[1] == 11 and kwargs["recurrence_depth"] == 3
        assert torch.equal(
            ids, torch.cat([b.input_ids[:, :11] for b in batches])[offset : offset + len(ids)]
        )
        assert kwargs["attention_mask"].shape == (15, 15) and kwargs["attention_mask"].all()
        random.random(), np.random.rand(), torch.rand(1)
        result = logits[offset : offset + len(ids), None, :].expand(-1, 15, -1)
        offset += len(ids)
        if fail:
            raise RuntimeError("injected failure")
        return result

    monkeypatch.setattr(model, "forward", forward)
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            evaluate_relational(model, batches, recurrence_depth=3)
    else:
        result = evaluate_relational(model, batches, recurrence_depth=3)
        assert result.total == 576 and result.correct == 572 and result.invalid_predictions == 4
        assert result.loss == pytest.approx(F.cross_entropy(logits, targets).item())
    assert_equal(state, _capture_rng())
    assert_equal(weights, model.state_dict())
    assert_equal(gradients, [p.grad for p in model.parameters()])
    assert modes == [m.training for m in model.modules()]


@pytest.mark.parametrize("behavior", ["perfect", "middle", "invalid"])
def test_grouped_metric_arithmetic_with_reordered_questions(dataset, behavior):
    tokenizer = RelationalColorTokenizer()
    predictions = []
    for record in dataset.records:
        for qa in record.questions:
            answer = qa.answer if behavior == "perfect" else record.scene.objects[1].color
            predictions.append(0 if behavior == "invalid" else tokenizer.encode_answer(answer))
    result = _grouped_metrics(dataset, predictions)
    expected = 1 if behavior == "perfect" else 0
    assert result["all_four"] == {"total": 144, "correct": 144 * expected, "accuracy": expected}
    assert result["different_answer_pairs"] == {
        "total": 720,
        "correct": 720 * expected,
        "accuracy": expected,
    }
    counts = list(result["by_question"].values())
    assert len(counts) == 6 and sum(c["total"] for c in counts) == 576
    assert (
        sum(c["correct"] for c in counts) / 576
        == {"perfect": 1, "middle": 0.5, "invalid": 0}[behavior]
    )
    with pytest.raises(ValueError, match="entire"):
        _grouped_metrics(dataset, predictions[:-1])


@pytest.mark.parametrize("condition", ["images", "questions", "blank"])
def test_controls_change_only_intended_inputs_across_group_boundaries(dataset, config, condition):
    batches = list(relational_loader(dataset, config, SyntheticTrainingConfig(batch_size=127)))
    originals = [(b.images.clone(), b.input_ids.clone()) for b in batches]
    permutation = list(reversed(range(144)))
    slots = [[1, 0, 3, 2] for _ in range(144)]
    kwargs = {
        "images": {"image_permutation": permutation},
        "questions": {"question_permutations": slots},
        "blank": {"blank": True},
    }[condition]
    changed = list(_control_batches(batches, dataset, **kwargs))
    tokenizer = RelationalColorTokenizer()
    offset = 0
    for before, after, (pixels, ids) in zip(batches, changed, originals, strict=True):
        assert torch.equal(before.images, pixels) and torch.equal(before.input_ids, ids)
        assert before.target_mask is after.target_mask
        assert before.attention_mask is after.attention_mask
        assert torch.equal(before.input_ids[:, -1], after.input_ids[:, -1])
        if condition == "questions":
            assert before.images is after.images
            expected = [
                tokenizer.encode_question(
                    dataset.records[i // 4].questions[slots[i // 4][i % 4]].question
                )
                for i in range(offset, offset + len(ids))
            ]
            assert torch.equal(after.input_ids[:, :11], torch.tensor(expected))
        else:
            assert before.input_ids is after.input_ids
            expected = (
                torch.zeros_like(pixels)
                if condition == "blank"
                else torch.stack(
                    [
                        dataset[4 * permutation[i // 4]].image
                        for i in range(offset, offset + len(ids))
                    ]
                )
            )
            assert torch.equal(after.images, expected)
        offset += len(ids)


class KnownInputReader(nn.Module):
    """Test lookup oracle keyed solely by rendered pixels and question IDs."""

    def __init__(self, dataset, config):
        super().__init__()
        self.config = config
        self.anchor = nn.Parameter(torch.zeros(()))
        tokenizer = RelationalColorTokenizer()
        self.lookup = {}
        for i in range(len(dataset)):
            ex = dataset[i]
            self.lookup[(ex.image.numpy().tobytes(), tokenizer.encode_question(ex.question))] = (
                tokenizer.encode_answer(ex.answer)
            )
        self.calls = []

    def forward(self, ids, images, *, recurrence_depth, attention_mask):
        assert not self.training and torch.is_inference_mode_enabled()
        assert ids.shape[1] == 11 and attention_mask.all()
        self.calls.append(recurrence_depth)
        predictions = [
            self.lookup.get((image.numpy().tobytes(), tuple(question.tolist())), 6)
            for image, question in zip(images, ids, strict=True)
        ]
        logits = torch.zeros(len(ids), 17)
        logits[torch.arange(len(ids)), predictions] = 5
        return logits[:, None, :].expand(-1, 15, -1)


def test_controls_oracle_reproducibility_coincidences_and_state(dataset, config):
    model = KnownInputReader(dataset, config)
    state = _capture_rng()
    first = evaluate_relational_controls(model, dataset, recurrence_depth=3, batch_size=127)
    repeat = evaluate_relational_controls(model, dataset, recurrence_depth=3, batch_size=127)
    other = evaluate_relational_controls(model, dataset, recurrence_depth=3, batch_size=113)
    assert_equal(first, repeat)
    assert_equal(state, _capture_rng())
    assert model.training and all(r == 3 for r in model.calls)
    assert first["correct"]["accuracy"] == 1
    assert first["correct"]["all_four"]["accuracy"] == 1
    assert first["correct"]["different_answer_pairs"]["accuracy"] == 1
    assert first["blank"]["accuracy"] == 0.25
    for seed, row in enumerate(first["shuffled_images"]):
        expected = list(range(144))
        random.Random(seed).shuffle(expected)
        assert row["permutation"] == expected
    for row in first["shuffled_questions"]:
        assert all(sorted(p) == [0, 1, 2, 3] for p in row["permutations"])
        assert row["metrics"]["accuracy"] == row["same_answer_pairing_fraction"]
    for kind in ("images", "questions"):
        for a, b in zip(first[f"shuffled_{kind}"], other[f"shuffled_{kind}"], strict=True):
            metric_a, metric_b = a.pop("metrics"), b.pop("metrics")
            assert_equal(a, b)
            assert metric_a.pop("loss") == pytest.approx(metric_b.pop("loss"))
            assert_equal(metric_a, metric_b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"recurrence_depth": 0},
        {"shuffle_seeds": ()},
        {"shuffle_seeds": (0, 0)},
        {"shuffle_seeds": (True,)},
    ],
)
def test_invalid_control_settings(dataset, config, kwargs):
    with pytest.raises((ValueError, TypeError)):
        evaluate_relational_controls(
            MultimodalLoopTransformer(config), dataset, **{"recurrence_depth": 2, **kwargs}
        )


def initial_checkpoint(path, manifest, config):
    seed_everything(23, torch.device("cpu"))
    model = MultimodalLoopTransformer(config)
    training = SyntheticTrainingConfig(batch_size=127, seed=23)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
        foreach=False,
        fused=False,
    )
    validation = evaluate_relational(
        model,
        relational_loader(RelationalColorDataset(manifest, "validation"), config, training),
        recurrence_depth=training.recurrence_depth,
    )
    history = [{"epoch": 0, "completed_steps": 0, "train": None, "validation": asdict(validation)}]
    save_relational_checkpoint(
        path,
        model,
        optimizer,
        manifest=manifest,
        training_config=training,
        completed_epochs=0,
        completed_steps=0,
        history=history,
    )
    return model, optimizer, training, history


def test_resume_exact_with_dropout_and_self_contained_manifest(manifest, config, tmp_path):
    path = tmp_path / "last.pt"
    model, optimizer, training, history = initial_checkpoint(path, manifest, config)
    data = RelationalColorDataset(manifest, "train")
    metrics = train_synthetic_epoch(
        model, optimizer, relational_loader(data, config, training, epoch=0), recurrence_depth=2
    )
    weights, optimizer_state, rng = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        _capture_rng(),
    )
    random.random(), np.random.rand(), torch.rand(3)
    saved = load_relational_checkpoint(path)
    assert saved.manifest.content == manifest.content and saved.history == history
    resumed = train_synthetic_epoch(
        saved.model,
        saved.optimizer,
        relational_loader(
            RelationalColorDataset(saved.manifest, "train"),
            saved.model.config,
            saved.training_config,
            epoch=0,
        ),
        recurrence_depth=2,
    )
    assert metrics == resumed
    assert_equal(weights, saved.model.state_dict())
    assert_equal(optimizer_state, saved.optimizer.state_dict())
    assert_equal(rng, _capture_rng())
    with pytest.raises(ValueError, match="format"):
        load_synthetic_checkpoint(path)


@pytest.mark.parametrize(
    "damage",
    [
        "kind",
        "version",
        "tokenizer",
        "hash",
        "steps",
        "epochs",
        "history",
        "count",
        "accuracy",
        "loss",
        "optimizer",
        "config",
        "missing",
        "backend",
    ],
)
def test_checkpoint_rejects_corruption_without_rng_changes(manifest, config, tmp_path, damage):
    path = tmp_path / "last.pt"
    initial_checkpoint(path, manifest, config)
    payload = torch.load(path, weights_only=True)
    if damage == "kind":
        payload["kind"] = "synthetic_color"
    elif damage == "version":
        payload["format_version"] = True
    elif damage == "tokenizer":
        payload["tokenizer"]["vocabulary"].reverse()
    elif damage == "hash":
        payload["manifest_sha256"] = "wrong"
    elif damage == "steps":
        payload["completed_steps"] = 1
    elif damage == "epochs":
        payload["completed_epochs"] = 1
    elif damage == "history":
        payload["history"] = []
    elif damage == "count":
        payload["history"][0]["validation"]["total"] = 144
    elif damage == "accuracy":
        payload["history"][0]["validation"]["accuracy"] = -1
    elif damage == "loss":
        payload["history"][0]["validation"]["loss"] = float("nan")
    elif damage == "optimizer":
        payload["optimizer_state"]["param_groups"][0]["lr"] = 0.5
    elif damage == "config":
        payload["training_config"].pop("seed")
    elif damage == "missing":
        payload.pop("rng_state")
    elif damage == "backend":
        payload["backend"] = "cuda"
    torch.save(payload, path)
    state = _capture_rng()
    with pytest.raises((ValueError, TypeError)):
        load_relational_checkpoint(path)
    assert_equal(state, _capture_rng())


@pytest.mark.parametrize(
    "overrides", [{"vocab_size": 10}, {"vocab_size": 18}, {"image_size": 32}, {"max_seq_len": 15}]
)
def test_task_model_config_validation(manifest, config, overrides):
    with pytest.raises(ValueError):
        validate_relational_config(replace(config, **overrides), manifest)


def run_cli(script, *args, ok=True):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / script),
            *map(str, args),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert (result.returncode == 0) == ok, result.stdout + result.stderr
    return result


def test_cli_process_resume_and_validation_controls(manifest, config, tmp_path):
    manifest_path, config_path = tmp_path / "manifest.json", tmp_path / "model.json"
    manifest_path.write_text(manifest.content)
    config_path.write_text(yaml.safe_dump(asdict(config)))
    whole, split = tmp_path / "whole", tmp_path / "split"
    fresh = [
        "--manifest",
        manifest_path,
        "--model-config",
        config_path,
        "--batch-size",
        127,
        "--seed",
        17,
        "--recurrence-depth",
        3,
    ]
    run_cli("train_relational.py", *fresh, "--epochs", 2, "--output-dir", whole)
    run_cli("train_relational.py", *fresh, "--epochs", 1, "--output-dir", split)
    # Resume needs neither external input file nor a current metrics sidecar.
    manifest_path.unlink()
    config_path.unlink()
    (split / "metrics.json").write_text("stale")
    run_cli(
        "train_relational.py", "--resume", split / "last.pt", "--epochs", 1, "--output-dir", split
    )
    a, b = [torch.load(p / "last.pt", weights_only=True) for p in (whole, split)]
    assert_equal(a, b)
    assert a["completed_steps"] == 10 and a["completed_epochs"] == 2
    assert a["training_config"]["recurrence_depth"] == 3
    assert json.loads((split / "metrics.json").read_text()) == a["history"]
    assert (split / "manifest.json").read_text() == manifest.content
    run_cli("train_relational.py", "--output-dir", split, ok=False)
    run_cli("train_relational.py", "--resume", split / "last.pt", "--seed", 2, ok=False)
    before = (split / "last.pt").read_bytes()
    output = tmp_path / "controls"
    args = [
        "--checkpoint",
        split / "last.pt",
        "--output-dir",
        output,
        "--batch-size",
        127,
        "--shuffle-seeds",
        0,
    ]
    run_cli("evaluate_relational.py", *args)
    report = json.loads((output / "controls.json").read_text())
    assert report["evaluation"]["split"] == "validation"
    assert report["evaluation"]["recurrence_depth"] == 3
    assert report["manifest_sha256"] == manifest.sha256
    assert report["results"]["correct"]["total"] == 576
    assert (split / "last.pt").read_bytes() == before
    run_cli("evaluate_relational.py", *args, ok=False)
    run_cli("evaluate_relational.py", *args, "--recurrence-depth", 1, ok=False)
