"""Direct grounding labels, leakage, exact budgets, frozen controls and artifact workflow."""

import copy
import json
import os
import random
import subprocess
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import (
    RelationalColorDataset,
    parse_relational_manifest,
)
from multimodal_loop.data.shape_grounding import (
    ShapeColorCollator,
    ShapeColorDataset,
    ShapeColorTokenizer,
)
from multimodal_loop.data.synthetic_shapes import COLORS, SHAPES
from multimodal_loop.data.text import RelationalColorTokenizer
from multimodal_loop.eval.kaggle_shape_grounding import ShapeGroundingRun, archive_shape_grounding
from multimodal_loop.eval.shape_grounding import (
    assess_shape_grounding,
    control_batches,
    diagnose_shape_grounding,
    evaluate_shape_controls,
)
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.checkpoint import _capture_rng, _restore_rng
from multimodal_loop.train.shape_grounding import (
    ShapeGroundingConfig,
    shape_color_loader,
    train_shape_grounding,
)
from multimodal_loop.train.shape_grounding_checkpoint import (
    load_shape_checkpoint,
    save_shape_checkpoint,
)
from multimodal_loop.train.synthetic import EpochMetrics, SyntheticTrainingConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest():
    config = RelationalCorpusConfig(
        image_size=16,
        object_sizes=(4,),
        train_geometry_count=1,
        validation_geometry_count=1,
        test_geometry_count=1,
    )
    return parse_relational_manifest(
        json.dumps(
            {
                "kind": "relational_color_rows",
                "format_version": 1,
                "software": {},
                "config": asdict(config),
                "splits": {
                    s: [asdict(r) for r in records]
                    for s, records in build_relational_splits(config).items()
                },
            }
        )
    )


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=17,
        image_size=16,
        patch_size=4,
        max_seq_len=32,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_prelude_layers=1,
        n_recurrent_layers=1,
        n_coda_layers=1,
        recurrence_depth=2,
    )


@pytest.fixture(autouse=True)
def preserve_rng():
    rng = _capture_rng()
    yield
    _restore_rng(rng, torch.device("cpu"))


def test_labels_balance_source_pixels_and_splits(manifest):
    geometries = {}
    for split in ("train", "validation"):
        data = ShapeColorDataset(manifest, split)
        original = RelationalColorDataset(manifest, split)
        counts = {(shape, color): 0 for shape in SHAPES for color in COLORS}
        assert len(data) == 432
        for i in range(len(data)):
            item = data[i]
            shape = ("square", "circle", "triangle")[i % 3]
            obj = [o for o in manifest.splits[split][i // 3].scene.objects if o.shape == shape][0]
            assert item.shape == shape and item.answer == obj.color
            assert item.question == f"What color is the {shape}?"
            assert torch.equal(item.image, original[4 * (i // 3)].image)
            counts[shape, item.answer] += 1
        assert set(counts.values()) == {36}
        geometries[split] = {
            tuple((o.left, o.top, o.size) for o in r.scene.objects) for r in data.records
        }
        data[0].image.zero_()
        assert data[0].image.any()  # Fresh render, not shared storage.
        assert data[-1].answer == data[len(data) - 1].answer
        with pytest.raises(IndexError):
            data[len(data)]
    assert not geometries["train"] & geometries["validation"]
    with pytest.raises(ValueError, match="only train and validation"):
        ShapeColorDataset(manifest, "test")
    # Relational QA metadata does not supply the new labels.
    altered = replace(
        manifest,
        splits={
            **manifest.splits,
            "train": tuple(replace(r, questions=()) for r in manifest.splits["train"]),
        },
    )
    assert (
        ShapeColorDataset(altered, "train")[0].answer
        == ShapeColorDataset(manifest, "train")[0].answer
    )


def test_tokenization_metadata_exclusion_and_shift(manifest, config):
    tokenizer = ShapeColorTokenizer()
    assert tokenizer.vocabulary == RelationalColorTokenizer.vocabulary
    for shape, token in (("square", 14), ("circle", 15), ("triangle", 16)):
        assert tokenizer.encode_question(f"What color is the {shape}?") == (0, 1, 2, 3, token, 5)
    for text in (
        "What color is the object?",
        "What color is the object immediately left of the circle?",
    ):
        with pytest.raises(ValueError):
            tokenizer.encode_question(text)
    item = ShapeColorDataset(manifest, "train")[0]

    class MetadataTrap:
        image, question, answer = item.image, item.question, item.answer

        def __getattr__(self, name):
            raise AssertionError(f"metadata accessed: {name}")

    batch = ShapeColorCollator(config)([MetadataTrap()])
    assert batch.input_ids.shape == (1, 7)
    assert batch.input_ids[0, -1] == tokenizer.encode_answer(item.answer)
    assert batch.target_mask.tolist() == [[False] * 6 + [True]]
    assert batch.target_mask[:, 1:].nonzero().tolist() == [[0, 5]]
    assert set(batch.model_inputs()) == {"input_ids", "images", "attention_mask"}


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_no_answer_leakage_and_backward(manifest, config, depth):
    batch = ShapeColorCollator(config)([ShapeColorDataset(manifest, "train")[0]])
    model = MultimodalLoopTransformer(config)
    changed = batch.input_ids.clone()
    changed[:, -1] = (changed[:, -1] - 6 + 1) % 4 + 6
    first = model(**batch.model_inputs(), recurrence_depth=depth)
    second = model(
        changed, batch.images, attention_mask=batch.attention_mask, recurrence_depth=depth
    )
    torch.testing.assert_close(first[:, :-1], second[:, :-1], rtol=0, atol=0)
    loss = torch.nn.functional.cross_entropy(
        first[:, config.num_patches + 5], batch.input_ids[:, -1]
    )
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_deterministic_loader_and_partial_final_batch(manifest, config):
    data = ShapeColorDataset(manifest, "train")
    training = SyntheticTrainingConfig()
    a = shape_color_loader(data, config, training, epoch=2)
    b = shape_color_loader(data, config, training, epoch=2)
    c = shape_color_loader(data, config, training, epoch=3)
    assert list(a.sampler) == list(b.sampler) != list(c.sampler)
    assert sorted(a.sampler) == list(range(len(data)))
    assert len(list(a)[-1].input_ids) == 16


def test_exact_research_budget(monkeypatch):
    import multimodal_loop.train.shape_grounding as module

    epochs, batches_seen = [], []
    monkeypatch.setattr(module, "ShapeColorDataset", lambda m, s: s)

    def loader(data, config, training, *, epoch=None):
        if data == "validation":
            return []
        epochs.append(epoch)
        return [(epoch, i) for i in range(216)]

    def train(model, optimizer, batches, *, recurrence_depth):
        batches = list(batches)
        batches_seen.extend(batches)
        return EpochMetrics(1.0, len(batches) * 32, len(batches))

    monkeypatch.setattr(module, "shape_color_loader", loader)
    monkeypatch.setattr(module, "train_synthetic_epoch", train)
    from multimodal_loop.eval.synthetic import EvaluationMetrics

    monkeypatch.setattr(
        module,
        "predict_color_answers",
        lambda *a, **k: (EvaluationMetrics(1728, 432, 0.25, 0, 1.0), []),
    )
    history = module.train_shape_grounding(
        SimpleNamespace(config=None), None, None, SyntheticTrainingConfig(), ShapeGroundingConfig()
    )
    assert epochs == list(range(14))
    assert len(batches_seen) == 2880 and batches_seen[-1] == (13, 71)
    assert [r["completed_steps"] for r in history] == list(range(0, 2881, 288))
    assert history[-1]["examples_seen"] == 92160
    assert (history[-1]["completed_passes"], history[-1]["batches_in_pass"]) == (13, 72)


class Oracle(nn.Module):
    """Read image identity and the requested shape, never the supervised answer."""

    def __init__(self, config, data, invalid=False):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.zeros(()))
        self.child = nn.Identity()
        self.invalid = invalid
        self.lookup = {}
        for i in range(len(data)):
            item = data[i]
            self.lookup[
                item.image.numpy().tobytes(), ShapeColorTokenizer.vocabulary.index(item.shape)
            ] = ShapeColorTokenizer().encode_answer(item.answer)

    def forward(self, input_ids, images, *, recurrence_depth, attention_mask):
        assert input_ids.shape[1] == 6
        assert attention_mask.all()
        random.random()
        np.random.random()
        torch.rand(1)
        logits = torch.zeros(len(input_ids), self.config.num_patches + 6, 17)
        for i, (ids, image) in enumerate(zip(input_ids, images, strict=True)):
            target = (
                11 if self.invalid else self.lookup.get((image.numpy().tobytes(), ids[4].item()), 6)
            )
            logits[i, -1, target] = 10
        return logits


def test_oracle_metrics_controls_and_rng_preservation(manifest, config):
    data = ShapeColorDataset(manifest, "validation")
    model = Oracle(config, data)
    model.train()
    model.child.eval()
    model.weight.grad = torch.tensor(2.0)
    training = SyntheticTrainingConfig(batch_size=31)
    rng = torch.get_rng_state().clone()
    py_rng = random.getstate()
    np_rng = np.random.get_state()
    summary, rows = diagnose_shape_grounding(model, data, training)
    assert summary["accuracy"] == summary["all_three"]["accuracy"] == 1
    assert summary["circle_square_pair"]["accuracy"] == 1
    assert summary["circle_square_same_prediction"]["fraction"] == 0
    controls = evaluate_shape_controls(model, data, training, seeds=(0, 1), correct=summary)
    assert controls["blank"]["accuracy"] == 0.25
    for row in controls["shuffled_questions"]:
        assert row["metrics"]["accuracy"] == row["same_answer_pairing_fraction"]
    for row in controls["shuffled_images"]:
        assert row["metrics"]["accuracy"] == row["same_answer_pairing_fraction"]
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == py_rng
    now = np.random.get_state()
    assert np_rng[0] == now[0] and np.array_equal(np_rng[1], now[1]) and np_rng[2:] == now[2:]
    assert model.training and not model.child.training and model.weight.grad.item() == 2
    assert len(rows) == len(data) and all(r["target_probability"] > 0.99 for r in rows)
    assert assess_shape_grounding(summary, summary, controls)["passed"]
    invalid, _ = diagnose_shape_grounding(Oracle(config, data, invalid=True), data, training)
    assert invalid["invalid_predictions"] == len(data) and invalid["accuracy"] == 0


def test_controls_keep_targets_and_layout_across_image_boundaries(manifest, config):
    data = ShapeColorDataset(manifest, "train")
    batches = list(shape_color_loader(data, config, SyntheticTrainingConfig(batch_size=5)))
    questions = [[1, 2, 0] for _ in data.records]
    images = list(reversed(range(len(data.records))))
    for kwargs in ({"blank": True}, {"images": images}, {"questions": questions}):
        modified = list(control_batches(batches, data, **kwargs))
        offset = 0
        for before, after in zip(batches, modified, strict=True):
            assert torch.equal(before.target_mask, after.target_mask)
            assert torch.equal(before.attention_mask, after.attention_mask)
            assert torch.equal(before.input_ids[:, -1], after.input_ids[:, -1])
            if "questions" in kwargs:
                assert torch.equal(before.images, after.images)
                for row in range(len(after.input_ids)):
                    shape = SHAPES[questions[(offset + row) // 3][(offset + row) % 3]]
                    assert after.input_ids[row, 4] == ShapeColorTokenizer.vocabulary.index(shape)
            else:
                assert torch.equal(before.input_ids, after.input_ids)
                for row in range(len(after.input_ids)):
                    expected = (
                        torch.zeros_like(before.images[row])
                        if "blank" in kwargs
                        else data[3 * images[(offset + row) // 3]].image
                    )
                    assert torch.equal(after.images[row], expected)
            offset += len(before.input_ids)


def test_partial_pass_training_and_frozen_checkpoint(manifest, config, tmp_path):
    model = MultimodalLoopTransformer(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0, foreach=False, fused=False
    )
    training = SyntheticTrainingConfig()
    budget = ShapeGroundingConfig(15, 5)
    history = train_shape_grounding(model, optimizer, manifest, training, budget)
    assert history[-1]["completed_passes"] == 1 and history[-1]["batches_in_pass"] == 1
    assert history[-1]["examples_seen"] == 464
    path = tmp_path / "last.pt"
    save_shape_checkpoint(
        path, model, optimizer, manifest=manifest, training=training, budget=budget, history=history
    )
    rng = torch.get_rng_state().clone()
    loaded, loaded_manifest, settings, payload = load_shape_checkpoint(path)
    assert torch.equal(rng, torch.get_rng_state())
    assert settings == training and loaded_manifest.content == manifest.content
    assert payload["completed_steps"] == 15 and "rng_state" in payload
    assert all(s["step"].item() == 15 for s in payload["optimizer_state"]["state"].values())
    data = ShapeColorDataset(manifest, "validation")
    before = diagnose_shape_grounding(model, data, training)
    after = diagnose_shape_grounding(loaded, data, training)
    assert before == after
    for key, value in (
        ("examples_seen", 999),
        ("batches_in_pass", 2),
        ("manifest_sha256", "wrong"),
    ):
        bad = copy.deepcopy(payload)
        bad[key] = value
        torch.save(bad, path)
        with pytest.raises(ValueError):
            load_shape_checkpoint(path)
    bad = copy.deepcopy(payload)
    bad["history"][1]["examples_seen"] += 1
    torch.save(bad, path)
    with pytest.raises(ValueError):
        load_shape_checkpoint(path)
    # A CUDA-origin checkpoint can be inspected on CPU without installing its RNG.
    cuda_origin = {
        **payload,
        "backend": "cuda",
        "device_rng_state": torch.zeros(20, dtype=torch.uint8),
    }
    torch.save(cuda_origin, path)
    loaded_cuda, _, _, _ = load_shape_checkpoint(path, device="cpu")
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), loaded_cuda.parameters(), strict=True)
    )
    payload["kind"] = "relational_color"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incompatible"):
        load_shape_checkpoint(path)


def test_cli_smoke_notebook_and_archive(manifest, config, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.content)
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)))
    root = tmp_path / "smoke"
    command = [
        sys.executable,
        "scripts/train_shape_grounding.py",
        "--manifest",
        str(manifest_path),
        "--model-config",
        str(config_path),
        "--output-dir",
        str(root / "training"),
        "--max-steps",
        "2",
        "--evaluation-interval",
        "1",
        "--smoke",
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    again = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert again.returncode and "fresh output" in again.stderr
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_shape_grounding.py",
            "--checkpoint",
            str(root / "training" / "last.pt"),
            "--output-dir",
            str(root / "diagnosis"),
            "--shuffle-seeds",
            "0",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads((root / "diagnosis" / "summary.json").read_text())
    assert set(summary["splits"]) == {"train", "validation"}
    assert summary["completed_steps"] == 2 and summary["examples_seen"] == 64
    assert summary["smoke"] is True
    assert len((root / "diagnosis" / "train_examples.jsonl").read_text().splitlines()) == 432
    assert "<svg" in (root / "diagnosis" / "inspection.html").read_text()
    run = ShapeGroundingRun(ROOT, root, tmp_path / "baseline", asdict(config))
    with pytest.raises(ValueError, match="complete training"):
        archive_shape_grounding(run)
    (root / "provenance").mkdir()
    (root / "provenance" / "final.json").write_text("{}")
    (root / "staging").mkdir()
    (root / "staging" / "excluded.txt").write_text("source archive")
    archive = archive_shape_grounding(run)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert any(n.endswith("training/last.pt") for n in names)
        assert not any("staging" in n for n in names)
    with pytest.raises(FileExistsError):
        archive_shape_grounding(run)
    notebook = json.loads((ROOT / "notebooks/kaggle_milestone2_shape_grounding.ipynb").read_text())
    code = []
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            assert c["execution_count"] is None and c["outputs"] == []
            compile("".join(c["source"]), "<notebook>", "exec")
            code.append("".join(c["source"]))
    assert "run_shape_grounding(run)" in "\n".join(code)


@pytest.mark.parametrize("field", ["max_steps", "evaluation_interval"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_budget(field, value):
    with pytest.raises((ValueError, TypeError)):
        ShapeGroundingConfig(**{field: value})


def test_assessment_boundaries_and_incomplete_groups():
    def metrics(total, correct):
        return {
            "breakdowns": {
                "shape": [
                    {
                        "shape": shape,
                        "total": total,
                        "correct": correct,
                        "accuracy": correct / total,
                    }
                    for shape in SHAPES
                ]
            }
        }

    train, validation = metrics(20, 19), metrics(10, 9)
    controls = {
        "accuracy_gaps": {
            key: 0.30
            for key in (
                "correct_minus_blank",
                "correct_minus_shuffled_images_mean",
                "correct_minus_shuffled_questions_mean",
            )
        }
    }
    result = assess_shape_grounding(train, validation, controls)
    assert result["passed"] and len(result["gates"]) == 9
    controls["accuracy_gaps"]["correct_minus_blank"] = 0.30 - 1e-12
    assert not assess_shape_grounding(train, validation, controls)["passed"]
    train["breakdowns"]["shape"].pop()
    with pytest.raises(ValueError, match="each shape"):
        assess_shape_grounding(train, validation, controls)


def test_notebook_fixed_workflow(monkeypatch, tmp_path, config):
    from multimodal_loop.eval import kaggle_shape_grounding as module
    from multimodal_loop.eval.baseline_artifacts import BASELINE_MANIFEST_SHA256
    from multimodal_loop.train.kaggle import file_hash

    repo, baseline, root = (tmp_path / name for name in ("repo", "baseline", "run"))
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "relational_baseline.yaml").write_text(yaml.safe_dump(asdict(config)))
    (baseline / "training").mkdir(parents=True)
    (baseline / "training" / "settings.json").write_text(
        json.dumps({"model": asdict(config), "training": asdict(SyntheticTrainingConfig())})
    )
    monkeypatch.setattr(module, "repository_revision", lambda p: "committed-revision")
    monkeypatch.setattr(module, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(module, "stage_baseline", lambda s, d: baseline)
    monkeypatch.setattr(module, "audit_baseline", lambda b: {"checkpoint_sha256": "original-hash"})
    run = module.prepare_shape_grounding(repo, root, baseline)
    protocol = json.loads((root / "provenance" / "protocol.json").read_text())
    assert protocol["max_steps"] == 2880 and protocol["qa_presentations"] == 92160
    assert protocol["splits"] == ["train", "validation"]
    calls = []

    def command(repo, output, name, args):
        calls.append((name, args))
        if name == "train":
            assert args[args.index("--max-steps") + 1] == "2880"
            assert "--resume" not in args and "--checkpoint" not in args
            (root / "training").mkdir()
            (root / "training" / "last.pt").write_bytes(b"new weights")
        else:
            assert args[args.index("--shuffle-seeds") + 1 :] == ["0", "1", "2", "3", "4"]
            (root / "diagnosis").mkdir()
            report = {
                "kind": "direct_shape_color_diagnostics",
                "format_version": 1,
                "smoke": False,
                "checkpoint_sha256": file_hash(root / "training" / "last.pt"),
                "manifest_sha256": BASELINE_MANIFEST_SHA256,
                "completed_steps": 2880,
                "examples_seen": 92160,
                "model": asdict(config),
                "training": asdict(SyntheticTrainingConfig()),
                "budget": {"max_steps": 2880, "evaluation_interval": 288},
                "evaluation": {
                    "splits": ["train", "validation"],
                    "shuffle_seeds": list(range(5)),
                    "recurrence_depth": 2,
                    "batch_size": 32,
                },
                "splits": {"train": {"total": 6912}, "validation": {"total": 1728}},
            }
            (root / "diagnosis" / "summary.json").write_text(json.dumps(report))
            (root / "diagnosis" / "controls.json").write_text("{}")

    monkeypatch.setattr(module, "run_logged_command", command)
    module.run_shape_grounding(run)
    assert [name for name, _ in calls] == ["train", "evaluate"]
    assert (root / "provenance" / "final.json").exists()
    with pytest.raises(ValueError, match="fresh run"):
        module.run_shape_grounding(run)
    with pytest.raises(ValueError, match="fresh run"):
        module.prepare_shape_grounding(repo, root, baseline)
