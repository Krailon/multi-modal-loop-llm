"""Independent CNN inputs, budget, frozen predictions and artifact comparison."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.nn import functional as F

from multimodal_loop.data.matched_size import derive_matched_size
from multimodal_loop.data.multi_arrangement import MultiArrangementDataset, derive_multi_arrangement
from multimodal_loop.data.quartet_fit import derive_quartet_fit
from multimodal_loop.data.quartet_transfer import QuartetTransferDataset
from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.eval.cnn_baseline import evaluate_cnn
from multimodal_loop.eval.cnn_reference import REFERENCE_HASHES, read_cnn_reference
from multimodal_loop.eval.focused_diagnosis import validate_rows
from multimodal_loop.eval.multi_arrangement import aggregate_role
from multimodal_loop.eval.quartet_transfer import population_metrics
from multimodal_loop.eval.shape_grounding import shape_prediction_rows
from multimodal_loop.model.cnn_baseline import CNNBaseline, CNNConfig, forward_macs
from multimodal_loop.train.cnn_baseline import (
    CNNTrainingConfig,
    cnn_loader,
    collate_cnn,
    predict_cnn,
)
from multimodal_loop.train.cnn_checkpoint import load_cnn_checkpoint, validate_cnn_protocol
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.multi_arrangement_checkpoint import load_multi_checkpoint
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig

ROOT = Path(__file__).resolve().parents[1]


def test_model_gradients_question_dependence_and_inputs():
    torch.manual_seed(8)
    model = CNNBaseline()
    assert sum(p.numel() for p in model.parameters()) == 145329
    assert forward_macs(model.config) == 133019776
    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    questions = torch.tensor([[0, 1, 2, 3, 14, 5], [0, 1, 2, 3, 15, 5]])
    logits = model(images, questions)
    assert logits.shape == (2, 17)
    assert not torch.equal(model(images[:1], questions[:1]), model(images[:1], questions[1:]))
    F.cross_entropy(logits, torch.tensor([6, 9])).backward()
    assert images.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.visual[0].weight.grad.abs().sum() > 0
    assert model.question.weight.grad[14].abs().sum() > 0
    with pytest.raises(ValueError, match="excluding answers"):
        model(images, torch.cat((questions, questions[:, :1]), dim=1))
    with pytest.raises(ValueError, match="images"):
        model(images[:, :, :16], questions)


def test_external_targets_and_metadata_exclusion():
    class Example:
        image = torch.zeros(3, 32, 32)
        question = "What color is the circle?"
        answer = "blue"

        def __getattr__(self, name):
            raise AssertionError(f"metadata accessed: {name}")

    a = Example()
    images, ids, targets = collate_cnn([a])
    assert ids.tolist() == [[0, 1, 2, 3, 15, 5]] and targets.tolist() == [8]
    a.answer = "yellow"
    changed = collate_cnn([a])
    assert torch.equal(images, changed[0]) and torch.equal(ids, changed[1])
    assert changed[2].tolist() == [9]


@pytest.fixture(scope="module")
def corpus():
    config = RelationalCorpusConfig(
        object_sizes=(8,),
        train_geometry_count=2,
        validation_geometry_count=1,
        test_geometry_count=1,
    )
    source = parse_relational_manifest(
        json.dumps(
            {
                "kind": "relational_color_rows",
                "format_version": 1,
                "software": {},
                "config": asdict(config),
                "splits": {
                    s: [asdict(r) for r in rs] for s, rs in build_relational_splits(config).items()
                },
            }
        )
    )
    return derive_multi_arrangement(derive_quartet_fit(derive_matched_size(source)))


def test_sampler_budget_and_protocol(corpus):
    import random

    dataset = MultiArrangementDataset(corpus)
    training = CNNTrainingConfig()
    expected = list(range(len(dataset)))
    random.Random("0:train:3").shuffle(expected)
    assert list(cnn_loader(dataset, training, epoch=3).sampler) == expected
    assert list(cnn_loader(dataset, training).sampler) == list(range(len(dataset)))
    with pytest.raises(ValueError, match="fixed CNN"):
        validate_cnn_protocol(corpus, CNNConfig(), training, ShapeGroundingConfig(8640, 216))
    pinned = SimpleNamespace(
        sha256=__import__(
            "multimodal_loop.train.cnn_checkpoint", fromlist=["MANIFEST_SHA256"]
        ).MANIFEST_SHA256
    )
    validate_cnn_protocol(pinned, CNNConfig(), training, ShapeGroundingConfig(8640, 216))
    for model, train, budget in (
        (CNNConfig(channels=(4,)), training, ShapeGroundingConfig(8640, 216)),
        (CNNConfig(), CNNTrainingConfig(seed=1), ShapeGroundingConfig(8640, 216)),
        (CNNConfig(), training, ShapeGroundingConfig(8641, 216)),
    ):
        with pytest.raises(ValueError, match="fixed CNN"):
            validate_cnn_protocol(pinned, model, train, budget)


def make_reference(root, corpus, model):
    rows, metrics = {}, {}
    for p in corpus.populations:
        dataset = QuartetTransferDataset(p)
        _, details = predict_cnn(model, dataset, CNNTrainingConfig())
        rows[p.name] = shape_prediction_rows(dataset, details)
        metrics[p.name] = population_metrics(rows[p.name], p.records)
    summary = {
        "manifest_sha256": corpus.sha256,
        "checkpoint_sha256": "fixture-weights",
        "smoke": True,
        "completed_steps": 1,
        "arrangements": metrics,
        "aggregates": {
            r: aggregate_role(corpus.populations, rows, r) for r in ("training_fit", "transfer")
        },
    }
    for name in ("data", "diagnosis", "provenance"):
        (root / name).mkdir(parents=True)
    (root / "data/manifest.json").write_text(corpus.content)
    (root / "diagnosis/summary.json").write_text(json.dumps(summary))
    (root / "diagnosis/predictions.jsonl").write_text(
        "".join(
            json.dumps({**r, "arrangement": p.name, "role": p.role}) + "\n"
            for p in corpus.populations
            for r in rows[p.name]
        )
    )
    (root / "provenance/final.json").write_text(
        json.dumps(
            {
                **{n: file_hash(root / n) for n in REFERENCE_HASHES},
                "training/last.pt": "fixture-weights",
            }
        )
    )
    return summary


def test_cli_checkpoint_evaluation_and_reference_audit(corpus, tmp_path):
    config = CNNConfig(channels=(4, 4), question_dim=4, hidden_dim=8)
    cfg, manifest = tmp_path / "config.yaml", tmp_path / "manifest.json"
    cfg.write_text(yaml.safe_dump(asdict(config)))
    manifest.write_text(corpus.content)
    command = [
        sys.executable,
        "scripts/train_cnn_baseline.py",
        "--manifest",
        str(manifest),
        "--model-config",
        str(cfg),
        "--output-dir",
        str(tmp_path / "training"),
        "--max-steps",
        "2",
        "--evaluation-interval",
        "1",
        "--smoke",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = tmp_path / "training/last.pt"
    rng = torch.get_rng_state().clone()
    model, restored, training, payload = load_cnn_checkpoint(checkpoint)
    assert torch.equal(rng, torch.get_rng_state()) and restored.sha256 == corpus.sha256
    assert payload["completed_steps"] == 2 and payload["examples_seen"] == 64
    assert payload["optimizer_state"]["state"] and payload["rng_state"]
    assert [h["completed_steps"] for h in payload["history"]] == [0, 1, 2]
    assert all("validation" not in h and "training_fit" in h for h in payload["history"])
    dataset = MultiArrangementDataset(corpus)
    model.train()
    before_rng = torch.get_rng_state().clone()
    first = predict_cnn(model, dataset, training)
    assert model.training and torch.equal(before_rng, torch.get_rng_state())
    assert first == predict_cnn(load_cnn_checkpoint(checkpoint)[0], dataset, training)
    with pytest.raises(ValueError, match="incompatible"):
        load_multi_checkpoint(checkpoint)
    bad = copy.deepcopy(payload)
    bad["completed_steps"] = 1
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="progress"):
        load_cnn_checkpoint(tmp_path / "bad.pt")
    reference = tmp_path / "reference"
    make_reference(reference, corpus, model)
    read_cnn_reference(reference, smoke=True)
    with pytest.raises(ValueError, match="identity"):
        read_cnn_reference(reference)
    digest = file_hash(checkpoint)
    report = evaluate_cnn(checkpoint, tmp_path / "diagnosis", reference)
    assert file_hash(checkpoint) == digest
    comparison = json.loads((tmp_path / "diagnosis/comparison.json").read_text())
    assert comparison["transfer_aggregate"]["accuracy"]["difference"] == 0
    assert all(v["prediction_changes"] == 0 for v in comparison["matched_changes"].values())
    records = [
        json.loads(line)
        for line in (tmp_path / "diagnosis/predictions.jsonl").read_text().splitlines()
    ]
    for p in corpus.populations:
        rows = [r for r in records if r["arrangement"] == p.name]
        validate_rows(
            rows,
            SimpleNamespace(splits={p.name: p.records}),
            p.name,
            report["arrangements"][p.name]["summary"],
        )
    with pytest.raises(ValueError, match="fresh"):
        evaluate_cnn(checkpoint, tmp_path / "diagnosis", reference)
    (reference / "diagnosis/predictions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash"):
        read_cnn_reference(reference, smoke=True)


def test_notebook_contract():
    notebook = json.loads((ROOT / "notebooks/kaggle_milestone2_cnn_baseline.ipynb").read_text())
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "<notebook>", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
            code.append(source)
    joined = "\n".join(code)
    assert 'SOURCE = REPO_DIR / "milestone2_multi_arrangement_artifacts.zip"' in joined
    assert "run_cnn_baseline(run)" in joined
    assert 'comparison["arrangements"]' in joined


@pytest.mark.parametrize("fault", [None, "exposures", "population"])
def test_kaggle_budget_and_archive_integrity(tmp_path, monkeypatch, corpus, fault):
    import zipfile

    from multimodal_loop.eval import kaggle_cnn_baseline as module

    repo, root = tmp_path / "repo", tmp_path / "run"
    source = repo / "milestone2_multi_arrangement_artifacts.zip"
    (repo / "configs").mkdir(parents=True)
    source.write_bytes(b"fixture")
    (repo / "configs/cnn_baseline.yaml").write_text(yaml.safe_dump(asdict(CNNConfig())))
    monkeypatch.setattr(module, "repository_revision", lambda _: "revision")
    monkeypatch.setattr(module, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(module, "read_cnn_reference", lambda _: (corpus, {}, {}, {}))
    run = module.prepare_cnn_baseline(repo, root, source)
    protocol = json.loads((root / "provenance/protocol.json").read_text())
    assert protocol["qa_presentations"] == 276480 and protocol["presentations_per_qa"] == 40
    assert protocol["parameters"] == 145329 and protocol["recurrence_depth"] is None
    monkeypatch.setattr(
        module,
        "load_cnn_checkpoint",
        lambda _: (None, corpus, None, {"smoke": False, "completed_steps": 8640}),
    )
    calls = []

    def command(repo, root, name, arguments):
        calls.append(name)
        assert "--smoke" not in arguments and "--resume" not in arguments
        if name == "train":
            assert arguments[arguments.index("--max-steps") + 1] == "8640"
            assert arguments[arguments.index("--evaluation-interval") + 1] == "216"
            assert "--reference-source" not in arguments
            (root / "training").mkdir()
            (root / "training/last.pt").write_bytes(b"cnn")
            (root / "training/presentations.json").write_bytes(
                b"{}" if fault == "exposures" else (root / "data/presentations.json").read_bytes()
            )
        else:
            assert arguments[arguments.index("--reference-source") + 1] == str(source)
            (root / "diagnosis").mkdir()
            module.write_json(
                root / "diagnosis/summary.json",
                {
                    "checkpoint_sha256": file_hash(root / "training/last.pt"),
                    "manifest_sha256": corpus.sha256,
                    "smoke": False,
                    "completed_steps": 8640,
                    "examples_seen": 276480,
                    "aggregates": {
                        role: {
                            "summary": {"total": 6912},
                            "families": {"total": 0 if fault == "population" else 576},
                        }
                        for role in ("training_fit", "transfer")
                    },
                },
            )

    monkeypatch.setattr(module, "run_logged_command", command)
    if fault:
        with pytest.raises(ValueError):
            module.run_cnn_baseline(run)
        assert not (root / "provenance/final.json").exists()
        return
    module.run_cnn_baseline(run)
    assert calls == ["train", "evaluate"]
    archive = module.archive_cnn_baseline(run)
    with zipfile.ZipFile(archive) as z:
        assert "run/training/last.pt" in z.namelist()
        assert not any("fixture" in n for n in z.namelist())
    (root / "training/last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.archive_cnn_baseline(run)
    with pytest.raises(ValueError, match="fresh"):
        module.run_cnn_baseline(run)
