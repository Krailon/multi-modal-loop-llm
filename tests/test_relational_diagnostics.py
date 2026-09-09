"""Frozen diagnostic metrics, metadata isolation, staging, and notebook orchestration."""

import copy
import json
import random
import stat
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import (
    RelationalColorDataset,
    parse_relational_manifest,
)
from multimodal_loop.data.text import RelationalColorTokenizer
from multimodal_loop.eval import baseline_artifacts, kaggle_diagnostics
from multimodal_loop.eval.baseline_artifacts import stage_baseline
from multimodal_loop.eval.relational import evaluate_relational
from multimodal_loop.eval.relational_diagnostics import diagnose_relational, inspection_html
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.checkpoint import _capture_rng, _restore_rng
from multimodal_loop.train.relational import relational_loader
from multimodal_loop.train.synthetic import SyntheticTrainingConfig

ROOT = Path(__file__).resolve().parents[1]


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
    records = build_relational_splits(config)
    return parse_relational_manifest(
        json.dumps(
            {
                "kind": "relational_color_rows",
                "format_version": 1,
                "software": {},
                "config": asdict(config),
                "splits": {
                    split: [
                        asdict(replace(r, questions=r.questions[1:] + r.questions[:1]))
                        for r in rows
                    ]
                    for split, rows in records.items()
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


class PixelQuestionOracle(nn.Module):
    def __init__(self, dataset, config, behavior):
        super().__init__()
        self.config = config
        self.anchor = nn.Parameter(torch.zeros(()))
        self.behavior = behavior
        tokenizer = RelationalColorTokenizer()
        self.lookup = {}
        for i in range(len(dataset)):
            ex = dataset[i]
            if behavior == "middle":
                answer = ex.scene.objects[1].color
            elif behavior == "absent":
                answer = next(
                    c
                    for c in ("red", "green", "blue", "yellow")
                    if c not in [o.color for o in ex.scene.objects]
                )
            else:
                answer = ex.answer
            self.lookup[(ex.image.numpy().tobytes(), tokenizer.encode_question(ex.question))] = (
                tokenizer.encode_answer(answer)
            )

    def forward(self, ids, images, *, recurrence_depth, attention_mask):
        assert not self.training and torch.is_inference_mode_enabled()
        assert ids.shape[1] == 11 and recurrence_depth == 3
        assert attention_mask.shape == (15, 15) and attention_mask.all()
        predicted = [
            0
            if self.behavior == "invalid"
            else self.lookup[(image.numpy().tobytes(), tuple(question.tolist()))]
            for image, question in zip(images, ids, strict=True)
        ]
        logits = torch.zeros(len(ids), 17)
        logits[torch.arange(len(ids)), predicted] = 5
        return logits[:, None].expand(-1, 15, -1)


@pytest.mark.parametrize(
    "behavior,accuracy", [("perfect", 1.0), ("middle", 0.5), ("absent", 0.0), ("invalid", 0.0)]
)
def test_detailed_predictions_groups_and_position_confusion(manifest, config, behavior, accuracy):
    dataset = RelationalColorDataset(manifest, "validation")
    model = PixelQuestionOracle(dataset, config, behavior)
    result = diagnose_relational(
        model, dataset, split="validation", recurrence_depth=3, batch_size=127
    )
    assert result.summary["accuracy"] == accuracy and len(result.examples) == 576
    expected_group_accuracy = 1.0 if behavior == "perfect" else 0.0
    assert result.summary["all_four"]["accuracy"] == expected_group_accuracy
    assert result.summary["different_answer_pairs"]["accuracy"] == expected_group_accuracy
    assert result.summary["invalid_predictions"] == (576 if behavior == "invalid" else 0)
    assert [row["example_index"] for row in result.examples] == list(range(576))
    for i, row in enumerate(result.examples):
        assert row["image_index"] == i // 4 and row["question_index"] == i % 4
        qa = dataset.records[i // 4].questions[i % 4]
        assert row["question"] == qa.question and row["anchor_shape"] == qa.query.anchor_shape
        logits = torch.zeros(17)
        logits[row["prediction_id"]] = 5
        assert row["loss"] == pytest.approx(
            F.cross_entropy(logits[None], torch.tensor([row["target_id"]])).item()
        )
        probabilities = logits.softmax(-1)
        assert row["confidence"] == pytest.approx(probabilities.max().item())
        assert row["target_probability"] == pytest.approx(probabilities[row["target_id"]].item())
    for axis, groups in result.summary["breakdowns"].items():
        if axis == "target_position_confusion":
            continue
        assert sum(row["total"] for row in groups) == 576
        assert sum(row["correct"] for row in groups) == int(576 * accuracy)
        assert sum(row["loss"] * row["total"] for row in groups) / 576 == pytest.approx(
            result.summary["loss"]
        )
    positions = result.summary["breakdowns"]["target_position_confusion"]
    for target, count in (("left", 144), ("middle", 288), ("right", 144)):
        expected = {
            "perfect": target,
            "middle": "middle",
            "absent": "absent_color",
            "invalid": "invalid_token",
        }[behavior]
        assert positions[target][expected] == count and sum(positions[target].values()) == count
    other = diagnose_relational(
        model, dataset, split="validation", recurrence_depth=3, batch_size=113
    )
    assert result.examples == other.examples
    assert result.summary["breakdowns"] == other.summary["breakdowns"]
    assert result.summary["loss"] == pytest.approx(other.summary["loss"])


@pytest.mark.parametrize("fail", [False, True])
def test_real_forward_preserves_modes_rng_gradients_and_metadata_is_post_inference(
    manifest, config, monkeypatch, fail
):
    base = RelationalColorDataset(manifest, "validation")

    # Avoid scene/query metadata on the samples consumed by the collator.
    class Dataset(RelationalColorDataset):
        def __getitem__(self, index):
            ex = super().__getitem__(index)
            return SimpleNamespace(image=ex.image, question=ex.question, answer=ex.answer)

    dataset = Dataset(manifest, "validation")
    model = MultimodalLoopTransformer(config)
    model.coda.eval()
    model.embeddings.text_embedding.weight.grad = torch.ones_like(
        model.embeddings.text_embedding.weight
    )
    modes = [m.training for m in model.modules()]
    weights = copy.deepcopy(model.state_dict())
    gradient = model.embeddings.text_embedding.weight.grad.clone()
    original_forward = model.forward

    def forward(ids, images, **kwargs):
        assert ids.shape[1] == 11
        random.random()
        np.random.rand()
        torch.rand(1)
        if fail:
            raise RuntimeError("injected")
        return original_forward(ids, images, **kwargs)

    monkeypatch.setattr(model, "forward", forward)
    state = _capture_rng()
    if fail:
        with pytest.raises(RuntimeError, match="injected"):
            diagnose_relational(
                model, dataset, split="validation", recurrence_depth=2, batch_size=127
            )
    else:
        diagnosis = diagnose_relational(
            model, dataset, split="validation", recurrence_depth=2, batch_size=127
        )
        ordinary = evaluate_relational(
            model,
            relational_loader(base, config, SyntheticTrainingConfig(batch_size=127)),
            recurrence_depth=2,
        )
        assert {key: diagnosis.summary[key] for key in asdict(ordinary)} == asdict(ordinary)
    after = _capture_rng()
    assert torch.equal(state.pop("torch_cpu"), after.pop("torch_cpu")) and state == after
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(gradient, model.embeddings.text_embedding.weight.grad)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    with pytest.raises(ValueError, match="only train and validation"):
        diagnose_relational(model, dataset, split="test", recurrence_depth=2)


def test_preview_selection_is_explicit_and_deterministic(manifest, config):
    datasets = {split: RelationalColorDataset(manifest, split) for split in ("train", "validation")}
    diagnoses = {
        split: diagnose_relational(
            PixelQuestionOracle(dataset, config, "invalid"),
            dataset,
            split=split,
            recurrence_depth=3,
            batch_size=127,
        )
        for split, dataset in datasets.items()
    }
    text = inspection_html(diagnoses, datasets)
    assert text.count("<article data-example=") == 16
    assert text.count("<tr><td>") == 64
    assert text.count('data-example="0"') == 2 and 'data-example="8"' not in text
    assert "not a representative" in text and "invalid token 0" in text
    assert text == inspection_html(diagnoses, datasets)


@pytest.mark.parametrize(
    "bad",
    [
        "../escape",
        "/absolute",
        "milestone2_baseline/../../escape",
        "milestone2_baseline\\escape",
        "symlink",
        "duplicate",
    ],
)
def test_archive_rejects_unsafe_members_before_writing(tmp_path, bad):
    source = tmp_path / "input.zip"
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("milestone2_baseline/training/last.pt", b"fixture")
        if bad == "symlink":
            item = zipfile.ZipInfo("milestone2_baseline/link")
            item.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(item, "../outside")
        elif bad == "duplicate":
            with pytest.warns(UserWarning):
                z.writestr("milestone2_baseline/training/last.pt", b"again")
        else:
            z.writestr(bad, b"escape")
    destination = tmp_path / "staging"
    with pytest.raises(ValueError):
        stage_baseline(source, destination)
    assert not destination.exists()


def test_archive_staging_and_directory_reuse_preserve_source(tmp_path):
    source = tmp_path / "input.zip"
    with zipfile.ZipFile(source, "w") as z:
        z.writestr("milestone2_baseline/training/last.pt", b"fixture")
    before = source.read_bytes()
    root = stage_baseline(source, tmp_path / "staging")
    assert (root / "training" / "last.pt").read_bytes() == b"fixture"
    assert source.read_bytes() == before
    assert stage_baseline(root, tmp_path / "unused") == root
    assert stage_baseline(root.parent, tmp_path / "unused") == root
    assert not (tmp_path / "unused").exists()
    with pytest.raises(ValueError, match="fresh"):
        stage_baseline(source, tmp_path / "staging")
    with pytest.raises(ValueError, match="recorded frozen"):
        baseline_artifacts.audit_baseline(root)


def test_notebook_orchestration_has_no_training_or_test_calls(tmp_path, monkeypatch):
    repo, root, source = tmp_path / "repo", tmp_path / "run", tmp_path / "source"
    repo.mkdir()
    (source / "training").mkdir(parents=True)
    (source / "training" / "last.pt").write_bytes(b"fixture")
    checkpoint_hash = kaggle_diagnostics.file_hash(source / "training" / "last.pt")
    monkeypatch.setattr(
        kaggle_diagnostics, "repository_revision", lambda repo: "diagnostic-revision"
    )
    monkeypatch.setattr(kaggle_diagnostics, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(
        kaggle_diagnostics,
        "audit_baseline",
        lambda path: {
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": "manifest",
            "training_revision": "training-revision",
        },
    )
    run = kaggle_diagnostics.prepare_diagnostics(repo, root, source)
    calls = []

    def command(repo, root, name, arguments):
        calls.append(arguments)
        assert arguments[0] == "scripts/diagnose_relational.py"
        output = Path(arguments[arguments.index("--output-dir") + 1])
        output.mkdir()
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "checkpoint_sha256": checkpoint_hash,
                    "evaluation": {
                        "splits": ["train", "validation"],
                        "recurrence_depth": 2,
                        "batch_size": 32,
                    },
                    "splits": {"train": {"total": 9216}, "validation": {"total": 2304}},
                }
            )
        )

    monkeypatch.setattr(kaggle_diagnostics, "run_logged_command", command)
    kaggle_diagnostics.run_diagnostics(run)
    assert len(calls) == 1 and "--resume" not in calls[0]
    archive = kaggle_diagnostics.diagnostic_archive(run)
    with zipfile.ZipFile(archive) as z:
        assert f"{root.name}/diagnosis/summary.json" in z.namelist()
        assert all("staging" not in n and "last.pt" not in n for n in z.namelist())
    assert (source / "training" / "last.pt").read_bytes() == b"fixture"
    with pytest.raises(ValueError, match="already exist"):
        kaggle_diagnostics.run_diagnostics(run)
    with pytest.raises(ValueError, match="fresh"):
        kaggle_diagnostics.prepare_diagnostics(repo, root, source)
    with pytest.raises(FileExistsError):
        kaggle_diagnostics.diagnostic_archive(run)


def test_notebook_compiles_with_empty_outputs():
    notebook = json.loads((ROOT / "notebooks/kaggle_milestone2_diagnostics.ipynb").read_text())
    assert notebook["nbformat"] == 4
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            source = "".join(cell["source"])
            compile(source, cell["id"], "exec")
            code.append(source)
    source = "\n".join(code)
    assert "report = run_diagnostics(run)" in source and "BASELINE_SOURCE" in source
    assert "train_run(" not in source and "load_relational_checkpoint(" not in source


def test_cli_writes_only_train_validation_diagnostics_and_preserves_sources(
    tmp_path, monkeypatch, manifest, config
):
    import runpy

    source = tmp_path / "source"
    (source / "training").mkdir(parents=True)
    (source / "validation").mkdir()
    checkpoint = source / "training" / "last.pt"
    checkpoint.write_bytes(b"fixture")
    controls = source / "validation" / "controls.json"
    controls.write_text("original controls")
    output = tmp_path / "diagnosis"
    model = MultimodalLoopTransformer(config)
    weights = copy.deepcopy(model.state_dict())
    globals_ = runpy.run_path(str(ROOT / "scripts/diagnose_relational.py"))["main"].__globals__
    loaded = []
    monkeypatch.setitem(
        globals_,
        "load_relational_checkpoint",
        lambda path, **kw: (
            loaded.append((path, kw))
            or SimpleNamespace(
                model=model, manifest=manifest, completed_epochs=10, completed_steps=2880
            )
        ),
    )
    old = dict(total=576, correct=0, accuracy=0.0, invalid_predictions=576, loss=3.0)
    monkeypatch.setitem(
        globals_,
        "audit_baseline",
        lambda path: {
            "checkpoint_sha256": globals_["file_hash"](checkpoint),
            "manifest_sha256": manifest.sha256,
            "training_revision": "original-revision",
            "controls": {"results": {"correct": old}},
        },
    )
    monkeypatch.setitem(globals_, "repository_revision", lambda repo: "diagnostic-revision")
    splits = []

    def dataset(manifest, split):
        assert split in ("train", "validation")
        splits.append(split)
        return RelationalColorDataset(manifest, split)

    monkeypatch.setitem(globals_, "RelationalColorDataset", dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        ["diagnose_relational.py", "--baseline-dir", str(source), "--output-dir", str(output)],
    )
    globals_["main"]()
    assert splits == ["train", "validation"]
    assert loaded == [(checkpoint, {"device": "cuda:0"})]
    report = json.loads((output / "summary.json").read_text())
    assert report["training_revision"] == "original-revision"
    assert report["diagnostic_revision"] == "diagnostic-revision"
    assert report["validation_comparison"]["original"] == old
    for split in ("train", "validation"):
        rows = [
            json.loads(line)
            for line in (output / f"{split}_examples.jsonl").read_text().splitlines()
        ]
        assert len(rows) == report["splits"][split]["total"] == 576
        assert all(row["split"] == split for row in rows)
    assert (output / "inspection.html").is_file()
    assert checkpoint.read_bytes() == b"fixture" and controls.read_text() == "original controls"
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    with pytest.raises(SystemExit):
        globals_["main"]()
    assert len(loaded) == 1  # Existing outputs are rejected before checkpoint loading.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose_relational.py",
            "--baseline-dir",
            str(source),
            "--output-dir",
            str(tmp_path / "other"),
            "--split",
            "test",
        ],
    )
    with pytest.raises(SystemExit):
        globals_["main"]()
    assert len(loaded) == 1
