"""Convolutional tokens, matching initialization, leakage and experiment integrity."""

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

from multimodal_loop.data.geometry_diversity import presentation_summary
from multimodal_loop.data.matched_size import derive_matched_size
from multimodal_loop.data.multi_arrangement import MultiArrangementDataset, derive_multi_arrangement
from multimodal_loop.data.quartet_fit import derive_quartet_fit
from multimodal_loop.data.quartet_transfer import QuartetTransferDataset
from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.data.shape_grounding import ShapeColorCollator
from multimodal_loop.eval.conv_stem import evaluate_conv_stem
from multimodal_loop.eval.conv_stem_reference import CNN_HASHES, read_conv_stem_references
from multimodal_loop.eval.focused_diagnosis import validate_rows
from multimodal_loop.eval.multi_arrangement import aggregate_role
from multimodal_loop.eval.quartet_transfer import population_metrics
from multimodal_loop.eval.shape_grounding import diagnose_shape_grounding
from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.conv_stem import (
    ConvStemConfig,
    ConvStemEmbedding,
    build_conv_stem_model,
    forward_macs,
    initialization_provenance,
)
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.cnn_checkpoint import load_cnn_checkpoint
from multimodal_loop.train.conv_stem_checkpoint import (
    MANIFEST_SHA256,
    load_conv_stem_checkpoint,
    validate_conv_stem_protocol,
)
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.losses import shifted_cross_entropy
from multimodal_loop.train.multi_arrangement_checkpoint import load_multi_checkpoint
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig, shape_color_loader
from multimodal_loop.train.synthetic import SyntheticTrainingConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channels": ()},
        {"channels": (0,)},
        {"channels": (True,)},
        {"pool_size": 0},
        {"pool_size": 2.5},
        {"init_seed": True},
        {"init_seed": -1},
    ],
)
def test_stem_config_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        ConvStemConfig(**kwargs)


def test_grid_order_and_global_average():
    config = ModelConfig(vocab_size=17, d_model=1, n_heads=1)
    stem = ConvStemEmbedding(config, ConvStemConfig(channels=(1,)))
    with torch.no_grad():
        stem.features[0].weight.zero_()
        stem.features[0].weight[0, 0, 1, 1] = 1
        stem.features[0].bias.zero_()
    images = torch.zeros(1, 3, 32, 32)
    for y in range(4):
        for x in range(4):
            images[0, 0, 8 * y : 8 * y + 8, 8 * x : 8 * x + 8] = 4 * y + x
    assert stem(images).flatten().tolist() == list(range(16))
    random_images = torch.rand(2, 3, 32, 32)
    torch.testing.assert_close(
        stem(random_images).mean(1), stem.features(random_images).mean((2, 3))
    )
    with pytest.raises(ValueError, match="shape"):
        stem(images[:, :, :16])
    with pytest.raises(TypeError, match="floating"):
        stem(images.long())
    with pytest.raises(ValueError, match="match"):
        ConvStemEmbedding(config, ConvStemConfig())


def test_initialization_parameter_count_and_compute():
    config = ModelConfig(vocab_size=17)
    torch.manual_seed(0)
    baseline = MultimodalLoopTransformer(config)
    tail = torch.get_rng_state().clone()
    torch.manual_seed(0)
    candidate = build_conv_stem_model(config)
    assert torch.equal(tail, torch.get_rng_state())
    for name, value in baseline.state_dict().items():
        if not name.startswith("embeddings.image_embedding."):
            assert torch.equal(value, candidate.state_dict()[name]), name
    assert sum(p.numel() for p in candidate.parameters()) == 290752
    provenance = initialization_provenance(candidate)
    assert provenance["stem_init_seed"] == 0
    assert len(provenance["non_image_sha256"]) == 64
    assert forward_macs(config, ConvStemConfig(), question_only=False) == {
        "sequence_length": 23,
        "direct_patch_transformer": 5014464,
        "conv_stem_transformer": 137823168,
    }
    assert (
        forward_macs(config, ConvStemConfig(), question_only=True)["conv_stem_transformer"]
        == 137602432
    )


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_mask_alignment_recurrence_and_gradients(depth):
    config = ModelConfig(vocab_size=17, d_model=8, n_heads=2, d_ff=16)
    model = build_conv_stem_model(config, ConvStemConfig(channels=(8, 8)))
    images = torch.rand(2, 3, 32, 32, requires_grad=True)
    ids = torch.tensor([[0, 1, 2, 3, 14, 5, 6], [0, 1, 2, 3, 15, 5, 9]])
    mask = build_prefix_mask(23, 22)
    calls = []
    handle = model.core.stack.register_forward_hook(
        lambda module, args, output: calls.append(tuple(id(p) for p in module.parameters()))
    )
    logits = model(ids, images, attention_mask=mask, recurrence_depth=depth)
    handle.remove()
    assert len(calls) == depth and all(c == calls[0] for c in calls)
    altered = ids.clone()
    altered[:, -1] = 7
    changed = model(altered, images, attention_mask=mask, recurrence_depth=depth)
    torch.testing.assert_close(logits[:, :22], changed[:, :22], rtol=0, atol=0)
    question_only = model(
        ids[:, :6], images, attention_mask=build_prefix_mask(22, 22), recurrence_depth=depth
    )
    torch.testing.assert_close(question_only[:, -1], logits[:, 21], rtol=1e-5, atol=1e-6)
    target_mask = torch.zeros_like(ids, dtype=torch.bool)
    target_mask[:, -1] = True
    loss = shifted_cross_entropy(logits, ids, num_image_tokens=16, target_mask=target_mask)
    torch.testing.assert_close(loss, F.cross_entropy(logits[:, 21], ids[:, -1]))
    loss.backward()
    assert images.grad.abs().sum() > 0
    assert model.embeddings.image_embedding.features[0].weight.grad.abs().sum() > 0
    assert model.core.stack.blocks[0].attn.qkv_proj.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    text = model(ids, recurrence_depth=depth)
    assert text.shape == (2, 7, 17)
    text.square().mean().backward()
    assert model.embeddings.text_embedding.weight.grad.abs().sum() > 0


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


def test_protocol_and_input_separation(corpus):
    model, stem, training, budget = (
        ModelConfig(vocab_size=17),
        ConvStemConfig(),
        SyntheticTrainingConfig(),
        ShapeGroundingConfig(8640, 216),
    )
    pinned = SimpleNamespace(sha256=MANIFEST_SHA256, config=corpus.config)
    validate_conv_stem_protocol(pinned, model, stem, training, budget)
    for replacement in (ConvStemConfig(init_seed=1), ConvStemConfig(channels=(64,))):
        with pytest.raises(ValueError, match="fixed"):
            validate_conv_stem_protocol(pinned, model, replacement, training, budget)
    with pytest.raises(ValueError, match="fixed"):
        validate_conv_stem_protocol(pinned, model, stem, training, ShapeGroundingConfig(8641, 216))
    with pytest.raises(ValueError, match="fixed"):
        validate_conv_stem_protocol(corpus, model, stem, training, budget)
    dataset = MultiArrangementDataset(corpus)
    example = dataset[0]

    class PublicOnly:
        image, question, answer = example.image, example.question, example.answer

        @property
        def scene(self):
            raise AssertionError("metadata accessed")

    batch = ShapeColorCollator(model)([PublicOnly()])
    assert batch.input_ids.shape == (1, 7) and batch.target_mask.sum() == 1
    assert batch.num_image_tokens == 16
    exposure = presentation_summary(corpus, max_steps=len(dataset) // 32 * 40)
    assert exposure["qa_visit_histogram"] == [{"visits": 40, "qa_count": len(dataset)}]


def reference_fixture(root, corpus, model, training):
    rows, metrics = {}, {}
    for p in corpus.populations:
        _, rows[p.name] = diagnose_shape_grounding(model, QuartetTransferDataset(p), training)
        metrics[p.name] = population_metrics(rows[p.name], p.records)
    for name in ("data", "diagnosis", "provenance"):
        (root / name).mkdir(parents=True)
    (root / "data/manifest.json").write_text(corpus.content)
    (root / "diagnosis/summary.json").write_text(
        json.dumps(
            {
                "manifest_sha256": corpus.sha256,
                "checkpoint_sha256": "fixture",
                "smoke": True,
                "arrangements": metrics,
                "aggregates": {
                    role: aggregate_role(corpus.populations, rows, role)
                    for role in ("training_fit", "transfer")
                },
            }
        )
    )
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
                **{name: file_hash(root / name) for name in CNN_HASHES},
                "training/last.pt": "fixture",
            }
        )
    )


def test_cli_checkpoint_and_both_reference_comparisons(corpus, tmp_path):
    config = ModelConfig(vocab_size=17, d_model=8, n_heads=2, d_ff=16)
    stem = ConvStemConfig(channels=(8, 8))
    for name, content in (
        ("model.yaml", yaml.safe_dump(asdict(config))),
        ("stem.yaml", yaml.safe_dump(asdict(stem))),
        ("manifest.json", corpus.content),
    ):
        (tmp_path / name).write_text(content)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_conv_stem.py",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--model-config",
            str(tmp_path / "model.yaml"),
            "--stem-config",
            str(tmp_path / "stem.yaml"),
            "--output-dir",
            str(tmp_path / "training"),
            "--max-steps",
            "2",
            "--evaluation-interval",
            "1",
            "--smoke",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = tmp_path / "training/last.pt"
    rng = torch.get_rng_state().clone()
    model, m, training, payload = load_conv_stem_checkpoint(checkpoint)
    assert torch.equal(rng, torch.get_rng_state()) and m.sha256 == corpus.sha256
    assert payload["completed_steps"] == 2 and payload["examples_seen"] == 64
    assert payload["optimizer_state"]["state"] and payload["rng_state"]
    assert all("training_fit" in r and "validation" not in r for r in payload["history"])
    for loader in (load_multi_checkpoint, load_cnn_checkpoint):
        with pytest.raises(ValueError, match="incompatible"):
            loader(checkpoint)
    batch = next(iter(shape_color_loader(MultiArrangementDataset(corpus), config, training)))
    with torch.no_grad():
        first = model(**batch.model_inputs())
        second = load_conv_stem_checkpoint(checkpoint)[0](**batch.model_inputs())
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    bad = copy.deepcopy(payload)
    bad["completed_steps"] = 1
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="progress"):
        load_conv_stem_checkpoint(tmp_path / "bad.pt")
    reference = tmp_path / "reference"
    reference_fixture(reference, corpus, model, training)
    with pytest.raises(ValueError, match="identity"):
        read_conv_stem_references(reference, reference)
    digest = file_hash(checkpoint)
    report = evaluate_conv_stem(checkpoint, tmp_path / "diagnosis", reference, reference)
    assert file_hash(checkpoint) == digest
    comparison = json.loads((tmp_path / "diagnosis/comparison.json").read_text())
    assert set(comparison) == {"transformer", "cnn"}
    for result in comparison.values():
        assert result["transfer_aggregate"]["accuracy"]["difference"] == 0
        assert all(v["prediction_changes"] == 0 for v in result["matched_changes"].values())
    rows = [
        json.loads(line)
        for line in (tmp_path / "diagnosis/predictions.jsonl").read_text().splitlines()
    ]
    for p in corpus.populations:
        validate_rows(
            [r for r in rows if r["arrangement"] == p.name],
            SimpleNamespace(splits={p.name: p.records}),
            p.name,
            report["arrangements"][p.name]["summary"],
        )
    assert (
        json.loads((tmp_path / "diagnosis/runtime.json").read_text())["final_evaluation_seconds"]
        > 0
    )
    with pytest.raises(ValueError, match="fresh"):
        evaluate_conv_stem(checkpoint, tmp_path / "diagnosis", reference, reference)
    (reference / "diagnosis/predictions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash"):
        read_conv_stem_references(reference, reference, smoke=True)


def test_notebook_contract():
    notebook = json.loads((ROOT / "notebooks/kaggle_milestone2_conv_stem.ipynb").read_text())
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "<notebook>", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
            sources.append(source)
    joined = "\n".join(sources)
    for name in (
        "milestone2_cnn_baseline_artifacts.zip",
        "milestone2_multi_arrangement_artifacts.zip",
    ):
        assert f'REPO_DIR / "{name}"' in joined
    assert "run_conv_stem(run)" in joined
    assert 'comparison["transformer"]' in joined and 'comparison["cnn"]' in joined


@pytest.mark.parametrize("fault", [None, "exposures", "missing", "metadata"])
def test_kaggle_budget_and_packaging(corpus, tmp_path, monkeypatch, fault):
    import zipfile

    from multimodal_loop.eval import kaggle_conv_stem as module

    repo, root = tmp_path / "repo", tmp_path / "run"
    (repo / "configs").mkdir(parents=True)
    for name, config in (
        ("relational_baseline.yaml", ModelConfig(vocab_size=17)),
        ("conv_stem.yaml", ConvStemConfig()),
    ):
        (repo / "configs" / name).write_text(yaml.safe_dump(asdict(config)))
    transformer, cnn = repo / "transformer.zip", repo / "cnn.zip"
    transformer.write_bytes(b"transformer")
    cnn.write_bytes(b"cnn")
    monkeypatch.setattr(module, "repository_revision", lambda _: "revision")
    monkeypatch.setattr(module, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(
        module,
        "read_conv_stem_references",
        lambda *args: {name: (corpus, {}, {}, {}) for name in ("transformer", "cnn")},
    )
    run = module.prepare_conv_stem(repo, root, transformer, cnn)
    protocol = json.loads((root / "provenance/protocol.json").read_text())
    assert protocol["parameters"] == 290752 and protocol["recurrence_depth"] == 2
    assert protocol["qa_presentations"] == 276480 and protocol["presentations_per_qa"] == 40
    payload = {
        "smoke": False,
        "completed_steps": 8640,
        "history": [{"completed_steps": 8640}],
        "config": asdict(ModelConfig(vocab_size=17)),
        "stem_config": asdict(ConvStemConfig()),
        "training_config": asdict(SyntheticTrainingConfig()),
        "budget": {"max_steps": 8640, "evaluation_interval": 216},
    }
    monkeypatch.setattr(
        module, "load_conv_stem_checkpoint", lambda _: (None, corpus, None, payload)
    )
    calls = []

    def command(repo, root, name, args):
        calls.append(name)
        assert "--smoke" not in args and "--resume" not in args
        if name == "train":
            assert args[args.index("--max-steps") + 1] == "8640"
            assert args[args.index("--evaluation-interval") + 1] == "216"
            assert "--transformer-source" not in args and "--cnn-source" not in args
            (root / "training").mkdir()
            (root / "training/last.pt").write_bytes(b"trained")
            (root / "training/presentations.json").write_bytes(
                b"{}" if fault == "exposures" else (root / "data/presentations.json").read_bytes()
            )
            for filename in ("settings.json", "manifest.json", "runtime.json"):
                (root / "training" / filename).write_text("{}")
            module.write_json(root / "training/metrics.json", payload["history"])
        else:
            assert args[args.index("--transformer-source") + 1] == str(transformer)
            assert args[args.index("--cnn-source") + 1] == str(cnn)
            (root / "diagnosis").mkdir()
            module.write_json(
                root / "diagnosis/summary.json",
                {
                    "kind": "conv_stem_direct_shape_color_diagnostics_v1",
                    "checkpoint_sha256": file_hash(root / "training/last.pt"),
                    "manifest_sha256": corpus.sha256,
                    "completed_steps": 8640,
                    "examples_seen": 276480,
                    "smoke": False,
                    "model": payload["config"],
                    "stem": {} if fault == "metadata" else payload["stem_config"],
                    "training": payload["training_config"],
                    "budget": payload["budget"],
                    "aggregates": {
                        role: {"summary": {"total": 6912}, "families": {"total": 576}}
                        for role in ("training_fit", "transfer")
                    },
                },
            )
            for filename in (
                "comparison.json",
                "reference_summary.json",
                "predictions.jsonl",
                "inspection.html",
                "runtime.json",
            ):
                if filename != "runtime.json" or fault != "missing":
                    (root / "diagnosis" / filename).write_text("{}")

    monkeypatch.setattr(module, "run_logged_command", command)
    if fault:
        with pytest.raises(ValueError):
            module.run_conv_stem(run)
        assert not (root / "provenance/final.json").exists()
        return
    module.run_conv_stem(run)
    assert calls == ["train", "evaluate"]
    archive = module.archive_conv_stem(run)
    with zipfile.ZipFile(archive) as z:
        assert "run/training/last.pt" in z.namelist()
        assert "run/diagnosis/runtime.json" in z.namelist()
        assert not any(n.endswith("cnn.zip") or n.endswith("transformer.zip") for n in z.namelist())
    (root / "training/last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.archive_conv_stem(run)
    with pytest.raises(ValueError, match="fresh"):
        module.run_conv_stem(run)
