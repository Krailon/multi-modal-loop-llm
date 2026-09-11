"""Matched-size corpus derivation and the unchanged direct-color training path."""

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

from multimodal_loop.data.geometry_diversity import derive_geometry_corpus, presentation_summary
from multimodal_loop.data.matched_size import (
    MatchedSizeDataset,
    derive_matched_size,
    origin_key,
    parse_matched_size,
    scene_key,
)
from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.data.relational_shapes import render_multi_object_scene
from multimodal_loop.data.shape_grounding import ShapeColorCollator
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.train.matched_size_checkpoint import load_matched_checkpoint
from multimodal_loop.train.shape_grounding_checkpoint import load_shape_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def source_manifest(config):
    return parse_relational_manifest(
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


@pytest.fixture(scope="module")
def small():
    source = source_manifest(
        RelationalCorpusConfig(
            object_sizes=(6,),
            train_geometry_count=1,
            validation_geometry_count=1,
            test_geometry_count=1,
        )
    )
    return source, derive_matched_size(source)


def test_full_counts_and_exposure():
    source, _ = derive_geometry_corpus(source_manifest(RelationalCorpusConfig()))
    corpus = derive_matched_size(source)
    d = corpus.derivation
    assert (
        d["original_images"],
        d["additional_images"],
        d["total_images"],
        d["eligible_families"],
    ) == (18432, 21024, 39456, 7008)
    keys = [scene_key(r.scene) for r in corpus.splits["train"]]
    assert len(set(keys)) == len(keys)
    assert keys[:18432] == [scene_key(r.scene) for r in source.splits["train"]]
    assert keys[18432:] == sorted(keys[18432:])
    reserved = {origin_key(r.scene) for s in ("validation", "test") for r in source.splits[s]}
    assert not any(origin_key(r.scene) in reserved for r in corpus.splits["train"])
    assert corpus.splits["validation"] == source.splits["validation"]
    assert corpus.splits["test"] == source.splits["test"]
    for family in d["families"]:
        scenes = [corpus.splits["train"][i].scene for i in family["variant_indices"]]
        assert len({scene_key(s) for s in scenes}) == 4
        original = source.splits["train"][family["source_index"]].scene
        for scene in scenes:
            assert origin_key(scene) == origin_key(original)
            assert [(o.shape, o.color) for o in scene.objects] == [
                (o.shape, o.color) for o in original.objects
            ]
            assert next(o for o in scene.objects if o.shape == "triangle") == next(
                o for o in original.objects if o.shape == "triangle"
            )
            assert all(o.left + o.size < 32 and o.top + o.size < 32 for o in scene.objects)
    exposure = presentation_summary(corpus, max_steps=5760)
    assert exposure["presentations"] == 184320 and exposure["qa_count"] == 118368
    assert exposure["qa_visit_histogram"] == [
        {"visits": 1, "qa_count": 52416},
        {"visits": 2, "qa_count": 65952},
    ]


def test_determinism_validation_and_pixels(small):
    source, corpus = small
    assert derive_matched_size(source).content == corpus.content
    assert parse_matched_size(corpus.content).sha256 == corpus.sha256
    train = MatchedSizeDataset(corpus, "train")
    for i in range(len(source.splits["train"])):
        assert torch.equal(
            train[3 * i].image, render_multi_object_scene(source.splits["train"][i].scene)
        )
    with pytest.raises(ValueError, match="excludes test"):
        MatchedSizeDataset(corpus, "test")
    bad = json.loads(corpus.content)
    bad["train"][0]["scene"]["objects"][0]["size"] = 8
    with pytest.raises(ValueError, match="derivation"):
        parse_matched_size(json.dumps(bad))
    # Color/shape identities do not bypass the reserved-origin rule.
    collided = SimpleNamespace(
        splits={**source.splits, "test": source.splits["train"]}, config=source.config
    )
    with pytest.raises(ValueError, match="held-out"):
        derive_matched_size(collided)
    example = train[0]

    class WithoutMetadata:
        image = example.image
        question = example.question
        answer = example.answer

        @property
        def scene(self):
            raise AssertionError("metadata entered collation")

    batch = ShapeColorCollator(ModelConfig(vocab_size=17))([WithoutMetadata()])
    assert batch.question_length == 6 and batch.input_ids.shape == (1, 7)


def test_cpu_training_checkpoint_evaluation(small, tmp_path):
    _, corpus = small
    manifest = tmp_path / "manifest.json"
    manifest.write_text(corpus.content)
    config = ModelConfig(vocab_size=17, d_model=8, n_heads=2, d_ff=16)
    cfg = tmp_path / "model.yaml"
    cfg.write_text(yaml.safe_dump(asdict(config)))
    training = tmp_path / "training"
    diagnosis = tmp_path / "diagnosis"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def command(args):
        result = subprocess.run(
            [sys.executable, *map(str, args)], cwd=ROOT, env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr

    command(
        [
            "scripts/train_matched_size.py",
            "--manifest",
            manifest,
            "--model-config",
            cfg,
            "--output-dir",
            training,
            "--max-steps",
            "2",
            "--evaluation-interval",
            "1",
            "--smoke",
        ]
    )
    checkpoint = training / "last.pt"
    model, restored, settings, payload = load_matched_checkpoint(checkpoint)
    assert restored.sha256 == corpus.sha256 and payload["completed_steps"] == 2
    assert payload["examples_seen"] == 64 and model.config == config
    with pytest.raises(ValueError, match="incompatible"):
        load_shape_checkpoint(checkpoint)
    bad = copy.deepcopy(payload)
    bad["completed_steps"] = 1
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="progress"):
        load_matched_checkpoint(tmp_path / "bad.pt")
    command(
        [
            "scripts/evaluate_matched_size.py",
            "--checkpoint",
            checkpoint,
            "--output-dir",
            diagnosis,
            "--shuffle-seeds",
            "0",
        ]
    )
    summary = json.loads((diagnosis / "summary.json").read_text())
    assert set(summary["splits"]) == {"train", "validation"}
    assert (
        summary["training_subsets"]["original"]["total"] == 3 * corpus.derivation["original_images"]
    )
    assert (
        summary["training_subsets"]["added"]["total"] == 3 * corpus.derivation["additional_images"]
    )
    assert len(summary["assessment"]["gates"]) == 9
    intervention = json.loads((diagnosis / "size_intervention.json").read_text())
    assert intervention["eligibility"]["eligible_images"] > 0
    assert (diagnosis / "size_intervention_predictions.jsonl").is_file()
    model2, _, _, _ = load_matched_checkpoint(checkpoint)
    assert all(torch.equal(v, model2.state_dict()[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("fault", [None, "budget", "subset", "exposure"])
def test_fixed_kaggle_workflow_and_comparisons(tmp_path, monkeypatch, fault):
    import zipfile

    from multimodal_loop.eval import kaggle_matched_size as module
    from multimodal_loop.train.synthetic import SyntheticTrainingConfig

    repo, root, reference, intervention_source = (
        tmp_path / n for n in ("repo", "run", "reference", "intervention")
    )
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "relational_baseline.yaml").write_text("{}")
    reference.mkdir()
    intervention_source.mkdir()
    training = asdict(SyntheticTrainingConfig())

    def metric(total, accuracy):
        return {
            "total": total,
            "accuracy": accuracy,
            "loss": 1.0,
            "circle_square_pair": {"accuracy": accuracy},
            "circle_square_same_prediction": {"fraction": 1 - accuracy},
            "all_three": {"accuracy": accuracy},
            "breakdowns": {
                "shape": [
                    {"shape": s, "accuracy": accuracy} for s in ("square", "circle", "triangle")
                ]
            },
        }

    old = {
        "manifest_sha256": "source",
        "splits": {"train": metric(55296, 0.75), "validation": metric(1728, 0.5)},
        "assessment": {"gates": []},
    }
    old_controls = {"results": {"accuracy_gaps": {"correct_minus_blank": 0.25}}}
    size = {
        "eligibility": {"eligible_images": 384},
        "metrics": {
            "conditions": {
                "original": {"images": 384, "accuracy": 0.5, "circle_square_pair_accuracy": 0.25}
            },
            "invariance": {},
            "size_reversal": {},
            "both_circle_square_correct_all_four": {"fraction": 0.1},
        },
    }
    monkeypatch.setattr(module, "read_intervention_reference", lambda _: size)
    monkeypatch.setattr(module, "stage_size_reference", lambda *a: reference)
    monkeypatch.setattr(
        module,
        "audit_size_reference",
        lambda _: {
            "manifest": None,
            "summary": old,
            "controls": old_controls,
            "settings": {"model": {}, "training": training},
        },
    )
    d = {
        "original_images": 18432,
        "additional_images": 21024,
        "total_images": 39456,
        "eligible_families": 7008,
    }
    monkeypatch.setattr(
        module,
        "derive_matched_size",
        lambda _: SimpleNamespace(
            derivation=d, content="{}", sha256="derived", source=SimpleNamespace(sha256="source")
        ),
    )
    monkeypatch.setattr(module, "presentation_summary", lambda *a, **kw: {"presentations": 184320})
    monkeypatch.setattr(module, "repository_revision", lambda _: "revision")
    monkeypatch.setattr(module, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(module, "reference_size_groups", lambda *args: {})
    run = module.prepare_matched_size(repo, root, reference, intervention_source)
    protocol = json.loads((root / "provenance" / "protocol.json").read_text())
    assert protocol["budget"]["max_steps"] == 5760 and protocol["qa_presentations"] == 184320
    calls = []

    def command(repo, root, name, args):
        calls.append(name)
        if name == "train":
            assert args[0] == "scripts/train_matched_size.py"
            assert args[args.index("--max-steps") + 1] == "5760"
            assert "--checkpoint" not in args and "--resume" not in args and "--smoke" not in args
            (root / "training").mkdir()
            (root / "training" / "last.pt").write_bytes(b"new")
            module.write_json(
                root / "training" / "presentations.json",
                {"presentations": 0 if fault == "exposure" else 184320},
            )
        else:
            assert args[0] == "scripts/evaluate_matched_size.py"
            assert args[args.index("--shuffle-seeds") + 1 :] == ["0", "1", "2", "3", "4"]
            digest = module.file_hash(root / "training" / "last.pt")
            (root / "diagnosis").mkdir()
            report = {
                "kind": "matched_size_diagnostics_v1",
                "smoke": False,
                "completed_steps": 2880 if fault == "budget" else 5760,
                "examples_seen": 184320,
                "manifest_sha256": "derived",
                "source_manifest_sha256": "source",
                "checkpoint_sha256": digest,
                "model": {},
                "training": training,
                "budget": {"max_steps": 5760, "evaluation_interval": 288},
                "evaluation": {
                    "splits": ["train", "validation"],
                    "shuffle_seeds": list(range(5)),
                    "recurrence_depth": 2,
                    "batch_size": 32,
                },
                "splits": {"train": metric(118368, 0.8), "validation": metric(1728, 0.75)},
                "relative_size_training_subsets": {"original": {}},
                "relative_size": {"validation": {}},
                "training_subsets": {
                    "original": metric(1 if fault == "subset" else 55296, 0.875),
                    "added": metric(63072, 0.7),
                },
            }
            module.write_json(root / "diagnosis" / "summary.json", report)
            module.write_json(
                root / "diagnosis" / "controls.json",
                {
                    "checkpoint_sha256": digest,
                    "results": {"accuracy_gaps": {"correct_minus_blank": 0.5}},
                },
            )
            module.write_json(
                root / "diagnosis" / "size_intervention.json", {"checkpoint_sha256": digest, **size}
            )
            for filename in (
                "train_examples.jsonl",
                "validation_examples.jsonl",
                "size_intervention_predictions.jsonl",
            ):
                (root / "diagnosis" / filename).write_text("{}\n")

    monkeypatch.setattr(module, "run_logged_command", command)
    with pytest.raises(ValueError, match="complete"):
        module.archive_matched_size(run)
    if fault:
        with pytest.raises(ValueError):
            module.run_matched_size(run)
        assert not (root / "provenance" / "final.json").exists()
        return
    module.run_matched_size(run)
    assert calls == ["train", "evaluate"]
    comparison = json.loads((root / "diagnosis" / "comparison.json").read_text())
    assert comparison["validation"]["accuracy"]["difference"] == 0.25
    assert comparison["original_training"]["accuracy"]["difference"] == 0.125
    path = module.archive_matched_size(run)
    with zipfile.ZipFile(path) as z:
        assert "run/training/last.pt" in z.namelist()
        assert not any("/staging/" in n for n in z.namelist())
    with pytest.raises(ValueError, match="fresh"):
        module.run_matched_size(run)


def test_notebook_and_intervention_reference_rejection(tmp_path):
    from multimodal_loop.eval.kaggle_matched_size import read_intervention_reference

    notebook = json.loads(
        (ROOT / "notebooks/kaggle_milestone2_matched_size_training.ipynb").read_text()
    )
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "<notebook>", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
            sources.append(source)
    code = "\n".join(sources)
    assert "INTERVENTION_SOURCE" in code and "run_matched_size(run)" in code
    (tmp_path / "diagnosis").mkdir()
    (tmp_path / "diagnosis" / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        read_intervention_reference(tmp_path)
