"""Geometry selection, held-out preservation, stream budgets and derived checkpoints."""

import copy
import json
import os
import random
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from multimodal_loop.data.geometry_diversity import (
    derive_geometry_corpus,
    geometry_of,
    presentation_summary,
    select_training_geometries,
    validate_geometry_derivation,
)
from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.data.shape_grounding import ShapeColorDataset
from multimodal_loop.data.synthetic_shapes import COLORS, SHAPES
from multimodal_loop.eval.geometry_reference import stage_geometry_reference
from multimodal_loop.eval.kaggle_geometry_diversity import (
    GeometryDiversityRun,
    archive_geometry_diversity,
)
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.train.shape_grounding_checkpoint import load_shape_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def make_manifest(config):
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


@pytest.fixture(scope="module")
def full_corpus():
    source = make_manifest(RelationalCorpusConfig())
    derived, provenance = derive_geometry_corpus(source)
    return source, derived, provenance


@pytest.fixture(scope="module")
def small_corpus():
    source = make_manifest(
        RelationalCorpusConfig(
            image_size=16,
            object_sizes=(4,),
            train_geometry_count=1,
            validation_geometry_count=1,
            test_geometry_count=1,
        )
    )
    derived, provenance = derive_geometry_corpus(source, geometry_count=2)
    return source, derived, provenance


def test_full_selection_coverage_and_size_quotas(full_corpus):
    source, derived, provenance = full_corpus
    original = {geometry_of(r) for r in source.splits["train"]}
    selected = {geometry_of(r) for r in derived.splits["train"]}
    reserved = {geometry_of(r) for s in ("validation", "test") for r in source.splits[s]}
    assert len(selected) == 128 and original <= selected and not selected & reserved
    old_sizes = Counter(tuple(o[2] for o in g) for g in original)
    new_sizes = Counter(tuple(o[2] for o in g) for g in selected)
    assert new_sizes == {s: n * 8 for s, n in old_sizes.items()}
    # Recompute coverage directly from coordinates, without the selector's feature helper.
    coordinates = {
        (i, size, axis, v)
        for g in selected
        for i, (x, y, size) in enumerate(g)
        for axis, v in [("x", x), ("y", y)]
    }
    offsets = {(i, size, x % 8, y % 8) for g in selected for i, (x, y, size) in enumerate(g)}
    assert len(coordinates) == 204 and len(offsets) == 329
    assert provenance["coverage"]["coordinates"]["missing_features"] == []
    assert (
        set(map(tuple, provenance["coverage"]["coordinates"]["selected_features"])) == coordinates
    )
    assert set(map(tuple, provenance["coverage"]["patch_offsets"]["selected_features"])) == offsets
    assert provenance["coverage"]["patch_offsets"]["attainable"] == 384
    for split in ("validation", "test"):
        assert source.splits[split] == derived.splits[split]
        assert (
            json.loads(source.content)["splits"][split]
            == json.loads(derived.content)["splits"][split]
        )
    assert len(derived.splits["train"]) == 18432
    counts = Counter(
        (geometry_of(r), o.shape, o.color) for r in derived.splits["train"] for o in r.scene.objects
    )
    assert set(counts.values()) == {36}
    assert len(counts) == 128 * len(SHAPES) * len(COLORS)


def test_fixed_budget_presentation_counts(full_corpus):
    source, derived, _ = full_corpus
    before, after = presentation_summary(source), presentation_summary(derived)
    assert before["qa_visit_histogram"] == [
        {"visits": 13, "qa_count": 4608},
        {"visits": 14, "qa_count": 2304},
    ]
    assert after["qa_visit_histogram"] == [
        {"visits": 1, "qa_count": 18432},
        {"visits": 2, "qa_count": 36864},
    ]
    assert after["unique_qas_seen"] == after["qa_count"] == 55296
    assert after["presentations"] == before["presentations"] == 92160
    for groups in after["exposure_totals"].values():
        assert sum(groups.values()) == 92160


def test_reproducible_derivation_and_rng_isolation(small_corpus):
    source, derived, provenance = small_corpus
    py = random.getstate()
    rng = torch.get_rng_state().clone()
    again, repeated = derive_geometry_corpus(source, geometry_count=2)
    assert again.content == derived.content and repeated == provenance
    assert random.getstate() == py and torch.equal(rng, torch.get_rng_state())
    assert validate_geometry_derivation(derived, provenance).content == source.content
    parsed = parse_relational_manifest(derived.content)
    assert parsed.sha256 == provenance["manifest_sha256"]
    data = ShapeColorDataset(parsed, "train")
    for i in range(len(data)):
        item = data[i]
        assert item.answer == next(o.color for o in item.scene.objects if o.shape == item.shape)


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 2),
        ("manifest_sha256", "wrong"),
        ("source_manifest_sha256", "wrong"),
        ("selection_seed", 1),
    ],
)
def test_tampered_derivation_rejected(small_corpus, key, value):
    _, derived, provenance = small_corpus
    bad = copy.deepcopy(provenance)
    bad[key] = value
    with pytest.raises(ValueError):
        validate_geometry_derivation(derived, bad)


def test_wrong_training_and_heldout_records_rejected(small_corpus):
    source, derived, provenance = small_corpus
    with pytest.raises(ValueError, match="disagrees"):
        validate_geometry_derivation(source, provenance)
    payload = json.loads(derived.content)
    payload["splits"]["validation"].reverse()
    changed = parse_relational_manifest(json.dumps(payload))
    with pytest.raises(ValueError, match="disagrees"):
        validate_geometry_derivation(changed, provenance)
    for count in (0, 1000):
        with pytest.raises(ValueError):
            select_training_geometries(source, geometry_count=count)


def test_cpu_generation_training_evaluation_and_authorization(small_corpus, tmp_path):
    source, _, _ = small_corpus
    path = tmp_path / "source.json"
    path.write_text(source.content)
    config = ModelConfig(
        vocab_size=17,
        image_size=16,
        patch_size=8,
        max_seq_len=32,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_prelude_layers=1,
        n_recurrent_layers=1,
        n_coda_layers=1,
        recurrence_depth=2,
    )
    cfg = tmp_path / "model.yaml"
    cfg.write_text(yaml.safe_dump(asdict(config)))
    data, training, diagnosis = (tmp_path / name for name in ("data", "training", "diagnosis"))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run(args, ok=True):
        result = subprocess.run(
            [sys.executable, *map(str, args)], cwd=ROOT, env=env, capture_output=True, text=True
        )
        assert (result.returncode == 0) == ok, result.stdout + result.stderr
        return result

    generation = [
        "scripts/generate_geometry_diversity.py",
        "--source-manifest",
        path,
        "--output-dir",
        data,
        "--geometry-count",
        "2",
        "--smoke",
    ]
    run(generation)
    assert "fresh output" in run(generation, ok=False).stderr
    command = [
        "scripts/train_shape_grounding.py",
        "--manifest",
        data / "manifest.json",
        "--geometry-source-manifest",
        path,
        "--model-config",
        cfg,
        "--output-dir",
        training,
        "--max-steps",
        "2",
        "--evaluation-interval",
        "1",
    ]
    assert "original source" in run(command, ok=False).stderr
    assert not training.exists()
    run([*command, "--smoke"])
    checkpoint = training / "last.pt"
    model, manifest, _, payload = load_shape_checkpoint(checkpoint)
    assert payload["corpus_derivation"]["source_manifest_sha256"] == source.sha256
    assert len(manifest.splits["train"]) == 288 and model.config == config
    run(
        [
            "scripts/evaluate_shape_grounding.py",
            "--checkpoint",
            checkpoint,
            "--output-dir",
            diagnosis,
            "--shuffle-seeds",
            "0",
        ]
    )
    summary = json.loads((diagnosis / "summary.json").read_text())
    groups = summary["training_geometry_subsets"]
    assert groups["original"]["total"] == groups["added"]["total"] == 432
    assert (
        groups["original"]["correct"] + groups["added"]["correct"]
        == summary["splits"]["train"]["correct"]
    )
    assert summary["corpus_derivation"] == payload["corpus_derivation"]
    assert summary["splits"]["validation"]["total"] == 432
    assert set(summary["splits"]) == {"train", "validation"}
    presentations = json.loads((training / "presentations.json").read_text())
    assert presentations["presentations"] == 64
    bad = copy.deepcopy(payload)
    bad["corpus_derivation"]["coverage"]["coordinates"]["selected"] += 1
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="disagrees"):
        load_shape_checkpoint(tmp_path / "bad.pt")


@pytest.mark.parametrize(
    "member",
    [
        "../outside",
        "/absolute",
        "milestone2_shape_grounding/../../bad",
        "wrong_root/training/last.pt",
    ],
)
def test_unsafe_reference_zip_rejected(tmp_path, member):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, "bad")
    staging = tmp_path / "staging"
    with pytest.raises(ValueError):
        stage_geometry_reference(path, staging)
    assert not staging.exists()


def test_notebook_and_archive_structure(tmp_path):
    notebook = json.loads(
        (ROOT / "notebooks/kaggle_milestone2_geometry_diversity.ipynb").read_text()
    )
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            text = "".join(cell["source"])
            compile(text, "<notebook>", "exec")
            sources.append(text)
    code = "\n".join(sources)
    assert "REFERENCE_SOURCE" in code and "run_geometry_diversity(run)" in code
    root = tmp_path / "run"
    run = GeometryDiversityRun(ROOT, root, tmp_path / "reference", {}, {})
    with pytest.raises(ValueError):
        archive_geometry_diversity(run)
    for directory in ("data", "training", "diagnosis", "provenance", "logs", "staging"):
        (root / directory).mkdir(parents=True)
        (root / directory / "example").write_text(directory)
    (root / "provenance" / "final.json").write_text("{}")
    path = archive_geometry_diversity(run)
    with zipfile.ZipFile(path) as z:
        assert not any("staging" in p for p in z.namelist())
        assert "run/data/example" in z.namelist()
    with pytest.raises(FileExistsError):
        archive_geometry_diversity(run)


def test_fixed_notebook_workflow_and_comparison(monkeypatch, tmp_path):
    from multimodal_loop.eval import kaggle_geometry_diversity as module
    from multimodal_loop.train.kaggle import file_hash
    from multimodal_loop.train.synthetic import SyntheticTrainingConfig

    repo, root, reference = (tmp_path / n for n in ("repo", "run", "reference"))
    (repo / "configs").mkdir(parents=True)
    model = yaml.safe_load((ROOT / "configs" / "relational_baseline.yaml").read_text())
    (repo / "configs" / "relational_baseline.yaml").write_text(yaml.safe_dump(model))
    reference.mkdir()
    training = asdict(SyntheticTrainingConfig())

    def metrics(total, accuracy):
        return {
            "total": total,
            "accuracy": accuracy,
            "loss": 1.0,
            "all_three": {"accuracy": accuracy},
            "circle_square_pair": {"accuracy": accuracy},
            "breakdowns": {"shape": [{"shape": s, "accuracy": accuracy} for s in SHAPES]},
        }

    old = {
        "splits": {"train": metrics(6912, 0.9), "validation": metrics(1728, 0.5)},
        "assessment": {
            "gates": [{"name": "train_circle", "threshold": 0.95, "value": 0.9, "passed": False}]
        },
    }
    old_controls = {"results": {"accuracy_gaps": {"correct_minus_blank": 0.25}}}
    monkeypatch.setattr(module, "repository_revision", lambda _: "committed")
    monkeypatch.setattr(module, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(module, "stage_geometry_reference", lambda *args: reference)
    monkeypatch.setattr(
        module,
        "audit_geometry_reference",
        lambda _: {
            "settings": {
                "model": model,
                "training": training,
                "budget": {"max_steps": 2880, "evaluation_interval": 288},
            },
            "manifest": SimpleNamespace(sha256="source-hash"),
            "summary": old,
            "controls": old_controls,
        },
    )
    run = module.prepare_geometry_diversity(repo, root, reference)
    protocol = json.loads((root / "provenance" / "protocol.json").read_text())
    assert protocol["geometry_count"] == 128 and protocol["qa_presentations"] == 92160
    assert protocol["criteria"] == [{"name": "train_circle", "threshold": 0.95}]
    monkeypatch.setattr(
        module, "load_relational_manifest", lambda _: SimpleNamespace(sha256="derived-hash")
    )
    calls = []
    provenance = {"kind": "mock"}
    presentations = {"presentations": 92160}

    def command(repo, output, name, args):
        calls.append((name, args))
        if name == "generate":
            assert args[args.index("--geometry-count") + 1] == "128"
            (root / "data").mkdir()
            module.write_json(
                root / "data" / "coverage.json",
                {
                    "coordinates": {"selected": 204, "missing_features": []},
                    "patch_offsets": {"selected": 329},
                },
            )
            module.write_json(root / "data" / "derivation.json", provenance)
            module.write_json(root / "data" / "presentations.json", {"derived": presentations})
        elif name == "train":
            assert "--geometry-source-manifest" in args
            assert args[args.index("--max-steps") + 1] == "2880"
            assert "--resume" not in args and "--checkpoint" not in args and "--smoke" not in args
            (root / "training").mkdir()
            (root / "training" / "last.pt").write_bytes(b"new weights")
            module.write_json(root / "training" / "presentations.json", presentations)
        else:
            assert args[args.index("--shuffle-seeds") + 1 :] == ["0", "1", "2", "3", "4"]
            (root / "diagnosis").mkdir()
            module.write_json(
                root / "diagnosis" / "summary.json",
                {
                    "kind": "direct_shape_color_diagnostics",
                    "format_version": 1,
                    "smoke": False,
                    "completed_steps": 2880,
                    "examples_seen": 92160,
                    "model": model,
                    "training": training,
                    "budget": {"max_steps": 2880, "evaluation_interval": 288},
                    "manifest_sha256": "derived-hash",
                    "checkpoint_sha256": file_hash(root / "training" / "last.pt"),
                    "evaluation": {
                        "splits": ["train", "validation"],
                        "shuffle_seeds": list(range(5)),
                        "recurrence_depth": 2,
                        "batch_size": 32,
                    },
                    "splits": {"train": metrics(55296, 0.8), "validation": metrics(1728, 0.75)},
                    "training_geometry_subsets": {
                        "original": metrics(6912, 0.85),
                        "added": metrics(48384, 0.8),
                    },
                    "corpus_derivation": provenance,
                },
            )
            module.write_json(
                root / "diagnosis" / "controls.json",
                {"results": {"accuracy_gaps": {"correct_minus_blank": 0.5}}},
            )

    monkeypatch.setattr(module, "run_logged_command", command)
    module.run_geometry_diversity(run)
    assert [name for name, _ in calls] == ["generate", "train", "evaluate"]
    comparison = json.loads((root / "diagnosis" / "comparison.json").read_text())
    assert comparison["validation"][0] == {
        "metric": "accuracy",
        "reference": 0.5,
        "current": 0.75,
        "difference": 0.25,
    }
    assert comparison["training"]["current_original"]["accuracy"] == 0.85
    assert (root / "provenance" / "final.json").exists()
    with pytest.raises(ValueError, match="fresh run"):
        module.run_geometry_diversity(run)
    with pytest.raises(ValueError, match="fresh run"):
        module.prepare_geometry_diversity(repo, root, reference)


def test_training_budget_exposure_counts(full_corpus):
    _, manifest, _ = full_corpus
    report = presentation_summary(manifest, max_steps=5760)
    assert report["presentations"] == 184320
    assert report["unique_qas_seen"] == 55296
    assert report["mean_presentations_per_qa"] == 10 / 3
    assert report["qa_visit_histogram"] == [
        {"visits": 3, "qa_count": 36864},
        {"visits": 4, "qa_count": 18432},
    ]
    for counts in report["exposure_totals"].values():
        assert sum(counts.values()) == 184320
