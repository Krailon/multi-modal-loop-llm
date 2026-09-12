"""Balanced training-fit selection, grouping, checkpoint and notebook contracts."""

import copy
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from multimodal_loop.data.geometry_diversity import derive_geometry_corpus, presentation_summary
from multimodal_loop.data.matched_size import derive_matched_size, origin_key, scene_key
from multimodal_loop.data.quartet_fit import (
    FIT_CONDITIONS,
    QuartetFitDataset,
    derive_quartet_fit,
    parse_quartet_fit,
)
from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.data.shape_grounding import ShapeColorCollator, ShapeColorTokenizer
from multimodal_loop.data.size_intervention import CONDITIONS, render_diagnostic_scene
from multimodal_loop.data.synthetic_shapes import SHAPES
from multimodal_loop.eval.focused_diagnosis import validate_rows
from multimodal_loop.eval.quartet_fit import assess_fit, family_metrics
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.train.matched_size_checkpoint import load_matched_checkpoint
from multimodal_loop.train.quartet_fit_checkpoint import (
    load_quartet_checkpoint,
    validate_fit_protocol,
)
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig
from multimodal_loop.train.synthetic import SyntheticTrainingConfig

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
            object_sizes=(8,),
            train_geometry_count=1,
            validation_geometry_count=1,
            test_geometry_count=1,
        )
    )
    return derive_quartet_fit(derive_matched_size(source))


def test_reference_arrangement_selection():
    source, _ = derive_geometry_corpus(source_manifest(RelationalCorpusConfig()))
    fit = derive_quartet_fit(derive_matched_size(source))
    assert fit.derivation["origins"] == [[1, 4], [10, 4], [20, 4]]
    assert fit.derivation["triangle_size"] == 8


def test_balance_quartets_pixels_and_exposures(small):
    fit = small
    assert derive_quartet_fit(fit.source).content == fit.content
    assert parse_quartet_fit(fit.content).sha256 == fit.sha256
    assert set(fit.splits) == {"train", "training_fit"}
    records = fit.splits["train"]
    assert len(records) == len({scene_key(r.scene) for r in records}) == 576
    assert len(fit.derivation["families"]) == 144
    reserved = {origin_key(r.scene) for s in ("validation", "test") for r in fit.source.splits[s]}
    assert {origin_key(r.scene) for r in records}.isdisjoint(reserved)
    counts = Counter()
    for r in records:
        for pos, obj in enumerate(r.scene.objects):
            counts[(obj.shape, obj.color, pos)] += 1
        assert all(
            o.left >= 1 and o.top >= 1 and o.left + o.size < 32 and o.top + o.size < 32
            for o in r.scene.objects
        )
        assert all(
            b.left - a.left - a.size >= 1
            for a, b in zip(r.scene.objects, r.scene.objects[1:], strict=False)
        )
    assert set(counts.values()) == {48}
    data = QuartetFitDataset(fit)
    for family in fit.derivation["families"]:
        scenes = [records[i].scene for i in family]
        assert (
            len({tuple((o.shape, o.color, o.left, o.top) for o in s.objects) for s in scenes}) == 1
        )
        for condition, i, scene in zip(FIT_CONDITIONS, family, scenes, strict=True):
            sizes = {o.shape: o.size for o in scene.objects}
            assert (sizes["circle"], sizes["square"]) == CONDITIONS[condition]
            assert sizes["triangle"] == 8
            assert torch.equal(data[3 * i].image, render_diagnostic_scene(scene))
    exposure = presentation_summary(fit, max_steps=2160)
    assert exposure["presentations"] == 69120
    assert exposure["qa_visit_histogram"] == [{"visits": 40, "qa_count": 1728}]
    assert presentation_summary(fit, max_steps=55)["presentations"] == 1760


def test_metadata_and_target_alignment(small):
    data = QuartetFitDataset(small)
    tokenizer = ShapeColorTokenizer()
    examples = [data[i] for i in range(12)]

    class InputsOnly:
        def __init__(self, e):
            self.image, self.question, self.answer = e.image, e.question, e.answer

        @property
        def scene(self):
            raise AssertionError("metadata entered model inputs")

        @property
        def shape(self):
            raise AssertionError("metadata entered model inputs")

    batch = ShapeColorCollator(ModelConfig(vocab_size=17))([InputsOnly(e) for e in examples])
    assert batch.question_length == 6 and batch.input_ids.shape == (12, 7)
    assert batch.input_ids[:, -1].tolist() == [tokenizer.encode_answer(e.answer) for e in examples]
    assert all(
        e.answer == next(o.color for o in e.scene.objects if o.shape == e.shape) for e in examples
    )


def test_reject_corruption_and_heldout_overlap(small):
    bad = json.loads(small.content)
    bad["derivation"]["source_indices"][0] += 1
    with pytest.raises(ValueError, match="derivation"):
        parse_quartet_fit(json.dumps(bad))
    bad["kind"] = "matched_size_color"
    with pytest.raises(ValueError, match="incompatible"):
        parse_quartet_fit(json.dumps(bad))
    collided = SimpleNamespace(
        derivation=small.source.derivation,
        splits={**small.source.splits, "test": small.source.splits["train"]},
    )
    with pytest.raises(ValueError, match="held-out"):
        derive_quartet_fit(collided)
    incomplete = SimpleNamespace(derivation={"families": []})
    with pytest.raises(ValueError, match="complete"):
        derive_quartet_fit(incomplete)


def test_independent_family_correctness_and_boundaries():
    families = [list(range(4 * i, 4 * i + 4)) for i in range(144)]
    rows = [
        {"image_index": i, "shape": shape, "correct": True} for i in range(576) for shape in SHAPES
    ]
    for i in range(7):
        rows[3 * (4 * i)]["correct"] = False
    result = family_metrics(rows, families)
    assert result["correct"] == 137 and result["fraction"] == 137 / 144
    summary = {"breakdowns": {"shape": [{"shape": s, "accuracy": 571 / 576} for s in SHAPES]}}
    assert assess_fit(summary, result)["passed"]
    rows[3 * (4 * 7) + 1]["correct"] = False
    assert not assess_fit(summary, family_metrics(rows, families))["passed"]
    summary["breakdowns"]["shape"][0]["accuracy"] = 570 / 576
    assert not assess_fit(summary, result)["passed"]
    summary["breakdowns"]["shape"][0]["accuracy"] = 0.99
    assert assess_fit(summary, {"fraction": 0.95})["passed"]
    for row in rows:
        row["correct"] = row["shape"] == "triangle"
    assert family_metrics(rows, families)["correct"] == 0  # Stable wrong answers do not pass.
    with pytest.raises(ValueError, match="coverage"):
        family_metrics(rows[:-1], families)
    with pytest.raises(ValueError, match="coverage"):
        family_metrics(rows, [families[0]] * 144)
    rows[0]["shape"] = "circle"
    with pytest.raises(ValueError, match="order"):
        family_metrics(rows, families)


def test_cpu_smoke_checkpoint_and_frozen_evaluation(small, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(small.content)
    config = ModelConfig(vocab_size=17, d_model=8, n_heads=2, d_ff=16)
    cfg = tmp_path / "model.yaml"
    cfg.write_text(yaml.safe_dump(asdict(config)))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def command(*args):
        result = subprocess.run(
            [sys.executable, *map(str, args)], cwd=ROOT, env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr

    command(
        "scripts/train_quartet_fit.py",
        "--manifest",
        manifest,
        "--model-config",
        cfg,
        "--output-dir",
        tmp_path / "training",
        "--max-steps",
        "2",
        "--evaluation-interval",
        "1",
        "--smoke",
    )
    path = tmp_path / "training/last.pt"
    model, restored, training, payload = load_quartet_checkpoint(path)
    assert restored.sha256 == small.sha256 and model.config == config
    assert payload["completed_steps"] == 2 and payload["examples_seen"] == 64
    assert payload["optimizer_state"]["state"] and payload["rng_state"]
    assert [h["completed_steps"] for h in payload["history"]] == [0, 1, 2]
    assert all("training_fit" in h and "validation" not in h for h in payload["history"])
    with pytest.raises(ValueError, match="incompatible"):
        load_matched_checkpoint(path)
    bad = copy.deepcopy(payload)
    bad["completed_steps"] = 1
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="progress"):
        load_quartet_checkpoint(tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="fixed"):
        validate_fit_protocol(small, config, training, ShapeGroundingConfig(2, 1))
    command(
        "scripts/evaluate_quartet_fit.py",
        "--checkpoint",
        path,
        "--output-dir",
        tmp_path / "diagnosis",
    )
    report = json.loads((tmp_path / "diagnosis/summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (tmp_path / "diagnosis/training_fit_examples.jsonl").read_text().splitlines()
    ]
    validate_rows(rows, small, "training_fit", report["training_fit"])
    assert report["evaluation"]["splits"] == ["training_fit"] and report["smoke"]
    assert report["families"]["total"] == 144 and len(rows) == 1728
    assert report["families"] == family_metrics(rows, small.derivation["families"])
    assert set(report["conditions"]) == set(FIT_CONDITIONS)
    assert (tmp_path / "diagnosis/inspection.html").is_file()
    again = load_quartet_checkpoint(path)[0]
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in model.state_dict().items())


def test_notebook_and_source_identity(tmp_path):
    from multimodal_loop.eval.kaggle_quartet_fit import read_fit_source

    notebook = json.loads((ROOT / "notebooks/kaggle_milestone2_quartet_fit.ipynb").read_text())
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "<notebook>", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
            code.append(source)
    assert "run_quartet_fit(run)" in "\n".join(code)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        read_fit_source(tmp_path)


@pytest.mark.parametrize("fault", [None, "budget", "population", "exposure"])
def test_kaggle_fixed_commands_and_archive(tmp_path, monkeypatch, fault):
    import zipfile

    from multimodal_loop.eval import kaggle_quartet_fit as module

    repo, root, source = (tmp_path / n for n in ("repo", "run", "source"))
    (repo / "configs").mkdir(parents=True)
    source.mkdir()
    (repo / "configs/relational_baseline.yaml").write_text(
        yaml.safe_dump(asdict(ModelConfig(vocab_size=17)))
    )
    monkeypatch.setattr(module, "repository_revision", lambda _: "revision")
    monkeypatch.setattr(module, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(module, "read_fit_source", lambda _: None)
    monkeypatch.setattr(
        module,
        "derive_quartet_fit",
        lambda _: SimpleNamespace(
            content="{}",
            sha256="fit",
            derivation={"origins": [[1, 4], [10, 4], [20, 4]], "triangle_size": 8},
        ),
    )
    monkeypatch.setattr(
        module,
        "presentation_summary",
        lambda *a, **kw: {
            "presentations": 69120,
            "qa_visit_histogram": [{"visits": 40, "qa_count": 1728}],
        },
    )
    run = module.prepare_quartet_fit(repo, root, source)
    protocol = json.loads((root / "provenance/protocol.json").read_text())
    assert protocol["qa_presentations"] == 69120 and protocol["passes"] == 40
    with pytest.raises(ValueError, match="complete"):
        module.archive_quartet_fit(run)
    calls = []

    def command(repo, root, name, args):
        calls.append(name)
        assert "--smoke" not in args and "--resume" not in args and "--split" not in args
        assert args[args.index("--device") + 1] == "cuda:0"
        if name == "train":
            assert args[0] == "scripts/train_quartet_fit.py"
            assert args[args.index("--max-steps") + 1] == "2160"
            assert args[args.index("--evaluation-interval") + 1] == "54"
            assert "--checkpoint" not in args
            (root / "training").mkdir()
            (root / "training/last.pt").write_bytes(b"fresh")
            exposures = json.loads((root / "data/presentations.json").read_text())
            if fault == "exposure":
                exposures["presentations"] = 1
            module.write_json(root / "training/presentations.json", exposures)
            module.write_json(
                root / "training/metrics.json",
                [
                    {"completed_steps": step, "completed_passes": step // 54, "training_fit": {}}
                    for step in range(0, 2161, 54)
                ],
            )
        else:
            assert args[0] == "scripts/evaluate_quartet_fit.py"
            (root / "diagnosis").mkdir()
            module.write_json(
                root / "diagnosis/summary.json",
                {
                    "kind": "quartet_training_fit_diagnostics_v1",
                    "smoke": False,
                    "completed_steps": 1 if fault == "budget" else 2160,
                    "examples_seen": 69120,
                    "manifest_sha256": "fit",
                    "source_manifest_sha256": module.SOURCE_SHA256,
                    "checkpoint_sha256": module.file_hash(root / "training/last.pt"),
                    "model": asdict(ModelConfig(vocab_size=17)),
                    "training": asdict(SyntheticTrainingConfig()),
                    "budget": {"max_steps": 2160, "evaluation_interval": 54},
                    "evaluation": {
                        "splits": ["training_fit"],
                        "recurrence_depth": 2,
                        "batch_size": 32,
                    },
                    "training_fit": {"total": 1 if fault == "population" else 1728},
                    "families": {"total": 144},
                },
            )

    monkeypatch.setattr(module, "run_logged_command", command)
    if fault:
        with pytest.raises(ValueError):
            module.run_quartet_fit(run)
        assert not (root / "provenance/final.json").exists()
        return
    module.run_quartet_fit(run)
    assert calls == ["train", "evaluate"]
    archive = module.archive_quartet_fit(run)
    with zipfile.ZipFile(archive) as z:
        assert "run/training/last.pt" in z.namelist()
        assert "run/provenance/final.json" in z.namelist()
        assert not any("staging" in n for n in z.namelist())
    (root / "training/last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.archive_quartet_fit(run)
    with pytest.raises(ValueError, match="fresh"):
        module.run_quartet_fit(run)
