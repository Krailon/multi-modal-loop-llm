"""Fixed assignment, repeated exposures and final same-population comparisons."""

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
from multimodal_loop.data.matched_size import derive_matched_size, origin_key, scene_key
from multimodal_loop.data.multi_arrangement import (
    TRAIN_NAMES,
    TRANSFER_NAMES,
    MultiArrangementDataset,
    derive_multi_arrangement,
    parse_multi_arrangement,
)
from multimodal_loop.data.quartet_fit import QuartetFitDataset, derive_quartet_fit
from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.data.shape_grounding import ShapeColorCollator, ShapeColorTokenizer
from multimodal_loop.eval.focused_diagnosis import validate_rows
from multimodal_loop.eval.multi_arrangement import (
    aggregate_role,
    assess_training,
    metric_comparison,
)
from multimodal_loop.eval.multi_arrangement_reference import TRANSFER_HASHES, read_multi_reference
from multimodal_loop.eval.quartet_fit import evaluate_quartet_checkpoint
from multimodal_loop.eval.quartet_transfer import evaluate_transfer
from multimodal_loop.eval.quartet_transfer_reference import REFERENCE_HASHES
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.multi_arrangement_checkpoint import (
    load_multi_checkpoint,
    validate_multi_protocol,
)
from multimodal_loop.train.quartet_fit_checkpoint import (
    load_quartet_checkpoint,
    save_quartet_checkpoint,
)
from multimodal_loop.train.shape_grounding import ShapeGroundingConfig, train_shape_datasets
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


def test_full_split_balance_and_exact_exposures():
    source, _ = derive_geometry_corpus(source_manifest(RelationalCorpusConfig()))
    fit = derive_quartet_fit(derive_matched_size(source))
    m = derive_multi_arrangement(fit)
    assert tuple(m.derivation["training_arrangements"]) == TRAIN_NAMES
    assert tuple(m.derivation["transfer_arrangements"]) == TRANSFER_NAMES
    assert m.derivation["training_questions"] == m.derivation["transfer_questions"] == 6912
    train_keys = {scene_key(r.scene) for r in m.splits["train"]}
    transfer_keys = {scene_key(r.scene) for r in m.splits["transfer"]}
    assert len(train_keys) == len(transfer_keys) == 2304 and train_keys.isdisjoint(transfer_keys)
    reserved = {origin_key(r.scene) for s in ("validation", "test") for r in fit.source.splits[s]}
    assert all(p.origins not in reserved for p in m.populations)
    assert len({p.origins for p in m.populations}) == 8
    assert all(len(p.records) == 576 and len(p.families) == 144 for p in m.populations)
    assert m.derivation["balance_per_arrangement"] == fit.derivation["balance"]
    exposure = presentation_summary(m, max_steps=8640)
    assert exposure["presentations"] == 276480
    assert exposure["qa_visit_histogram"] == [{"visits": 40, "qa_count": 6912}]
    assert presentation_summary(m, max_steps=217)["presentations"] == 6944
    assert parse_multi_arrangement(m.content).sha256 == m.sha256
    bad = json.loads(m.content)
    bad["derivation"]["training_arrangements"].reverse()
    with pytest.raises(ValueError, match="derivation"):
        parse_multi_arrangement(json.dumps(bad))


def test_training_criteria_each_arrangement_and_denominators():
    ps = [
        SimpleNamespace(name=n, role="training_fit" if n != "reserved" else "transfer")
        for n in ("a", "b", "reserved")
    ]

    def metrics(accuracy):
        return {
            "summary": {
                "total": 1728,
                "accuracy": accuracy,
                "loss": 0.1,
                "circle_square_pair": {"accuracy": accuracy},
                "breakdowns": {
                    "shape": [
                        {"shape": s, "accuracy": accuracy} for s in ("circle", "square", "triangle")
                    ]
                },
            },
            "families": {"total": 144, "fraction": 0.95},
        }

    ms = {"a": metrics(0.99), "b": metrics(0.99), "reserved": metrics(0)}
    assert assess_training(ps, ms)["passed"]
    ms["b"]["summary"]["breakdowns"]["shape"][0]["accuracy"] = 0.989
    assert not assess_training(ps, ms)["passed"]
    assert set(assess_training(ps, ms)["arrangements"]) == {"a", "b"}
    diff = metric_comparison(metrics(0.25), metrics(0.75))
    assert diff["accuracy"]["difference"] == 0.5
    wrong = metrics(0.75)
    wrong["summary"]["total"] = 12096
    with pytest.raises(ValueError, match="denominators"):
        metric_comparison(metrics(0.25), wrong)


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("multi-arrangement")
    source = source_manifest(
        RelationalCorpusConfig(
            object_sizes=(8,),
            train_geometry_count=2,
            validation_geometry_count=1,
            test_geometry_count=1,
        )
    )
    fit = derive_quartet_fit(derive_matched_size(source))
    fit_root = root / "fit"
    for name in ("data", "training", "provenance"):
        (fit_root / name).mkdir(parents=True)
    (fit_root / "data/manifest.json").write_text(fit.content)
    model = MultimodalLoopTransformer(ModelConfig(vocab_size=17, d_model=8, n_heads=2, d_ff=16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    training, budget = SyntheticTrainingConfig(), ShapeGroundingConfig(1, 1)

    def publish(history):
        save_quartet_checkpoint(
            fit_root / "training/last.pt",
            model,
            optimizer,
            manifest=fit,
            training=training,
            budget=budget,
            history=history,
            smoke=True,
        )

    train_shape_datasets(
        model,
        optimizer,
        QuartetFitDataset(fit),
        QuartetFitDataset(fit),
        training,
        budget,
        publish=publish,
        evaluation_name="training_fit",
    )
    evaluate_quartet_checkpoint(fit_root / "training/last.pt", fit_root / "diagnosis")
    (fit_root / "provenance/final.json").write_text(
        json.dumps({p: file_hash(fit_root / p) for p in REFERENCE_HASHES})
    )
    transfer = root / "transfer"
    evaluate_transfer(fit_root, transfer / "diagnosis", smoke=True)
    (transfer / "data").mkdir()
    (transfer / "provenance").mkdir()
    (transfer / "data/transfer_manifest.json").write_bytes(
        (transfer / "diagnosis/transfer_manifest.json").read_bytes()
    )
    (transfer / "provenance/final.json").write_text(
        json.dumps({p: file_hash(transfer / p) for p in TRANSFER_HASHES})
    )
    return root, derive_multi_arrangement(fit), transfer


def test_training_inputs_and_identity_rejection(fixture_run):
    _, m, reference = fixture_run
    data = MultiArrangementDataset(m)
    assert set(data.records).isdisjoint(m.splits["transfer"])
    example = data[0]

    class InputsOnly:
        image, question, answer = example.image, example.question, example.answer

        @property
        def scene(self):
            raise AssertionError("metadata entered inputs")

    batch = ShapeColorCollator(ModelConfig(vocab_size=17))([InputsOnly()])
    assert batch.input_ids[0, -1].item() == ShapeColorTokenizer().encode_answer(example.answer)
    assert batch.question_length == 6
    with pytest.raises(ValueError, match="identity"):
        read_multi_reference(reference, m.source)
    read_multi_reference(reference, m.source, smoke=True)
    with pytest.raises(ValueError, match="fixed"):
        validate_multi_protocol(
            m,
            ModelConfig(vocab_size=17),
            SyntheticTrainingConfig(),
            ShapeGroundingConfig(8640, 216),
        )


def test_cpu_cli_fresh_training_checkpoint_and_final_comparison(fixture_run, tmp_path):
    _, m, reference = fixture_run
    manifest = tmp_path / "manifest.json"
    manifest.write_text(m.content)
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
        "scripts/train_multi_arrangement.py",
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
    checkpoint = tmp_path / "training/last.pt"
    model, restored, _, payload = load_multi_checkpoint(checkpoint)
    assert (
        restored.sha256 == m.sha256
        and payload["completed_steps"] == 2
        and payload["examples_seen"] == 64
    )
    assert payload["optimizer_state"]["state"] and payload["rng_state"]
    assert all("training_fit" in h and "validation" not in h for h in payload["history"])
    assert all(
        h["training_fit"]["total"] == len(MultiArrangementDataset(m)) for h in payload["history"]
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_quartet_checkpoint(checkpoint)
    bad = copy.deepcopy(payload)
    bad["completed_steps"] = 1
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="progress"):
        load_multi_checkpoint(tmp_path / "bad.pt")
    before = file_hash(checkpoint)
    command(
        "scripts/evaluate_multi_arrangement.py",
        "--checkpoint",
        checkpoint,
        "--reference-source",
        reference,
        "--output-dir",
        tmp_path / "diagnosis",
    )
    assert file_hash(checkpoint) == before
    report = json.loads((tmp_path / "diagnosis/summary.json").read_text())
    comparison = json.loads((tmp_path / "diagnosis/comparison.json").read_text())
    rows = [
        json.loads(line)
        for line in (tmp_path / "diagnosis/predictions.jsonl").read_text().splitlines()
    ]
    by_name = {p.name: [r for r in rows if r["arrangement"] == p.name] for p in m.populations}
    for p in m.populations:
        validate_rows(
            by_name[p.name],
            SimpleNamespace(splits={p.name: p.records}),
            p.name,
            report["arrangements"][p.name]["summary"],
        )
    for role in ("training_fit", "transfer"):
        assert report["aggregates"][role] == aggregate_role(m.populations, by_name, role)
        assert report["aggregates"][role]["summary"]["total"] == 1728
    assert set(comparison["matched_changes"]) == set(m.derivation["transfer_arrangements"])
    again = load_multi_checkpoint(checkpoint)[0]
    assert all(torch.equal(v, again.state_dict()[k]) for k, v in model.state_dict().items())


def test_notebook_contract():
    n = json.loads((ROOT / "notebooks/kaggle_milestone2_multi_arrangement.ipynb").read_text())
    code = []
    for c in n["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            compile(s, "<notebook>", "exec")
            code.append(s)
            assert c["execution_count"] is None and c["outputs"] == []
    joined = "\n".join(code)
    assert (
        "run_multi_arrangement(run)" in joined
        and "FIT_SOURCE" in joined
        and "TRANSFER_SOURCE" in joined
    )


@pytest.mark.parametrize("fault", [None, "budget", "exposures"])
def test_kaggle_fixed_budget_and_packaging(tmp_path, monkeypatch, fault):
    import zipfile

    from multimodal_loop.eval import kaggle_multi_arrangement as module

    repo, root, fit, transfer = (tmp_path / n for n in ("repo", "run", "fit", "transfer"))
    (repo / "configs").mkdir(parents=True)
    fit.mkdir()
    transfer.mkdir()
    (repo / "configs/relational_baseline.yaml").write_text(
        yaml.safe_dump(asdict(ModelConfig(vocab_size=17)))
    )
    monkeypatch.setattr(module, "repository_revision", lambda _: "revision")
    monkeypatch.setattr(module, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(module, "read_transfer_reference", lambda _: (None, None, None, None, {}))
    monkeypatch.setattr(module, "read_multi_reference", lambda *args: ({}, {}, {}))
    monkeypatch.setattr(
        module,
        "derive_multi_arrangement",
        lambda _: SimpleNamespace(
            sha256="multi",
            content="{}",
            derivation={
                "training_arrangements": list(TRAIN_NAMES),
                "transfer_arrangements": list(TRANSFER_NAMES),
            },
        ),
    )
    monkeypatch.setattr(module, "presentation_summary", lambda *a, **kw: {"presentations": 276480})
    run = module.prepare_multi_arrangement(repo, root, fit, transfer)
    protocol = json.loads((root / "provenance/protocol.json").read_text())
    assert protocol["passes"] == 40 and protocol["qa_presentations"] == 276480
    calls = []

    def command(repo, root, name, args):
        calls.append(name)
        assert "--smoke" not in args and "--resume" not in args
        if name == "train":
            assert args[0] == "scripts/train_multi_arrangement.py"
            assert args[args.index("--max-steps") + 1] == "8640"
            assert args[args.index("--evaluation-interval") + 1] == "216"
            assert "--checkpoint" not in args and "--reference-source" not in args
            (root / "training").mkdir()
            (root / "training/last.pt").write_bytes(b"fresh")
            for p in ("settings.json", "manifest.json"):
                (root / "training" / p).write_text("{}")
            module.write_json(
                root / "training/presentations.json",
                {"presentations": 1 if fault == "exposures" else 276480},
            )
            module.write_json(
                root / "training/metrics.json",
                [
                    {"completed_steps": i, "completed_passes": i // 216, "training_fit": {}}
                    for i in range(0, 8641, 216)
                ],
            )
        else:
            assert args[0] == "scripts/evaluate_multi_arrangement.py"
            (root / "diagnosis").mkdir()
            module.write_json(
                root / "diagnosis/summary.json",
                {
                    "kind": "multi_arrangement_quartet_diagnostics_v1",
                    "smoke": False,
                    "completed_steps": 1 if fault == "budget" else 8640,
                    "examples_seen": 276480,
                    "checkpoint_sha256": module.file_hash(root / "training/last.pt"),
                    "manifest_sha256": "multi",
                    "model": asdict(ModelConfig(vocab_size=17)),
                    "training": asdict(SyntheticTrainingConfig()),
                    "budget": {"max_steps": 8640, "evaluation_interval": 216},
                    "evaluation": {
                        "training_arrangements": list(TRAIN_NAMES),
                        "transfer_arrangements": list(TRANSFER_NAMES),
                        "batch_size": 32,
                        "recurrence_depth": 2,
                    },
                    "aggregates": {
                        role: {"summary": {"total": 6912}, "families": {"total": 576}}
                        for role in ("training_fit", "transfer")
                    },
                },
            )
            for p in (
                "comparison.json",
                "reference_summary.json",
                "reference_transfer_aggregate.json",
                "predictions.jsonl",
                "inspection.html",
            ):
                (root / "diagnosis" / p).write_text("{}")

    monkeypatch.setattr(module, "run_logged_command", command)
    if fault:
        with pytest.raises(ValueError):
            module.run_multi_arrangement(run)
        assert not (root / "provenance/final.json").exists()
        return
    module.run_multi_arrangement(run)
    assert calls == ["train", "evaluate"]
    archive = module.archive_multi_arrangement(run)
    with zipfile.ZipFile(archive) as z:
        assert "run/training/last.pt" in z.namelist()
        assert not any("staging" in name for name in z.namelist())
    (root / "training/last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.archive_multi_arrangement(run)
    with pytest.raises(ValueError, match="fresh"):
        module.run_multi_arrangement(run)
