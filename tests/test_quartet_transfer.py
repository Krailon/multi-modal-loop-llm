"""Unseen-arrangement selection, frozen evaluation and reproduction accounting."""

import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from multimodal_loop.data.geometry_diversity import derive_geometry_corpus
from multimodal_loop.data.matched_size import derive_matched_size, origin_key, scene_key
from multimodal_loop.data.quartet_fit import QuartetFitDataset, derive_quartet_fit
from multimodal_loop.data.quartet_transfer import build_transfer_populations, transfer_manifest
from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.eval.quartet_fit import evaluate_quartet_checkpoint
from multimodal_loop.eval.quartet_transfer import (
    aggregate_transfer,
    evaluate_transfer,
    matched_changes,
)
from multimodal_loop.eval.quartet_transfer_reference import (
    REFERENCE_HASHES,
    read_transfer_reference,
)
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.kaggle import file_hash
from multimodal_loop.train.quartet_fit_checkpoint import save_quartet_checkpoint
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


def test_full_selection_counts_and_matching():
    source, _ = derive_geometry_corpus(source_manifest(RelationalCorpusConfig()))
    fit = derive_quartet_fit(derive_matched_size(source))
    populations = build_transfer_populations(fit)
    assert len(populations) == 8
    assert populations[0].origins == ((1, 4), (10, 4), (20, 4))
    assert {p.triangle_size for p in populations} == {8}
    assert len({p.origins for p in populations}) == 8
    assert sum(len(p.records) * 3 for p in populations if p.role == "transfer") == 12096
    assert len({scene_key(r.scene) for p in populations for r in p.records}) == 4608
    reserved = {origin_key(r.scene) for s in ("validation", "test") for r in fit.source.splits[s]}
    assert all(p.origins not in reserved for p in populations)
    assert populations[0].records == fit.splits["train"]
    for p in populations:
        assert len(p.records) == 576 and len(p.families) == 144
        for original, transferred in zip(populations[0].records, p.records, strict=True):
            assert [(o.shape, o.color, o.size) for o in original.scene.objects] == [
                (o.shape, o.color, o.size) for o in transferred.scene.objects
            ]
            assert all(
                o.left >= 1 and o.top >= 1 and o.left + o.size < 32 and o.top + o.size < 32
                for o in transferred.scene.objects
            )
    assert transfer_manifest(fit, populations) == transfer_manifest(
        fit, build_transfer_populations(fit)
    )
    bad = SimpleNamespace(
        source=SimpleNamespace(
            derivation=fit.source.derivation,
            splits={**fit.source.splits, "test": populations[1].records},
        ),
        derivation=fit.derivation,
    )
    with pytest.raises(ValueError, match="reserved"):
        build_transfer_populations(bad)


def test_matched_accounting():
    rows = [
        {
            "image_index": 0,
            "question_index": i,
            "shape": "square",
            "question": "q",
            "answer": "red",
            "target_id": 6,
            "prediction_id": token,
            "correct": token == 6,
        }
        for i, token in enumerate([6, 7, 6, 7])
    ]
    current = [
        {**r, "prediction_id": token, "correct": token == 6}
        for r, token in zip(rows, [7, 6, 6, 8], strict=True)
    ]
    result = matched_changes(rows, current)
    assert result["prediction_changes"] == 3
    assert [
        result[k] for k in ("correct_to_wrong", "wrong_to_correct", "both_correct", "both_wrong")
    ] == [1] * 4
    with pytest.raises(ValueError, match="equal"):
        matched_changes(rows, current[:-1])
    current[0]["answer"] = "blue"
    with pytest.raises(ValueError, match="identity"):
        matched_changes(rows, current)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("transfer-reference")
    source = source_manifest(
        RelationalCorpusConfig(
            object_sizes=(8,),
            train_geometry_count=2,
            validation_geometry_count=1,
            test_geometry_count=1,
        )
    )
    fit = derive_quartet_fit(derive_matched_size(source))
    (root / "data").mkdir()
    (root / "training").mkdir()
    (root / "provenance").mkdir()
    (root / "data/manifest.json").write_text(fit.content)
    model = MultimodalLoopTransformer(ModelConfig(vocab_size=17, d_model=8, n_heads=2, d_ff=16))
    training, budget = SyntheticTrainingConfig(), ShapeGroundingConfig(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate)

    def publish(history):
        save_quartet_checkpoint(
            root / "training/last.pt",
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
    evaluate_quartet_checkpoint(root / "training/last.pt", root / "diagnosis")
    (root / "provenance/final.json").write_text(
        json.dumps({p: file_hash(root / p) for p in REFERENCE_HASHES})
    )
    return root, fit


def test_cpu_frozen_smoke_and_aggregate(reference, tmp_path, monkeypatch):
    root, fit = reference
    before = file_hash(root / "training/last.pt")

    def no_optimizer(*args, **kwargs):
        raise AssertionError("frozen evaluation created optimizer")

    monkeypatch.setattr(torch.optim, "AdamW", no_optimizer)
    out = tmp_path / "diagnosis"
    report = evaluate_transfer(root, out, smoke=True)
    assert report["reproduction"]["passed"] and report["reproduction"]["prediction_changes"] == 0
    assert report["transfer_aggregate"]["summary"]["total"] == 1728
    assert report["transfer_aggregate"]["families"]["total"] == 144
    assert report["evaluation"]["total_questions"] == 3456
    assert "assessment" not in report
    assert file_hash(root / "training/last.pt") == before
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    populations = build_transfer_populations(fit)
    by_name = {p.name: [r for r in rows if r["arrangement"] == p.name] for p in populations}
    assert report["transfer_aggregate"] == aggregate_transfer(populations, by_name)
    assert report["transfer_aggregate"]["summary"]["correct"] == sum(
        r["correct"] for r in by_name["transfer_01"]
    )
    assert len(report["matched_changes"]) == 1
    with pytest.raises(ValueError, match="fresh"):
        evaluate_transfer(root, out, smoke=True)


def test_reference_identity_zip_and_corruption(reference, tmp_path):
    root, _ = reference
    with pytest.raises(ValueError, match="identity"):
        read_transfer_reference(root)
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in [*REFERENCE_HASHES, "provenance/final.json"]:
            z.write(root / name, arcname="milestone2_quartet_fit/" + name)
    assert (
        read_transfer_reference(archive, smoke=True)[4]
        == read_transfer_reference(root, smoke=True)[4]
    )
    corrupt = tmp_path / "bad"
    for name in [*REFERENCE_HASHES, "provenance/final.json"]:
        p = corrupt / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((root / name).read_bytes())
    (corrupt / "training/last.pt").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash"):
        read_transfer_reference(corrupt, smoke=True)


def test_reproduction_failure_keeps_reports(reference, tmp_path, monkeypatch):
    from multimodal_loop.eval import quartet_transfer as module

    original = module.matched_changes
    calls = 0

    def mismatch(a, b):
        nonlocal calls
        result = original(a, b)
        if calls == 0:
            result["prediction_changes"] = 1
        calls += 1
        return result

    monkeypatch.setattr(module, "matched_changes", mismatch)
    out = tmp_path / "diagnosis"
    report = module.evaluate_transfer(reference[0], out, smoke=True)
    assert not report["reproduction"]["passed"]
    assert "FAILED" in (out / "inspection.html").read_text()
    assert (out / "predictions.jsonl").is_file() and (out / "matched_comparisons.json").is_file()


def test_notebook():
    n = json.loads((ROOT / "notebooks/kaggle_milestone2_quartet_transfer.ipynb").read_text())
    code = []
    for c in n["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            compile(s, "<notebook>", "exec")
            code.append(s)
            assert c["execution_count"] is None and c["outputs"] == []
    assert "run_quartet_transfer(run)" in "\n".join(code)
    assert "REPRODUCTION FAILED" in "\n".join(code)


@pytest.mark.parametrize("reproduced", [True, False])
def test_kaggle_evaluation_only_and_archive(tmp_path, monkeypatch, reproduced):
    from multimodal_loop.eval import kaggle_quartet_transfer as module

    root, repo, source = (tmp_path / n for n in ("run", "repo", "source"))
    repo.mkdir()
    source.mkdir()
    monkeypatch.setattr(module, "repository_revision", lambda _: "revision")
    monkeypatch.setattr(module, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(
        module, "read_transfer_reference", lambda _: (None, None, None, None, REFERENCE_HASHES)
    )
    monkeypatch.setattr(module, "build_transfer_populations", lambda _: [None] * 8)
    monkeypatch.setattr(module, "transfer_manifest", lambda *a: ("{}", "manifest"))
    run = module.prepare_quartet_transfer(repo, root, source)
    protocol = json.loads((root / "provenance/protocol.json").read_text())
    assert protocol["model_updates"] == 0 and protocol["accuracy_gates"] == []
    with pytest.raises(ValueError, match="complete"):
        module.archive_quartet_transfer(run)
    calls = []

    def command(repo, root, name, args):
        calls.append(name)
        assert name == "evaluate" and args[0] == "scripts/evaluate_quartet_transfer.py"
        assert "--smoke" not in args and "--max-steps" not in args
        assert args[args.index("--device") + 1] == "cuda:0"
        d = root / "diagnosis"
        d.mkdir()
        (d / "transfer_manifest.json").write_text("{}")
        module.write_json(
            d / "summary.json",
            {
                "kind": "quartet_transfer_diagnostics_v1",
                "smoke": False,
                "checkpoint_sha256": REFERENCE_HASHES["training/last.pt"],
                "transfer_manifest_sha256": "manifest",
                "evaluation": {"total_questions": 13824, "transfer_questions": 12096},
                "transfer_aggregate": {"summary": {"total": 12096}, "families": {"total": 1008}},
                "reproduction": {"passed": reproduced},
            },
        )
        for name in (
            "reference_summary.json",
            "reference_hashes.json",
            "matched_comparisons.json",
            "predictions.jsonl",
            "inspection.html",
        ):
            (d / name).write_text("{}")

    monkeypatch.setattr(module, "run_logged_command", command)
    report = module.run_quartet_transfer(run)
    assert calls == ["evaluate"] and report["reproduction"]["passed"] == reproduced
    archive = module.archive_quartet_transfer(run)
    with zipfile.ZipFile(archive) as z:
        assert "run/diagnosis/predictions.jsonl" in z.namelist()
        assert not any(n.endswith(".pt") for n in z.namelist())
    (root / "diagnosis/predictions.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        module.archive_quartet_transfer(run)
    with pytest.raises(ValueError, match="fresh"):
        module.run_quartet_transfer(run)
