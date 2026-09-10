"""Budget protocol orchestration, reference safety and fixed-corpus comparisons."""

import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from multimodal_loop.data.synthetic_shapes import SHAPES
from multimodal_loop.eval.budget_reference import audit_budget_reference, stage_budget_reference
from multimodal_loop.eval.kaggle_training_budget import TrainingBudgetRun, archive_training_budget

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "member",
    [
        "../outside",
        "/absolute",
        "milestone2_geometry_diversity/../../bad",
        "wrong_root/training/last.pt",
    ],
)
def test_unsafe_reference_zip_rejected(tmp_path, member):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, "bad")
    staging = tmp_path / "staging"
    with pytest.raises(ValueError):
        stage_budget_reference(path, staging)
    assert not staging.exists()


def test_notebook_and_archive_structure(tmp_path):
    notebook = json.loads((ROOT / "notebooks/kaggle_milestone2_training_budget.ipynb").read_text())
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            text = "".join(cell["source"])
            compile(text, "<notebook>", "exec")
            sources.append(text)
    code = "\n".join(sources)
    assert "REFERENCE_SOURCE" in code and "run_training_budget(run)" in code
    root = tmp_path / "run"
    run = TrainingBudgetRun(ROOT, root, tmp_path / "reference", {}, {})
    with pytest.raises(ValueError):
        archive_training_budget(run)
    for directory in ("data", "training", "diagnosis", "provenance", "logs", "staging"):
        (root / directory).mkdir(parents=True)
        (root / directory / "example").write_text(directory)
    (root / "provenance" / "final.json").write_text("{}")
    path = archive_training_budget(run)
    with zipfile.ZipFile(path) as z:
        assert not any("staging" in p for p in z.namelist())
        assert "run/data/example" in z.namelist()
    with pytest.raises(FileExistsError):
        archive_training_budget(run)


@pytest.mark.parametrize("fault", [None, "budget", "split", "exposures", "controls", "checkpoint"])
def test_fixed_notebook_workflow_and_comparison(monkeypatch, tmp_path, fault):
    from multimodal_loop.eval import kaggle_training_budget as module
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
            "circle_square_same_prediction": {"fraction": 1 - accuracy},
            "breakdowns": {"shape": [{"shape": s, "accuracy": accuracy} for s in SHAPES]},
        }

    old = {
        "manifest_sha256": "derived-hash",
        "splits": {"train": metrics(55296, 0.9), "validation": metrics(1728, 0.5)},
        "training_geometry_subsets": {"original": metrics(6912, 0.9), "added": metrics(48384, 0.9)},
        "assessment": {
            "gates": [{"name": "train_circle", "threshold": 0.95, "value": 0.9, "passed": False}]
        },
    }
    old_controls = {"results": {"accuracy_gaps": {"correct_minus_blank": 0.25}}}
    monkeypatch.setattr(module, "repository_revision", lambda _: "committed")
    monkeypatch.setattr(module, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(module, "stage_budget_reference", lambda *args: reference)
    monkeypatch.setattr(
        module,
        "audit_budget_reference",
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
    run = module.prepare_training_budget(repo, root, reference)
    protocol = json.loads((root / "provenance" / "protocol.json").read_text())
    assert protocol["geometry_count"] == 128 and protocol["qa_presentations"] == 184320
    assert protocol["criteria"] == [{"name": "train_circle", "threshold": 0.95}]
    monkeypatch.setattr(
        module, "load_relational_manifest", lambda _: SimpleNamespace(sha256="derived-hash")
    )
    calls = []
    provenance = {"kind": "mock"}
    presentations = {"presentations": 184320}

    (reference / "data").mkdir()
    for name in ("manifest.json", "source_manifest.json", "coverage.json"):
        (reference / "data" / name).write_text("{}")
    module.write_json(reference / "data" / "derivation.json", provenance)
    monkeypatch.setattr(module, "presentation_summary", lambda *a, **kw: presentations)

    def command(repo, output, name, args):
        calls.append((name, args))
        if name == "train":
            assert "--geometry-source-manifest" in args
            assert args[args.index("--max-steps") + 1] == "5760"
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
                    "completed_steps": 5760,
                    "examples_seen": 184320,
                    "model": model,
                    "training": training,
                    "budget": {"max_steps": 5760, "evaluation_interval": 288},
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
                {
                    "checkpoint_sha256": file_hash(root / "training" / "last.pt"),
                    "manifest_sha256": "derived-hash",
                    "results": {
                        "accuracy_gaps": {"correct_minus_blank": 0.5},
                        "correct": metrics(1728, 0.75),
                        "shuffled_images": [{"seed": i} for i in range(5)],
                        "shuffled_questions": [{"seed": i} for i in range(5)],
                    },
                },
            )

            path = root / "diagnosis" / "summary.json"
            report = json.loads(path.read_text())
            if fault == "budget":
                report["completed_steps"] = 2880
            if fault == "split":
                report["splits"]["test"] = metrics(1728, 0.75)
            module.write_json(path, report)
            if fault == "exposures":
                module.write_json(root / "training" / "presentations.json", {})
            if fault == "controls":
                path = root / "diagnosis" / "controls.json"
                controls = json.loads(path.read_text())
                controls["results"]["shuffled_questions"].pop()
                module.write_json(path, controls)
            if fault == "checkpoint":
                (root / "training" / "last.pt").write_bytes(b"changed")

    monkeypatch.setattr(module, "run_logged_command", command)
    if fault:
        with pytest.raises(ValueError):
            module.run_training_budget(run)
        assert not (root / "provenance" / "final.json").exists()
        return
    module.run_training_budget(run)
    assert [name for name, _ in calls] == ["train", "evaluate"]
    comparison = json.loads((root / "diagnosis" / "comparison.json").read_text())
    assert comparison["validation"][0] == {
        "metric": "accuracy",
        "reference": 0.5,
        "current": 0.75,
        "difference": 0.25,
    }
    assert comparison["train"][0]["difference"] == pytest.approx(-0.1)
    paired = {r["metric"]: r for r in comparison["validation"]}
    assert paired["circle_square_same_prediction_fraction"]["difference"] == -0.25
    assert (root / "provenance" / "final.json").exists()
    with pytest.raises(ValueError, match="fresh run"):
        module.run_training_budget(run)
    with pytest.raises(ValueError, match="fresh run"):
        module.prepare_training_budget(repo, root, reference)


def test_reference_identity_rejected_before_loading(tmp_path):
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "last.pt").write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="identity mismatch"):
        audit_budget_reference(tmp_path)


def test_staging_zip_and_directory_interfaces(tmp_path, monkeypatch):
    from multimodal_loop.eval import budget_reference as module

    monkeypatch.setattr(module, "REFERENCE_HASHES", {"training/last.pt": "unused"})
    archive = tmp_path / "reference.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("milestone2_geometry_diversity/training/last.pt", b"reference")
    root = stage_budget_reference(archive, tmp_path / "staging")
    assert (root / "training" / "last.pt").read_bytes() == b"reference"
    assert stage_budget_reference(root, tmp_path / "unused") == root
    assert stage_budget_reference(root.parent, tmp_path / "unused") == root
    assert not (tmp_path / "unused").exists()
    with pytest.raises(ValueError, match="fresh"):
        stage_budget_reference(archive, tmp_path / "staging")


def test_comparison_rejects_changed_corpus():
    from multimodal_loop.eval.kaggle_training_budget import compare_reference

    with pytest.raises(ValueError, match="identical manifests"):
        compare_reference({"manifest_sha256": "old"}, {}, {"manifest_sha256": "new"}, {})
