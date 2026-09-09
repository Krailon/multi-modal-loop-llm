"""Notebook structure, mocked workflows, provenance, and acceptance gates."""

import copy
import json
import subprocess
import sys
import zipfile
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import pytest
import torch
import yaml

from multimodal_loop.data.relational_corpus import build_relational_splits
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.data.relational_shapes import RelationalColorQuestion
from multimodal_loop.data.synthetic_shapes import SHAPES
from multimodal_loop.eval.relational_protocol import (
    assess_validation_gates,
)
from multimodal_loop.train import kaggle
from multimodal_loop.train.relational_checkpoint import tokenizer_metadata

ROOT = Path(__file__).resolve().parents[1]


@lru_cache
def manifest_content(corpus):
    return json.dumps(
        {
            "kind": "relational_color_rows",
            "format_version": 1,
            "software": {},
            "config": asdict(corpus),
            "splits": {
                split: [asdict(r) for r in records]
                for split, records in build_relational_splits(corpus).items()
            },
        }
    )


def payload_for(run, epochs):
    content = manifest_content(run.corpus)
    manifest = parse_relational_manifest(content)
    training_count = 576 * run.corpus.train_geometry_count
    validation_count = 576 * run.corpus.validation_geometry_count
    steps = training_count // 32
    history = [
        {
            "epoch": e,
            "completed_steps": e * steps,
            "train": None if e == 0 else {"loss": 1.0, "examples": training_count, "steps": steps},
            "validation": {
                "total": validation_count,
                "correct": validation_count // 4,
                "accuracy": 0.25,
                "invalid_predictions": 0,
                "loss": 1.0,
            },
        }
        for e in range(epochs + 1)
    ]
    return {
        "kind": "relational_color",
        "format_version": 1,
        "backend": "cuda",
        "model_dtype": "float32",
        "config": run.model,
        "training_config": asdict(run.training),
        "tokenizer": tokenizer_metadata(),
        "manifest_content": content,
        "manifest_sha256": manifest.sha256,
        "completed_epochs": epochs,
        "completed_steps": epochs * steps,
        "history": history,
        "runtime": {"device": "cuda:0", "torch": "test fixture"},
        "optimizer_state": {
            "param_groups": [
                {
                    "lr": 0.001,
                    "weight_decay": 0.0,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "foreach": False,
                    "fused": False,
                }
            ]
        },
    }


def write_checkpoint(run, epochs, path=None):
    path = path or run.checkpoint
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload_for(run, epochs), path)
    return path


def metrics(images, accuracy):
    def counts(n, a):
        return {"total": n, "correct": int(n * a), "accuracy": a}

    return {
        **counts(4 * images, accuracy),
        "invalid_predictions": 0,
        "loss": 1.0,
        "by_question": {
            RelationalColorQuestion(shape, direction).text: counts(4 * images // 6, accuracy)
            for shape in SHAPES
            for direction in ("left", "right")
        },
        "all_four": counts(images, 1.0 if accuracy == 1 else 0.0),
        "different_answer_pairs": counts(5 * images, 1.0 if accuracy == 1 else 0.0),
    }


def report_for(run):
    payload = kaggle.inspect_checkpoint(run, run.checkpoint)
    images = 144 * run.corpus.validation_geometry_count
    result = {
        "correct": metrics(images, 1.0),
        "blank": metrics(images, 0.25),
        "accuracy_gaps": {"correct_minus_blank": 0.75},
    }
    for kind, accuracy in (("images", 0.25), ("questions", 0.375)):
        rows = []
        for seed in run.seeds:
            row = {"seed": seed, "metrics": metrics(images, accuracy), "accuracy_gap": 1 - accuracy}
            if kind == "images":
                row["permutation"] = list(range(images))
            else:
                row["permutations"] = [[0, 1, 2, 3] for _ in range(images)]
                row["same_answer_pairing_fraction"] = 1.0
            rows.append(row)
        result[f"shuffled_{kind}"] = rows
        result[f"shuffled_{kind}_summary"] = {
            "accuracy": {k: accuracy for k in ("mean", "min", "max")},
            "loss": {k: 1.0 for k in ("mean", "min", "max")},
        }
        result["accuracy_gaps"][f"correct_minus_shuffled_{kind}_mean"] = 1 - accuracy
    return {
        "kind": "relational_color_controls",
        "format_version": 1,
        "checkpoint_sha256": kaggle.file_hash(run.checkpoint),
        "manifest_sha256": payload["manifest_sha256"],
        "model": run.model,
        "training": asdict(run.training),
        "completed_epochs": run.epochs,
        "completed_steps": payload["completed_steps"],
        "evaluation": {
            "split": "validation",
            "batch_size": 32,
            "recurrence_depth": 2,
            "shuffle_seeds": run.seeds,
        },
        "results": result,
    }


def validate_report(run, report):
    payload = kaggle.inspect_checkpoint(run, run.checkpoint)
    kaggle.validate_control_report(
        report,
        checkpoint_hash=kaggle.file_hash(run.checkpoint),
        manifest_hash=payload["manifest_sha256"],
        model=run.model,
        training=asdict(run.training),
        epochs=run.epochs,
        steps=payload["completed_steps"],
        images=144 * run.corpus.validation_geometry_count,
        seeds=run.seeds,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    path = tmp_path / "repo"
    (path / "configs").mkdir(parents=True)
    (path / "configs" / "relational_baseline.yaml").write_bytes(
        (ROOT / "configs" / "relational_baseline.yaml").read_bytes()
    )
    monkeypatch.setattr(kaggle, "repository_revision", lambda repo: "a" * 40)
    monkeypatch.setattr(kaggle, "cuda_runtime", lambda: {"device": "cuda:0", "torch": "fixture"})
    return path


@pytest.fixture
def run(repo, tmp_path):
    return kaggle.prepare_run(repo, tmp_path / "baseline")


@pytest.fixture
def commands(monkeypatch):
    calls = []

    def command(run, name, arguments):
        calls.append((name, arguments))
        if name == "generate":
            path = run.root / "data" / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(manifest_content(run.corpus))
        elif name == "train":
            completed = 0
            if "--resume" in arguments:
                source = Path(arguments[arguments.index("--resume") + 1])
                completed = kaggle.inspect_checkpoint(run, source)["completed_epochs"]
            additional = int(arguments[arguments.index("--epochs") + 1])
            write_checkpoint(run, completed + additional)
            # Match the CLI's authoritative history sidecar after every completed epoch.
            payload = kaggle.inspect_checkpoint(run, run.checkpoint)
            (run.checkpoint.parent / "metrics.json").write_text(kaggle._json(payload["history"]))
        elif name == "evaluate":
            assert arguments[arguments.index("--split") + 1] == "validation"
            path = run.root / "validation" / "controls.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report_for(run)))
        else:
            raise AssertionError(name)

    monkeypatch.setattr(kaggle, "run_command", command)
    return calls


def test_fresh_baseline_and_completed_rerun(run, commands):
    kaggle.train_run(run)
    assert [name for name, _ in commands] == ["generate", "train"]
    args = commands[-1][1]
    assert args[args.index("--epochs") + 1] == "10" and "--resume" not in args
    report = kaggle.evaluate_run(run)
    assert report["completed_steps"] == 2880
    assert json.loads((run.root / "validation" / "acceptance.json").read_text())["passed"]
    original_checkpoint = run.checkpoint.read_bytes()
    original_report = (run.root / "validation" / "controls.json").read_bytes()
    calls = len(commands)
    repeated = kaggle.prepare_run(run.repo, run.root)
    kaggle.train_run(repeated)
    kaggle.evaluate_run(repeated)
    assert len(commands) == calls
    assert run.checkpoint.read_bytes() == original_checkpoint
    assert (run.root / "validation" / "controls.json").read_bytes() == original_report
    archive = kaggle.archive_run(run)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert any(p.endswith("training/last.pt") for p in names)
        assert any(p.endswith("validation/acceptance.json") for p in names)
        assert any(p.endswith("provenance/protocol.json") for p in names)
        assert not any(p.endswith(".zip") for p in names)


def test_partial_resume_budget_and_embedded_manifest(run, commands):
    write_checkpoint(run, 4)
    kaggle.train_run(run)
    assert len(commands) == 1
    args = commands[0][1]
    assert args[args.index("--epochs") + 1] == "6"
    assert args[args.index("--resume") + 1] == str(run.checkpoint)
    assert "--manifest" not in args and "--seed" not in args
    assert (run.root / "data" / "manifest.json").read_text() == manifest_content(run.corpus)


@pytest.mark.parametrize("epochs", [4, 10])
def test_external_resume_into_fresh_output(run, tmp_path, commands, epochs):
    source = write_checkpoint(run, epochs)
    original = source.read_bytes()
    resumed = kaggle.prepare_run(run.repo, tmp_path / "continued", resume_checkpoint=source)
    kaggle.train_run(resumed)
    assert source.read_bytes() == original
    assert len(commands) == (0 if epochs == 10 else 1)
    assert resumed.checkpoint.exists()
    assert (resumed.checkpoint.parent / "settings.json").exists()
    kaggle.evaluate_run(resumed)
    assert json.loads((resumed.root / "provenance" / "resume_source.json").read_text())[
        "sha256"
    ] == kaggle.file_hash(source)


def test_smoke_runs_two_epochs_and_one_shuffle_only(repo, tmp_path, commands):
    smoke = kaggle.prepare_run(repo, tmp_path / "smoke", smoke=True)
    kaggle.train_run(smoke)
    report = kaggle.evaluate_run(smoke)
    training = [args for name, args in commands if name == "train"]
    assert len(training) == 2 and all(a[a.index("--epochs") + 1] == "1" for a in training)
    assert "--resume" not in training[0] and "--resume" in training[1]
    assert report["completed_steps"] == 36 and report["results"]["correct"]["total"] == 576
    assert report["evaluation"]["shuffle_seeds"] == [0]
    assert not (smoke.root / "validation" / "acceptance.json").exists()
    with pytest.raises(ValueError, match="fresh"):
        kaggle.prepare_run(repo, smoke.root, smoke=True)


@pytest.mark.parametrize(
    "damage", ["depth", "corpus", "epochs", "steps", "optimizer", "backend", "dtype", "tokenizer"]
)
def test_resume_rejects_mismatched_protocol_before_commands(run, commands, damage):
    payload = payload_for(run, 4)
    if damage == "depth":
        payload["training_config"]["recurrence_depth"] = 3
    elif damage == "corpus":
        payload["manifest_sha256"] = "wrong"
    elif damage == "epochs":
        payload = payload_for(run, 11)
    elif damage == "steps":
        payload["completed_steps"] = 1
    elif damage == "optimizer":
        payload["optimizer_state"]["param_groups"][0]["lr"] = 0.5
    elif damage == "backend":
        payload["backend"] = "cpu"
    elif damage == "dtype":
        payload["model_dtype"] = "float64"
    elif damage == "tokenizer":
        payload["tokenizer"]["version"] = 99
    run.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, run.checkpoint)
    with pytest.raises(ValueError):
        kaggle.train_run(run)
    assert not commands


def test_setup_guards(repo, tmp_path, run, monkeypatch):
    with pytest.raises(ValueError, match="separate"):
        kaggle.prepare_run(repo, repo / "artifacts")
    (run.root / "training").mkdir()
    (run.root / "training" / "metrics.json").write_text("old")
    with pytest.raises(ValueError, match="without a checkpoint"):
        kaggle.prepare_run(repo, run.root)
    source = write_checkpoint(run, 4)
    with pytest.raises(ValueError, match="fresh"):
        kaggle.prepare_run(repo, run.root, resume_checkpoint=source)
    monkeypatch.setattr(kaggle, "repository_revision", lambda repo: "b" * 40)
    with pytest.raises(ValueError, match="disagrees"):
        kaggle.prepare_run(repo, run.root)
    monkeypatch.setattr(
        kaggle, "cuda_runtime", lambda: (_ for _ in ()).throw(ValueError("CUDA unavailable"))
    )
    root = tmp_path / "no_gpu"
    with pytest.raises(ValueError, match="CUDA unavailable"):
        kaggle.prepare_run(repo, root)
    assert not root.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "checkpoint",
        "manifest",
        "split",
        "seed",
        "model",
        "count",
        "nonfinite",
        "missing",
        "summary",
        "permutation",
        "gap",
    ],
)
def test_existing_bad_report_is_rejected_without_overwrite(run, commands, damage):
    write_checkpoint(run, 10)
    report = report_for(run)
    if damage == "checkpoint":
        report["checkpoint_sha256"] = "wrong"
    elif damage == "manifest":
        report["manifest_sha256"] = "wrong"
    elif damage == "split":
        report["evaluation"]["split"] = "test"
    elif damage == "seed":
        report["results"]["shuffled_images"].pop()
    elif damage == "model":
        report["model"] = {**run.model, "d_model": 128}
    elif damage == "count":
        report["results"]["correct"]["total"] = 144
    elif damage == "nonfinite":
        report["results"]["blank"]["loss"] = float("nan")
    elif damage == "missing":
        report["results"]["correct"]["by_question"].popitem()
    elif damage == "summary":
        report["results"]["shuffled_images_summary"]["accuracy"]["mean"] = 0.5
    elif damage == "permutation":
        report["results"]["shuffled_images"][0]["permutation"] = [0]
    elif damage == "gap":
        report["results"]["accuracy_gaps"]["correct_minus_blank"] = 0.6
    path = run.root / "validation" / "controls.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        kaggle.evaluate_run(run)
    assert path.read_bytes() == before and not commands


def boundary_results():
    result = {
        "correct": metrics(576, 1.0),
        "accuracy_gaps": {
            "correct_minus_blank": 0.3,
            "correct_minus_shuffled_images_mean": 0.3,
            "correct_minus_shuffled_questions_mean": 0.3,
        },
    }
    result["correct"]["accuracy"] = 0.9
    result["correct"]["all_four"]["accuracy"] = 0.8
    for row in result["correct"]["by_question"].values():
        row["accuracy"] = 0.8
    return result


def test_inclusive_gate_boundaries():
    result = assess_validation_gates(boundary_results())
    assert result["passed"] and len(result["gates"]) == 6
    assert all(row["value"] == row["threshold"] for row in result["gates"])


@pytest.mark.parametrize("gate", range(6))
def test_each_gate_can_fail_without_changing_thresholds(gate):
    data = boundary_results()
    if gate == 0:
        data["correct"]["accuracy"] -= 1e-8
    elif gate == 1:
        next(iter(data["correct"]["by_question"].values()))["accuracy"] -= 1e-8
    elif gate == 2:
        data["correct"]["all_four"]["accuracy"] -= 1e-8
    else:
        data["accuracy_gaps"][list(data["accuracy_gaps"])[gate - 3]] -= 1e-8
    original = copy.deepcopy(data)
    summary = assess_validation_gates(data)
    assert not summary["passed"] and sum(not row["passed"] for row in summary["gates"]) == 1
    assert data == original


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True, "0.9"])
def test_invalid_gate_values(value):
    data = boundary_results()
    data["correct"]["accuracy"] = value
    with pytest.raises(ValueError):
        assess_validation_gates(data)


def test_missing_gate_fields():
    data = boundary_results()
    del data["accuracy_gaps"]["correct_minus_blank"]
    with pytest.raises(ValueError):
        assess_validation_gates(data)


def test_subprocess_streams_and_logs_failure(run, capsys):
    with pytest.raises(subprocess.CalledProcessError):
        kaggle.run_command(
            run, "failure", ["-c", "print('retained output', flush=True); raise SystemExit(3)"]
        )
    assert "retained output" in capsys.readouterr().out
    logs = list((run.root / "logs").glob("failure-*.log"))
    assert len(logs) == 1 and "retained output" in logs[0].read_text()
    assert sys.executable in logs[0].read_text()


@pytest.mark.parametrize("name", ["kaggle_relational_smoke", "kaggle_milestone2_baseline"])
def test_notebook_structure_compilation_and_thin_workflow(name):
    notebook = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    ids = [c["id"] for c in notebook["cells"]]
    assert len(ids) == len(set(ids))
    sources = []
    for cell in notebook["cells"]:
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            source = "".join(cell["source"])
            compile(source, f"{name}:{cell['id']}", "exec")
            sources.append(source)
    code = "\n".join(sources)
    assert "checkpoint = train_run(run)" in code and "report = evaluate_run(run)" in code
    assert "archive = archive_run(run)" in code
    assert '"--split", "test"' not in code
    assert ("smoke=True" in code) == (name == "kaggle_relational_smoke")
    assert ("resume_checkpoint=RESUME_CHECKPOINT" in code) == (name != "kaggle_relational_smoke")


@pytest.mark.parametrize(
    "existing,mismatch,dirty",
    [(False, False, False), (True, False, False), (True, True, False), (True, False, True)],
)
def test_bootstrap_checkout_guards_without_network(
    tmp_path, monkeypatch, existing, mismatch, dirty
):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    notebook = json.loads((ROOT / "notebooks" / "kaggle_milestone2_baseline.ipynb").read_text())
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code"
        and "git clone" not in "".join(c["source"])
        and "def git(" in "".join(c["source"])
    )
    repo = tmp_path / "checkout"
    if existing:
        repo.mkdir()
    url = "https://github.com/Krailon/multi-modal-loop-llm.git"
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))

    def output(args, **kwargs):
        if args[1:3] == ["remote", "get-url"]:
            return url
        if args[1:3] == ["status", "--porcelain"]:
            return " M changed.py" if dirty else ""
        return "new" if mismatch and args[-1] == "FETCH_HEAD" else "old"

    monkeypatch.setattr(subprocess, "check_output", output)
    namespace = {
        "REPO_DIR": str(repo),
        "RUN_ROOT": str(tmp_path / "output"),
        "REPO_URL": url,
        "REPO_REF": "milestone2",
    }
    if mismatch or dirty:
        with pytest.raises(ValueError):
            exec(source, namespace)
        assert not any("pip" in a for a in calls)
    else:
        exec(source, namespace)
        assert any("pip" in a for a in calls)
        assert any("checkout" in a for a in calls) == (not existing)


def test_completed_checkpoint_repairs_stale_history_without_training(run, commands):
    write_checkpoint(run, 10)
    sidecar = run.checkpoint.parent / "metrics.json"
    sidecar.write_text("stale")
    kaggle.train_run(run)
    assert not commands
    assert json.loads(sidecar.read_text()) == payload_for(run, 10)["history"]


def test_wrong_model_yaml_rejected_before_artifacts(repo, tmp_path):
    path = repo / "configs" / "relational_baseline.yaml"
    model = yaml.safe_load(path.read_text())
    model["d_model"] = 128
    path.write_text(yaml.safe_dump(model))
    with pytest.raises(ValueError, match="Model YAML"):
        kaggle.prepare_run(repo, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_notebook_interrupt_stops_subprocess(run, monkeypatch, timeout):
    actions = []

    class InterruptedProcess:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @property
        def stdout(self):
            yield "training started\n"
            raise KeyboardInterrupt

        def terminate(self):
            actions.append("terminate")

        def kill(self):
            actions.append("kill")

        def wait(self, **kwargs):
            actions.append("wait")
            if timeout and kwargs:
                raise subprocess.TimeoutExpired("fixture", 5)
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: InterruptedProcess())
    with pytest.raises(KeyboardInterrupt):
        kaggle.run_command(run, "interrupt", ["unused.py"])
    assert actions == (["terminate", "wait", "kill", "wait"] if timeout else ["terminate", "wait"])
