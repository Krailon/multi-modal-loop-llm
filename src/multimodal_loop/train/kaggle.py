"""Thin notebook orchestration for the fixed relational CUDA protocols.

Training and model evaluation stay in the existing CLI scripts. These helpers
manage paths, subprocess logs, resume budgets and report provenance.
"""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from multimodal_loop.data.relational_corpus import RelationalCorpusConfig
from multimodal_loop.data.relational_dataset import (
    load_relational_manifest,
    parse_relational_manifest,
)
from multimodal_loop.eval.relational_protocol import (
    assess_validation_gates,
    validate_control_report,
)
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.train.relational_checkpoint import _validate_progress, tokenizer_metadata
from multimodal_loop.train.runtime import resolve_device, runtime_metadata
from multimodal_loop.train.synthetic import SyntheticTrainingConfig


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write_once(path: Path, content: str) -> None:
    """Reuse identical provenance/data; never silently replace a different file."""
    if path.exists():
        if path.read_bytes() != content.encode("utf-8"):
            raise ValueError(f"Existing file disagrees with this run: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)


def _json(value) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def repository_revision(repo: Path) -> str:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    if dirty:
        raise ValueError("Notebook checkout must be clean; commit changes before the research run")
    return revision


def cuda_runtime() -> dict:
    return {
        **runtime_metadata(resolve_device("cuda:0")),
        "python": platform.python_version(),
        "torch_num_threads": torch.get_num_threads(),
    }


@dataclass(frozen=True)
class NotebookRun:
    repo: Path
    root: Path
    model: dict
    corpus: RelationalCorpusConfig
    training: SyntheticTrainingConfig
    smoke: bool
    resume_checkpoint: Path | None = None

    @property
    def epochs(self) -> int:
        return 2 if self.smoke else 10

    @property
    def checkpoint(self) -> Path:
        return self.root / "training" / "last.pt"

    @property
    def seeds(self) -> list[int]:
        return [0] if self.smoke else list(range(5))


def inspect_checkpoint(run: NotebookRun, path: Path) -> dict:
    """Read on CPU without creating a model or restoring RNGs."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or type(payload.get("format_version")) is not int:
        raise ValueError("Invalid relational checkpoint format")
    expected = {
        "kind": "relational_color",
        "format_version": 1,
        "backend": "cuda",
        "model_dtype": "float32",
        "config": run.model,
        "training_config": asdict(run.training),
        "tokenizer": tokenizer_metadata(),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Checkpoint {key} disagrees with notebook protocol")
    manifest = parse_relational_manifest(payload["manifest_content"])
    if manifest.config != run.corpus or manifest.sha256 != payload["manifest_sha256"]:
        raise ValueError("Checkpoint corpus disagrees with notebook protocol")
    _validate_progress(
        manifest,
        run.training,
        payload["completed_epochs"],
        payload["completed_steps"],
        payload["history"],
    )
    if payload["completed_epochs"] > run.epochs:
        raise ValueError("Checkpoint exceeds the fixed epoch budget")
    group = payload["optimizer_state"]["param_groups"]
    expected_optimizer = {
        "lr": 0.001,
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "foreach": False,
        "fused": False,
    }
    if len(group) != 1 or any(group[0].get(k) != v for k, v in expected_optimizer.items()):
        raise ValueError("Checkpoint optimizer disagrees with protocol")
    return payload


def prepare_run(
    repo: str | Path,
    root: str | Path,
    *,
    smoke: bool = False,
    resume_checkpoint: str | Path | None = None,
) -> NotebookRun:
    repo, root = Path(repo).resolve(), Path(root).resolve()
    source = Path(resume_checkpoint).resolve() if resume_checkpoint else None
    if root == repo or root.is_relative_to(repo) or repo.is_relative_to(root):
        raise ValueError("Artifacts and repository must be separate directory trees")
    if smoke and (source is not None or root.exists()):
        raise ValueError("Smoke checks require a fresh output directory and no resume override")
    if source is not None and root.exists() and any(root.iterdir()):
        raise ValueError("An external resume checkpoint requires a fresh output directory")
    model = yaml.safe_load((repo / "configs" / "relational_baseline.yaml").read_text())
    # Pin all settings rather than silently inheriting later changes to defaults/YAML.
    expected_model = dict(
        vocab_size=17,
        max_seq_len=128,
        d_model=64,
        n_heads=4,
        d_ff=256,
        n_prelude_layers=1,
        n_recurrent_layers=1,
        n_coda_layers=1,
        recurrence_depth=2,
        image_size=32,
        patch_size=8,
        num_channels=3,
        dropout=0.0,
        layer_norm_eps=1e-5,
    )
    if model != expected_model:
        raise ValueError("Model YAML disagrees with the fixed Milestone 2 protocol")
    ModelConfig(**model)
    corpus = RelationalCorpusConfig(
        image_size=32,
        object_sizes=(6, 8),
        seed=1 if smoke else 0,
        train_geometry_count=1 if smoke else 16,
        validation_geometry_count=1 if smoke else 4,
        test_geometry_count=1 if smoke else 4,
    )
    training = SyntheticTrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.0,
        seed=1 if smoke else 0,
        recurrence_depth=2,
    )
    run = NotebookRun(repo, root, model, corpus, training, smoke, source)
    checkpoint = run.checkpoint if run.checkpoint.exists() else source
    if checkpoint is not None:
        inspect_checkpoint(run, checkpoint)
    elif (root / "training").exists() and any((root / "training").iterdir()):
        raise ValueError("Training artifacts exist without a checkpoint; use a fresh directory")
    revision = repository_revision(repo)
    runtime = cuda_runtime()  # Fail before making run artifacts if CUDA is unavailable.
    provenance = {
        "revision": revision,
        "model": model,
        "corpus": asdict(corpus),
        "training": asdict(run.training),
        "target_epochs": run.epochs,
        "smoke": smoke,
    }
    provenance_dir = root / "provenance"
    _write_once(provenance_dir / "protocol.json", _json(provenance))
    _write_once(
        provenance_dir / "model.yaml", (repo / "configs" / "relational_baseline.yaml").read_text()
    )
    if source is not None:
        _write_once(
            provenance_dir / "resume_source.json",
            _json({"path": str(source), "sha256": file_hash(source)}),
        )
    # Retain every session's runtime rather than overwriting the original environment.
    _write_once(provenance_dir / f"runtime-{uuid.uuid4().hex}.json", _json(runtime))
    print(_json({"revision": revision, "runtime": runtime, "artifacts": str(root)}))
    return run


def run_command(run: NotebookRun, name: str, arguments: list[str]) -> None:
    """Stream a checked subprocess to the cell and a unique persistent log."""
    log = run.root / "logs" / f"{name}-{uuid.uuid4().hex}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *map(str, arguments)]
    print(f"Running: {command}\nLog: {log}", flush=True)
    with log.open("x", encoding="utf-8") as handle:
        handle.write(_json({"command": command}))
        with subprocess.Popen(
            command,
            cwd=run.repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
                    handle.flush()
                code = process.wait()
            except BaseException:
                # A notebook interrupt must not leave training running while
                # a later cell attempts to resume the same checkpoint.
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def prepare_data(run: NotebookRun) -> Path:
    destination = run.root / "data" / "manifest.json"
    checkpoint = run.checkpoint if run.checkpoint.exists() else run.resume_checkpoint
    if checkpoint is not None:
        payload = inspect_checkpoint(run, checkpoint)
        _write_once(destination, payload["manifest_content"])
    elif not destination.exists():
        run_command(
            run,
            "generate",
            [
                "scripts/generate_relational_data.py",
                "--image-size",
                "32",
                "--object-sizes",
                "6",
                "8",
                "--seed",
                str(run.corpus.seed),
                "--train-geometry-count",
                str(run.corpus.train_geometry_count),
                "--validation-geometry-count",
                str(run.corpus.validation_geometry_count),
                "--test-geometry-count",
                str(run.corpus.test_geometry_count),
                "--preview-count",
                "1" if run.smoke else "12",
                "--output-dir",
                str(destination.parent),
            ],
        )
    manifest = load_relational_manifest(destination)
    if manifest.config != run.corpus:
        raise ValueError("Existing corpus disagrees with notebook protocol")
    _write_once(run.root / "provenance" / "manifest.sha256", manifest.sha256 + "\n")
    return destination


def train_run(run: NotebookRun) -> Path:
    manifest = prepare_data(run)
    checkpoint = run.checkpoint if run.checkpoint.exists() else run.resume_checkpoint
    completed = inspect_checkpoint(run, checkpoint)["completed_epochs"] if checkpoint else 0
    while completed < run.epochs:
        additional = 1 if run.smoke else run.epochs - completed
        args = [
            "scripts/train_relational.py",
            "--device",
            "cuda:0",
            "--epochs",
            str(additional),
            "--output-dir",
            str(run.checkpoint.parent),
        ]
        if checkpoint is not None:
            args += ["--resume", str(checkpoint)]
        else:
            args += [
                "--manifest",
                str(manifest),
                "--model-config",
                str(run.repo / "configs" / "relational_baseline.yaml"),
                "--batch-size",
                "32",
                "--recurrence-depth",
                "2",
                "--learning-rate",
                "0.001",
                "--weight-decay",
                "0",
                "--seed",
                str(run.training.seed),
            ]
        run_command(run, "train", args)
        checkpoint = run.checkpoint
        payload = inspect_checkpoint(run, checkpoint)
        if payload["completed_epochs"] != completed + additional:
            raise ValueError("Trainer did not complete the requested number of epochs")
        completed = payload["completed_epochs"]
    # A completed external checkpoint needs no model update; retain it with its sidecars.
    if checkpoint != run.checkpoint:
        run.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, run.checkpoint)
    payload = inspect_checkpoint(run, run.checkpoint)
    _write_once(run.checkpoint.parent / "manifest.json", payload["manifest_content"])
    # A crash after checkpoint publication can leave this derived sidecar stale.
    # The embedded history is authoritative even when no further epochs remain.
    (run.checkpoint.parent / "metrics.json").write_text(_json(payload["history"]), encoding="utf-8")
    if not (run.checkpoint.parent / "settings.json").exists():
        _write_once(
            run.checkpoint.parent / "settings.json",
            _json(
                {
                    "model": payload["config"],
                    "training": payload["training_config"],
                    "manifest_sha256": payload["manifest_sha256"],
                    "tokenizer": payload["tokenizer"],
                    "runtime": payload["runtime"],
                    "source": "reconstructed from completed checkpoint",
                }
            ),
        )
    _write_once(
        run.root / "provenance" / "final_checkpoint.sha256", file_hash(run.checkpoint) + "\n"
    )
    print(f"Training complete: epoch {completed}, updates {payload['completed_steps']}")
    return run.checkpoint


def evaluate_run(run: NotebookRun) -> dict:
    payload = inspect_checkpoint(run, run.checkpoint)
    if payload["completed_epochs"] != run.epochs:
        raise ValueError("Complete the fixed training budget before evaluating controls")
    report_path = run.root / "validation" / "controls.json"
    checkpoint_hash = file_hash(run.checkpoint)
    if not report_path.exists():
        run_command(
            run,
            "evaluate",
            [
                "scripts/evaluate_relational.py",
                "--checkpoint",
                str(run.checkpoint),
                "--split",
                "validation",
                "--batch-size",
                "32",
                "--shuffle-seeds",
                *map(str, run.seeds),
                "--device",
                "cuda:0",
                "--output-dir",
                str(report_path.parent),
            ],
        )
    if file_hash(run.checkpoint) != checkpoint_hash:
        raise ValueError("Evaluation changed the selected checkpoint")
    report = json.loads(report_path.read_text())
    validate_control_report(
        report,
        checkpoint_hash=checkpoint_hash,
        manifest_hash=payload["manifest_sha256"],
        model=run.model,
        training=asdict(run.training),
        epochs=run.epochs,
        steps=payload["completed_steps"],
        images=144 * run.corpus.validation_geometry_count,
        seeds=run.seeds,
    )
    if run.smoke:
        print("CUDA smoke check passed: training, resume, and validation controls completed.")
    else:
        assessment = assess_validation_gates(report["results"])
        summary = {
            "protocol": "milestone2_baseline_v1",
            "checkpoint_sha256": checkpoint_hash,
            "controls_sha256": file_hash(report_path),
            **assessment,
        }
        _write_once(run.root / "validation" / "acceptance.json", _json(summary))
        print(f"{'Gate':<44} {'Value':>9} {'Required':>9} {'Result':>8}")
        for gate in assessment["gates"]:
            print(
                f"{gate['name']:<44} {gate['value']:>9.2%} {gate['threshold']:>9.2%} "
                f"{'PASS' if gate['passed'] else 'FAIL':>8}"
            )
        print(
            "All validation gates passed; frozen test evaluation remains a separate step."
            if assessment["passed"]
            else "Validation gates not met. Preserve this baseline and diagnose before another run."
        )
    correct = report["results"]["correct"]
    print(
        _json(
            {
                "correct_input_diagnostics": correct,
                "accuracy_gaps": report["results"]["accuracy_gaps"],
            }
        )
    )
    return report


def archive_run(run: NotebookRun) -> Path:
    """Keep the archive outside its input tree; retain the original files too."""
    if not (run.root / "validation" / "controls.json").exists():
        raise ValueError("Evaluate validation controls before creating the completed-run archive")
    destination = shutil.make_archive(
        str(run.root.parent / f"{run.root.name}_artifacts"),
        "zip",
        root_dir=run.root.parent,
        base_dir=run.root.name,
    )
    return Path(destination)
