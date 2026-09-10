"""Matched size variants, frozen CPU prediction and Kaggle artifact contracts."""

import copy
import json
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import parse_relational_manifest
from multimodal_loop.data.relational_shapes import MultiObjectScene, render_multi_object_scene
from multimodal_loop.data.shape_grounding import ShapeColorCollator, ShapeColorTokenizer
from multimodal_loop.data.size_intervention import (
    CONDITIONS,
    DiagnosticScene,
    InterventionCase,
    SizeInterventionDataset,
    build_size_cases,
    render_diagnostic_scene,
    variant_scene,
)
from multimodal_loop.data.synthetic_shapes import ShapeScene
from multimodal_loop.eval.size_intervention import (
    compare_original,
    evaluate_cases,
    summarize_intervention,
)
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scene():
    return MultiObjectScene(
        (
            ShapeScene("square", "red", 4, 23, 8),
            ShapeScene("circle", "blue", 14, 24, 6),
            ShapeScene("triangle", "green", 23, 24, 6),
        )
    )


def cases_for(scene):
    return [
        InterventionCase(0, "validation:g003", name, variant_scene(scene, sizes))
        for name, sizes in CONDITIONS.items()
    ]


def test_validity_pixels_and_preserved_fields(scene):
    assert torch.equal(
        render_diagnostic_scene(variant_scene(scene)), render_multi_object_scene(scene)
    )
    original = render_multi_object_scene(scene)
    for sizes in CONDITIONS.values():
        variant = variant_scene(scene, sizes)
        assert variant.objects[2] == scene.objects[2]
        assert [(o.left, o.top, o.color, o.shape) for o in variant.objects] == [
            (o.left, o.top, o.color, o.shape) for o in scene.objects
        ]
        pixels = render_diagnostic_scene(variant)
        assert pixels.shape == (3, 32, 32) and pixels.device.type == "cpu"
        assert torch.equal(pixels[:, :, 23:], original[:, :, 23:])
    edge = variant_scene(scene, (8, 8))
    assert edge.objects[1].top + edge.objects[1].size == 32
    assert (
        render_diagnostic_scene(edge)[:, :, -1].sum() == 0
    )  # no right-edge object in this fixture
    assert render_diagnostic_scene(edge)[:, 31, :].sum() > 0  # fully visible circle bottom row
    with pytest.raises(ValueError, match="outside_canvas"):
        DiagnosticScene((replace(scene.objects[0], left=26), *scene.objects[1:]))
    with pytest.raises(ValueError, match="horizontal_gap"):
        DiagnosticScene((replace(scene.objects[0], left=7), *scene.objects[1:]))
    with pytest.raises(ValueError):
        MultiObjectScene(edge.objects)  # Original rules were not weakened.


def test_full_eligibility():
    config = RelationalCorpusConfig()
    manifest = parse_relational_manifest(
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
    cases, eligibility = build_size_cases(manifest)
    assert eligibility["source_images"] == 576 and eligibility["eligible_images"] == 384
    assert eligibility["eligible_by_geometry"] == {
        "validation:g000": 48,
        "validation:g001": 144,
        "validation:g002": 48,
        "validation:g003": 144,
    }
    assert len(cases) == 1920
    assert sum(not r["eligible"] for r in eligibility["records"]) == 192
    assert all(r["failures"] for r in eligibility["records"] if not r["eligible"])
    assert all(
        len([c for c in cases if c.source_index == index]) == 5
        for index in {c.source_index for c in cases}
    )


def test_batching_ignores_metadata(scene):
    collator = ShapeColorCollator(ModelConfig(vocab_size=17))
    for condition in CONDITIONS:
        data = SizeInterventionDataset(cases_for(scene), condition)
        example = data[0]

        class InputOnly:
            image = example.image
            question = example.question
            answer = example.answer

            @property
            def scene(self):
                raise AssertionError("metadata read")

        batch = collator([InputOnly()])
        tokenizer = ShapeColorTokenizer()
        assert batch.input_ids[0].tolist() == list(tokenizer.encode_question(example.question)) + [
            tokenizer.encode_answer("red")
        ]


class Oracle(torch.nn.Module):
    def __init__(self, cases):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.config = ModelConfig(vocab_size=17)
        tokenizer = ShapeColorTokenizer()
        self.lookup = {}
        for name in CONDITIONS:
            data = SizeInterventionDataset(cases, name)
            for i in range(len(data)):
                e = data[i]
                self.lookup[e.image.numpy().tobytes(), tokenizer.encode_question(e.question)] = (
                    tokenizer.encode_answer(e.answer)
                )

    def forward(self, tokens, images, *, recurrence_depth, attention_mask):
        assert tokens.shape[1] == 6 and recurrence_depth == 2
        assert attention_mask.all() and not torch.is_grad_enabled()
        out = torch.zeros(len(tokens), 1, 17)
        for i in range(len(tokens)):
            target = self.lookup[images[i].numpy().tobytes(), tuple(tokens[i].tolist())]
            out[i, 0, target] = 10
        return out


def test_perfect_oracle_and_invariance_vs_stable_wrong(scene):
    cases = cases_for(scene)
    model = Oracle(cases)
    report, rows = evaluate_cases(model, cases, batch_size=2)
    assert len(rows) == 15 and report["both_circle_square_correct_all_four"]["fraction"] == 1
    assert all(m["accuracy"] == 1 for m in report["conditions"].values())
    assert report["invariance"]["triangle"]["changed_prediction"] == 0
    wrong = copy.deepcopy(rows)
    for row in wrong:
        if row["shape"] == "circle":
            row.update(correct=False, prediction_id=6, prediction_kind="other_circle_square")
    summary = summarize_intervention(wrong)
    assert summary["invariance"]["circle"]["invariant_incorrect"] == 1
    assert summary["both_circle_square_correct_all_four"]["correct"] == 0
    next(r for r in wrong if r["shape"] == "circle" and r["condition"] == "circle8_square6").update(
        prediction_id=9, correct=True, prediction_kind="correct"
    )
    summary = summarize_intervention(wrong)
    assert summary["size_reversal"]["circle"]["prediction_changes"] == 1
    assert summary["size_reversal"]["circle"]["category_transitions"] == {
        "other_circle_square -> correct": 1
    }
    with pytest.raises(ValueError, match="identical"):
        summarize_intervention(rows[:-1])
    reference = [
        dict(r, image_index=r["source_index"]) for r in rows if r["condition"] == "original"
    ]
    assert compare_original(rows, reference)["disagreements"] == 0
    reference[0]["prediction_id"] += 1
    assert compare_original(rows, reference)["disagreements"] == 1


def test_real_cpu_forward_preserves_parameters_and_rng(scene):
    model = MultimodalLoopTransformer(ModelConfig(vocab_size=17, d_model=16, n_heads=2, d_ff=32))
    parameters = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    report, rows = evaluate_cases(model, cases_for(scene), batch_size=2)
    assert len(rows) == 15 and report["conditions"]["original"]["questions"] == 3
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, parameters[k]) for k, v in model.state_dict().items())
    assert model.training and all(p.grad is None for p in model.parameters())


def test_notebook_and_reference_rejection(tmp_path):
    from multimodal_loop.eval.size_reference import audit_size_reference, stage_size_reference

    notebook = json.loads(
        (ROOT / "notebooks/kaggle_milestone2_size_intervention.ipynb").read_text()
    )
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "<notebook>", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
            code.append(source)
    assert "run_size_intervention(run)" in "\n".join(code)
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "last.pt").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="identity"):
        audit_size_reference(tmp_path)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape", b"bad")
    with pytest.raises(ValueError):
        stage_size_reference(archive, tmp_path / "staging")
    assert not (tmp_path / "staging").exists()


@pytest.mark.parametrize("fault", [None, "count", "checkpoint"])
def test_kaggle_evaluation_only_and_archive(tmp_path, monkeypatch, scene, fault):
    from multimodal_loop.eval import kaggle_size_intervention as module

    repo, root, reference = (tmp_path / n for n in ("repo", "run", "reference"))
    repo.mkdir()
    (reference / "data").mkdir(parents=True)
    (reference / "training").mkdir()
    (reference / "data" / "manifest.json").write_text("{}")
    checkpoint = reference / "training" / "last.pt"
    checkpoint.write_bytes(b"frozen")
    eligibility = {"eligible_images": 384, "eligible_by_geometry": module.EXPECTED_GEOMETRIES}
    monkeypatch.setattr(module, "repository_revision", lambda _: "commit")
    monkeypatch.setattr(module, "cuda_runtime", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(module, "stage_size_reference", lambda *args: reference)
    monkeypatch.setattr(
        module,
        "audit_size_reference",
        lambda _: {"manifest": None, "summary": {}, "settings": {"model": {}}},
    )
    monkeypatch.setattr(module, "checked_cases", lambda _: (cases_for(scene), eligibility))
    run = module.prepare_size_intervention(repo, root, reference)
    protocol = json.loads((root / "provenance" / "protocol.json").read_text())
    assert protocol["qa_evaluations"] == 5760 and protocol["training"] == "none"
    calls = []

    def command(repo_arg, root_arg, name, args):
        calls.append(name)
        assert name == "evaluate" and args[0] == "scripts/evaluate_size_intervention.py"
        assert args[-2:] == ["--device", "cuda:0"] and "--max-steps" not in args
        diagnosis = root / "diagnosis"
        diagnosis.mkdir()
        module.write_json(
            diagnosis / "summary.json",
            {
                "checkpoint_sha256": module.file_hash(checkpoint),
                "manifest_sha256": module.file_hash(root / "data" / "manifest.json"),
                "qa_evaluations": 1 if fault == "count" else 5760,
                "source_completed_steps": 5760,
                "split": "validation",
                "recurrence_depth": 2,
                "batch_size": 32,
                "eligibility": eligibility,
            },
        )
        (diagnosis / "predictions.jsonl").write_text("{}\n")
        (diagnosis / "inspection.html").write_text("<html></html>")
        if fault == "checkpoint":
            checkpoint.write_bytes(b"changed")

    monkeypatch.setattr(module, "run_logged_command", command)
    with pytest.raises(ValueError, match="complete"):
        module.archive_size_intervention(run)
    if fault:
        with pytest.raises(ValueError, match="protocol"):
            module.run_size_intervention(run)
        assert not (root / "provenance" / "final.json").exists()
        return
    module.run_size_intervention(run)
    assert calls == ["evaluate"] and checkpoint.read_bytes() == b"frozen"
    archive = module.archive_size_intervention(run)
    with zipfile.ZipFile(archive) as z:
        assert "run/data/cases.json" in z.namelist()
        assert not any(p.endswith(".pt") or "/staging/" in p for p in z.namelist())
    with pytest.raises(ValueError, match="fresh"):
        module.run_size_intervention(run)
    with pytest.raises(FileExistsError):
        module.archive_size_intervention(run)
