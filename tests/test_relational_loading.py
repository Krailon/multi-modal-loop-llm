"""Relational manifest integrity, stored ordering, and question-wise datasets."""

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, asdict

import pytest
import torch

from multimodal_loop.data.relational_corpus import RelationalCorpusConfig, build_relational_splits
from multimodal_loop.data.relational_dataset import (
    RelationalColorDataset,
    load_relational_manifest,
    parse_relational_manifest,
)
from multimodal_loop.data.relational_shapes import render_multi_object_scene
from multimodal_loop.train.checkpoint import _capture_rng, _restore_rng


@pytest.fixture
def payload():
    config = RelationalCorpusConfig(
        image_size=16,
        object_sizes=(4,),
        train_geometry_count=1,
        validation_geometry_count=1,
        test_geometry_count=1,
    )
    return json.loads(
        json.dumps(
            {
                "kind": "relational_color_rows",
                "format_version": 1,
                "software": {"python": "fixture", "torch": "fixture"},
                "config": asdict(config),
                "splits": {
                    split: [asdict(record) for record in records]
                    for split, records in build_relational_splits(config).items()
                },
            }
        )
    )


def test_loading_keeps_exact_bytes_order_and_rng_without_rendering(payload, tmp_path, monkeypatch):
    payload["splits"]["train"].reverse()
    for record in payload["splits"]["train"]:
        record["questions"].reverse()
    path = tmp_path / "manifest.json"
    content = json.dumps(payload, indent=2).replace("\n", "\r\n")
    path.write_bytes(content.encode())

    def forbidden(*args, **kwargs):
        raise AssertionError("loader must neither render nor regenerate")

    monkeypatch.setattr(
        "multimodal_loop.data.relational_dataset.render_multi_object_scene", forbidden
    )
    monkeypatch.setattr("multimodal_loop.data.relational_shapes.render_scene", forbidden)
    monkeypatch.setattr("multimodal_loop.data.relational_corpus.build_relational_splits", forbidden)
    state = _capture_rng()
    manifest = load_relational_manifest(path)
    after = _capture_rng()
    assert torch.equal(state.pop("torch_cpu"), after.pop("torch_cpu"))
    assert state == after
    assert manifest.content == content
    assert manifest.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    for split, records in manifest.splits.items():
        assert json.loads(json.dumps([asdict(r) for r in records])) == payload["splits"][split]
    with pytest.raises(TypeError):
        manifest.splits["train"] = ()
    with pytest.raises(FrozenInstanceError):
        manifest.content = "changed"


@pytest.mark.parametrize(
    "damage",
    [
        "kind",
        "version",
        "bool_version",
        "missing_config",
        "bad_count",
        "software",
        "missing_split",
        "scene_size",
        "spatial_order",
        "object_size",
        "bounds",
        "duplicate_shapes",
        "duplicate_colors",
        "query_count",
        "duplicate_query",
        "missing_neighbor",
        "wrong_text",
        "wrong_answer",
        "missing_query_field",
        "duplicate_variant",
        "incomplete_expansion",
        "overlap",
    ],
)
def test_corrupt_manifest_rejected(payload, damage):
    record = payload["splits"]["train"][0]
    objects = record["scene"]["objects"]
    qa = record["questions"][0]
    if damage == "kind":
        payload["kind"] = "single_color"
    elif damage == "version":
        payload["format_version"] = 2
    elif damage == "bool_version":
        payload["format_version"] = True
    elif damage == "missing_config":
        del payload["config"]["seed"]
    elif damage == "bad_count":
        payload["splits"]["train"].pop()
    elif damage == "software":
        payload["software"] = []
    elif damage == "missing_split":
        del payload["splits"]["test"]
    elif damage == "scene_size":
        record["scene"]["image_size"] = 32
    elif damage == "spatial_order":
        objects.reverse()
    elif damage == "object_size":
        payload["config"]["image_size"] = 32
        payload["config"]["object_sizes"] = [6]
        for records in payload["splits"].values():
            for item in records:
                item["scene"]["image_size"] = 32
    elif damage == "bounds":
        objects[2]["left"] = 12
    elif damage == "duplicate_shapes":
        objects[1]["shape"] = objects[0]["shape"]
    elif damage == "duplicate_colors":
        objects[1]["color"] = objects[0]["color"]
    elif damage == "query_count":
        record["questions"].pop()
    elif damage == "duplicate_query":
        record["questions"][1] = copy.deepcopy(qa)
    elif damage == "missing_neighbor":
        qa["query"] = {"anchor_shape": objects[0]["shape"], "direction": "left"}
    elif damage == "wrong_text":
        qa["question"] += " "
    elif damage == "wrong_answer":
        qa["answer"] = objects[0]["color"]
    elif damage == "missing_query_field":
        del qa["query"]["direction"]
    elif damage == "duplicate_variant":
        payload["splits"]["train"][1] = copy.deepcopy(record)
    elif damage == "incomplete_expansion":
        new_top = 1 if objects[0]["top"] != 1 else 2
        for obj in objects:
            obj["top"] = new_top
    else:
        payload["splits"]["test"] = copy.deepcopy(payload["splits"]["train"])
    with pytest.raises((TypeError, ValueError)):
        parse_relational_manifest(json.dumps(payload))


def test_dataset_flattening_bounds_storage_and_explicit_supervision(payload, monkeypatch):
    manifest = parse_relational_manifest(json.dumps(payload))
    dataset = RelationalColorDataset(manifest, "train")
    assert dataset.records is manifest.splits["train"]
    assert len(dataset) == 576 and dataset.image_size == 16

    def forbidden(*args, **kwargs):
        raise AssertionError("dataset must use stored text and answers")

    monkeypatch.setattr(
        "multimodal_loop.data.relational_dataset.answer_relational_question", forbidden
    )
    state = _capture_rng()
    try:
        for index, example in enumerate(dataset):
            record = dataset.records[index // 4]
            qa = record.questions[index % 4]
            assert example.question == qa.question and example.answer == qa.answer
            assert example.scene is record.scene and example.query is qa.query
            assert torch.equal(example.image, render_multi_object_scene(record.scene))
        assert dataset[-1].question == dataset[len(dataset) - 1].question
        assert dataset[-len(dataset)].answer == dataset[0].answer
        with pytest.raises(IndexError):
            dataset[len(dataset)]
        with pytest.raises(IndexError):
            dataset[-len(dataset) - 1]
        with pytest.raises(TypeError):
            dataset[0.5]
        first, second = dataset[0], dataset[1]
        assert torch.equal(first.image, second.image)
        assert first.image.data_ptr() != second.image.data_ptr()
        first.image.zero_()
        assert second.image.any() and dataset[0].image.any()
        after = _capture_rng()
        assert torch.equal(state["torch_cpu"], after["torch_cpu"])
        assert {k: v for k, v in state.items() if k != "torch_cpu"} == {
            k: v for k, v in after.items() if k != "torch_cpu"
        }
    finally:
        _restore_rng(state, torch.device("cpu"))
    with pytest.raises(ValueError, match="split"):
        RelationalColorDataset(manifest, "unknown")
