"""Geometry enumeration, exact balancing, split integrity and generated previews."""

import json
import os
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError, asdict, replace
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np
import pytest
import torch

from multimodal_loop.data.relational_corpus import (
    RelationalCorpusConfig,
    _geometry_catalog,
    build_relational_splits,
)
from multimodal_loop.data.relational_shapes import render_multi_object_scene
from multimodal_loop.data.synthetic_shapes import COLORS, SHAPES


def small_config(**kwargs):
    return RelationalCorpusConfig(
        **{
            "image_size": 16,
            "object_sizes": (4,),
            "train_geometry_count": 1,
            "validation_geometry_count": 1,
            "test_geometry_count": 1,
            **kwargs,
        }
    )


def geometry(record):
    return tuple((o.left, o.top, o.size) for o in record.scene.objects)


def test_catalog_known_order_and_independent_exhaustive_capacity():
    simple = small_config()
    assert simple.capacity == 11
    assert _geometry_catalog(simple) == [
        ((1, top, 4), (6, top, 4), (11, top, 4)) for top in range(1, 12)
    ]
    config = small_config(image_size=18, object_sizes=(4, 6))
    expected = set()
    for sizes in product((4, 6), repeat=3):
        for lefts in combinations(range(1, 18), 3):
            if any(left + size >= 18 for left, size in zip(lefts, sizes, strict=True)):
                continue
            if lefts[1] <= lefts[0] + sizes[0] or lefts[2] <= lefts[1] + sizes[1]:
                continue
            for center in range(1, 18):
                tops = tuple(center - size // 2 for size in sizes)
                if all(top >= 1 and top + size < 18 for top, size in zip(tops, sizes, strict=True)):
                    expected.add(tuple(zip(lefts, tops, sizes, strict=True)))
    actual = _geometry_catalog(config)
    assert len(actual) == len(set(actual)) == config.capacity == len(expected)
    assert set(actual) == expected
    assert RelationalCorpusConfig().capacity == 25136
    assert RelationalCorpusConfig(object_sizes=(8,)).capacity == 805


def test_default_counts_expansion_balance_and_question_targets():
    config = RelationalCorpusConfig()
    splits = build_relational_splits(config)
    seen = set()
    for split, expected_geometries in (("train", 16), ("validation", 4), ("test", 4)):
        records = splits[split]
        assert len(records) == expected_geometries * 144
        groups = defaultdict(list)
        for record in records:
            groups[geometry(record)].append(record)
        assert len(groups) == expected_geometries
        assert not seen.intersection(groups)
        seen.update(groups)
        counts_by_question = defaultdict(Counter)
        middle_correct = 0
        for records_for_geometry in groups.values():
            variants = set()
            for record in records_for_geometry:
                objects = record.scene.objects
                assert [o.left for o in objects] == sorted(o.left for o in objects)
                variants.add((tuple(o.shape for o in objects), tuple(o.color for o in objects)))
                assert len(record.questions) == 4
                expected = [(0, "right", 1), (1, "left", 0), (1, "right", 2), (2, "left", 1)]
                for qa, (anchor, direction, target) in zip(record.questions, expected, strict=True):
                    assert qa.query.anchor_shape == objects[anchor].shape
                    assert qa.query.direction == direction
                    assert (
                        qa.question == f"What color is the object immediately {direction} "
                        f"of the {objects[anchor].shape}?"
                    )
                    assert qa.answer == objects[target].color
                    counts_by_question[qa.question][qa.answer] += 1
                    middle_correct += qa.answer == objects[1].color
                assert sorted(Counter(qa.answer for qa in record.questions).values()) == [1, 1, 2]
            assert variants == set(product(permutations(SHAPES), permutations(COLORS, 3)))
        assert len(counts_by_question) == 6
        for counts in counts_by_question.values():
            assert counts == Counter({color: expected_geometries * 24 for color in COLORS})
        assert middle_correct / (len(records) * 4) == 0.5


def test_exhaustion_no_duplicate_images_or_global_rng_changes():
    config = small_config(test_geometry_count=9)
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    splits = build_relational_splits(config)
    assert splits == build_relational_splits(config)
    assert random.getstate() == python_state
    current = np.random.get_state()
    assert current[0] == numpy_state[0] and np.array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    expected_geometries = _geometry_catalog(config)
    random.Random(0).shuffle(expected_geometries)
    assert {geometry(record) for record in splits["train"]} == {expected_geometries[0]}
    assert {geometry(record) for record in splits["validation"]} == {expected_geometries[1]}
    images = set()
    for records in splits.values():
        for record in records:
            pixels = render_multi_object_scene(record.scene).numpy().tobytes()
            assert pixels not in images
            images.add(pixels)
    assert len(images) == 11 * 144
    assert build_relational_splits(replace(config, seed=1)) != splits
    with pytest.raises(FrozenInstanceError):
        splits["train"][0].questions = ()
    with pytest.raises(FrozenInstanceError):
        splits["train"][0].questions[0].answer = "red"


@pytest.mark.parametrize(
    "changes",
    [
        {"image_size": True},
        {"image_size": 5},
        {"image_size": 15},
        {"object_sizes": []},
        {"object_sizes": ()},
        {"object_sizes": (4, 4)},
        {"object_sizes": (3,)},
        {"object_sizes": (5,)},
        {"object_sizes": (True,)},
        {"object_sizes": (16,)},
        {"object_sizes": (6,)},
        {"seed": 0.5},
        {"seed": False},
        {"train_geometry_count": 0},
        {"validation_geometry_count": -1},
        {"test_geometry_count": True},
        {"train_geometry_count": 1.5},
        {"test_geometry_count": 10},
    ],
)
def test_invalid_config(changes):
    with pytest.raises((TypeError, ValueError)):
        small_config(**changes)


def run_cli(output, *extra):
    return subprocess.run(
        [
            sys.executable,
            "scripts/generate_relational_data.py",
            "--output-dir",
            str(output),
            "--image-size",
            "16",
            "--object-sizes",
            "4",
            "--train-geometry-count",
            "1",
            "--validation-geometry-count",
            "1",
            "--test-geometry-count",
            "1",
            *extra,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )


def test_manifest_and_preview_match_records_pixels_and_grouped_questions(tmp_path):
    result = run_cli(tmp_path, "--preview-count", "2")
    assert result.returncode == 0, result.stderr
    assert "1 geometries; 144 images; 576 QA examples" in result.stdout
    manifest_path, preview_path = tmp_path / "manifest.json", tmp_path / "preview.html"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["kind"] == "relational_color_rows" and manifest["format_version"] == 1
    assert manifest["config"] == json.loads(json.dumps(asdict(small_config())))
    assert set(manifest["software"]) == {"python", "torch"}
    splits = build_relational_splits(small_config())
    assert manifest["splits"] == json.loads(
        json.dumps({s: [asdict(r) for r in rs] for s, rs in splits.items()})
    )
    figures = re.findall(
        r'<figure data-split="(.*?)" data-index="(\d+)">(.*?)</figure>',
        preview_path.read_text(),
        re.S,
    )
    assert len(figures) == 6
    for split, index, figure in figures:
        record = splits[split][int(index)]
        for qa in record.questions:
            assert f"<li>{qa.question} <strong>{qa.answer}</strong></li>" in figure
        svg = ET.fromstring(re.search(r"<svg.*?</svg>", figure, re.S).group())
        pixels = torch.zeros(3, 16, 16)
        for rect in svg:
            color = rect.attrib["fill"]
            if color == "#000000":
                continue
            x, y, width, height = (int(rect.attrib[name]) for name in ("x", "y", "width", "height"))
            rgb = torch.tensor([int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)])
            pixels[:, y : y + height, x : x + width] = rgb[:, None, None]
        assert torch.equal(pixels, render_multi_object_scene(record.scene))
    before = (manifest_path.read_bytes(), preview_path.read_bytes())
    assert run_cli(tmp_path, "--preview-count", "2").returncode == 0
    assert before == (manifest_path.read_bytes(), preview_path.read_bytes())


@pytest.mark.parametrize(
    "extra", [("--preview-count", "0"), ("--train-geometry-count", "10"), ("--object-sizes", "5")]
)
def test_invalid_cli_does_not_write(tmp_path, extra):
    output = tmp_path / "absent"
    result = run_cli(output, *extra)
    assert result.returncode != 0
    assert not output.exists()
