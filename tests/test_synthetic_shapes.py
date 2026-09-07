"""Independent geometry, split integrity, reproducibility, and preview checks."""

import hashlib
import json
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from multimodal_loop.data.synthetic_shapes import (
    COLORS,
    QUESTION,
    SHAPES,
    ShapeScene,
    SyntheticShapesConfig,
    build_scene_splits,
    make_example,
    render_scene,
)


@pytest.mark.parametrize(
    "shape, rows",
    [
        ("square", ["1111", "1111", "1111", "1111"]),
        ("circle", ["0110", "1111", "1111", "0110"]),
        ("triangle", ["0000", "0110", "0110", "1111"]),
    ],
)
@pytest.mark.parametrize(
    "color, rgb",
    [
        ("red", (1, 0, 0)),
        ("green", (0, 1, 0)),
        ("blue", (0, 0, 1)),
        ("yellow", (1, 1, 0)),
    ],
)
def test_render_matches_hand_specified_pixels(shape, rows, color, rgb):
    scene = ShapeScene(shape, color, left=2, top=3, size=4)
    actual = render_scene(scene, image_size=10)
    expected = torch.zeros(3, 10, 10)
    for y, row in enumerate(rows):
        for x, occupied in enumerate(row):
            if occupied == "1":
                expected[:, 3 + y, 2 + x] = torch.tensor(rgb)
    assert actual.shape == (3, 10, 10)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert actual.min() == 0 and actual.max() == 1
    example = make_example(scene, image_size=10)
    assert example.question == "What color is the object?"
    assert example.answer == color
    assert example.scene == scene
    torch.testing.assert_close(example.image, expected, rtol=0, atol=0)


def test_render_owns_pixels_and_ignores_default_device_and_dtype():
    scene = ShapeScene("circle", "red", 1, 1, 4)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            first = make_example(scene, image_size=6)
            second = make_example(scene, image_size=6)
        assert first.image.dtype == torch.float32
        assert first.image.device.type == "cpu"
        first.image.zero_()
        assert second.image.sum() > 0
    finally:
        torch.set_default_dtype(original_dtype)
    with pytest.raises(FrozenInstanceError):
        scene.color = "blue"


def test_default_splits_balance_colors_and_hold_out_layouts():
    config = SyntheticShapesConfig()
    assert config.capacity == 13380
    splits = build_scene_splits(config)
    assert list(splits) == ["train", "validation", "test"]
    prior_layouts, prior_images = set(), set()
    for split, count in [("train", 1024), ("validation", 256), ("test", 256)]:
        scenes = splits[split]
        assert len(scenes) == count
        assert Counter(s.color for s in scenes) == {color: count // 4 for color in COLORS}
        assert {s.shape for s in scenes} == set(SHAPES)
        layouts = defaultdict(list)
        images = set()
        for scene in scenes:
            layouts[scene.layout].append(scene.color)
            image = render_scene(scene, image_size=config.image_size)
            assert image[:, 0].count_nonzero() == image[:, -1].count_nonzero() == 0
            assert image[:, :, 0].count_nonzero() == image[:, :, -1].count_nonzero() == 0
            images.add(hashlib.sha256(image.numpy().tobytes()).digest())
        assert len(images) == count
        assert all(Counter(colors) == Counter(COLORS) for colors in layouts.values())
        assert prior_layouts.isdisjoint(layouts)
        assert prior_images.isdisjoint(images)
        prior_layouts.update(layouts)
        prior_images.update(images)


def test_small_catalog_can_be_exhausted_without_duplicate_pixels():
    config = SyntheticShapesConfig(
        image_size=10, object_sizes=(4, 6), train_size=168, validation_size=120, test_size=120
    )
    scenes = [scene for split in build_scene_splits(config).values() for scene in split]
    assert len(scenes) == config.capacity == 408
    fingerprints = {render_scene(s, image_size=10).numpy().tobytes() for s in scenes}
    assert len(fingerprints) == 408


def test_generation_is_repeatable_and_preserves_all_global_rngs():
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        config = SyntheticShapesConfig()
        first = build_scene_splits(config)
        for scenes in first.values():
            make_example(scenes[-1])
        assert random.getstate() == states[0]
        numpy_state = np.random.get_state()
        assert numpy_state[0] == states[1][0]
        np.testing.assert_array_equal(numpy_state[1], states[1][1])
        assert numpy_state[2:] == states[1][2:]
        torch.testing.assert_close(torch.get_rng_state(), states[2], rtol=0, atol=0)
        random.seed(876)
        np.random.seed(876)
        torch.manual_seed(876)
        assert build_scene_splits(config) == first
        assert build_scene_splits(replace(config, seed=1)) != first
        # Access order cannot change later examples or split membership.
        expected = make_example(first["train"][0]).image
        make_example(first["test"][-1])
        torch.testing.assert_close(make_example(first["train"][0]).image, expected, rtol=0, atol=0)
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"image_size": True}, TypeError),
        ({"image_size": 5}, ValueError),
        ({"seed": 0.5}, TypeError),
        ({"object_sizes": [8]}, TypeError),
        ({"object_sizes": ()}, ValueError),
        ({"object_sizes": (8, 8)}, ValueError),
        ({"object_sizes": (True,)}, TypeError),
        ({"object_sizes": (2,)}, ValueError),
        ({"object_sizes": (7,)}, ValueError),
        ({"object_sizes": (32,)}, ValueError),
        ({"train_size": 0}, ValueError),
        ({"validation_size": 5}, ValueError),
        ({"test_size": False}, TypeError),
        ({"test_size": 14000}, ValueError),
    ],
)
def test_invalid_configuration(changes, error):
    with pytest.raises(error):
        SyntheticShapesConfig(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"shape": "star"},
        {"color": "black"},
        {"left": 0},
        {"top": -1},
        {"size": 3},
    ],
)
def test_invalid_scene(changes):
    fields = dict(shape="square", color="red", left=1, top=1, size=4)
    fields.update(changes)
    with pytest.raises(ValueError):
        ShapeScene(**fields)


@pytest.mark.parametrize("left, top", [(3, 1), (1, 3)])
def test_out_of_bounds_render(left, top):
    with pytest.raises(ValueError, match="margin"):
        render_scene(ShapeScene("square", "blue", left, top, 4), image_size=7)


def run_generator(output, *args):
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_synthetic_data.py"
    return subprocess.run(
        [sys.executable, str(script), "--output-dir", str(output), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_manifest_and_preview_reproduce_actual_examples(tmp_path):
    args = (
        "--train-size",
        "12",
        "--validation-size",
        "12",
        "--test-size",
        "12",
        "--preview-count",
        "20",
    )
    first = run_generator(tmp_path / "first", *args)
    second = run_generator(tmp_path / "second", *args)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    for filename in ("manifest.json", "preview.html"):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
    manifest = json.loads((tmp_path / "first/manifest.json").read_text())
    assert manifest["format_version"] == 1
    config_fields = manifest["config"]
    config_fields["object_sizes"] = tuple(config_fields["object_sizes"])
    config = SyntheticShapesConfig(**config_fields)
    for split, scenes in build_scene_splits(config).items():
        assert manifest["splits"][split] == [
            {"scene": asdict(s), "question": QUESTION, "answer": s.color} for s in scenes
        ]
        assert f"{split}: 12 examples" in first.stdout
    page = (tmp_path / "first/preview.html").read_text()
    figures = re.findall(
        r'<figure data-split="(\w+)" data-index="(\d+)">(.*?)</figure>', page, re.S
    )
    assert len(figures) == 36
    for split, index, figure in figures:
        record = manifest["splits"][split][int(index)]
        assert QUESTION in figure and f"<strong>{record['answer']}</strong>" in figure
        svg = ET.fromstring(re.search(r"<svg.*?</svg>", figure, re.S)[0])
        pixels = torch.zeros(3, config.image_size, config.image_size)
        for rect in svg:
            fields = rect.attrib
            left, top = int(fields.get("x", 0)), int(fields.get("y", 0))
            width, height = int(fields["width"]), int(fields["height"])
            rgb = torch.tensor([int(fields["fill"][i : i + 2], 16) / 255 for i in (1, 3, 5)])
            pixels[:, top : top + height, left : left + width] = rgb[:, None, None]
        torch.testing.assert_close(
            pixels, render_scene(ShapeScene(**record["scene"])), rtol=0, atol=0
        )


def test_invalid_cli_does_not_write_output(tmp_path):
    output = tmp_path / "invalid"
    result = run_generator(output, "--train-size", "5")
    assert result.returncode == 2 and "divisible by four" in result.stderr
    assert not output.exists()
