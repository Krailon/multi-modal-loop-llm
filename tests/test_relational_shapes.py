"""Independent row geometry, spatial labels, validation, and determinism."""

import random
from dataclasses import FrozenInstanceError, replace
from itertools import permutations, product

import numpy as np
import pytest
import torch

from multimodal_loop.data.relational_shapes import (
    MultiObjectScene,
    RelationalColorQuestion,
    answer_relational_question,
    make_relational_example,
    render_multi_object_scene,
    resolve_relation,
)
from multimodal_loop.data.synthetic_shapes import COLORS, SHAPES, ShapeScene


@pytest.fixture
def scene():
    return MultiObjectScene(
        (
            ShapeScene("circle", "green", 1, 6, 4),
            ShapeScene("square", "red", 6, 6, 4),
            ShapeScene("triangle", "blue", 11, 6, 4),
        ),
        image_size=16,
    )


def test_complete_pixels_match_hand_specified_geometry(scene):
    expected = torch.zeros(3, 16, 16)
    for left, channel, rows in (
        (1, 1, ("0110", "1111", "1111", "0110")),
        (6, 0, ("1111", "1111", "1111", "1111")),
        (11, 2, ("0000", "0110", "0110", "1111")),
    ):
        for y, row in enumerate(rows):
            for x, occupied in enumerate(row):
                expected[channel, 6 + y, left + x] = int(occupied)
    actual = render_multi_object_scene(scene)
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    assert torch.isfinite(actual).all() and actual.min() == 0 and actual.max() == 1
    assert not actual[:, :, [0, 5, 10, 15]].any()
    assert not actual[:, [0, 15], :].any()


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_tuple_order_does_not_change_pixels_or_relations(scene, order):
    reordered = replace(scene, objects=tuple(scene.objects[i] for i in order))
    assert torch.equal(render_multi_object_scene(scene), render_multi_object_scene(reordered))
    for anchor, direction, index in (
        ("circle", "right", 1),
        ("square", "left", 0),
        ("square", "right", 2),
        ("triangle", "left", 1),
    ):
        query = RelationalColorQuestion(anchor, direction)
        assert resolve_relation(reordered, query) is scene.objects[index]
    assert reordered.objects == tuple(scene.objects[i] for i in order)


def test_label_oracle_for_all_shape_orders_and_color_assignments():
    # Unequal sizes/gaps: top coordinates align centers at y=10.
    for shapes in permutations(SHAPES):
        for colors in product(COLORS, repeat=3):
            objects = tuple(
                ShapeScene(shape, color, left, top, size)
                for shape, color, (left, top, size) in zip(
                    shapes, colors, ((1, 8, 4), (9, 6, 8), (20, 7, 6)), strict=True
                )
            )
            scene = MultiObjectScene(tuple(reversed(objects)))
            for anchor_index, direction, target_index in (
                (0, "right", 1),
                (1, "left", 0),
                (1, "right", 2),
                (2, "left", 1),
            ):
                query = RelationalColorQuestion(shapes[anchor_index], direction)
                assert resolve_relation(scene, query) is objects[target_index]
                assert answer_relational_question(scene, query) == colors[target_index]


def test_two_questions_same_pixels_different_answers(scene):
    left = make_relational_example(scene, RelationalColorQuestion("square", "left"))
    right = make_relational_example(scene, RelationalColorQuestion("square", "right"))
    assert left.question == "What color is the object immediately left of the square?"
    assert right.question == "What color is the object immediately right of the square?"
    assert left.answer == "green" and right.answer == "blue"
    assert torch.equal(left.image, right.image)
    assert left.image.data_ptr() != right.image.data_ptr()
    assert left.scene is scene and left.query.direction == "left"
    left.image.zero_()
    assert right.image.any() and render_multi_object_scene(scene).any()
    with pytest.raises(FrozenInstanceError):
        left.answer = "red"
    with pytest.raises(FrozenInstanceError):
        scene.image_size = 20
    with pytest.raises(FrozenInstanceError):
        left.query.direction = "right"


def test_answers_do_not_render(monkeypatch, scene):
    def fail(*args, **kwargs):
        raise AssertionError("unexpected rendering")

    monkeypatch.setattr("multimodal_loop.data.relational_shapes.render_scene", fail)
    monkeypatch.setattr("multimodal_loop.data.relational_shapes.render_multi_object_scene", fail)
    assert answer_relational_question(scene, RelationalColorQuestion("square", "left")) == "green"
    with pytest.raises(ValueError, match="no neighbor"):
        make_relational_example(scene, RelationalColorQuestion("circle", "left"))


@pytest.mark.parametrize("anchor,direction", [("circle", "left"), ("triangle", "right")])
def test_missing_neighbor(scene, anchor, direction):
    with pytest.raises(ValueError, match="no neighbor"):
        resolve_relation(scene, RelationalColorQuestion(anchor, direction))


def test_repeated_shapes_only_make_matching_anchors_ambiguous(scene):
    scene = replace(
        scene,
        objects=(scene.objects[0], scene.objects[1], replace(scene.objects[2], shape="circle")),
    )
    assert answer_relational_question(scene, RelationalColorQuestion("square", "right")) == "blue"
    with pytest.raises(ValueError, match="ambiguous anchor"):
        resolve_relation(scene, RelationalColorQuestion("circle", "right"))
    with pytest.raises(ValueError, match="missing anchor"):
        resolve_relation(scene, RelationalColorQuestion("triangle", "left"))


@pytest.mark.parametrize("value", [[], (), (None, None, None), (1, 2), (1, 2, 3, 4)])
def test_invalid_objects(value):
    with pytest.raises((TypeError, ValueError)):
        MultiObjectScene(value)


@pytest.mark.parametrize("value", [True, 16.0, "16", 0, 5, 15])
def test_invalid_canvas(scene, value):
    with pytest.raises((TypeError, ValueError)):
        replace(scene, image_size=value)


@pytest.mark.parametrize("left", [2, 4, 5])
def test_overlap_or_touching_boxes_rejected(scene, left):
    objects = (scene.objects[0], replace(scene.objects[1], left=left), scene.objects[2])
    with pytest.raises(ValueError, match="blank pixel"):
        replace(scene, objects=objects)


def test_misaligned_centers_and_bottom_margin(scene):
    with pytest.raises(ValueError, match="vertically aligned"):
        replace(scene, objects=(replace(scene.objects[0], top=5), *scene.objects[1:]))
    with pytest.raises(ValueError, match="canvas margin"):
        replace(scene, objects=tuple(replace(obj, top=12) for obj in scene.objects))


@pytest.mark.parametrize(
    "anchor,direction",
    [
        ("star", "left"),
        ("Square", "right"),
        ("square", "above"),
        ("square", "Left"),
        (None, "left"),
        ("circle", 1),
    ],
)
def test_invalid_question(anchor, direction):
    with pytest.raises((TypeError, ValueError)):
        RelationalColorQuestion(anchor, direction)


def test_yellow_and_unequal_sizes_render_as_separated_boxes():
    scene = MultiObjectScene(
        (
            ShapeScene("square", "yellow", 1, 5, 4),
            ShapeScene("square", "yellow", 6, 3, 8),
            ShapeScene("square", "yellow", 15, 4, 6),
        ),
        image_size=24,
    )
    expected = torch.zeros(3, 24, 24)
    expected[:2, 5:9, 1:5] = 1
    expected[:2, 3:11, 6:14] = 1
    expected[:2, 4:10, 15:21] = 1
    assert torch.equal(render_multi_object_scene(scene), expected)


def test_renderer_ignores_default_device_dtype_and_preserves_rng(scene):
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            example = make_relational_example(scene, RelationalColorQuestion("square", "left"))
        assert example.image.device.type == "cpu" and example.image.dtype == torch.float32
        assert torch.equal(example.image, render_multi_object_scene(scene))
        assert random.getstate() == python_state
        current = np.random.get_state()
        assert current[0] == numpy_state[0] and np.array_equal(current[1], numpy_state[1])
        assert current[2:] == numpy_state[2:]
        assert torch.equal(torch.get_rng_state(), torch_state)
    finally:
        torch.set_default_dtype(dtype)
