"""Fixed relational vocabulary, metadata isolation and end-to-end answer alignment."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from multimodal_loop.data.collator import RelationalColorCollator
from multimodal_loop.data.relational_shapes import (
    MultiObjectScene,
    RelationalColorQuestion,
    make_relational_example,
)
from multimodal_loop.data.synthetic_shapes import COLORS, ShapeScene
from multimodal_loop.data.text import ColorQuestionTokenizer, RelationalColorTokenizer
from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.losses import shifted_cross_entropy


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=17, image_size=16, patch_size=4, max_seq_len=28, d_model=8, n_heads=2, d_ff=16
    )


@pytest.fixture
def examples():
    scene = MultiObjectScene(
        (
            ShapeScene("circle", "green", 1, 6, 4),
            ShapeScene("square", "red", 6, 6, 4),
            ShapeScene("triangle", "blue", 11, 6, 4),
        ),
        image_size=16,
    )
    return [
        make_relational_example(scene, RelationalColorQuestion("square", direction))
        for direction in ("left", "right")
    ]


def test_complete_vocabulary_exact_encodings_and_color_roundtrips():
    tokenizer = RelationalColorTokenizer()
    assert tokenizer.version == 1 and tokenizer.question_length == 11 and tokenizer.vocab_size == 17
    assert tokenizer.vocabulary == ColorQuestionTokenizer.vocabulary + (
        "immediately",
        "left",
        "right",
        "of",
        "square",
        "circle",
        "triangle",
    )
    for (shape, shape_id), (direction, direction_id) in product(
        zip(("square", "circle", "triangle"), (14, 15, 16), strict=True),
        zip(("left", "right"), (11, 12), strict=True),
    ):
        text = f"What color is the object immediately {direction} of the {shape}?"
        assert tokenizer.encode_question(text) == (
            0,
            1,
            2,
            3,
            4,
            10,
            direction_id,
            13,
            3,
            shape_id,
            5,
        )
    for token, color in enumerate(COLORS, 6):
        assert tokenizer.encode_answer(color) == token
        assert tokenizer.decode_answer(token) == color


@pytest.mark.parametrize(
    "question",
    [
        "What color is the object?",
        "What color is the object immediately left of the square",
        "What color is the object immediately above of the square?",
        "what color is the object immediately left of the square?",
        "What color is the object immediately left of the square? ",
        None,
    ],
)
def test_invalid_question(question):
    with pytest.raises((TypeError, ValueError)):
        RelationalColorTokenizer().encode_question(question)


@pytest.mark.parametrize("value", [-1, 0, 5, 10, 11, 16, 17, True, 6.0, "6"])
def test_noncolor_predictions_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        RelationalColorTokenizer().decode_answer(value)


@pytest.mark.parametrize("answer", ["Red", "purple", " red", 6, None])
def test_invalid_answer(answer):
    with pytest.raises((TypeError, ValueError)):
        RelationalColorTokenizer().encode_answer(answer)


def test_batch_metadata_exclusion_exact_masks_and_transfer(config, examples):
    collate = RelationalColorCollator(config)
    bare = [SimpleNamespace(image=e.image, question=e.question, answer=e.answer) for e in examples]
    batch = collate(bare)
    assert batch.input_ids.tolist() == [
        [0, 1, 2, 3, 4, 10, 11, 13, 3, 14, 5, 7],
        [0, 1, 2, 3, 4, 10, 12, 13, 3, 14, 5, 8],
    ]
    assert batch.input_ids.dtype == torch.int64 and batch.images.shape == (2, 3, 16, 16)
    assert batch.images.dtype == torch.float32
    assert batch.question_length == 11 and batch.num_image_tokens == 16
    assert batch.target_mask.dtype == torch.bool and batch.target_mask.shape == (2, 12)
    assert batch.target_mask[:, -1].all() and not batch.target_mask[:, :-1].any()
    assert batch.attention_mask.shape == (28, 28) and batch.attention_mask.dtype == torch.bool
    assert batch.attention_mask[:27, :27].all() and not batch.attention_mask[:27, 27].any()
    assert batch.attention_mask[27].all()
    assert set(vars(batch)) == {
        "images",
        "input_ids",
        "target_mask",
        "attention_mask",
        "question_length",
        "num_image_tokens",
    }
    assert set(batch.model_inputs()) == {"input_ids", "images", "attention_mask"}
    cpu = batch.to("cpu")
    assert cpu is not batch
    meta = batch.to("meta")
    for name in ("images", "input_ids", "target_mask", "attention_mask"):
        assert torch.equal(getattr(cpu, name), getattr(batch, name))
        assert getattr(meta, name).device.type == "meta"
    changed = collate(
        [replace(e, image=torch.zeros_like(e.image), answer="yellow") for e in examples]
    )
    assert torch.equal(changed.input_ids[:, :-1], batch.input_ids[:, :-1])
    assert changed.input_ids[:, -1].tolist() == [9, 9]
    assert not changed.images.any()
    assert examples[0].image.any()
    batch.images.zero_()
    assert examples[0].image.any()


@pytest.mark.parametrize("changes", [{"vocab_size": 16}, {"num_channels": 1}, {"max_seq_len": 27}])
def test_invalid_config(config, changes):
    with pytest.raises(ValueError):
        RelationalColorCollator(replace(config, **changes))


@pytest.mark.parametrize(
    "damage", ["empty", "shape", "dtype", "device", "nan", "negative", "large"]
)
def test_invalid_examples(config, examples, damage):
    image = examples[0].image.clone()
    if damage == "shape":
        image = image[:, :8]
    elif damage == "dtype":
        image = image.double()
    elif damage == "device":
        image = image.to("meta")
    elif damage == "nan":
        image[0, 0, 0] = float("nan")
    elif damage == "negative":
        image[0, 0, 0] = -1
    elif damage == "large":
        image[0, 0, 0] = 2
    with pytest.raises((TypeError, ValueError)):
        RelationalColorCollator(config)(
            [] if damage == "empty" else [replace(examples[0], image=image)]
        )


def test_shifted_loss_matches_final_question_logit_and_gradient(config, examples):
    batch = RelationalColorCollator(config)(examples)
    logits = (
        torch.arange(2 * 28 * 17, dtype=torch.float64).reshape(2, 28, 17).remainder(19) / 7
    ).requires_grad_()
    loss = shifted_cross_entropy(
        logits, batch.input_ids, num_image_tokens=16, target_mask=batch.target_mask
    )
    expected = F.cross_entropy(logits[:, 26], torch.tensor([7, 8]))
    torch.testing.assert_close(loss, expected)
    gradients = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    loss.backward()
    torch.testing.assert_close(logits.grad, gradients)
    assert logits.grad[:, 26].abs().sum() > 0
    assert not logits.grad[:, :26].any() and not logits.grad[:, 27].any()


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_real_model_forward_backward_question_only_and_answer_isolation(config, examples, depth):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = MultimodalLoopTransformer(config).eval()
    batch = RelationalColorCollator(config)(examples)
    batch.images.requires_grad_()
    logits = model(**batch.model_inputs(), recurrence_depth=depth)
    changed_ids = batch.input_ids.clone()
    changed_ids[:, -1] = 9
    changed = model(
        changed_ids, batch.images, attention_mask=batch.attention_mask, recurrence_depth=depth
    )
    torch.testing.assert_close(logits[:, :27], changed[:, :27], rtol=0, atol=0)
    assert not torch.allclose(logits[:, 27], changed[:, 27])
    question_only = model(
        batch.input_ids[:, :11],
        batch.images,
        attention_mask=build_prefix_mask(27, 27),
        recurrence_depth=depth,
    )
    torch.testing.assert_close(question_only[:, -1], logits[:, 26], rtol=1e-5, atol=1e-6)
    loss = shifted_cross_entropy(
        logits, batch.input_ids, num_image_tokens=16, target_mask=batch.target_mask
    )
    loss.backward()
    assert batch.images.grad is not None and batch.images.grad.abs().sum() > 0
    for component in (
        model.embeddings.image_embedding,
        model.embeddings.text_embedding,
        model.core,
    ):
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters()
        )
        assert sum(p.grad.abs().sum() for p in component.parameters()) > 0
