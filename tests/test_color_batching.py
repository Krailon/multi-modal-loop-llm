"""Task vocabulary, metadata isolation, and end-to-end answer alignment."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from multimodal_loop.data.collator import ColorQuestionBatch, SyntheticColorCollator
from multimodal_loop.data.synthetic_shapes import COLORS, QUESTION, ShapeScene, make_example
from multimodal_loop.data.text import ColorQuestionTokenizer
from multimodal_loop.model.attention import build_prefix_mask
from multimodal_loop.model.config import ModelConfig
from multimodal_loop.model.model import MultimodalLoopTransformer
from multimodal_loop.train.losses import shifted_cross_entropy
from multimodal_loop.train.trainer import train_on_batch


@pytest.fixture
def config():
    return ModelConfig(
        vocab_size=10, image_size=8, patch_size=4, d_model=8, n_heads=2, d_ff=16, max_seq_len=11
    )


@pytest.fixture
def examples():
    return [make_example(ShapeScene("square", color, 1, 1, 4), image_size=8) for color in COLORS]


def test_exact_vocabulary_and_round_trips():
    tokenizer = ColorQuestionTokenizer()
    assert tokenizer.version == 1
    assert tokenizer.vocabulary == (
        "What",
        "color",
        "is",
        "the",
        "object",
        "?",
        "red",
        "green",
        "blue",
        "yellow",
    )
    assert tokenizer.vocab_size == 10 and tokenizer.question_length == 6
    assert tokenizer.encode_question(QUESTION) == (0, 1, 2, 3, 4, 5)
    for answers in (COLORS, tuple(reversed(COLORS))):
        for answer in answers:
            expected = {"red": 6, "green": 7, "blue": 8, "yellow": 9}[answer]
            assert tokenizer.encode_answer(answer) == expected
            assert tokenizer.decode_answer(expected) == answer


@pytest.mark.parametrize("question", ["what color is the object?", QUESTION + " ", "red", ""])
def test_unsupported_question(question):
    with pytest.raises(ValueError, match="supported question"):
        ColorQuestionTokenizer().encode_question(question)


@pytest.mark.parametrize("answer", ["RED", "red ", "blue square", "", "purple"])
def test_unsupported_answer(answer):
    with pytest.raises(ValueError, match="answer must be"):
        ColorQuestionTokenizer().encode_answer(answer)


@pytest.mark.parametrize("token", [-1, 0, 5, 10, 100])
def test_nonanswer_id_rejected(token):
    with pytest.raises(ValueError, match="answer token ID"):
        ColorQuestionTokenizer().decode_answer(token)


@pytest.mark.parametrize(
    "method, value",
    [
        ("encode_question", None),
        ("encode_answer", 6),
        ("decode_answer", True),
        ("decode_answer", 6.0),
    ],
)
def test_tokenizer_type_errors(method, value):
    with pytest.raises(TypeError):
        getattr(ColorQuestionTokenizer(), method)(value)


def test_batch_order_storage_masks_and_model_keys(config, examples):
    original = [e.image.clone() for e in examples]
    batch = SyntheticColorCollator(config)(examples)
    assert batch.input_ids.tolist() == [[0, 1, 2, 3, 4, 5, answer] for answer in (6, 7, 8, 9)]
    assert batch.input_ids.dtype == torch.int64
    assert batch.images.shape == (4, 3, 8, 8) and batch.images.dtype == torch.float32
    assert batch.target_mask.dtype == batch.attention_mask.dtype == torch.bool
    assert batch.target_mask.tolist() == [[False] * 6 + [True]] * 4
    assert batch.question_length == 6 and batch.num_image_tokens == 4
    assert batch.attention_mask.shape == (11, 11)
    assert batch.attention_mask[:10, :10].all()
    assert not batch.attention_mask[:10, 10].any()
    assert batch.attention_mask[10].all()
    assert set(batch.model_inputs()) == {"input_ids", "images", "attention_mask"}
    for actual, expected in zip(batch.images, original, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    batch.images.zero_()
    for example, expected in zip(examples, original, strict=True):
        torch.testing.assert_close(example.image, expected, rtol=0, atol=0)


def test_batch_transfer_preserves_all_fields(config, examples):
    original = SyntheticColorCollator(config)(examples)
    same_device = original.to("cpu")
    assert same_device is not original
    moved = original.to("meta")
    for name in ("images", "input_ids", "target_mask", "attention_mask"):
        before, after = getattr(original, name), getattr(moved, name)
        assert before.device.type == "cpu" and after.device.type == "meta"
        assert before.shape == after.shape and before.dtype == after.dtype
        torch.testing.assert_close(getattr(same_device, name), before, rtol=0, atol=0)
    assert moved.question_length == 6 and moved.num_image_tokens == 4


def test_metadata_is_never_read_or_returned(config, examples):
    collate = SyntheticColorCollator(config)
    original = collate(examples)
    altered = collate(
        [replace(e, scene=ShapeScene("triangle", "yellow", 2, 2, 4)) for e in examples]
    )
    # Objects without a scene field establish that metadata is not required even
    # for validation. Only the three explicit data/supervision fields are read.
    bare = collate(
        [SimpleNamespace(image=e.image, question=e.question, answer=e.answer) for e in examples]
    )
    assert {f.name for f in fields(ColorQuestionBatch)} == {
        "images",
        "input_ids",
        "target_mask",
        "attention_mask",
        "question_length",
        "num_image_tokens",
    }
    for other in (altered, bare):
        for name in ("images", "input_ids", "target_mask", "attention_mask"):
            torch.testing.assert_close(
                getattr(other, name), getattr(original, name), rtol=0, atol=0
            )


def test_only_explicit_answers_determine_target_ids(config, examples):
    collate = SyntheticColorCollator(config)
    original = collate(examples)
    changed = collate([replace(e, answer=COLORS[(i + 1) % 4]) for i, e in enumerate(examples)])
    assert changed.input_ids[:, -1].tolist() == [7, 8, 9, 6]
    torch.testing.assert_close(changed.input_ids[:, :6], original.input_ids[:, :6], rtol=0, atol=0)
    for name in ("images", "attention_mask", "target_mask"):
        torch.testing.assert_close(getattr(changed, name), getattr(original, name), rtol=0, atol=0)


def test_masked_loss_selects_only_final_question_logit(config, examples):
    batch = SyntheticColorCollator(config)(examples)
    logits = (
        torch.arange(440, dtype=torch.float64).remainder(17).reshape(4, 11, 10) / 7
    ).requires_grad_()
    actual = shifted_cross_entropy(
        logits, batch.input_ids, num_image_tokens=4, target_mask=batch.target_mask
    )
    expected = F.cross_entropy(logits[:, 9], torch.tensor([6, 7, 8, 9]))
    # Masked and compact CE reductions may differ at float64 rounding precision.
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual.backward()
    torch.testing.assert_close(logits.grad, expected_grad, rtol=1e-14, atol=1e-14)
    assert logits.grad[:, 9].abs().sum() > 0
    assert logits.grad[:, :9].count_nonzero() == 0
    assert logits.grad[:, 10:].count_nonzero() == 0


@pytest.mark.parametrize("depth", [1, 3])
def test_real_model_answer_isolation_and_question_only_equivalence(config, examples, depth):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = MultimodalLoopTransformer(config).eval()
    batch = SyntheticColorCollator(config)(examples)
    batch.images.requires_grad_()
    states = []

    def retain(module, args, output):
        output.retain_grad()
        states.append(output)

    handle = model.embeddings.register_forward_hook(retain)
    try:
        logits = model(**batch.model_inputs(), recurrence_depth=depth)
    finally:
        handle.remove()
    changed_ids = batch.input_ids.clone()
    changed_ids[:, -1] = (changed_ids[:, -1] - 6 + 1) % 4 + 6
    changed = model(
        changed_ids, batch.images, attention_mask=batch.attention_mask, recurrence_depth=depth
    )
    torch.testing.assert_close(changed[:, :10], logits[:, :10], rtol=0, atol=0)
    assert not torch.allclose(changed[:, 10], logits[:, 10])
    question_only = model(
        batch.input_ids[:, :6],
        batch.images,
        attention_mask=build_prefix_mask(10, 10),
        recurrence_depth=depth,
    )
    torch.testing.assert_close(question_only[:, -1], logits[:, 9], rtol=1e-5, atol=1e-6)
    loss = shifted_cross_entropy(
        logits, batch.input_ids, num_image_tokens=4, target_mask=batch.target_mask
    )
    loss.backward()
    assert states[0].grad[:, -1].count_nonzero() == 0
    assert states[0].grad[:, :10].abs().sum() > 0
    assert batch.images.grad is not None and batch.images.grad.abs().sum() > 0
    for component in (
        model.embeddings.text_embedding,
        model.embeddings.image_embedding,
        model.core,
    ):
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters()
        )
        assert sum(p.grad.abs().sum() for p in component.parameters()) > 0


def test_batch_works_with_existing_trainer(config, examples):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        model = MultimodalLoopTransformer(config)
    batch = SyntheticColorCollator(config)(examples)
    expected = shifted_cross_entropy(
        model(**batch.model_inputs()),
        batch.input_ids,
        num_image_tokens=4,
        target_mask=batch.target_mask,
    ).item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = train_on_batch(
        model,
        optimizer,
        batch.input_ids,
        batch.images,
        steps=1,
        question_length=batch.question_length,
    )
    assert losses == [expected]
    assert optimizer.state[model.lm_head.weight]["step"] == 1


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"num_channels": 1}, "three"),
        ({"vocab_size": 9}, "vocab_size"),
        ({"max_seq_len": 10}, "max_seq_len"),
    ],
)
def test_config_must_fit_task(config, changes, message):
    with pytest.raises(ValueError, match=message):
        SyntheticColorCollator(replace(config, **changes))


@pytest.mark.parametrize(
    "kind, error",
    [
        ("empty", ValueError),
        ("shape", ValueError),
        ("device", ValueError),
        ("dtype", TypeError),
        ("nan", ValueError),
        ("high", ValueError),
        ("low", ValueError),
        ("not_tensor", TypeError),
        ("question", ValueError),
        ("answer", ValueError),
    ],
)
def test_invalid_examples(config, examples, kind, error):
    example = examples[0]
    if kind == "shape":
        example = replace(example, image=torch.zeros(3, 7, 8))
    elif kind == "device":
        example = replace(example, image=example.image.to("meta"))
    elif kind == "dtype":
        example = replace(example, image=example.image.double())
    elif kind in ("nan", "high", "low"):
        image = example.image.clone()
        image[0, 0, 0] = {"nan": float("nan"), "high": 1.01, "low": -0.01}[kind]
        example = replace(example, image=image)
    elif kind == "not_tensor":
        example = replace(example, image=None)
    elif kind == "question":
        example = replace(example, question="What shape is the object?")
    elif kind == "answer":
        example = replace(example, answer="purple")
    with pytest.raises(error):
        SyntheticColorCollator(config)([] if kind == "empty" else [example])
