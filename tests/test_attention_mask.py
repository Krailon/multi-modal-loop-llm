"""Allowed and blocked attention patterns for text and multimodal inputs."""

import pytest
import torch

from multimodal_loop.model.attention import build_causal_mask, build_prefix_mask


def test_causal_mask_exact_matrix() -> None:
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    assert torch.equal(build_causal_mask(4), expected)


def test_prefix_mask_exact_matrix() -> None:
    # Three image/question inputs followed by three answer inputs.
    expected = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
            [True, True, True, True, True, True],
        ]
    )

    assert torch.equal(build_prefix_mask(6, 3), expected)


@pytest.mark.parametrize("prefix_len", [1, 3, 5])
def test_prefix_context_and_answer_visibility(prefix_len: int) -> None:
    seq_len = 7
    mask = build_prefix_mask(seq_len, prefix_len)

    assert mask[:prefix_len, :prefix_len].all()
    assert not mask[:prefix_len, prefix_len:].any()
    assert mask[prefix_len:, :prefix_len].all()
    for query in range(prefix_len, seq_len):
        assert mask[query, : query + 1].all()
        assert not mask[query, query + 1 :].any()


@pytest.mark.parametrize("seq_len", [1, 4, 9])
def test_zero_prefix_is_causal(seq_len: int) -> None:
    assert torch.equal(build_prefix_mask(seq_len, 0), build_causal_mask(seq_len))


@pytest.mark.parametrize("seq_len", [1, 4, 9])
def test_full_prefix_allows_all_attention(seq_len: int) -> None:
    mask = build_prefix_mask(seq_len, seq_len)

    assert mask.shape == (seq_len, seq_len)
    assert mask.all()


def test_single_token_can_attend_to_itself() -> None:
    expected = torch.tensor([[True]])

    assert torch.equal(build_causal_mask(1), expected)
    assert torch.equal(build_prefix_mask(1, 0), expected)
    assert torch.equal(build_prefix_mask(1, 1), expected)


@pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu")])
def test_mask_shape_dtype_and_cpu_placement(device: torch.device | str | None) -> None:
    masks = (build_causal_mask(5, device=device), build_prefix_mask(5, 2, device=device))

    for mask in masks:
        assert mask.shape == (5, 5)
        assert mask.dtype == torch.bool
        assert mask.device == torch.device("cpu")


@pytest.mark.parametrize("seq_len", [True, False, 4.0, 1.5, "4", None])
def test_sequence_length_requires_an_integer(seq_len: object) -> None:
    with pytest.raises(TypeError, match="seq_len must be an integer"):
        build_causal_mask(seq_len)
    with pytest.raises(TypeError, match="seq_len must be an integer"):
        build_prefix_mask(seq_len, 0)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_sequence_length_must_be_positive(seq_len: int) -> None:
    with pytest.raises(ValueError, match="seq_len must be positive"):
        build_causal_mask(seq_len)
    with pytest.raises(ValueError, match="seq_len must be positive"):
        build_prefix_mask(seq_len, 0)


@pytest.mark.parametrize("prefix_len", [True, False, 2.0, 1.5, "2", None])
def test_prefix_length_requires_an_integer(prefix_len: object) -> None:
    with pytest.raises(TypeError, match="prefix_len must be an integer"):
        build_prefix_mask(4, prefix_len)


@pytest.mark.parametrize("prefix_len", [-1, 5])
def test_prefix_length_must_fit_sequence(prefix_len: int) -> None:
    with pytest.raises(ValueError, match="prefix_len must be in"):
        build_prefix_mask(4, prefix_len)
