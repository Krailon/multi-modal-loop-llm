"""Fixed version-1 vocabulary for the initial synthetic color task."""

from multimodal_loop.data.synthetic_shapes import COLORS, QUESTION


class ColorQuestionTokenizer:
    """Strict task tokenizer; no fitting, normalization, padding, or special IDs.

    IDs 0..5 represent What/color/is/the/object/?, and IDs 6..9 represent
    red/green/blue/yellow. Zero is an ordinary question token. The mapping is
    independent of dataset contents and ordering. Unsupported predictions must
    be counted as incorrect by evaluation rather than decoded as a color.
    """

    version = 1
    vocabulary = ("What", "color", "is", "the", "object", "?", "red", "green", "blue", "yellow")
    vocab_size = len(vocabulary)
    question_length = 6

    def encode_question(self, question: str) -> tuple[int, ...]:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if question != QUESTION:
            raise ValueError(f"supported question is {QUESTION!r}")
        return (0, 1, 2, 3, 4, 5)

    def encode_answer(self, answer: str) -> int:
        if not isinstance(answer, str):
            raise TypeError("answer must be a string")
        if answer not in COLORS:
            raise ValueError(f"answer must be one of {COLORS}")
        return self.vocabulary.index(answer)

    def decode_answer(self, token_id: int) -> str:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("answer token ID must be an integer")
        if not self.question_length <= token_id < self.vocab_size:
            raise ValueError("answer token ID must be in [6, 10)")
        return self.vocabulary[token_id]
