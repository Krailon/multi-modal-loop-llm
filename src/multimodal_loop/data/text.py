"""Fixed task vocabularies for single-object and relational color questions."""

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


class RelationalColorTokenizer:
    """Version-1 relational vocabulary; preserve all original color-task IDs.

    Only six canonical relational question strings are supported. Questions
    contain 11 tokens, including two occurrences of 'the'. Answer IDs remain
    6..9 and must not be inferred from question length or total vocabulary size.
    """

    version = 1
    vocabulary = ColorQuestionTokenizer.vocabulary + (
        "immediately",
        "left",
        "right",
        "of",
        "square",
        "circle",
        "triangle",
    )
    vocab_size = len(vocabulary)
    question_length = 11

    def encode_question(self, question: str) -> tuple[int, ...]:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        for shape in ("square", "circle", "triangle"):
            for direction in ("left", "right"):
                if question == f"What color is the object immediately {direction} of the {shape}?":
                    return (
                        0,
                        1,
                        2,
                        3,
                        4,
                        10,
                        self.vocabulary.index(direction),
                        13,
                        3,
                        self.vocabulary.index(shape),
                        5,
                    )
        raise ValueError("unsupported relational color question")

    def encode_answer(self, answer: str) -> int:
        return ColorQuestionTokenizer().encode_answer(answer)

    def decode_answer(self, token_id: int) -> str:
        return ColorQuestionTokenizer().decode_answer(token_id)
