"""Prompt compilation (prompts.py): the answer letters and how they are checked."""

import re
import string
from typing import Any

import pytest
from test_service import CharacterTokenizer

from rizzo_flow.prompts import TAIL_CHARS, compile_request
from rizzo_flow.schema import Request

CTX = 100_000
MERGED = 0x110000  # a token id no character has


def exactly(message: str) -> str:
    """A `match` pattern for pytest.raises that accepts the message and nothing around it."""
    return f"^{re.escape(message)}$"


class MergingTokenizer(CharacterTokenizer):
    """Characters, except that one multi-character string is a single token (a BPE merge)."""

    def __init__(self, merge: str) -> None:
        self.merge = merge

    def encode(self, value: str, **kwargs: Any) -> list[int]:
        assert kwargs == {"add_special_tokens": False}
        ids: list[int] = []
        index = 0
        while index < len(value):
            if value.startswith(self.merge, index):
                ids.append(MERGED)
                index += len(self.merge)
            else:
                ids.append(ord(value[index]))
                index += 1
        return ids


class RecordingTokenizer(CharacterTokenizer):
    """Remembers how long every text was that it was asked to encode."""

    def __init__(self) -> None:
        self.lengths: list[int] = []

    def encode(self, value: str, **kwargs: Any) -> list[int]:
        self.lengths.append(len(value))
        return super().encode(value, **kwargs)


class TwoTokensPerCharacter(CharacterTokenizer):
    """Spells every character with two tokens, the answer letters included, so that a prompt plus
    a letter is always the prompt's tokens plus the letter's."""

    def encode(self, value: str, **kwargs: Any) -> list[int]:
        return [token for char in value for token in (ord(char), ord(char) + 1000)]


def boolean_request() -> Request:
    body = {
        "state": "Some evidence",
        "questions": {"q": {"type": "boolean", "instructions": "Is it?"}},
    }
    return Request.model_validate(body)


def long_request(state_length: int) -> Request:
    options = [{"id": f"o{i}", "description": f"option {i}"} for i in range(26)]
    question = {
        "type": "choice",
        "instructions": "Choose",
        "options": options,
        "policy": {"allow_abstain": False},
    }
    return Request.model_validate({"state": "x" * state_length, "questions": {"q": question}})


# answer letters ---------------------------------------------------------------------------------


def test_the_answer_letters_are_checked_on_the_end_of_the_prompt_only():
    # Against the whole prompt, 26 letters cost 26 more tokenizations of it: 19 s of the 24 s of a
    # request with 64 such questions on a state of 7,000 tokens.
    tokenizer = RecordingTokenizer()
    _, (job,) = compile_request(tokenizer, long_request(4500), CTX)
    assert len(job.slots) == 26
    long = [length for length in tokenizer.lengths if length > TAIL_CHARS + 1]
    assert len(long) == 2  # the whole prompt once, and the head that ends with the evidence
    assert len(tokenizer.lengths) == 1 + 1 + 2 * 26 + 1  # + the tail, and each letter twice


def test_a_letter_that_merges_with_the_prompt_end_is_refused():
    # "ASSISTANT:" followed by "A" would tokenize differently from the prompt plus the letter.
    message = "Tokenizer does not support exact single-token answer slot A"
    with pytest.raises(ValueError, match=exactly(message)):
        compile_request(MergingTokenizer(":A"), boolean_request(), CTX)


@pytest.mark.parametrize("state_length", [1, 5000])
def test_a_letter_that_merges_with_the_end_of_the_prompt_is_refused_at_any_length(state_length):
    message = "Tokenizer does not support exact single-token answer slot A"
    with pytest.raises(ValueError, match=exactly(message)):
        compile_request(MergingTokenizer(":A"), long_request(state_length), CTX)


def test_an_answer_letter_that_takes_two_tokens_is_refused_even_when_it_extends_the_prompt():
    """The check that prompt plus letter is prompt plus the letter's tokens is not enough on its
    own: the second token of the letter would be dropped."""
    message = "Tokenizer does not support exact single-token answer slot A"
    with pytest.raises(ValueError, match=exactly(message)):
        compile_request(TwoTokensPerCharacter(), boolean_request(), CTX)


def test_the_slots_are_the_tokens_of_the_letters_in_order():
    _, (job,) = compile_request(CharacterTokenizer(), long_request(10), CTX)
    assert job.slots == [ord(letter) for letter in string.ascii_uppercase]
