"""Mutation tests of `rizzo_flow.prompts`."""

import pytest
from test_mutation_support import exactly

from rizzo_flow.prompts import compile_request
from rizzo_flow.schema import Request


class TwoTokensPerCharacter:
    """A tokenizer that spells every character with two tokens, the answer letters included, so
    that a prompt plus a letter is always the prompt's tokens plus the letter's."""

    def apply_chat_template(self, messages, tokenize=False, **variables):
        return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"

    def encode(self, text, add_special_tokens=False):
        return [token for char in text for token in (ord(char), ord(char) + 1000)]


def test_an_answer_letter_that_takes_two_tokens_is_refused_even_when_it_extends_the_prompt():
    """`compile_request` checking that prompt plus letter is prompt plus the letter's tokens,
    but not that the letter is one token: the second token of the letter would be dropped."""
    request = Request.model_validate(
        {
            "state": "Some evidence",
            "questions": {"q": {"type": "boolean", "instructions": "Is it?"}},
        }
    )
    message = "Tokenizer does not support exact single-token answer slot A"
    with pytest.raises(ValueError, match=exactly(message)):
        compile_request(TwoTokensPerCharacter(), request, 8192)
