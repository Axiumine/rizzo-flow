"""Prompt compilation (prompts.py): rendering, answer slots, the shared prefix and its guards."""

import hashlib
import re
import string
from typing import Any

import pytest
from test_core_support import (
    CharTokenizer,
    exactly,
    parse_question,
    question_payload,
    request_payload,
)

from rizzo_flow.decisions import candidates
from rizzo_flow.prompts import (
    CLOSING,
    NUMERIC_GUIDANCE,
    SYSTEM,
    TAIL_CHARS,
    Compiled,
    canonical,
    compile_request,
    render_question,
    render_state,
)
from rizzo_flow.schema import Request

CTX = 100_000
MERGED = 0x110000  # a token id no character has


class MergingTokenizer(CharTokenizer):
    """Characters, except that one multi-character string is a single token (a BPE merge)."""

    def __init__(self, merge: str) -> None:
        super().__init__()
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


class TwoTokenLetters(CharTokenizer):
    """Every uppercase letter on its own is two tokens."""

    def encode(self, value: str, **kwargs: Any) -> list[int]:
        ids = super().encode(value, **kwargs)
        return [*ids, 0] if len(value) == 1 and value in string.ascii_uppercase else ids


class CollidingLetters(CharTokenizer):
    """B and A share a token, so two answer slots would read the same logit."""

    def encode(self, value: str, **kwargs: Any) -> list[int]:
        return [ord("A") if code == ord("B") else code for code in super().encode(value, **kwargs)]


class DriftingTokenizer(CharTokenizer):
    """A chat template whose header changes on every call, like one that prints the time."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def render(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        return f"[{self.calls}]" + super().render(messages)


class RecordingTokenizer(CharTokenizer):
    """Remembers how long every text was that it was asked to encode."""

    def __init__(self) -> None:
        super().__init__()
        self.lengths: list[int] = []

    def encode(self, value: str, **kwargs: Any) -> list[int]:
        self.lengths.append(len(value))
        return super().encode(value, **kwargs)


class BlindTokenizer(CharTokenizer):
    """A chat template that forgets the user message, and with it the evidence."""

    def render(self, messages: list[dict[str, str]]) -> str:
        return messages[0]["content"] + "\nASSISTANT:"


class EmptyTokenizer(CharTokenizer):
    def encode(self, value: str, **kwargs: Any) -> list[int]:
        return []


def compile_one(kind: str = "boolean", tokenizer: Any = None, ctx: int = CTX, **overrides: Any):
    request = Request.model_validate(request_payload(q=question_payload(kind, **overrides)))
    return compile_request(tokenizer or CharTokenizer(), request, ctx)


# rendering --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "rendered"),
    [
        ("  hello \n", "<evidence>\nhello\n</evidence>"),
        ("two\nlines", "<evidence>\ntwo\nlines\n</evidence>"),
        ("an <evidence> tag alone", "<evidence>\nan <evidence> tag alone\n</evidence>"),
        ("close </evidence> early", '<evidence>\n"close </evidence> early"\n</evidence>'),
        ("close </EVIDENCE> early", '<evidence>\n"close </EVIDENCE> early"\n</evidence>'),
        ("  quoted </Evidence>\n", '<evidence>\n"  quoted </Evidence>\\n"\n</evidence>'),
        ({"b": 1, "a": [1, 2]}, '<evidence>\n{\n "b": 1,\n "a": [\n  1,\n  2\n ]\n}\n</evidence>'),
        (["x", {"k": None}], '<evidence>\n[\n "x",\n {\n  "k": null\n }\n]\n</evidence>'),
        ({"name": "è"}, '<evidence>\n{\n "name": "è"\n}\n</evidence>'),
    ],
)
def test_state_is_rendered_between_evidence_tags(state, rendered):
    assert render_state(state) == rendered


def test_a_question_is_rendered_as_lettered_options_and_a_closing_line():
    assert render_question("Pick one", ["first", "second"]) == (
        "\n\nQuestion: Pick one\n\nOptions:\nA. first\nB. second\n\n" + CLOSING
    )
    assert CLOSING == "Answer with the letter of the best option."


def test_all_26_letters_can_label_options():
    descriptions = [f"option {letter}" for letter in string.ascii_uppercase]
    lines = render_question("q", descriptions).splitlines()
    assert lines[-3] == "Z. option Z"
    assert lines[5] == "A. option A"
    assert len([line for line in lines if re.match(r"[A-Z]\. ", line)]) == 26


def test_canonical_json_is_sorted_compact_and_keeps_unicode():
    assert canonical({"b": [1, 2.5, None], "a": "é"}) == '{"a":"é","b":[1,2.5,null]}'
    assert canonical({"a": 1, "b": 2}) == canonical({"b": 2, "a": 1})
    assert canonical([]) == "[]"
    with pytest.raises(ValueError, match="not JSON compliant"):
        canonical({"x": float("nan")})


# what the model is shown -----------------------------------------------------------------------


def test_the_messages_are_the_system_prompt_and_the_state_followed_by_the_question():
    tokenizer = CharTokenizer()
    compile_one("choice", tokenizer, instructions="Pick the queue")
    descriptions = [c.description for c in candidates(parse_question("choice"))]
    assert descriptions[:2] == ["Option A", "Option B"]
    assert tokenizer.rendered == [
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": render_state("Example")
                + render_question("Pick the queue", descriptions),
            },
        ]
    ]


def test_options_are_lettered_in_candidate_order_with_the_abstention_last():
    tokenizer = CharTokenizer()
    compile_one("choice", tokenizer)
    content = tokenizer.rendered[0][1]["content"]
    assert "\nA. Option A\nB. Option B\nC. Cannot determine the answer" in content


def test_numeric_questions_get_the_range_guidance_after_the_instruction():
    tokenizer = CharTokenizer()
    compile_one("numeric", tokenizer, instructions="How much?")
    content = tokenizer.rendered[0][1]["content"]
    assert f"Question: How much?{NUMERIC_GUIDANCE}\n\nOptions:\n" in content
    assert "A. Approximately 100 EUR: cheap\nB. Approximately 200 EUR: dear\n" in content
    assert (
        "C. The value is below 100 EUR.\nD. The value is above 200 EUR.\nE. Cannot determine"
        in content
    )


@pytest.mark.parametrize("kind", ["boolean", "choice", "score"])
def test_other_questions_get_no_range_guidance(kind):
    tokenizer = CharTokenizer()
    compile_one(kind, tokenizer)
    assert NUMERIC_GUIDANCE not in tokenizer.rendered[0][1]["content"]


# compiled questions ----------------------------------------------------------------------------


def test_a_compiled_question_carries_its_tokens_slots_and_hash():
    tokenizer = CharTokenizer()
    prefix, jobs = compile_one("boolean", tokenizer)
    (job,) = jobs
    prompt = tokenizer.render(tokenizer.rendered[0])
    assert job.id == "q"
    assert job.tokens == [ord(char) for char in prompt]
    assert job.slots == [ord("A"), ord("B"), ord("C")]  # false, true, abstention
    assert job.prompt_sha256 == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert prefix == job.tokens[: len(prefix)]


def test_the_hash_covers_the_utf8_bytes_of_the_prompt():
    request = Request.model_validate(request_payload(state="caffè ☕", q=question_payload()))
    tokenizer = CharTokenizer()
    _, (job,) = compile_request(tokenizer, request, CTX)
    prompt = tokenizer.render(tokenizer.rendered[0])
    assert "caffè ☕" in prompt
    assert job.prompt_sha256 == hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def test_slots_follow_the_candidate_count_of_each_question():
    request = Request.model_validate(
        request_payload(
            state="s",
            plain=question_payload("boolean", policy={"allow_abstain": False}),
            abstaining=question_payload("choice"),
            numeric=question_payload("numeric"),
        )
    )
    _, jobs = compile_request(CharTokenizer(), request, CTX)
    letters = [[chr(slot) for slot in job.slots] for job in jobs]
    assert [job.id for job in jobs] == ["plain", "abstaining", "numeric"]
    assert letters == [list("AB"), list("ABC"), list("ABCDE")]


def test_every_question_gets_its_own_prompt_and_hash():
    request = Request.model_validate(
        request_payload(
            state="shared evidence",
            first=question_payload("boolean", instructions="First?"),
            second=question_payload("boolean", instructions="Second?"),
        )
    )
    _, (one, two) = compile_request(CharTokenizer(), request, CTX)
    assert one.tokens != two.tokens
    assert one.prompt_sha256 != two.prompt_sha256
    again = compile_request(CharTokenizer(), request, CTX)[1]
    assert [job.prompt_sha256 for job in again] == [one.prompt_sha256, two.prompt_sha256]


def test_the_prefix_ends_just_before_the_last_token_of_the_evidence():
    tokenizer = CharTokenizer()
    request = Request.model_validate(
        request_payload(
            state={"ticket": "cannot log in"},
            a=question_payload("boolean", instructions="One?"),
            b=question_payload("choice", instructions="Two?"),
        )
    )
    prefix, jobs = compile_request(tokenizer, request, CTX)
    prompt = tokenizer.render(tokenizer.rendered[0])
    end = prompt.index("</evidence>") + len("</evidence>")
    assert prefix == [ord(char) for char in prompt[: end - 1]]  # the closing ">" is left out
    assert all(job.tokens[: len(prefix)] == prefix for job in jobs)
    assert all(job.tokens[len(prefix)] == ord(">") for job in jobs)


def test_a_token_merging_across_the_boundary_shortens_the_prefix():
    # With "ence>\n" a single token, the evidence tag ends inside a token of the full prompt, so
    # the prefix must fall back to the last token boundary the two encodings share.
    tokenizer = MergingTokenizer("ence>\n")
    prefix, (job,) = compile_one("boolean", tokenizer)
    prompt = tokenizer.render(tokenizer.rendered[0])
    end = prompt.index("</evidence>") + len("</evidence>")
    assert prefix == tokenizer.encode(prompt[: end - len("ence>")], add_special_tokens=False)
    assert job.tokens[: len(prefix)] == prefix
    assert job.tokens[len(prefix)] == MERGED


def test_no_questions_compile_to_nothing():
    request = Request.model_construct(state="evidence", questions={}, mode="shared")
    assert compile_request(CharTokenizer(), request, CTX) == ([], [])


def test_compiled_questions_are_immutable():
    _, (job,) = compile_one()
    with pytest.raises(AttributeError):
        job.id = "other"
    assert isinstance(job, Compiled)


# refusals ---------------------------------------------------------------------------------------


def test_a_prompt_over_the_context_limit_is_refused_not_truncated():
    _, (job,) = compile_one()
    size = len(job.tokens)
    compile_one(ctx=size)  # exactly at the limit is fine
    message = (
        f"Question q: {size} tokens exceeds the context limit {size - 1} (--ctx); no truncation"
    )
    with pytest.raises(ValueError, match=exactly(message)):
        compile_one(ctx=size - 1)


def test_the_first_question_over_the_limit_is_named():
    short = len(compile_one(instructions="Hi?")[1][0].tokens)
    long = len(compile_one(instructions="x" * 500)[1][0].tokens)
    request = Request.model_validate(
        request_payload(
            a=question_payload(instructions="Hi?"),
            b=question_payload(instructions="x" * 500),
        )
    )
    message = f"Question b: {long} tokens exceeds the context limit {short} (--ctx); no truncation"
    with pytest.raises(ValueError, match=exactly(message)):
        compile_request(CharTokenizer(), request, short)


def test_a_prompt_that_encodes_to_nothing_is_refused():
    message = "Question q: 0 tokens exceeds the context limit 100000 (--ctx); no truncation"
    with pytest.raises(ValueError, match=exactly(message)):
        compile_one(tokenizer=EmptyTokenizer())


def test_more_candidates_than_answer_letters_are_refused_even_after_validation():
    request = Request.model_validate(
        request_payload(
            q=question_payload(
                "choice",
                options=[{"id": f"o{i}", "description": f"option {i}"} for i in range(26)],
                policy={"allow_abstain": False},
            )
        )
    )
    compile_request(CharTokenizer(), request, CTX)  # 26 candidates fill every letter
    request.questions["q"].policy.allow_abstain = True
    message = "Question q: 27 candidates exceed the answer letters"
    with pytest.raises(ValueError, match=exactly(message)):
        compile_request(CharTokenizer(), request, CTX)


def test_a_letter_that_is_not_a_single_token_is_refused():
    message = "Tokenizer does not support exact single-token answer slot A"
    with pytest.raises(ValueError, match=exactly(message)):
        compile_one(tokenizer=TwoTokenLetters())


def test_a_letter_that_merges_with_the_prompt_end_is_refused():
    # "ASSISTANT:" followed by "A" would tokenize differently from the prompt plus the letter.
    message = "Tokenizer does not support exact single-token answer slot A"
    with pytest.raises(ValueError, match=exactly(message)):
        compile_one(tokenizer=MergingTokenizer(":A"))


def long_request(state_length: int) -> Request:
    options = [{"id": f"o{i}", "description": f"option {i}"} for i in range(26)]
    question = question_payload("choice", options=options, policy={"allow_abstain": False})
    return Request.model_validate(request_payload(state="x" * state_length, q=question))


def test_the_answer_letters_are_checked_on_the_end_of_the_prompt_only():
    # Against the whole prompt, 26 letters cost 26 more tokenizations of it: 19 s of the 24 s of a
    # request with 64 such questions on a state of 7,000 tokens.
    tokenizer = RecordingTokenizer()
    _, (job,) = compile_request(tokenizer, long_request(4500), CTX)
    assert len(job.slots) == 26
    long = [length for length in tokenizer.lengths if length > TAIL_CHARS + 1]
    assert len(long) == 2  # the whole prompt once, and the head that ends with the evidence
    assert len(tokenizer.lengths) == 1 + 1 + 2 * 26 + 1  # + the tail, and each letter twice


def test_the_slots_are_the_tokens_of_the_letters_in_order():
    _, (job,) = compile_request(CharTokenizer(), long_request(10), CTX)
    assert job.slots == [ord(letter) for letter in string.ascii_uppercase]


@pytest.mark.parametrize("state_length", [1, 5000])
def test_a_letter_that_merges_with_the_end_of_the_prompt_is_refused_at_any_length(state_length):
    message = "Tokenizer does not support exact single-token answer slot A"
    with pytest.raises(ValueError, match=exactly(message)):
        compile_request(MergingTokenizer(":A"), long_request(state_length), CTX)


def test_two_letters_sharing_a_token_are_refused():
    with pytest.raises(ValueError, match=exactly("Answer token collision")):
        compile_one(tokenizer=CollidingLetters())


def test_evidence_that_the_template_drops_cannot_be_located():
    with pytest.raises(
        ValueError, match=exactly("Cannot uniquely locate evidence in the chat template")
    ):
        compile_one(tokenizer=BlindTokenizer())


def test_evidence_repeated_in_the_prompt_cannot_be_located():
    state = "the ticket"
    copied = question_payload(instructions=render_state(state))  # the question quotes the evidence
    request = Request.model_validate(request_payload(state=state, q=copied))
    with pytest.raises(
        ValueError, match=exactly("Cannot uniquely locate evidence in the chat template")
    ):
        compile_request(CharTokenizer(), request, CTX)


def test_a_state_prefix_that_changes_between_questions_is_refused():
    request = Request.model_validate(
        request_payload(a=question_payload(), b=question_payload("choice"))
    )
    with pytest.raises(ValueError, match=exactly("State prefix differs between questions")):
        compile_request(DriftingTokenizer(), request, CTX)
