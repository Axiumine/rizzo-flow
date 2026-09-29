"""Request contract (schema.py): text that JSON can spell but UTF-8 cannot encode."""

import json
import re
from typing import Any

import pytest
from pydantic import ValidationError

from rizzo_flow.schema import Request, require_unicode

LONE_SURROGATE = (
    "Strings must be valid Unicode text: a lone surrogate, such as the JSON escape \\ud800, "
    "cannot be encoded as UTF-8"
)


def exactly(message: str) -> str:
    """A `match` pattern for pytest.raises that accepts the message and nothing around it."""
    return f"^{re.escape(message)}$"


def question_payload(kind: str = "boolean", **overrides: Any) -> dict[str, Any]:
    """A minimal valid question of the given kind; `overrides` replace or extend its fields."""
    payload: dict[str, Any] = {"type": kind, "instructions": "Evaluate the evidence"}
    if kind == "choice":
        payload["options"] = [
            {"id": "a", "description": "Option A"},
            {"id": "b", "description": "Option B"},
        ]
    elif kind == "score":
        payload["levels"] = ["low", "medium", "high"]
    elif kind == "numeric":
        payload["unit"] = "EUR"
        payload["anchors"] = [
            {"value": 100, "description": "cheap"},
            {"value": 200, "description": "dear"},
        ]
    payload.update(overrides)
    return payload


def request_payload(state: Any = "Example", **questions: dict[str, Any]) -> dict[str, Any]:
    """A request body; every keyword argument is one named question."""
    return {"state": state, "questions": questions or {"q": question_payload()}}


# lone surrogates ----------------------------------------------------------------------------
# JSON can spell one ("\ud800", which json.loads accepts); UTF-8 cannot encode it.


@pytest.mark.parametrize(
    "value", ["text", "é😀", {"a": [1, None, True, 1.5, "é"]}, [], {}, 5, 1.5, None, True, b"raw"]
)
def test_require_unicode_hands_back_what_it_checked(value):
    assert require_unicode(value) is value


@pytest.mark.parametrize(
    "value",
    [
        "\ud800",
        "a\udfffb",
        {"key": "\ud800"},
        {"\ud800": 1},
        ["ok", ["ok", {"deep": ["\udc00"]}]],
        {"a": {"b": {"c\udbff": None}}},
    ],
)
def test_require_unicode_refuses_a_lone_surrogate_anywhere(value):
    with pytest.raises(ValueError, match=exactly(LONE_SURROGATE)):
        require_unicode(value)


@pytest.mark.parametrize(
    ("code_point", "accepted"),
    [
        (0xD7FF, True),  # the last scalar value before the surrogates
        (0xD800, False),  # the first high surrogate
        (0xDBFF, False),  # the last high surrogate
        (0xDC00, False),  # the first low surrogate
        (0xDFFF, False),  # the last low surrogate
        (0xE000, True),  # the first scalar value after them
        (0x1F600, True),
        (0x10FFFF, True),
    ],
)
def test_require_unicode_refuses_exactly_the_surrogate_range(code_point, accepted):
    character = chr(code_point)
    if accepted:
        assert require_unicode(character) == character
    else:
        with pytest.raises(ValueError, match="lone surrogate"):
            require_unicode(character)


def test_require_unicode_takes_a_pair_of_escapes_for_the_character_it_spells():
    assert require_unicode(json.loads(r'"😀"')) == "\U0001f600"
    with pytest.raises(ValueError, match="lone surrogate"):
        require_unicode(json.loads(r'"\ud83d"'))
    with pytest.raises(ValueError, match="lone surrogate"):
        require_unicode(json.loads(r'"\ude00\ud83d"'))  # a low surrogate first is not a pair


def test_require_unicode_does_not_recurse():
    nesting = 10_000  # ten times the default recursion limit
    for bottom, refused in (("fine", False), ("\ud800", True)):
        value: Any = bottom
        for _ in range(nesting):
            value = [{"k": value}]
        if refused:
            with pytest.raises(ValueError, match="lone surrogate"):
                require_unicode(value)
        else:
            assert require_unicode(value) is value


@pytest.mark.parametrize(
    ("body", "field"),
    [
        (request_payload(state="\ud800"), "state"),
        (request_payload(state=["ok", {"k": ["\ud800"]}]), "state"),
        (request_payload(state={"ok": {"\udfff": 1}}), "state"),
        (request_payload(**{"\ud800": question_payload()}), "questions"),
        (request_payload(q=question_payload(instructions="x\ud800")), "questions"),
        (request_payload(q=question_payload(true_description="\ud800")), "questions"),
        (
            request_payload(
                q=question_payload(
                    "choice",
                    options=[{"id": "a", "description": "A"}, {"id": "b", "description": "\ud800"}],
                )
            ),
            "questions",
        ),
        (
            request_payload(
                q=question_payload(
                    "choice",
                    options=[{"id": "a", "description": "A"}, {"id": "\ud800", "description": "B"}],
                )
            ),
            "questions",
        ),
        (request_payload(q=question_payload("score", levels=["low", "\ud800"])), "questions"),
        (request_payload(q=question_payload("numeric", unit="\ud800")), "questions"),
        (
            request_payload(
                q=question_payload(
                    "numeric",
                    anchors=[
                        {"value": 1, "description": "a"},
                        {"value": 2, "description": "\ud800"},
                    ],
                )
            ),
            "questions",
        ),
    ],
    ids=[
        "state",
        "state-value",
        "state-key",
        "question-id",
        "instructions",
        "boolean-description",
        "option-description",
        "option-id",
        "level",
        "unit",
        "anchor-description",
    ],
)
def test_request_refuses_a_lone_surrogate_wherever_it_carries_text(body, field):
    with pytest.raises(ValidationError) as raised:
        Request.model_validate(body)
    (error,) = raised.value.errors()
    assert error["loc"] == (field,)
    assert error["msg"] == f"Value error, {LONE_SURROGATE}"


def test_request_reports_the_state_and_the_questions_separately():
    body = request_payload(state="\ud800", **{"\udfff": question_payload()})
    with pytest.raises(ValidationError) as raised:
        Request.model_validate(body)
    assert sorted(error["loc"] for error in raised.value.errors()) == [("questions",), ("state",)]


def test_request_accepts_astral_text_in_the_state_and_in_ids():
    body = request_payload(state={"😀": ["\U0001f600"]}, **{"\U0001f600": question_payload()})
    request = Request.model_validate(body)
    assert request.state == {"😀": ["😀"]}
    assert list(request.questions) == ["😀"]
