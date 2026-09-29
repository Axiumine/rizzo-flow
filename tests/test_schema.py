"""Request contract (schema.py): every limit is enforced, and at the exact boundary."""

import json
import string
from collections.abc import Sequence
from functools import partial
from typing import Any

import pytest
from pydantic import ValidationError
from test_core_support import exactly, question_payload, request_payload, value_errors

from rizzo_flow.schema import (
    MAX_SLOTS,
    Anchor,
    BaseQuestion,
    BooleanQuestion,
    ChoiceQuestion,
    NumericQuestion,
    Option,
    Policy,
    Request,
    ScoreQuestion,
    require_unicode,
)

UNIQUE_IDS = "Option IDs must be unique"
DISTINCT_LEVELS = "Levels must have distinct descriptions"
INCREASING = "Numeric anchors must be strictly increasing"
EMPTY_STATE = "State must not be empty"
STATE_LIMIT = "State exceeds 256 KB; no silent truncation"
QUESTION_IDS = "Question IDs must have 1–128 nonblank characters"
LONE_SURROGATE = (
    "Strings must be valid Unicode text: a lone surrogate, such as the JSON escape \\ud800, "
    "cannot be encoded as UTF-8"
)


def slots(limit: int, reserved: int) -> str:
    return (
        f"At most {limit} entries fit here: 26 answer letters, "
        f"{reserved} reserved for abstention/out-of-range"
    )


def choice(count: int, abstain: bool | None = None) -> dict[str, Any]:
    options = [{"id": f"o{i}", "description": f"option {i}"} for i in range(count)]
    payload = question_payload("choice", options=options)
    if abstain is not None:
        payload["policy"] = {"allow_abstain": abstain}
    return payload


def score(count: int, abstain: bool | None = None) -> dict[str, Any]:
    payload = question_payload("score", levels=[f"level {i}" for i in range(count)])
    if abstain is not None:
        payload["policy"] = {"allow_abstain": abstain}
    return payload


def numeric(
    values: Sequence[float], abstain: bool | None = None, **overrides: Any
) -> dict[str, Any]:
    anchors = [{"value": v, "description": f"anchor {i}"} for i, v in enumerate(values)]
    payload = question_payload("numeric", anchors=anchors, **overrides)
    if abstain is not None:
        payload["policy"] = {"allow_abstain": abstain}
    return payload


def request_with_state(state: Any) -> Request:
    return Request.model_validate(request_payload(state=state))


# limits shared by every question type -------------------------------------------------------


def test_the_slot_limit_is_the_number_of_uppercase_letters():
    assert MAX_SLOTS == len(string.ascii_uppercase) == 26


def test_policy_defaults():
    policy = Policy()
    assert policy.allow_abstain is True
    assert policy.max_unavailable_probability == 0.5
    assert policy.min_top_probability == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_unavailable_probability", 0),
        ("max_unavailable_probability", -0.1),
        ("max_unavailable_probability", 1.01),
        ("min_top_probability", -0.01),
        ("min_top_probability", 1.01),
        ("allow_abstain", "yes"),
        ("allow_abstain", 1),
    ],
)
def test_policy_rejects_values_outside_its_bounds(field, value):
    with pytest.raises(ValidationError):
        Policy.model_validate({field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_unavailable_probability", 1.0),
        ("max_unavailable_probability", 0.001),
        ("min_top_probability", 0.0),
        ("min_top_probability", 1.0),
        ("allow_abstain", False),
    ],
)
def test_policy_accepts_its_boundaries(field, value):
    assert getattr(Policy.model_validate({field: value}), field) == value


def test_policy_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Policy.model_validate({"allow_abstain": True, "temperature": 2})


def test_text_is_stripped_and_bounded_between_one_and_8000_characters():
    padded = "  " + "x" * 8000 + "\n"
    question = BooleanQuestion.model_validate(question_payload(instructions=padded))
    assert question.instructions == "x" * 8000
    for bad in ("", "  \n\t", "x" * 8001, 5, None):
        with pytest.raises(ValidationError):
            BooleanQuestion.model_validate(question_payload(instructions=bad))


def test_question_fields_are_strictly_typed_and_unknown_fields_refused():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BooleanQuestion.model_validate(question_payload(unsupported=True))
    with pytest.raises(ValidationError):
        BooleanQuestion.model_validate(question_payload(policy={"allow_abstain": "false"}))


def test_the_slot_check_counts_the_abstention_and_the_reserved_slots():
    question = BaseQuestion(instructions="x", policy=Policy(allow_abstain=False))
    question.require_slots([0] * 26)
    question.require_slots([0] * 24, reserved=2)
    with pytest.raises(ValueError, match=exactly(slots(25, 1))):
        BaseQuestion(instructions="x").require_slots([0] * 26)
    with pytest.raises(ValueError, match=exactly(slots(24, 2))):
        question.require_slots([0] * 25, reserved=2)
    abstaining = BaseQuestion(instructions="x")
    abstaining.require_slots([0] * 23, reserved=2)
    with pytest.raises(ValueError, match=exactly(slots(23, 3))):
        abstaining.require_slots([0] * 24, reserved=2)


# boolean ------------------------------------------------------------------------------------


def test_boolean_defaults_and_custom_descriptions():
    default = BooleanQuestion.model_validate(question_payload("boolean"))
    assert default.true_description.startswith("Yes.")
    assert default.false_description.startswith("No.")
    assert default.policy == Policy()
    custom = BooleanQuestion.model_validate(
        question_payload("boolean", true_description="  Yes, really ", false_description="Nope")
    )
    assert (custom.true_description, custom.false_description) == ("Yes, really", "Nope")
    with pytest.raises(ValidationError):
        BooleanQuestion.model_validate(question_payload("boolean", true_description=""))


# choice -------------------------------------------------------------------------------------


@pytest.mark.parametrize("identifier", ["a", "A", "0", "a1", "a_b", "a-b", "Z9_-x", "a" * 64])
def test_option_ids_accept_letters_digits_underscore_and_dash(identifier):
    option = Option.model_validate({"id": identifier, "description": "d"})
    assert option.id == identifier


@pytest.mark.parametrize(
    "identifier",
    [
        "",
        "_a",
        "-a",
        "a b",
        "a.b",
        "a" * 65,
        "é",
        "a/b",
        "a\n",
        "__insufficient__",
        "__above_range__",
    ],
)
def test_option_ids_refuse_everything_else(identifier):
    with pytest.raises(ValidationError):
        Option.model_validate({"id": identifier, "description": "d"})


def test_choice_needs_unique_ids():
    payload = choice(3)
    payload["options"][2]["id"] = "o0"
    assert value_errors(lambda: ChoiceQuestion.model_validate(payload)) == [UNIQUE_IDS]


def test_choice_needs_between_two_and_26_options():
    ChoiceQuestion.model_validate(choice(2))
    with pytest.raises(ValidationError, match="at least 2 items"):
        ChoiceQuestion.model_validate(choice(1))
    with pytest.raises(ValidationError, match="at most 26 items"):
        ChoiceQuestion.model_validate(choice(27, abstain=False))


def test_choice_abstention_takes_one_of_the_26_letters():
    assert len(ChoiceQuestion.model_validate(choice(26, abstain=False)).options) == 26
    assert len(ChoiceQuestion.model_validate(choice(25, abstain=True)).options) == 25
    assert value_errors(lambda: ChoiceQuestion.model_validate(choice(26, abstain=True))) == [
        slots(25, 1)
    ]
    assert value_errors(lambda: ChoiceQuestion.model_validate(choice(26))) == [slots(25, 1)]


# score --------------------------------------------------------------------------------------


def test_score_levels_must_be_distinct_once_stripped():
    payload = question_payload("score", levels=["low", "high", "low"])
    assert value_errors(lambda: ScoreQuestion.model_validate(payload)) == [DISTINCT_LEVELS]
    payload["levels"] = ["low", " low "]
    assert value_errors(lambda: ScoreQuestion.model_validate(payload)) == [DISTINCT_LEVELS]


def test_score_needs_between_two_and_26_levels():
    ScoreQuestion.model_validate(score(2))
    with pytest.raises(ValidationError, match="at least 2 items"):
        ScoreQuestion.model_validate(score(1))
    with pytest.raises(ValidationError, match="at most 26 items"):
        ScoreQuestion.model_validate(score(27, abstain=False))


def test_score_abstention_takes_one_of_the_26_letters():
    assert len(ScoreQuestion.model_validate(score(26, abstain=False)).levels) == 26
    assert len(ScoreQuestion.model_validate(score(25, abstain=True)).levels) == 25
    assert value_errors(lambda: ScoreQuestion.model_validate(score(26))) == [slots(25, 1)]


# numeric ------------------------------------------------------------------------------------


def test_numeric_anchors_must_strictly_increase():
    NumericQuestion.model_validate(numeric([1, 2]))
    for values in ([1, 1], [2, 1], [1, 3, 2], [1, 2, 2], [0, 5, 5, 6]):
        payload = numeric(values, abstain=False)
        assert value_errors(partial(NumericQuestion.model_validate, payload)) == [INCREASING]


def test_numeric_needs_between_two_and_26_anchors():
    with pytest.raises(ValidationError, match="at least 2 items"):
        NumericQuestion.model_validate(numeric([1]))
    with pytest.raises(ValidationError, match="at most 26 items"):
        NumericQuestion.model_validate(numeric(list(range(27)), abstain=False))


def test_numeric_reserves_below_above_and_the_abstention():
    assert len(NumericQuestion.model_validate(numeric(list(range(24)), False)).anchors) == 24
    assert len(NumericQuestion.model_validate(numeric(list(range(23)), True)).anchors) == 23
    too_many = numeric(list(range(25)), False)
    assert value_errors(lambda: NumericQuestion.model_validate(too_many)) == [slots(24, 2)]
    with_abstention = numeric(list(range(24)), True)
    assert value_errors(lambda: NumericQuestion.model_validate(with_abstention)) == [slots(23, 3)]
    by_default = numeric(list(range(24)))  # abstention is on by default
    assert value_errors(lambda: NumericQuestion.model_validate(by_default)) == [slots(23, 3)]


def test_anchor_values_are_finite_numbers_within_1e100():
    for good in (1e100, -1e100, 0, 0.5):
        assert Anchor.model_validate({"value": good, "description": "d"}).value == good
    for bad in (1.0000001e100, -1.0000001e100, float("nan"), float("inf"), "1", None):
        with pytest.raises(ValidationError):
            Anchor.model_validate({"value": bad, "description": "d"})


def test_numeric_unit_is_stripped_and_at_most_64_characters():
    assert NumericQuestion.model_validate(numeric([1, 2], unit="  EUR ")).unit == "EUR"
    assert len(NumericQuestion.model_validate(numeric([1, 2], unit="u" * 64)).unit) == 64
    for bad in ("", "   ", "u" * 65, 3):
        with pytest.raises(ValidationError):
            NumericQuestion.model_validate(numeric([1, 2], unit=bad))


# request ------------------------------------------------------------------------------------


def test_request_defaults_to_the_shared_mode():
    request = Request.model_validate(request_payload())
    assert request.mode == "shared"
    assert Request.model_validate({**request_payload(), "mode": "direct"}).mode == "direct"


@pytest.mark.parametrize("mode", ["fast", "", None, 1, "Shared"])
def test_request_mode_is_shared_or_direct(mode):
    with pytest.raises(ValidationError):
        Request.model_validate({**request_payload(), "mode": mode})


@pytest.mark.parametrize(
    "state", ["text", {"a": 1}, [1, 2], {"a": None, "b": [True, 1.5, {"c": "d"}]}, [{}], {"": ""}]
)
def test_request_accepts_text_objects_and_arrays_as_state(state):
    assert request_with_state(state).state == state


@pytest.mark.parametrize("state", ["", "   ", "\n\t ", {}, []])
def test_request_state_must_not_be_empty(state):
    assert value_errors(lambda: request_with_state(state)) == [EMPTY_STATE]


@pytest.mark.parametrize("state", [5, 1.5, None, True])
def test_request_state_must_be_text_object_or_array(state):
    with pytest.raises(ValidationError):
        request_with_state(state)


def test_request_state_rejects_non_finite_numbers_anywhere():
    for state in ({"a": float("nan")}, [1, [float("inf")]], {"a": {"b": float("-inf")}}):
        with pytest.raises(ValidationError):
            request_with_state(state)


def test_state_of_exactly_256000_encoded_bytes_is_accepted():
    # The JSON rendering of a string adds two quotes.
    assert len(request_with_state("x" * 255_998).state) == 255_998
    assert value_errors(lambda: request_with_state("x" * 255_999)) == [STATE_LIMIT]


def test_the_state_limit_counts_utf8_bytes_of_the_unescaped_text():
    assert len(request_with_state("é" * 127_999).state) == 127_999  # 2 bytes each
    assert value_errors(lambda: request_with_state("é" * 128_000)) == [STATE_LIMIT]


def test_the_state_limit_applies_to_structured_state_too():
    request_with_state({"k": "x" * 200_000})
    request_with_state(["x" * 100_000, "y" * 100_000])
    assert value_errors(lambda: request_with_state({"k": "x" * 300_000})) == [STATE_LIMIT]
    assert value_errors(lambda: request_with_state([["x" * 100_000] * 3])) == [STATE_LIMIT]


def test_request_needs_between_one_and_64_questions():
    with pytest.raises(ValidationError, match="at least 1 item"):
        Request.model_validate({"state": "s", "questions": {}})
    body = {f"q{i}": question_payload() for i in range(64)}
    assert len(Request.model_validate({"state": "s", "questions": body}).questions) == 64
    body["q64"] = question_payload()
    with pytest.raises(ValidationError, match="at most 64 items"):
        Request.model_validate({"state": "s", "questions": body})


@pytest.mark.parametrize("identifier", ["", " ", "\t\n", "x" * 129, " " * 129])
def test_question_ids_must_be_nonblank_and_at_most_128_characters(identifier):
    body = request_payload(**{identifier: question_payload()})
    assert value_errors(lambda: Request.model_validate(body)) == [QUESTION_IDS]


@pytest.mark.parametrize("identifier", ["q", "x" * 128, "my question", "  padded  ", "é", "a.b/c"])
def test_question_ids_accept_free_text(identifier):
    request = Request.model_validate(request_payload(**{identifier: question_payload()}))
    assert list(request.questions) == [identifier]


def test_the_first_blank_id_is_enough_to_refuse_the_request():
    questions = {"fine": question_payload(), " ": question_payload()}
    assert value_errors(lambda: Request.model_validate({"state": "s", "questions": questions})) == [
        QUESTION_IDS
    ]


def test_request_refuses_unknown_fields_and_unknown_question_types():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Request.model_validate({**request_payload(), "temperature": 1})
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        Request.model_validate(request_payload(q={"type": "text", "instructions": "x"}))
    with pytest.raises(ValidationError, match="Unable to extract tag"):
        Request.model_validate(request_payload(q={"instructions": "x"}))


def test_the_question_type_selects_the_question_class():
    request = Request.model_validate(
        request_payload(
            a=question_payload("boolean"),
            b=question_payload("choice"),
            c=question_payload("score"),
            d=question_payload("numeric"),
        )
    )
    assert [type(q) for q in request.questions.values()] == [
        BooleanQuestion,
        ChoiceQuestion,
        ScoreQuestion,
        NumericQuestion,
    ]
    with pytest.raises(ValidationError):
        Request.model_validate(request_payload(q={"type": "choice", "instructions": "x"}))


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
    assert require_unicode(json.loads(r'"\ud83d\ude00"')) == "\U0001f600"
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
