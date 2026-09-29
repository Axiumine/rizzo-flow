"""Pure-function tests of compat.py: wire validation, translation to native and back."""

import json
import re
from typing import Any

import pytest
from pydantic import ValidationError
from test_core_support import parse_question, value_errors

from rizzo_flow.compat import (
    LOCAL_ALIAS,
    SystemOneRequest,
    UnknownModel,
    confidence,
    from_native,
    list_models,
    model_name,
    resolve_model,
    text,
    to_native,
)
from rizzo_flow.config import MODEL_ID

BASE = {"source": "XHToken/Spark-X2.5-4B", "precision": "q8_0"}
SERVED = "rizzo-spark-x2.5-4b-q8_0"


def wire(**questions: dict[str, Any]) -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {"state": "evidence", "model": LOCAL_ALIAS, "questions": questions}
    )


def noul(criteria: Any = None, instructions: Any = "Is it urgent?") -> dict[str, Any]:
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        question["criteria"] = criteria
    return question


def native_question(question: dict[str, Any]) -> Any:
    native, _ = to_native(wire(q=question))
    return native.questions["q"]


# wire validation --------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
def test_choice_option_keys_must_not_be_blank(blank):
    question = {"type": "choice", "instructions": "i", "criteria": {"valid": "x", blank: "y"}}
    assert value_errors(lambda: wire(q=question)) == ["Choice option keys must not be blank"]


def test_choice_option_keys_with_inner_spaces_are_fine():
    question = {"type": "choice", "instructions": "i", "criteria": {"Billing team": None, "é": "x"}}
    parsed: Any = wire(q=question).questions["q"]
    assert list(parsed.criteria) == ["Billing team", "é"]


@pytest.mark.parametrize(("kind", "low", "high"), [("choice", 2, 26), ("score", 2, 10)])
def test_criteria_size_limits(kind, low, high):
    def build(count: int) -> Any:
        criteria: Any = {f"k{i}": None for i in range(count)} if kind == "choice" else ["l"] * count
        return wire(q={"type": kind, "instructions": "i", "criteria": criteria})

    build(low)
    build(high)
    for count in (low - 1, high + 1):
        with pytest.raises(ValidationError):
            build(count)


def test_unknown_fields_are_ignored_on_the_request_but_not_in_questions():
    body: dict[str, Any] = {
        "state": "s",
        "model": "m",
        "questions": {"q": noul()},
        "trace": "abc",
    }
    assert not hasattr(SystemOneRequest.model_validate(body), "trace")
    body["questions"]["q"]["surprise"] = 1
    with pytest.raises(ValidationError, match="surprise"):
        SystemOneRequest.model_validate(body)


# JSON can spell a lone surrogate ("\ud800"); UTF-8 cannot encode it, so the wire format refuses
# it wherever it carries text, as the native format does, before anything is translated.


def choice(criteria: dict[str, Any]) -> dict[str, Any]:
    return {"type": "choice", "instructions": "i", "criteria": criteria}


def levels(criteria: list[Any]) -> dict[str, Any]:
    return {"type": "score", "instructions": "i", "criteria": criteria}


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"state": "\ud800"}, "state"),
        ({"state": {"k": ["ok", {"\udfff": 1}]}}, "state"),
        ({"model": "\ud800"}, "model"),
        ({"questions": {"\ud800": noul()}}, "questions"),
        ({"questions": {"q": noul(instructions="\ud800")}}, "questions"),
        ({"questions": {"q": noul(instructions={"k": ["\ud800"]})}}, "questions"),
        ({"questions": {"q": noul({"true": "\ud800"})}}, "questions"),
        ({"questions": {"q": noul({"false": {"\ud800": 1}})}}, "questions"),
        ({"questions": {"q": choice({"a": "\ud800", "b": None})}}, "questions"),
        ({"questions": {"q": choice({"a": None, "\ud800": None})}}, "questions"),
        ({"questions": {"q": levels(["low", "\ud800"])}}, "questions"),
    ],
    ids=[
        "state",
        "state-key",
        "model",
        "question-id",
        "instructions",
        "structured-instructions",
        "true-criterion",
        "false-criterion-key",
        "option-detail",
        "option-key",
        "level",
    ],
)
def test_a_lone_surrogate_is_refused_wherever_the_wire_request_carries_text(override, field):
    body = {"state": "s", "model": LOCAL_ALIAS, "questions": {"q": noul()}, **override}
    with pytest.raises(ValidationError) as raised:
        SystemOneRequest.model_validate(body)
    (error,) = raised.value.errors()
    assert error["loc"] == (field,)
    assert error["msg"].startswith("Value error, Strings must be valid Unicode text")


def test_a_lone_surrogate_in_an_ignored_field_is_ignored_with_it():
    body = {"state": "s", "model": "m", "questions": {"q": noul()}, "trace\ud800": ["\udc00"]}
    assert set(SystemOneRequest.model_validate(body).model_dump()) == {
        "state",
        "model",
        "questions",
    }


def test_astral_text_and_surrogate_pairs_go_through_the_translation():
    emoji = "\U0001f600"
    pair = json.loads(r'"\ud83d\ude00"')
    assert pair == emoji
    request = SystemOneRequest.model_validate(
        {"state": pair, "model": LOCAL_ALIAS, "questions": {pair: choice({pair: pair, "b": None})}}
    )
    native, options = to_native(request)
    assert (native.state, list(native.questions)) == (emoji, [emoji])
    assert options == {emoji: [emoji, "b"]}


# to_native: noul --------------------------------------------------------------------------


def test_noul_without_criteria_keeps_the_default_descriptions():
    native = native_question(noul())
    default = parse_question("boolean")
    assert native.type == "boolean"
    assert native.instructions == "Is it urgent?"
    assert native.true_description == default.true_description
    assert native.false_description == default.false_description


def test_noul_with_empty_criteria_keeps_the_default_descriptions():
    native = native_question(noul({}))
    default = parse_question("boolean")
    assert native.true_description == default.true_description
    assert native.false_description == default.false_description


def test_noul_true_criterion_only():
    native = native_question(noul({"true": "  Explicitly time-sensitive  "}))
    assert native.true_description == "Yes. Explicitly time-sensitive"
    assert native.false_description == parse_question("boolean").false_description


def test_noul_false_criterion_only():
    native = native_question(noul({"false": "Routine request"}))
    assert native.false_description == "No. Routine request"
    assert native.true_description == parse_question("boolean").true_description


def test_noul_both_criteria_and_structured_text():
    native = native_question(noul({"true": {"b": 1, "a": [None]}, "false": ["x", 2]}))
    assert native.true_description == 'Yes. {"a":[null],"b":1}'
    assert native.false_description == 'No. ["x",2]'


def test_translated_questions_never_abstain_and_return_no_option_keys_for_noul():
    native, options = to_native(wire(a=noul(), b=noul({"true": "t"})))
    assert [q.policy.allow_abstain for q in native.questions.values()] == [False, False]
    assert options == {}


# to_native: choice, score and the request as a whole ----------------------------------------


def test_choice_options_are_positional_and_keep_the_caller_keys():
    question = {
        "type": "choice",
        "instructions": {"k": 1},
        "criteria": {"Billing team": "Payments", "tech": None, "ops": {"x": [1, 2]}, "sales": []},
    }
    native, options = to_native(wire(q=question))
    assert options == {"q": ["Billing team", "tech", "ops", "sales"]}
    translated = native.questions["q"]
    assert translated.type == "choice"
    assert translated.instructions == '{"k":1}'
    assert [o.id for o in translated.options] == ["o0", "o1", "o2", "o3"]
    assert [o.description for o in translated.options] == [
        "Billing team: Payments",
        "tech",
        'ops: {"x":[1,2]}',
        "sales: []",
    ]


def test_score_levels_are_rendered_as_text():
    question = {"type": "score", "instructions": "i", "criteria": ["  Calm ", {"a": 1}, "Angry"]}
    translated = native_question(question)
    assert translated.type == "score"
    assert translated.levels == ["Calm", '{"a":1}', "Angry"]


def test_score_levels_that_collapse_to_the_same_text_are_rejected():
    question = {"type": "score", "instructions": "i", "criteria": ["Calm", " Calm "]}
    request = wire(q=question)
    assert value_errors(lambda: to_native(request)) == ["Levels must have distinct descriptions"]


@pytest.mark.parametrize("state", ["plain text", {"a": [1, {"b": None}]}, ["x", 2]])
def test_the_state_reaches_the_native_request_untouched(state):
    request = SystemOneRequest.model_validate(
        {"state": state, "model": LOCAL_ALIAS, "questions": {"q": noul()}}
    )
    native, _ = to_native(request)
    assert native.state == state
    assert native.mode == "shared"


def test_options_are_only_recorded_for_choice_questions_and_keep_the_question_order():
    choice = {"type": "choice", "instructions": "i", "criteria": {"b": None, "a": None}}
    score = {"type": "score", "instructions": "i", "criteria": ["l", "m"]}
    native, options = to_native(wire(first=choice, second=noul(), third=score, fourth=choice))
    assert list(native.questions) == ["first", "second", "third", "fourth"]
    assert options == {"first": ["b", "a"], "fourth": ["b", "a"]}


# text and confidence ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  hi  ", "hi"),
        ({"b": 1, "a": [True, None]}, '{"a":[true,null],"b":1}'),
        ([1, "é"], '[1,"é"]'),
        (7, "7"),
        (None, "null"),
    ],
)
def test_text_strips_strings_and_serializes_the_rest_canonically(value, expected):
    assert text(value) == expected


def test_confidence_is_the_peak_over_uniform_statistic():
    assert confidence([1, 0, 0]) == 1
    assert confidence([0.5, 0.5]) == 0
    assert confidence([0.9, 0.06, 0.04]) == pytest.approx(0.85)
    assert confidence([0.6, 0.4]) == pytest.approx(0.2)
    assert confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)


def test_confidence_is_clamped_to_the_unit_interval():
    assert confidence([0.4, 0.3]) == 0.0  # an unnormalized input would give -0.2
    assert confidence([2.0, 0.0]) == 1.0  # ... and this one 3


# model names ------------------------------------------------------------------------------


def test_model_name_of_the_base_and_fine_tuned_weights():
    assert model_name(BASE) == SERVED
    assert model_name({**BASE, "weights": "base"}) == SERVED
    assert model_name({**BASE, "weights": "flow"}) == "rizzo-flow-4b-q8_0"
    small = {"source": "XHToken/Spark-X2.5-1.7B", "precision": "bf16"}
    assert model_name(small) == "rizzo-spark-x2.5-1.7b-bf16"
    assert model_name({**small, "weights": "flow"}) == "rizzo-flow-1.7b-bf16"


def test_model_name_defaults_to_the_configured_checkpoint():
    expected = "rizzo-" + MODEL_ID.split("/")[-1].lower() + "-unknown"
    assert model_name({}) == expected
    assert model_name({"source": "local-model-7B", "precision": "q4"}) == "rizzo-local-model-7b-q4"
    assert model_name({"source": "plain", "precision": "q4", "weights": "flow"}) == (
        "rizzo-flow-plain-q4"
    )


def test_resolve_model_accepts_the_alias_the_served_id_and_jev_names():
    for requested in (LOCAL_ALIAS, SERVED, "jev-latest", "jev-1.13.0", "jev-"):
        assert resolve_model(requested, BASE) == SERVED
    assert resolve_model("rizzo-flow-4b-q8_0", {**BASE, "weights": "flow"}) == (
        "rizzo-flow-4b-q8_0"
    )


@pytest.mark.parametrize(
    "requested", ["gpt-4", "jev", "Jev-latest", "rizzo", "", "rizzo-flow-4b-q8_0"]
)
def test_resolve_model_refuses_other_names(requested):
    with pytest.raises(UnknownModel) as raised:
        resolve_model(requested, BASE)
    assert isinstance(raised.value, ValueError)
    assert str(raised.value) == (
        f"Unknown model {requested!r}. Use 'rizzo-latest', {SERVED!r} or a jev-* alias."
    )


def test_list_models_names_the_alias_the_served_model_and_the_compatibility_alias():
    listing = list_models(BASE)
    assert set(listing) == {"models"}
    models = listing["models"]
    assert [m["name"] for m in models] == [LOCAL_ALIAS, SERVED, "jev-latest"]
    local = "Local XHToken/Spark-X2.5-4B scored with typed option logits."
    assert models[0]["description"] == models[1]["description"] == local
    assert models[2]["description"] == (
        f"Compatibility alias: answered by {SERVED}, not by TypeSafe Jev."
    )
    for model in models:
        assert set(model) == {"name", "description", "release_date"}
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", model["release_date"])


def test_list_models_without_a_source_says_spark():
    described = list_models({"precision": "q8_0"})["models"][0]["description"]
    assert described == "Local Spark scored with typed option logits."


# from_native ------------------------------------------------------------------------------


def native_answer(
    probabilities: dict[str, float], input_tokens: int, status: str | None = None, **extra: Any
) -> dict[str, Any]:
    return {
        "probabilities": probabilities,
        "input_tokens": input_tokens,
        "probability_status": status or "uncalibrated_conditional_option_scores",
        **extra,
    }


def three_questions() -> SystemOneRequest:
    return wire(
        urgent=noul(),
        team={
            "type": "choice",
            "instructions": "Who?",
            "criteria": {"billing": None, "tech": "Bugs", "sales": None},
        },
        mood={"type": "score", "instructions": "How?", "criteria": ["Calm", {"a": 1}, "Angry"]},
    )


def test_from_native_maps_every_question_type():
    request = three_questions()
    _, options = to_native(request)
    response = {
        "model": {"fingerprint": "fp-1"},
        "answers": {
            "urgent": native_answer({"false": 0.2, "true": 0.8}, 120),
            "team": native_answer({"o0": 0.1, "o1": 0.6, "o2": 0.3}, 130),
            "mood": native_answer(
                {"0": 0.5, "1": 0.3, "2": 0.2},
                140,
                "temperature_scaled_requires_held_out_validation",
                score=0.7,
            ),
        },
        "timing": {"shared_prefix_tokens": 100, "total_seconds": 0.5},
    }
    result = from_native(request, response, options, "rizzo-served")
    assert list(result) == ["model", "answers", "usage", "x_rizzo"]
    assert result["model"] == "rizzo-served"
    assert list(result["answers"]) == ["urgent", "team", "mood"]
    assert result["answers"]["urgent"] == {"type": "noul", "noul": 0.8}
    assert result["answers"]["team"] == {
        "type": "choice",
        "choice": "tech",
        "probabilities": {"billing": 0.1, "tech": 0.6, "sales": 0.3},
        "confidence": pytest.approx(0.4),
    }
    assert result["answers"]["mood"] == {
        "type": "score",
        "score": 0.7,
        "legend": {"0": "Calm", "1": '{"a":1}', "2": "Angry"},
        "probabilities": {"0": 0.5, "1": 0.3, "2": 0.2},
        "confidence": pytest.approx(0.25),
    }
    # The 100 shared tokens are evaluated once, not once per question.
    assert result["usage"] == {"input_tokens": 120 + 130 + 140 - 100 * 2, "output_tokens": 0}
    assert result["x_rizzo"] == {
        "timing": {"shared_prefix_tokens": 100, "total_seconds": 0.5},
        "probability_status": [
            "temperature_scaled_requires_held_out_validation",
            "uncalibrated_conditional_option_scores",
        ],
        "fingerprint": "fp-1",
    }


def test_from_native_without_shared_prefix_or_fingerprint():
    request = wire(a=noul(), b=noul())
    response = {
        "model": {},
        "answers": {
            "a": native_answer({"false": 0.5, "true": 0.5}, 60),
            "b": native_answer({"false": 0.5, "true": 0.5}, 70),
        },
        "timing": {"total_seconds": 1.0},
    }
    result = from_native(request, response, {}, "m")
    assert result["usage"] == {"input_tokens": 130, "output_tokens": 0}
    assert result["x_rizzo"]["fingerprint"] is None
    assert result["x_rizzo"]["probability_status"] == ["uncalibrated_conditional_option_scores"]


def test_from_native_counts_a_single_question_in_full():
    request = wire(a=noul())
    response = {
        "model": {"fingerprint": "f"},
        "answers": {"a": native_answer({"false": 0.1, "true": 0.9}, 60)},
        "timing": {"shared_prefix_tokens": 0},
    }
    assert from_native(request, response, {}, "m")["usage"]["input_tokens"] == 60
    response["timing"] = {"shared_prefix_tokens": 40}
    assert from_native(request, response, {}, "m")["usage"]["input_tokens"] == 60


def test_from_native_breaks_a_choice_tie_toward_the_first_option():
    request = wire(
        q={"type": "choice", "instructions": "i", "criteria": {"first": None, "second": None}}
    )
    _, options = to_native(request)
    response = {
        "model": {},
        "answers": {"q": native_answer({"o0": 0.5, "o1": 0.5}, 10)},
        "timing": {},
    }
    answer = from_native(request, response, options, "m")["answers"]["q"]
    assert answer["choice"] == "first"
    assert answer["confidence"] == 0.0
