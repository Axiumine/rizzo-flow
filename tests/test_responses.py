"""Output contract (responses.py): the response is re-validated before it leaves the service."""

import copy
import math
from functools import partial
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from test_core_support import parse_question, value_errors

from rizzo_flow.decisions import decode
from rizzo_flow.responses import (
    Answer,
    BooleanAnswer,
    ChoiceAnswer,
    NumericAnswer,
    Response,
    ScoreAnswer,
    Statistics,
    TypedAnswer,
    Uncertainty,
)

CLASSES: dict[str, type[Answer]] = {
    "boolean": BooleanAnswer,
    "choice": ChoiceAnswer,
    "score": ScoreAnswer,
    "numeric": NumericAnswer,
}
PRIMARY = {"boolean": "value", "choice": "choice", "score": "score", "numeric": "value"}
LOGITS = {
    "boolean": [0.0, 2.0],
    "choice": [0.0, 2.0],
    "score": [0.0, 1.0, 2.0],
    "numeric": [0.0, 2.0, 0.0, 0.0],
}
KINDS = list(CLASSES)
SUM_MESSAGE = "Output probabilities must sum to one"
MAPPINGS_MESSAGE = "Output candidate mappings must agree"
PRIMARY_MESSAGE = "Only an accepted decision may have a non-null primary value"


def answer(kind: str) -> dict[str, Any]:
    """What the engine hands to the validator: decode's result plus the prompt bookkeeping."""
    question = parse_question(kind, policy={"allow_abstain": False})
    payload = decode(question, LOGITS[kind])
    assert payload["status"] == "ok"
    payload["prompt_sha256"] = "0" * 64
    payload["input_tokens"] = 12
    return payload


def rejected(kind: str, payload: dict[str, Any], message: str) -> None:
    """The answer is refused with one of pydantic's own messages (a substring of it)."""
    with pytest.raises(ValidationError, match=message):
        CLASSES[kind].model_validate(payload)


def refused(kind: str, payload: dict[str, Any], message: str) -> None:
    """The answer is refused by our validator, with exactly this message and no other error."""
    assert value_errors(partial(CLASSES[kind].model_validate, payload)) == [message]


# what decode produces is valid --------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_decode_results_are_valid_answers_of_their_own_class(kind):
    payload = answer(kind)
    validated = CLASSES[kind].model_validate(payload)
    assert validated.model_dump() == payload
    selected: Any = TypeAdapter(TypedAnswer).validate_python(payload)
    assert type(selected) is CLASSES[kind]


def test_an_unknown_answer_type_is_refused():
    payload = answer("boolean")
    payload["type"] = "text"
    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        TypeAdapter(TypedAnswer).validate_python(payload)


@pytest.mark.parametrize("kind", KINDS)
def test_answers_refuse_unknown_fields(kind):
    payload = answer(kind)
    payload["surprise"] = 1
    rejected(kind, payload, "Extra inputs are not permitted")


# probabilities must be a distribution --------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_probabilities_must_sum_to_one_within_a_millionth(kind):
    first = next(iter(answer(kind)["probabilities"]))
    for drift in (5e-7, -5e-7):
        payload = answer(kind)
        payload["probabilities"][first] += drift
        CLASSES[kind].model_validate(payload)
    for drift in (2e-6, -2e-6, 0.5):
        payload = answer(kind)
        payload["probabilities"][first] += drift
        refused(kind, payload, SUM_MESSAGE)


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf])
def test_each_probability_is_a_number_in_the_unit_interval(bad):
    payload = answer("choice")
    payload["probabilities"]["a"] = bad
    with pytest.raises(ValidationError):
        ChoiceAnswer.model_validate(payload)


def test_probabilities_and_option_logits_must_name_the_same_candidates():
    payload = answer("choice")
    payload["option_logits"]["z"] = payload["option_logits"].pop("a")
    refused("choice", payload, MAPPINGS_MESSAGE)


def test_probabilities_and_legend_must_name_the_same_candidates():
    payload = answer("choice")
    payload["legend"]["z"] = payload["legend"].pop("b")
    refused("choice", payload, MAPPINGS_MESSAGE)


def test_a_candidate_missing_from_a_mapping_is_refused():
    for mapping in ("option_logits", "legend"):
        payload = answer("choice")
        del payload[mapping]["a"]
        refused("choice", payload, MAPPINGS_MESSAGE)


def test_an_extra_candidate_in_a_mapping_is_refused():
    for mapping in ("option_logits", "legend"):
        payload = answer("choice")
        payload[mapping]["extra"] = 0.0 if mapping == "option_logits" else "extra"
        refused("choice", payload, MAPPINGS_MESSAGE)


# only an accepted decision has a value ---------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_an_accepted_decision_needs_a_primary_value(kind):
    payload = answer(kind)
    payload[PRIMARY[kind]] = None
    refused(kind, payload, PRIMARY_MESSAGE)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("status", ["insufficient_evidence", "out_of_range", "uncertain"])
def test_a_rejected_decision_must_not_carry_a_primary_value(kind, status):
    payload = answer(kind)
    payload["status"] = status
    refused(kind, payload, PRIMARY_MESSAGE)
    payload[PRIMARY[kind]] = None
    CLASSES[kind].model_validate(payload)


def test_status_is_one_of_the_four_outcomes():
    payload = answer("choice")
    payload["status"] = "maybe"
    payload["choice"] = None
    with pytest.raises(ValidationError):
        ChoiceAnswer.model_validate(payload)


# field constraints ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("temperature",), 0),
        (("temperature",), -1.0),
        (("temperature",), math.nan),
        (("input_tokens",), 0),
        (("input_tokens",), -3),
        (("uncertainty", "entropy_nats"), -0.1),
        (("uncertainty", "top_probability"), 1.1),
        (("uncertainty", "concentration"), -0.1),
        (("uncertainty", "concentration"), 1.1),
        (("uncertainty", "unavailable_probability"), 1.1),
        (("option_logits", "a"), math.nan),
        (("option_logits", "a"), math.inf),
        (("probability_status",), "calibrated"),
        (("prompt_sha256",), None),
    ],
)
def test_scalar_fields_are_constrained(path, value):
    payload = answer("choice")
    target = payload
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        ChoiceAnswer.model_validate(payload)


def test_statistics_have_a_nonnegative_spread_and_only_the_two_quantiles():
    good = {"mean": 1.0, "stddev": 0.0, "median": 1.0, "anchor_quantiles": {"p10": 0, "p90": 2}}
    assert Statistics.model_validate(good).anchor_quantiles == {"p10": 0, "p90": 2}
    for bad in (
        {**good, "stddev": -0.1},
        {**good, "mean": math.nan},
        {**good, "anchor_quantiles": {"p10": 0, "p50": 2}},
        {**good, "median": math.inf},
    ):
        with pytest.raises(ValidationError):
            Statistics.model_validate(bad)


def test_uncertainty_accepts_the_boundaries():
    Uncertainty.model_validate(
        {
            "top_probability": 1.0,
            "entropy_nats": 0.0,
            "concentration": 0.0,
            "unavailable_probability": 0.0,
        }
    )


@pytest.mark.parametrize("support", [[1.0], [1.0, 2.0, 3.0]])
def test_support_has_exactly_two_bounds(support):
    payload = answer("score")
    payload["support"] = support
    rejected("score", payload, "List should have")


def test_numeric_range_probabilities_are_named_below_and_above():
    payload = answer("numeric")
    payload["range_probabilities"] = {"under": 0.1, "above": 0.1}
    rejected("numeric", payload, "range_probabilities")


# the response envelope -------------------------------------------------------------------------


def envelope(**answers: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": {"fingerprint": "fp"},
        "mode": "shared",
        "answers": answers or {"a": answer("boolean")},
        "calibration": None,
        "timing": {"generated_tokens": 0, "total_seconds": 0.25},
    }


def test_response_round_trips_through_validation_and_dump():
    body = envelope(a=answer("boolean"), b=answer("choice"), c=answer("score"), d=answer("numeric"))
    dumped = Response.model_validate(copy.deepcopy(body)).model_dump()
    assert dumped == body
    assert [a["type"] for a in dumped["answers"].values()] == list(CLASSES)


@pytest.mark.parametrize("field", ["model", "mode", "answers", "calibration", "timing"])
def test_response_fields_are_all_required(field):
    body = envelope()
    del body[field]
    with pytest.raises(ValidationError, match="Field required"):
        Response.model_validate(body)


def test_response_mode_and_extra_fields():
    for mode in ("fast", None, "Shared"):
        with pytest.raises(ValidationError):
            Response.model_validate({**envelope(), "mode": mode})
    Response.model_validate({**envelope(), "mode": "direct"})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Response.model_validate({**envelope(), "extra": 1})


def test_response_calibration_is_a_mapping_or_null():
    assert Response.model_validate({**envelope(), "calibration": {"a": 1}}).calibration == {"a": 1}
    with pytest.raises(ValidationError):
        Response.model_validate({**envelope(), "calibration": "fitted"})


def test_response_timing_holds_finite_numbers_only():
    Response.model_validate({**envelope(), "timing": {"tokens": 3, "seconds": 0.5}})
    for timing in ({"seconds": math.nan}, {"seconds": math.inf}, {"seconds": "slow"}):
        with pytest.raises(ValidationError):
            Response.model_validate({**envelope(), "timing": timing})


def test_response_validates_every_answer():
    broken = answer("boolean")
    broken["probabilities"]["true"] = 0.0
    assert value_errors(
        lambda: Response.model_validate(envelope(good=answer("choice"), broken=broken))
    ) == [SUM_MESSAGE]
