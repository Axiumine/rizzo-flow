"""Benchmark reports (evaluation.py): what `evaluate` refuses before it runs anything.

The engine is a scripted stand-in that records every request it gets, so a test can tell that a
fixture was refused before the warmup call, which is before any model time was spent.
"""

import copy
import math
import re
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from rizzo_flow.evaluation import check_fixtures, check_requests, evaluate
from rizzo_flow.schema import require_unicode

Reply = dict[str, Any]


def exactly(message: str) -> str:
    """A `match` pattern for pytest.raises that accepts the message and nothing around it."""
    return f"^{re.escape(message)}$"


class ScriptedEngine:
    """Answers `decide` from a function of (request, call number) and records every request."""

    def __init__(self, respond: Callable[[dict[str, Any], int], Reply]) -> None:
        self.respond = respond
        self.calls: list[dict[str, Any]] = []

    def decide(self, request: dict[str, Any]) -> Reply:
        self.calls.append(copy.deepcopy(request))
        return copy.deepcopy(self.respond(request, len(self.calls)))


def reply(answers: dict[str, Reply], seconds: float, mode: str = "shared") -> Reply:
    return {"mode": mode, "timing": {"total_seconds": seconds}, "answers": answers}


def pick(probabilities: dict[str, float], status: str = "ok") -> Reply:
    return {"type": "choice", "status": status, "probabilities": probabilities}


def money(value: float | None) -> Reply:
    return {
        "type": "numeric",
        "status": "ok" if value is not None else "insufficient_evidence",
        "probabilities": {"0": 1.0},
        "unit": "EUR",
        "support": [50, 150],
        "value": value,
    }


def fixture(name: str, **expected: dict[str, Any]) -> dict[str, Any]:
    return {"id": name, "request": {"tag": name}, "expected": expected}


# what the engine could not score ----------------------------------------------------------------


def test_an_expected_question_that_was_not_answered_is_reported_with_its_fixture():
    bad = [fixture("odd", q1={"label": "yes"}, q9={"label": "yes"})]
    engine = ScriptedEngine(lambda request, call: reply({"q1": pick({"yes": 0.6, "no": 0.4})}, 0.1))
    with pytest.raises(ValueError, match=exactly("Unknown expected question q9 in fixture odd")):
        evaluate(engine, bad)


@pytest.mark.parametrize(
    "target", ["100", None, math.nan, math.inf, -math.inf, [1], 10**400, -(10**400), 1e101, -1e101]
)
def test_numeric_targets_must_be_finite_numbers(target):
    # Bounded like the anchors (1e100): a bigger int does not even convert to a float.
    bad = [fixture("fine", money={"value": 100}), fixture("bad", money={"value": target})]
    engine = ScriptedEngine(lambda request, call: reply({"money": money(100.0)}, 0.1))
    with pytest.raises(ValueError, match=exactly("Expected numeric targets must be finite")):
        evaluate(engine, bad)
    assert engine.calls == []  # not even the warmup


@pytest.mark.parametrize("target", [0, -3, 100.5, 1e100, -1e100, 10**100, True])
def test_numeric_targets_within_the_bound_of_the_anchors_are_accepted(target):
    assert check_fixtures([fixture("ok", money={"value": target})]) is None


# fixtures that cannot be evaluated ---------------------------------------------------------------

NOT_AN_OBJECT = "is not a JSON object"


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (None, f"Fixture 2 {NOT_AN_OBJECT}"),
        ([1], f"Fixture 2 {NOT_AN_OBJECT}"),
        ("text", f"Fixture 2 {NOT_AN_OBJECT}"),
        ({}, 'Fixture 2 has no "id"'),
        ({"request": {}}, 'Fixture 2 has no "id"'),
        ({"id": "b"}, 'Fixture 2 has no "request"'),
        ({"id": "b", "request": None}, f'Fixture 2: "request" {NOT_AN_OBJECT}'),
        ({"id": "b", "request": "text"}, f'Fixture 2: "request" {NOT_AN_OBJECT}'),
        ({"id": "b", "request": [{}]}, f'Fixture 2: "request" {NOT_AN_OBJECT}'),
        ({"id": "b", "request": {}, "expected": None}, f'Fixture 2: "expected" {NOT_AN_OBJECT}'),
        ({"id": "b", "request": {}, "expected": [{}]}, f'Fixture 2: "expected" {NOT_AN_OBJECT}'),
        (
            {"id": "b", "request": {}, "expected": {"q1": "yes"}},
            f"Fixture 2: expected q1 {NOT_AN_OBJECT}",
        ),
        (
            {"id": "b", "request": {}, "expected": {"q1": None}},
            f"Fixture 2: expected q1 {NOT_AN_OBJECT}",
        ),
    ],
)
def test_a_malformed_fixture_is_refused_before_anything_runs(bad, message):
    engine = ScriptedEngine(lambda request, call: reply({}, 0.1))
    with pytest.raises(ValueError, match=exactly(message)):
        evaluate(engine, [fixture("fine"), bad])  # the second one: every fixture is looked at
    assert engine.calls == []  # not even the warmup


def unicode_refusal() -> str:
    """What `require_unicode` says about a lone surrogate."""
    with pytest.raises(ValueError, match="lone surrogate") as refused:
        require_unicode("\ud800")
    return str(refused.value)


@pytest.mark.parametrize(
    "spoil",
    [
        lambda row: {**row, "id": "bad\ud800"},
        lambda row: {**row, "expected": {"q1": {"label": "yes\ud800"}}},
        lambda row: {**row, "expected": {"q1\ud800": {"label": "yes"}}},
        lambda row: {**row, "request": {"state": {"deep": ["x", {"k": "\ud800"}]}}},
    ],
    ids=["id", "expected-label", "expected-question", "request"],
)
def test_text_that_utf8_cannot_encode_is_refused_before_anything_runs(spoil):
    # It used to fail when the report was hashed, after every decision.
    engine = ScriptedEngine(lambda request, call: reply({}, 0.1))
    message = exactly(f"Fixture 2: {unicode_refusal()}")  # by position: the text is the culprit
    with pytest.raises(ValueError, match=message):
        evaluate(engine, [fixture("fine"), spoil(fixture("bad"))])
    assert engine.calls == []


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "spoil",
    [
        lambda row, number: {**row, "id": number},
        lambda row, number: {**row, "expected": {"q1": {"note": number}}},
        lambda row, number: {**row, "request": {"state": [{"deep": [number]}]}},
    ],
    ids=["id", "expected", "request"],
)
def test_nan_and_infinity_in_a_fixture_are_refused_before_anything_runs(spoil, number):
    # JSON has neither, json.loads takes both, and the hash of the report cannot: it used to
    # fail there, after every decision.
    engine = ScriptedEngine(lambda request, call: reply({}, 0.1))
    with pytest.raises(ValueError, match=r"^Fixture 2: .*not JSON compliant") as refused:
        evaluate(engine, [fixture("fine"), spoil(fixture("bad"), number)])
    assert engine.calls == []
    assert refused.value.__cause__ is None


def test_the_culprit_of_a_lone_surrogate_is_named_by_position_not_by_text():
    with pytest.raises(ValueError, match=r"^Fixture 1: ") as refused:
        check_fixtures([{"id": "culprit\ud800", "request": {}}])
    assert "culprit" not in str(refused.value)
    assert refused.value.__cause__ is None  # the codec's own error would repeat the text


def test_check_fixtures_accepts_what_evaluate_can_use():
    usable = [fixture("a", q1={"label": "yes"}), {"id": 7, "request": {}}, fixture("b")]
    assert check_fixtures(usable, repeats=3) is None
    assert check_fixtures(usable) is None


# requests and expectations the engine would refuse -----------------------------------------------

REQUEST = {
    "state": "Cannot log in",
    "questions": {"flag": {"type": "boolean", "instructions": "Is it urgent?"}},
}


def expecting(question: str, **expected: Any) -> list[dict[str, Any]]:
    return [{"id": "odd", "request": REQUEST, "expected": {question: expected}}]


@pytest.mark.parametrize("label", ["maybe", 3, None, ["true"], {"true": 1}, [[]]])
def test_check_requests_refuses_a_label_the_question_has_not_whatever_it_is(label):
    # A list or an object is not hashable: still an unknown label, not a TypeError.
    message = f"Unknown expected label {label} in fixture odd"
    with pytest.raises(ValueError, match=exactly(message)):
        check_requests(expecting("flag", label=label))


@pytest.mark.parametrize("expected", [{"label": "true"}, {"status": "ok"}, {}])
def test_check_requests_refuses_a_question_the_request_does_not_ask(expected):
    with pytest.raises(ValueError, match=exactly("Unknown expected question ghost in fixture odd")):
        check_requests(expecting("ghost", **expected))


def test_check_requests_names_the_fixture_whose_request_is_refused():
    broken = {"id": "broken", "request": {"state": "x", "questions": {}}}
    # What pydantic says about the request follows the name, over several lines.
    with pytest.raises(ValueError, match=r"(?s)^Fixture broken: .*questions") as refused:
        check_requests([{"id": "fine", "request": REQUEST}, broken])
    assert isinstance(refused.value.__cause__, ValidationError)
