"""Engine (engine.py): what a failure of the model is, and what the request gets wrong instead."""

import math
import re
from typing import Any

import pytest
from test_service import FakeBackend

from rizzo_flow.engine import BackendError, Engine


def exactly(message: str) -> str:
    """A `match` pattern for pytest.raises that accepts the message and nothing around it."""
    return f"^{re.escape(message)}$"


class StubBackend(FakeBackend):
    """Scores like the FakeBackend of test_service (it favors the second candidate), and keeps a
    record of what it was asked."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def logits_for(self, job: Any) -> list[float]:
        return [0.0, 10.0] + [0.0] * (len(job.slots) - 2)

    def score(self, prefix: list[int], jobs: list[Any], mode: str) -> tuple[dict, dict]:
        self.calls.append([job.id for job in jobs])
        return {job.id: self.logits_for(job) for job in jobs}, {"generated_tokens": 0}


def choice(**overrides: Any) -> dict[str, Any]:
    question: dict[str, Any] = {
        "type": "choice",
        "instructions": "Choose a queue",
        "options": [
            {"id": "a", "description": "Option A"},
            {"id": "b", "description": "Option B"},
        ],
    }
    question.update(overrides)
    return question


def payload() -> dict[str, Any]:
    return {
        "state": {"ticket": "Cannot log in"},
        "questions": {
            "supported": {"type": "boolean", "instructions": "Does the user need login help?"},
            "route": choice(),
        },
    }


# a model that fails on a valid request -----------------------------------------------------------


def test_a_backend_that_returns_the_wrong_number_of_logits_is_caught():
    class ShortBackend(StubBackend):
        def logits_for(self, job: Any) -> list[float]:
            return [0.0, 1.0]  # one short for the boolean question, which abstains

    message = "Logit count does not match the declared candidates"
    with pytest.raises(BackendError, match=exactly(message)):
        Engine(ShortBackend()).decide(payload())


def test_a_value_error_of_the_runtime_is_a_backend_error():
    class BrokenBackend(StubBackend):
        def score(self, prefix: list[int], jobs: list[Any], mode: str) -> tuple[dict, dict]:
            raise ValueError("llama_decode returned -3: compute error")

    with pytest.raises(BackendError, match=exactly("llama_decode returned -3: compute error")) as e:
        Engine(BrokenBackend()).decide(payload())
    assert isinstance(e.value, ValueError)  # what callers that catch ValueError still get
    assert isinstance(e.value.__cause__, ValueError)
    assert not isinstance(e.value.__cause__, BackendError)  # the runtime's own error, kept


@pytest.mark.parametrize(
    ("logits", "message"),
    [
        ([math.nan, 1.0, 2.0], "At least two finite logits are required"),
        ([math.inf, 1.0, 2.0], "At least two finite logits are required"),
        ([1.0], "Logit count does not match the declared candidates"),
    ],
    ids=["nan", "inf", "missing-row"],
)
def test_logits_that_cannot_be_decoded_are_a_backend_error(logits, message):
    class OddBackend(StubBackend):
        def logits_for(self, job: Any) -> list[float]:
            return logits

    body = {"state": "Cannot log in", "questions": {"q": choice()}}
    with pytest.raises(BackendError, match=exactly(message)):
        Engine(OddBackend()).decide(body)


def test_what_the_request_gets_wrong_is_not_a_backend_error():
    backend = StubBackend()
    body = payload()
    body["questions"]["route"]["options"][1]["id"] = "a"  # a duplicate option id
    with pytest.raises(ValueError, match="Option IDs must be unique") as invalid:
        Engine(backend).decide(body)
    with pytest.raises(ValueError, match="exceeds the context limit 10") as too_long:
        Engine(backend, ctx=10).decide(payload())
    assert not isinstance(invalid.value, BackendError)
    assert not isinstance(too_long.value, BackendError)
    assert backend.calls == []


def test_a_runtime_error_that_is_not_a_value_error_passes_through_unchanged():
    class CrashingBackend(StubBackend):
        def score(self, prefix: list[int], jobs: list[Any], mode: str) -> tuple[dict, dict]:
            raise RuntimeError("the model fell over")

    with pytest.raises(RuntimeError, match="the model fell over") as raised:
        Engine(CrashingBackend()).decide(payload())
    assert not isinstance(raised.value, ValueError)


def test_after_a_backend_error_the_next_request_is_served():
    class FlakyBackend(StubBackend):
        failures = 1

        def score(self, prefix: list[int], jobs: list[Any], mode: str) -> tuple[dict, dict]:
            if self.failures:
                self.failures -= 1
                raise ValueError("llama_decode returned 1: the context is full")
            return super().score(prefix, jobs, mode)

    engine = Engine(FlakyBackend())
    with pytest.raises(BackendError):
        engine.decide(payload())
    assert engine.decide(payload())["answers"]["supported"]["value"] is True  # lock is free
