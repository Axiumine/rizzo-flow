"""Shared builders and fakes for the tests of the pure core modules.

Not a test module: pytest collects it because of its name and finds nothing to run.
"""

import re
import threading
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from rizzo_flow.schema import Request


def exactly(message: str) -> str:
    """A `match` pattern for pytest.raises that accepts the message and nothing around it.

    `match` searches, so a plain substring would also accept a message with text added to it.
    """
    return f"^{re.escape(message)}$"


def value_errors(build: Callable[[], Any]) -> list[str]:
    """The messages of the ValueErrors that model validators raised while `build` ran.

    Pydantic wraps them in a ValidationError and prefixes "Value error, "; both are removed so
    that a test can compare the exact text.
    """
    with pytest.raises(ValidationError) as raised:
        build()
    return [error["msg"].removeprefix("Value error, ") for error in raised.value.errors()]


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


def parse_question(kind: str = "boolean", **overrides: Any) -> Any:
    """The validated question object, so that tests exercise the real schema classes."""
    payload = request_payload(q=question_payload(kind, **overrides))
    return Request.model_validate(payload).questions["q"]


class CharTokenizer:
    """One token per character and a chat template that concatenates the messages.

    Deliberately unlike the real Spark tokenizer: it only has to honor the calls
    `compile_request` makes, and it checks their arguments.
    """

    def __init__(self) -> None:
        self.rendered: list[list[dict[str, str]]] = []  # messages of every prompt built

    def encode(self, value: str, **kwargs: Any) -> list[int]:
        assert kwargs == {"add_special_tokens": False}
        return [ord(char) for char in value]

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        self.rendered.append(messages)
        return self.render(messages)

    def render(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(message["content"] for message in messages) + "\nASSISTANT:"


class StubBackend:
    """Scores like the existing FakeBackend (favors the second candidate), and keeps a record."""

    def __init__(
        self,
        tokenizer: Any = None,
        fingerprint: str = "test-fingerprint",
        timing: dict[str, Any] | None = None,
    ) -> None:
        self.tokenizer = tokenizer if tokenizer is not None else CharTokenizer()
        self.metadata = {
            "fingerprint": fingerprint,
            "source": "XHToken/Spark-X2.5-4B",
            "precision": "q8_0",
        }
        self.timing = {"generated_tokens": 0} if timing is None else timing
        self.calls: list[dict[str, Any]] = []
        self.threads: list[threading.Thread] = []

    def logits_for(self, job: Any) -> list[float]:
        return [0.0, 10.0] + [0.0] * (len(job.slots) - 2)

    def score(self, prefix: list[int], jobs: list[Any], mode: str) -> tuple[dict, dict]:
        self.calls.append({"prefix": list(prefix), "ids": [job.id for job in jobs], "mode": mode})
        self.threads.append(threading.current_thread())
        return {job.id: self.logits_for(job) for job in jobs}, dict(self.timing)
