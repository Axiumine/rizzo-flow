"""Engine (engine.py): what a failure of the model is, and how the engine is closed."""

import math
import re
import threading
from concurrent.futures import Future
from typing import Any

import pytest
from test_service import CharacterTokenizer, FakeBackend

from rizzo_flow.engine import BackendError, Engine

JOIN_SECONDS = 10


def exactly(message: str) -> str:
    """A `match` pattern for pytest.raises that accepts the message and nothing around it."""
    return f"^{re.escape(message)}$"


class CharTokenizer(CharacterTokenizer):
    """The tokenizer of test_service, which keeps the messages of every prompt it renders."""

    def __init__(self) -> None:
        self.rendered: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.rendered.append(messages)
        return self.render(messages)

    def render(self, messages: list[dict[str, str]]) -> str:
        return "\n".join(message["content"] for message in messages) + "\nASSISTANT:"


class StubBackend(FakeBackend):
    """Scores like the FakeBackend of test_service (it favors the second candidate), and keeps a
    record of what it was asked and of the threads that were asked."""

    def __init__(self, tokenizer: Any = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer if tokenizer is not None else CharTokenizer()
        self.calls: list[list[str]] = []
        self.threads: list[threading.Thread] = []

    def logits_for(self, job: Any) -> list[float]:
        return [0.0, 10.0] + [0.0] * (len(job.slots) - 2)

    def score(self, prefix: list[int], jobs: list[Any], mode: str) -> tuple[dict, dict]:
        self.calls.append([job.id for job in jobs])
        self.threads.append(threading.current_thread())
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


# close ------------------------------------------------------------------------------------------


def test_close_accepts_no_more_work_and_leaves_the_inference_thread_alone():
    # The thread that ran the model must live as long as the process: MLX's CUDA backend aborts
    # the process when it exits.
    backend = StubBackend()
    engine = Engine(backend)
    engine.decide(payload())
    (worker,) = set(backend.threads)
    engine.close()
    assert worker.is_alive()
    engine.close()  # a second call has nothing left to wait for
    with pytest.raises(RuntimeError, match=exactly("The engine is closed")):
        engine.decide(payload())
    assert len(backend.calls) == 1


def test_a_closed_engine_refuses_a_request_before_it_compiles_it():
    # Compiling tokenizes with the model, which the caller may have freed by now.
    backend = StubBackend()
    engine = Engine(backend)
    engine.close()
    with pytest.raises(RuntimeError, match=exactly("The engine is closed")):
        engine.decide(payload())
    assert backend.tokenizer.rendered == []
    assert backend.calls == []


@pytest.mark.parametrize("stage", ["compiling", "deciding"])
def test_close_waits_for_the_request_under_way(stage):
    """A caller that frees the model after close() must not do so under a request that is
    running, or that is about to run: one that got past the check that refuses a closed engine."""
    reached, release = threading.Event(), threading.Event()
    journal: list[str] = []

    def hold(where: str) -> None:
        if where == stage:
            reached.set()
            release.wait(JOIN_SECONDS)

    class HeldTokenizer(CharTokenizer):
        def render(self, messages: list[dict[str, str]]) -> str:
            hold("compiling")
            return super().render(messages)

    class HeldBackend(StubBackend):
        def score(self, prefix: list[int], jobs: list[Any], mode: str) -> tuple[dict, dict]:
            hold("deciding")
            journal.append("score:end")
            return super().score(prefix, jobs, mode)

    engine = Engine(HeldBackend(tokenizer=HeldTokenizer()))
    errors: list[BaseException] = []

    def decide() -> None:
        try:
            engine.decide(payload())
        except BaseException as error:  # noqa: BLE001 - reported to the test, not swallowed
            errors.append(error)

    def close() -> None:
        engine.close()
        journal.append("closed")

    deciding = threading.Thread(target=decide)
    closing = threading.Thread(target=close)
    try:
        deciding.start()
        assert reached.wait(JOIN_SECONDS)
        closing.start()
        closing.join(timeout=0.3)
        assert closing.is_alive()  # still waiting for the request
        assert journal == []
    finally:
        release.set()
        deciding.join(timeout=JOIN_SECONDS)
        closing.join(timeout=JOIN_SECONDS)
    assert journal == ["score:end", "closed"]
    assert errors == []  # the request under way was not cut short


def test_close_waits_for_a_decode_whose_caller_gave_up(monkeypatch):
    """Ctrl-C ends the wait of the caller, not the decode on the inference thread."""
    started, finish = threading.Event(), threading.Event()
    journal: list[str] = []

    class SlowBackend(StubBackend):
        def score(self, prefix: list[int], jobs: list[Any], mode: str) -> tuple[dict, dict]:
            started.set()
            finish.wait(JOIN_SECONDS)
            journal.append("score:end")
            return super().score(prefix, jobs, mode)

    def interrupted(self: Future, timeout: float | None = None) -> Any:
        started.wait(JOIN_SECONDS)
        raise KeyboardInterrupt

    engine = Engine(SlowBackend())
    real_result = Future.result
    monkeypatch.setattr(Future, "result", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            engine.decide(payload())
        monkeypatch.setattr(Future, "result", real_result)  # close() waits on a future as well
        threading.Timer(0.3, finish.set).start()
        engine.close()
        journal.append("closed")
        assert journal == ["score:end", "closed"]
    finally:
        finish.set()
