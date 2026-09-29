"""Engine (engine.py): validation, calibration binding, the inference thread and the timings."""

import json
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from test_core_support import (
    CharTokenizer,
    StubBackend,
    exactly,
    question_payload,
    request_payload,
    value_errors,
)

from rizzo_flow import engine as engine_module
from rizzo_flow.calibration import Calibration
from rizzo_flow.decisions import softmax
from rizzo_flow.engine import BackendError, Engine
from rizzo_flow.prompts import compile_request
from rizzo_flow.schema import Request

JOIN_SECONDS = 10


def payload() -> dict[str, Any]:
    return request_payload(
        state={"ticket": "Cannot log in"},
        supported=question_payload("boolean", instructions="Does the user need login help?"),
        route=question_payload("choice", instructions="Choose a queue"),
    )


def all_types() -> dict[str, Any]:
    return request_payload(
        state="Cannot log in",
        yes_no=question_payload("boolean"),
        queue=question_payload("choice"),
        mood=question_payload("score"),
        price=question_payload("numeric"),
    )


def calibration(fingerprint: str = "test-fingerprint", **temperatures: float) -> Calibration:
    return Calibration.model_validate(
        {
            "fingerprint": fingerprint,
            "dataset_sha256": "0" * 64,
            "temperatures": temperatures or {"choice": 2.0},
            "fit_metrics": {},
        }
    )


def call_from_threads(engine: Engine, count: int, body: dict[str, Any]) -> list[BaseException]:
    """Run `count` simultaneous decide() calls, each on a thread of its own; return the errors."""
    barrier = threading.Barrier(count)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            barrier.wait(timeout=JOIN_SECONDS)
            engine.decide(body)
        except BaseException as error:  # noqa: BLE001 - reported to the test, not swallowed
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=JOIN_SECONDS)
    assert not any(thread.is_alive() for thread in threads), "a decide() call never returned"
    return errors


# construction -----------------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", [0, -1, -8192])
def test_the_context_limit_must_be_positive(ctx):
    with pytest.raises(ValueError, match=exactly("ctx must be positive")):
        Engine(StubBackend(), ctx=ctx)


def test_a_context_limit_of_one_is_accepted_and_enforced_per_prompt():
    backend = StubBackend()
    engine = Engine(backend, ctx=1)
    assert engine.ctx == 1
    _, jobs = compile_request(backend.tokenizer, Request.model_validate(payload()), 10**6)
    message = f"Question supported: {len(jobs[0].tokens)} tokens exceeds the context limit 1 "
    with pytest.raises(ValueError, match=exactly(message + "(--ctx); no truncation")):
        engine.decide(payload())
    assert backend.calls == []


def test_the_default_context_limit():
    assert Engine(StubBackend()).ctx == 8192


def test_a_calibration_must_belong_to_the_backend_configuration():
    Engine(StubBackend(fingerprint="fp-1"), calibration=calibration("fp-1"))
    message = "Calibration was fitted for a different model/runtime/prompt configuration"
    with pytest.raises(ValueError, match=exactly(message)):
        Engine(StubBackend(fingerprint="fp-2"), calibration=calibration("fp-1"))


def test_without_a_calibration_the_backend_fingerprint_is_not_needed():
    backend = StubBackend()
    del backend.metadata["fingerprint"]
    assert Engine(backend).calibration is None


# the answer -------------------------------------------------------------------------------------


def test_the_response_reports_model_mode_answers_and_prompt_bookkeeping():
    backend = StubBackend()
    engine = Engine(backend, ctx=5000)
    response = engine.decide(payload())
    assert list(response) == ["model", "mode", "answers", "calibration", "timing"]
    assert response["model"] == backend.metadata
    assert response["mode"] == "shared"
    assert response["calibration"] is None
    assert list(response["answers"]) == ["supported", "route"]
    prefix, jobs = compile_request(backend.tokenizer, Request.model_validate(payload()), 5000)
    for job in jobs:
        answer = response["answers"][job.id]
        assert answer["prompt_sha256"] == job.prompt_sha256
        assert answer["input_tokens"] == len(job.tokens)
    assert response["answers"]["supported"]["value"] is True
    assert response["answers"]["route"]["choice"] == "b"
    assert backend.calls == [{"prefix": prefix, "ids": ["supported", "route"], "mode": "shared"}]
    json.dumps(response)  # plain data, ready to serialize


@pytest.mark.parametrize("mode", ["shared", "direct"])
def test_the_requested_mode_reaches_the_backend_and_the_response(mode):
    backend = StubBackend()
    response = Engine(backend).decide({**payload(), "mode": mode})
    assert response["mode"] == mode
    assert backend.calls[0]["mode"] == mode


def test_a_request_object_and_its_dict_give_the_same_answers():
    engine = Engine(StubBackend())
    from_dict = engine.decide(payload())
    from_object = engine.decide(Request.model_validate(payload()))
    assert from_object["answers"] == from_dict["answers"]
    assert from_object["model"] == from_dict["model"]


def test_an_invalid_dict_is_rejected_before_any_model_work():
    backend = StubBackend()
    body = payload()
    body["questions"]["route"]["options"][1]["id"] = "a"  # duplicate option ids
    assert value_errors(lambda: Engine(backend).decide(body)) == ["Option IDs must be unique"]
    assert backend.calls == []


def test_a_request_object_changed_after_validation_is_validated_again():
    request = Request.model_validate(
        request_payload(
            q=question_payload(
                "choice",
                options=[{"id": f"o{i}", "description": f"option {i}"} for i in range(26)],
                policy={"allow_abstain": False},
            )
        )
    )
    backend = StubBackend()
    Engine(backend).decide(request)
    request.questions["q"].policy.allow_abstain = True  # now 27 answer letters would be needed
    message = (
        "At most 25 entries fit here: 26 answer letters, 1 reserved for abstention/out-of-range"
    )
    assert value_errors(lambda: Engine(backend).decide(request)) == [message]
    assert len(backend.calls) == 1


def test_the_callers_request_is_not_modified():
    request = Request.model_validate(payload())
    before = request.model_dump()
    Engine(StubBackend()).decide(request)
    assert request.model_dump() == before


def test_the_context_limit_applies_to_every_prompt_at_the_exact_size():
    backend = StubBackend()
    _, jobs = compile_request(backend.tokenizer, Request.model_validate(payload()), 10**6)
    largest = max(len(job.tokens) for job in jobs)
    Engine(backend, ctx=largest).decide(payload())
    first = next(job for job in jobs if len(job.tokens) == largest)
    message = f"Question {first.id}: {largest} tokens exceeds the context limit {largest - 1} "
    with pytest.raises(ValueError, match=exactly(message + "(--ctx); no truncation")):
        Engine(backend, ctx=largest - 1).decide(payload())


def test_a_backend_that_returns_the_wrong_number_of_logits_is_caught():
    class ShortBackend(StubBackend):
        def logits_for(self, job: Any) -> list[float]:
            return [0.0, 1.0]  # one short for the boolean question, which abstains

    message = "Logit count does not match the declared candidates"
    with pytest.raises(BackendError, match=exactly(message)):
        Engine(ShortBackend()).decide(payload())


def test_an_answer_that_breaks_the_output_contract_never_leaves_the_engine(monkeypatch):
    real = engine_module.decode

    def leaky(question, logits, temperature):
        answer = real(question, logits, temperature)
        answer["probabilities"] = {k: p / 2 for k, p in answer["probabilities"].items()}
        return answer

    monkeypatch.setattr(engine_module, "decode", leaky)
    with pytest.raises(ValueError, match="Output probabilities must sum to one"):
        Engine(StubBackend()).decide(payload())


def test_timings_that_are_not_numbers_do_not_leave_the_engine_either():
    backend = StubBackend(timing={"generated_tokens": 0, "forward_seconds": math.nan})
    with pytest.raises(ValueError, match="forward_seconds"):
        Engine(backend).decide(payload())


# a model that fails on a valid request -----------------------------------------------------------


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

    with pytest.raises(BackendError, match=exactly(message)):
        Engine(OddBackend()).decide(request_payload(q=question_payload("choice")))


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


# calibration in use -----------------------------------------------------------------------------


def test_each_question_type_is_scaled_by_its_own_calibrated_temperature():
    fitted = calibration(boolean=0.5, choice=2.0)  # score and numeric were not calibrated
    backend = StubBackend()
    response = Engine(backend, calibration=fitted).decide(all_types())
    answers = response["answers"]
    assert {name: a["temperature"] for name, a in answers.items()} == {
        "yes_no": 0.5,
        "queue": 2.0,
        "mood": 1.0,
        "price": 1.0,
    }
    scaled = "temperature_scaled_requires_held_out_validation"
    raw = "uncalibrated_conditional_option_scores"
    assert {name: a["probability_status"] for name, a in answers.items()} == {
        "yes_no": scaled,
        "queue": scaled,
        "mood": raw,
        "price": raw,
    }
    assert list(answers["queue"]["probabilities"].values()) == pytest.approx(
        softmax([0.0, 10.0, 0.0], 2.0)
    )
    assert list(answers["mood"]["probabilities"].values()) == pytest.approx(
        softmax([0.0, 10.0, 0.0, 0.0])
    )
    assert answers["queue"]["option_logits"] == {"a": 0.0, "b": 10.0, "__insufficient__": 0.0}
    assert response["calibration"] == fitted.model_dump()


def test_without_a_calibration_every_temperature_is_one():
    answers = Engine(StubBackend()).decide(all_types())["answers"]
    assert {a["temperature"] for a in answers.values()} == {1.0}


# threads and timing -----------------------------------------------------------------------------


def test_inference_always_runs_on_the_same_dedicated_thread():
    backend = StubBackend()
    engine = Engine(backend)
    callers: list[threading.Thread] = []

    def work() -> None:
        callers.append(threading.current_thread())
        engine.decide(payload())

    for _ in range(3):
        thread = threading.Thread(target=work)
        thread.start()
        thread.join(timeout=JOIN_SECONDS)
    engine.decide(payload())
    callers.append(threading.current_thread())
    assert len(backend.threads) == 4
    worker = backend.threads[0]
    assert set(backend.threads) == {worker}
    assert worker.name.startswith("rizzo-inference")
    assert worker not in callers
    assert worker.is_alive()  # it outlives the short-lived threads that asked for the work


def test_the_inference_executor_has_one_worker_and_a_recognizable_name(monkeypatch):
    seen: list[dict[str, Any]] = []

    class RecordingExecutor(ThreadPoolExecutor):
        def __init__(self, max_workers: int | None = None, thread_name_prefix: str = "") -> None:
            seen.append({"max_workers": max_workers, "thread_name_prefix": thread_name_prefix})
            super().__init__(max_workers, thread_name_prefix)

    monkeypatch.setattr(engine_module, "ThreadPoolExecutor", RecordingExecutor)
    Engine(StubBackend())
    assert seen == [{"max_workers": 1, "thread_name_prefix": "rizzo-inference"}]


def test_every_engine_has_its_own_inference_thread():
    first, second = StubBackend(), StubBackend()
    Engine(first).decide(payload())
    Engine(second).decide(payload())
    assert first.threads[0] is not second.threads[0]


def test_concurrent_requests_do_not_overlap_in_the_tokenizer():
    class OverlapProbe(CharTokenizer):
        """Records how many threads are inside the chat template at the same moment."""

        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.most_active = 0
            self._guard = threading.Lock()

        def render(self, messages: list[dict[str, str]]) -> str:
            with self._guard:
                self.active += 1
                self.most_active = max(self.most_active, self.active)
            time.sleep(0.02)
            with self._guard:
                self.active -= 1
            return super().render(messages)

    probe = OverlapProbe()
    engine = Engine(StubBackend(tokenizer=probe))
    assert call_from_threads(engine, 4, payload()) == []
    assert probe.most_active == 1
    assert len(probe.rendered) == 4 * 2  # every thread compiled both of its questions


def test_a_failing_backend_raises_to_the_caller_and_does_not_block_the_next_request():
    class FlakyBackend(StubBackend):
        failures = 1

        def score(self, prefix: list[int], jobs: list[Any], mode: str) -> tuple[dict, dict]:
            if self.failures:
                self.failures -= 1
                raise RuntimeError("the model fell over")
            return super().score(prefix, jobs, mode)

    backend = FlakyBackend()
    engine = Engine(backend)
    with pytest.raises(RuntimeError, match="the model fell over"):
        engine.decide(payload())
    assert call_from_threads(engine, 1, payload()) == []  # the lock was released
    assert len(backend.calls) == 1


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


def test_timings_split_queueing_compilation_and_the_whole_call(monkeypatch):
    ticks = iter([10.0, 10.5, 11.25, 14.0])  # start, lock acquired, compiled, finished
    fake_time = SimpleNamespace(perf_counter=lambda: next(ticks))
    monkeypatch.setattr(engine_module, "time", fake_time)
    backend = StubBackend(timing={"generated_tokens": 0, "forward_seconds": 0.75})
    timing = Engine(backend).decide(payload())["timing"]
    assert timing == {
        "generated_tokens": 0,
        "forward_seconds": 0.75,
        "queue_seconds": 0.5,
        "compile_seconds": 0.75,
        "total_seconds": 4.0,
    }


def test_the_engines_own_timings_take_precedence_over_the_backends(monkeypatch):
    ticks = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(engine_module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    backend = StubBackend(timing={"total_seconds": 99.0, "queue_seconds": 98.0})
    timing = Engine(backend).decide(payload())["timing"]
    assert timing["total_seconds"] == 3.0
    assert timing["queue_seconds"] == 1.0
    assert timing["compile_seconds"] == 1.0


def test_real_timings_are_nonnegative_and_add_up():
    timing = Engine(StubBackend()).decide(payload())["timing"]
    assert timing["queue_seconds"] >= 0
    assert timing["compile_seconds"] >= 0
    assert timing["total_seconds"] + 1e-9 >= timing["queue_seconds"] + timing["compile_seconds"]
