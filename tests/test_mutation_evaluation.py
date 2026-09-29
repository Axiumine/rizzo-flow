"""Mutation tests of `rizzo_flow.evaluation`."""

import pytest

from rizzo_flow.evaluation import evaluate


class Engine:
    """`decide` answers from a function of (request, number of the call)."""

    def __init__(self, respond):
        self.respond = respond
        self.calls = 0

    def decide(self, request):
        self.calls += 1
        return self.respond(request, self.calls)


def reply(probabilities, seconds=0.5, mode="shared", **extra):
    answer = {"status": "ok", "probabilities": probabilities}
    return {"mode": mode, "timing": {"total_seconds": seconds}, "answers": {"q": answer}, **extra}


FIXTURES = [{"id": "f", "request": {"tag": "f"}, "expected": {}}]


def test_the_largest_shift_of_an_answer_is_the_largest_size_of_change_of_any_option():
    """`evaluate` reporting the smallest change of an answer's options (`min` for `max`), or the
    largest signed one (no `abs`). With two options the changes are equal in size and opposite in
    sign, so it takes three, and a largest change that is a fall."""

    def respond(request, call):
        if request.get("mode") == "direct":
            return reply({"a": 0.6, "b": 0.2, "c": 0.2}, mode="direct")
        return reply({"a": 0.3, "b": 0.3, "c": 0.4})

    comparison = evaluate(Engine(respond), FIXTURES, compare_modes=True)["summary"][
        "mode_comparison"
    ]
    assert comparison["decisions"] == 1
    assert comparison["changed_argmaxes"] == 1  # from c to a
    assert comparison["max_probability_delta"] == pytest.approx(0.3)  # a rose; b and c fell


def test_the_report_keeps_the_first_repetition_as_its_evidence():
    """`evaluate` keeping the last repetition instead of the first, which only shows when the
    repetitions differ."""

    def respond(request, call):
        return reply({"a": 0.5, "b": 0.5}, seconds=float(call), call=call)

    row = evaluate(Engine(respond), FIXTURES, repeats=3)["rows"][0]
    assert row["response"]["call"] == 2  # the first call is the warmup
    assert [t["total_seconds"] for t in row["repeat_timings"]] == [2.0, 3.0, 4.0]
