"""Benchmark reports (evaluation.py): metrics, warmup, repeats and the mode comparison.

The engine is a scripted stand-in, so every latency and probability is exact and the expected
statistics can be worked out by hand.
"""

import copy
import hashlib
import json
import math
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError
from test_core_support import StubBackend, exactly, question_payload, request_payload

from rizzo_flow.engine import Engine
from rizzo_flow.evaluation import check_fixtures, check_requests, evaluate
from rizzo_flow.schema import require_unicode

Reply = dict[str, Any]
WARMUP_SECONDS = 100.0  # a decoy: the warmup call must never reach the timings


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


def fixture(name: str, **expected: dict[str, Any]) -> dict[str, Any]:
    return {"id": name, "request": {"tag": name}, "expected": expected}


def run(fixtures: list[dict[str, Any]], respond: Callable, **options: Any) -> tuple[Reply, Any]:
    engine = ScriptedEngine(respond)
    return evaluate(engine, fixtures, **options), engine


# categorical metrics ---------------------------------------------------------------------------

CATEGORICAL = [
    fixture("f1", q1={"label": "yes", "status": "ok"}, q2={"label": "b"}),
    fixture("f2", q1={"label": "yes", "status": "insufficient_evidence"}),
    fixture("f3", q1={"label": "no"}),
    fixture("f4", q1={"label": "no"}),
]
CATEGORICAL_REPLIES = {
    "f1": reply(
        {
            "q1": pick({"yes": 0.95, "no": 0.05}),
            "q2": pick({"a": 0.15, "b": 0.85}),
        },
        seconds=0.2,
    ),
    "f2": reply({"q1": pick({"yes": 0.65, "no": 0.35}, status="uncertain")}, seconds=0.4),
    "f3": reply({"q1": pick({"yes": 0.75, "no": 0.25})}, seconds=0.6),
    "f4": reply({"q1": pick({"yes": 1.0, "no": 0.0})}, seconds=0.8),
}


def answer_categorical(request: dict[str, Any], call: int) -> Reply:
    if call == 1:
        return reply(CATEGORICAL_REPLIES["f1"]["answers"], seconds=WARMUP_SECONDS)
    return CATEGORICAL_REPLIES[request["tag"]]


@pytest.fixture
def categorical() -> tuple[Reply, ScriptedEngine]:
    return run(copy.deepcopy(CATEGORICAL), answer_categorical)


def test_the_warmup_is_the_first_request_and_stays_out_of_the_timings(categorical):
    report, engine = categorical
    assert [call["tag"] for call in engine.calls] == ["f1", "f1", "f2", "f3", "f4"]
    summary = report["summary"]
    assert summary["warmup_excluded"] is True
    assert summary["requests"] == 4
    assert summary["repeats"] == 1
    assert summary["latency_seconds"] == pytest.approx({"median": 0.5, "p95": 0.8})
    assert summary["decisions_per_second"] == pytest.approx(5 / 2.0)


def test_coverage_and_status_accuracy_count_every_expected_decision(categorical):
    summary = categorical[0]["summary"]
    assert summary["labeled_decision_coverage"] == pytest.approx(4 / 5)  # f2.q1 was not accepted
    assert summary["status_accuracy"] == pytest.approx(1 / 2)


def test_categorical_accuracy_and_proper_scoring_rules(categorical):
    stats = categorical[0]["summary"]["categorical"]
    assert stats["rows"] == 5
    assert stats["accuracy"] == pytest.approx(3 / 5)
    assert stats["accepted_accuracy"] == pytest.approx(1 / 2)  # f2.q1 is right but not accepted
    losses = [-math.log(p) for p in (0.95, 0.85, 0.65, 0.25)] + [-math.log(1e-300)]
    assert stats["nll"] == pytest.approx(sum(losses) / 5)  # a zero probability is floored
    assert stats["brier"] == pytest.approx((0.005 + 0.045 + 0.245 + 1.125 + 2.0) / 5)


def test_reliability_bins_and_expected_calibration_error(categorical):
    stats = categorical[0]["summary"]["categorical"]
    assert stats["reliability_bins"] == [
        pytest.approx({"lower": 0.6, "count": 1, "accuracy": 1, "mean_top_probability": 0.65}),
        pytest.approx({"lower": 0.7, "count": 1, "accuracy": 0, "mean_top_probability": 0.75}),
        pytest.approx({"lower": 0.8, "count": 1, "accuracy": 1, "mean_top_probability": 0.85}),
        pytest.approx({"lower": 0.9, "count": 2, "accuracy": 0.5, "mean_top_probability": 0.975}),
    ]
    # A probability of exactly 1.0 lands in the last bin, not in a bin 10.
    assert stats["ece_10_bins"] == pytest.approx((0.35 + 0.75 + 0.15) / 5 + 2 / 5 * 0.475)


def test_the_report_keeps_the_raw_evidence_and_a_dataset_hash(categorical):
    report, _ = categorical
    canonical = json.dumps(
        CATEGORICAL, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert report["dataset_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert [row["id"] for row in report["rows"]] == ["f1", "f2", "f3", "f4"]
    first = report["rows"][0]
    assert set(first) == {"id", "expected", "response", "repeat_timings"}
    assert first["expected"] == CATEGORICAL[0]["expected"]
    assert first["response"] == CATEGORICAL_REPLIES["f1"]
    assert first["repeat_timings"] == [{"total_seconds": 0.2}]
    assert report["summary"]["mode_comparison"] is None
    assert set(report) == {"dataset_sha256", "summary", "rows"}


def test_evaluation_does_not_modify_its_fixtures():
    fixtures = copy.deepcopy(CATEGORICAL)
    run(fixtures, answer_categorical, compare_modes=True)
    assert fixtures == CATEGORICAL


def test_an_unknown_label_is_reported_with_its_fixture():
    bad = [fixture("broken", q1={"label": "maybe"})]
    engine = ScriptedEngine(lambda request, call: reply({"q1": pick({"yes": 0.6, "no": 0.4})}, 0.1))
    with pytest.raises(ValueError, match=exactly("Unknown expected label maybe in fixture broken")):
        evaluate(engine, bad)


def test_an_expected_question_that_was_not_answered_is_reported_with_its_fixture():
    bad = [fixture("odd", q1={"label": "yes"}, q9={"label": "yes"})]
    engine = ScriptedEngine(lambda request, call: reply({"q1": pick({"yes": 0.6, "no": 0.4})}, 0.1))
    with pytest.raises(ValueError, match=exactly("Unknown expected question q9 in fixture odd")):
        evaluate(engine, bad)


def test_status_only_expectations_do_not_produce_categorical_rows():
    only_status = [fixture("s", q1={"status": "uncertain"})]
    report, _ = run(
        only_status, lambda request, call: reply({"q1": pick({"a": 1.0, "b": 0.0})}, 0.1)
    )
    summary = report["summary"]
    assert summary["status_accuracy"] == 0
    assert summary["labeled_decision_coverage"] == 1
    assert summary["categorical"]["rows"] == 0
    assert summary["numeric"]["rows"] == 0


def test_without_expectations_only_the_timings_are_reported():
    bare = [{"id": "bare", "request": {"tag": "bare"}}]
    report, _ = run(bare, lambda request, call: reply({"q1": pick({"a": 0.5, "b": 0.5})}, 0.25))
    summary = report["summary"]
    assert report["rows"][0]["expected"] == {}
    assert summary["labeled_decision_coverage"] is None
    assert summary["status_accuracy"] is None
    assert summary["categorical"] == {
        "rows": 0,
        "accuracy": None,
        "accepted_accuracy": None,
        "nll": None,
        "brier": None,
        "ece_10_bins": None,
        "reliability_bins": [],
    }
    assert summary["numeric"] == {"rows": 0, "by_type_unit_and_support": {}}
    assert summary["decisions_per_second"] == pytest.approx(1 / 0.25)


# numeric targets ---------------------------------------------------------------------------------


def money(
    value: float | None, unit: str = "EUR", support: tuple[float, float] = (50, 150)
) -> Reply:
    status = "ok" if value is not None else "insufficient_evidence"
    return {
        "type": "numeric",
        "status": status,
        "probabilities": {"0": 1.0},
        "unit": unit,
        "support": list(support),
        "value": value,
    }


def mood(score: float) -> Reply:
    return {
        "type": "score",
        "status": "ok",
        "probabilities": {"0": 1.0},
        "support": [0, 3],
        "score": score,
    }


NUMERIC = [
    fixture("g1", mood={"value": 2}, money={"value": 100}),
    fixture("g2", money={"value": 100}),
    fixture("g3", money={"value": 100.0}),
    fixture("g4", money={"value": 10.0}),
]
NUMERIC_REPLIES = {
    "g1": {"mood": mood(1.5), "money": money(130.0)},
    "g2": {"money": money(None)},
    "g3": {"money": money(90)},
    "g4": {"money": money(None, unit="USD", support=(0, 20))},
}


def answer_numeric(request: dict[str, Any], call: int) -> Reply:
    return reply(NUMERIC_REPLIES[request["tag"]], seconds=0.5)


def test_numeric_errors_are_grouped_by_type_unit_and_support():
    report, _ = run(NUMERIC, answer_numeric)
    numeric = report["summary"]["numeric"]
    assert numeric["rows"] == 5
    groups = numeric["by_type_unit_and_support"]
    score_group = '{"support":[0,3],"type":"score","unit":null}'
    eur_group = '{"support":[50,150],"type":"numeric","unit":"EUR"}'
    usd_group = '{"support":[0,20],"type":"numeric","unit":"USD"}'
    assert list(groups) == [score_group, eur_group, usd_group]
    assert groups[score_group] == {
        "rows": 1,
        "answered": 1,
        "mae_on_answered": 0.5,
        "rmse_on_answered": 0.5,
    }
    # Errors +30 and -10 on the two answered rows; the abstention counts as a row only.
    assert groups[eur_group] == {
        "rows": 3,
        "answered": 2,
        "mae_on_answered": pytest.approx(20.0),
        "rmse_on_answered": pytest.approx(math.sqrt(500)),
    }
    assert groups[usd_group] == {
        "rows": 1,
        "answered": 0,
        "mae_on_answered": None,
        "rmse_on_answered": None,
    }
    assert report["summary"]["labeled_decision_coverage"] == pytest.approx(3 / 5)


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


# repeats and the mode comparison ---------------------------------------------------------------

COMPARISON = [fixture("h1", q1={"label": "yes"}), fixture("h2", q1={"label": "yes"})]
SHARED = {
    "h1": ({"yes": 0.75, "no": 0.25}, (0.1, 0.3)),
    "h2": ({"yes": 0.6, "no": 0.4}, (0.5, 0.7)),
}
DIRECT = {"h1": ({"yes": 0.7, "no": 0.3}, 0.9), "h2": ({"yes": 0.4, "no": 0.6}, 1.1)}


def answer_both_modes(request: dict[str, Any], call: int) -> Reply:
    tag = request["tag"]
    if request.get("mode") == "direct":
        probabilities, seconds = DIRECT[tag]
        return reply({"q1": pick(probabilities)}, seconds, mode="direct")
    probabilities, times = SHARED[tag]
    # Calls: warmup, then h1 twice, its direct twin, then h2 twice, its direct twin.
    repeat = {2: 0, 3: 1, 5: 0, 6: 1}.get(call, 0)
    return reply({"q1": pick(probabilities)}, times[repeat])


def test_repeats_and_the_mode_comparison_are_separate_measurements():
    report, engine = run(COMPARISON, answer_both_modes, repeats=2, compare_modes=True)
    assert [(c["tag"], c.get("mode")) for c in engine.calls] == [
        ("h1", None),  # warmup
        ("h1", None),
        ("h1", None),
        ("h1", "direct"),
        ("h2", None),
        ("h2", None),
        ("h2", "direct"),
    ]
    summary = report["summary"]
    assert summary["repeats"] == 2
    # Only the four repeated requests are timed; the direct twins are not.
    assert summary["latency_seconds"] == pytest.approx({"median": 0.4, "p95": 0.7})
    assert summary["decisions_per_second"] == pytest.approx(4 / 1.6)
    assert [r["repeat_timings"] for r in report["rows"]] == [
        [{"total_seconds": 0.1}, {"total_seconds": 0.3}],
        [{"total_seconds": 0.5}, {"total_seconds": 0.7}],
    ]


def test_the_mode_comparison_reports_decisions_and_the_median_time_of_each_mode():
    report, _ = run(COMPARISON, answer_both_modes, repeats=2, compare_modes=True)
    comparison = report["summary"]["mode_comparison"]
    assert comparison["decisions"] == 2
    assert comparison["changed_argmaxes"] == 1  # h2 flips from yes to no
    assert comparison["max_probability_delta"] == pytest.approx(0.2)
    assert comparison["median_seconds"] == pytest.approx({"shared": 0.4, "direct": 1.0})
    first = report["rows"][0]
    assert first["alternate_mode_response"]["mode"] == "direct"
    assert first["alternate_mode_response"]["answers"]["q1"]["probabilities"] == DIRECT["h1"][0]
    assert first["response"]["answers"]["q1"]["probabilities"] == SHARED["h1"][0]


# Per question: the probabilities of yes and no in shared mode, then in direct mode.
FLIPS = {
    "q1": ((0.3, 0.7), (0.6, 0.4)),  # no -> yes: the argmax changes
    "q2": ((0.2, 0.8), (0.55, 0.45)),  # no -> yes: the argmax changes
    "q3": ((0.4, 0.6), (0.45, 0.55)),  # no -> no
    "q4": ((0.9, 0.1), (0.85, 0.15)),  # yes -> yes
    "q5": ((0.7, 0.3), (0.7, 0.3)),  # yes -> yes, nothing moved
}


def answer_flips(request: dict[str, Any], call: int) -> Reply:
    mode = request.get("mode", "shared")
    index = 1 if mode == "direct" else 0
    answers = {
        name: pick({"yes": pair[index][0], "no": pair[index][1]}) for name, pair in FLIPS.items()
    }
    return reply(answers, 0.5, mode=mode)


def test_the_mode_comparison_counts_every_argmax_change_and_the_largest_shift():
    fixtures = [fixture("flips", **{name: {"label": "yes"} for name in FLIPS})]
    report, _ = run(fixtures, answer_flips, compare_modes=True)
    comparison = report["summary"]["mode_comparison"]
    assert comparison["decisions"] == 5
    assert comparison["changed_argmaxes"] == 2
    assert comparison["max_probability_delta"] == pytest.approx(0.35)


def test_the_alternate_request_is_an_independent_copy_of_the_fixture():
    request = {"tag": "nested", "state": {"seen": []}}
    fixtures = [{"id": "nested", "request": request, "expected": {}}]

    def scribble_on_the_twin(received: dict[str, Any], call: int) -> Reply:
        if "mode" in received:
            received["state"]["seen"].append(call)
        return reply({"q1": pick({"a": 0.5, "b": 0.5})}, 0.5)

    run(fixtures, scribble_on_the_twin, compare_modes=True)
    assert request == {"tag": "nested", "state": {"seen": []}}


def test_the_alternate_mode_is_the_opposite_of_the_one_that_answered():
    def answer_direct_first(request: dict[str, Any], call: int) -> Reply:
        mode = "shared" if request.get("mode") == "shared" else "direct"
        return reply({"q1": pick({"yes": 0.6, "no": 0.4})}, 0.5, mode=mode)

    report, engine = run(COMPARISON[:1], answer_direct_first, compare_modes=True)
    assert engine.calls[-1]["mode"] == "shared"
    assert report["summary"]["mode_comparison"]["median_seconds"] == {"shared": 0.5, "direct": 0.5}


def test_a_comparison_without_decisions_reports_no_shift():
    report, _ = run(
        [{"id": "empty", "request": {"tag": "empty"}}],
        lambda request, call: reply({}, 0.5),
        compare_modes=True,
    )
    comparison = report["summary"]["mode_comparison"]
    assert comparison["decisions"] == 0
    assert comparison["changed_argmaxes"] == 0
    assert comparison["max_probability_delta"] == 0
    assert report["summary"]["decisions_per_second"] == 0


def test_the_95th_percentile_is_the_nearest_rank():
    def answer_by_call(request: dict[str, Any], call: int) -> Reply:
        return reply(
            {"q1": pick({"a": 0.5, "b": 0.5})}, WARMUP_SECONDS if call == 1 else call - 1.0
        )

    report, engine = run([fixture("only", q1={"label": "a"})], answer_by_call, repeats=20)
    assert len(engine.calls) == 21
    latency = report["summary"]["latency_seconds"]
    assert latency == {"median": 10.5, "p95": 19.0}  # 1..20 seconds
    assert report["summary"]["decisions_per_second"] == pytest.approx(20 / 210)
    assert len(report["rows"][0]["repeat_timings"]) == 20


def test_a_single_measurement_is_its_own_median_and_percentile():
    report, _ = run(
        [fixture("one", q1={"label": "a"})],
        lambda request, call: reply({"q1": pick({"a": 0.5, "b": 0.5})}, 0.3),
    )
    assert report["summary"]["latency_seconds"] == {"median": 0.3, "p95": 0.3}


# arguments ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixtures", "repeats"), [([], 1), ([fixture("x")], 0), ([fixture("x")], -1)]
)
def test_evaluation_needs_fixtures_and_a_repetition(fixtures, repeats):
    engine = ScriptedEngine(lambda request, call: reply({}, 0.1))
    with pytest.raises(ValueError, match=exactly("Provide fixtures and at least one repetition")):
        evaluate(engine, fixtures, repeats=repeats)
    assert engine.calls == []  # nothing ran, not even the warmup


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


def test_a_malformed_fixture_is_reported_before_a_lone_surrogate_in_an_earlier_one():
    fixtures = [{"id": "bad\ud800", "request": {}}, {"request": {}}]
    with pytest.raises(ValueError, match=exactly('Fixture 2 has no "id"')):
        check_fixtures(fixtures)


def test_check_fixtures_accepts_what_evaluate_can_use():
    usable = [fixture("a", q1={"label": "yes"}), {"id": 7, "request": {}}, fixture("b")]
    assert check_fixtures(usable, repeats=3) is None
    assert check_fixtures(usable) is None


@pytest.mark.parametrize(("fixtures", "repeats"), [([], 1), ([fixture("x")], 0)])
def test_check_fixtures_needs_fixtures_and_a_repetition(fixtures, repeats):
    with pytest.raises(ValueError, match=exactly("Provide fixtures and at least one repetition")):
        check_fixtures(fixtures, repeats)


VALID_REQUEST = {"state": "x", "questions": {"q": {"type": "boolean", "instructions": "Is it?"}}}


def test_check_requests_accepts_the_requests_the_engine_accepts():
    fixtures = [{"id": "a", "request": VALID_REQUEST}, {"id": "b", "request": VALID_REQUEST}]
    assert check_requests(fixtures) is None
    assert fixtures[0]["request"] == VALID_REQUEST  # only looked at


EVERY_KIND = request_payload(
    state="Cannot log in",
    flag=question_payload("boolean"),
    queue=question_payload("choice"),
    mood=question_payload("score"),
    price=question_payload("numeric"),
    strict=question_payload("choice", policy={"allow_abstain": False}),
)


def expecting(question: str, **expected: Any) -> list[dict[str, Any]]:
    return [{"id": "odd", "request": EVERY_KIND, "expected": {question: expected}}]


def test_check_requests_accepts_exactly_the_labels_the_engine_reports():
    answers = Engine(StubBackend()).decide(EVERY_KIND)["answers"]
    assert set(answers) == set(EVERY_KIND["questions"])
    for question, answer in answers.items():
        assert len(answer["probabilities"]) >= 2
        for label in answer["probabilities"]:
            assert check_requests(expecting(question, label=label)) is None
        with pytest.raises(ValueError, match="Unknown expected label no-such-label"):
            check_requests(expecting(question, label="no-such-label"))
    # A question that cannot abstain has no abstention to expect.
    assert "__insufficient__" not in answers["strict"]["probabilities"]
    with pytest.raises(ValueError, match="Unknown expected label __insufficient__"):
        check_requests(expecting("strict", label="__insufficient__"))


@pytest.mark.parametrize(
    "expected",
    [{"status": "ok"}, {"value": 100}, {"status": "uncertain", "value": 1.5}, {}],
    ids=["status", "value", "both", "nothing"],
)
def test_check_requests_looks_at_labels_only_where_there_is_one(expected):
    assert check_requests(expecting("flag", **expected)) is None


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


def test_check_requests_checks_every_fixture_and_stops_at_the_first_wrong_one():
    fixtures = [
        *expecting("flag", label="true"),
        {"id": "two", "request": EVERY_KIND, "expected": {"flag": {"label": "yes"}}},
        {"id": "three", "request": EVERY_KIND, "expected": {"ghost": {}}},
    ]
    with pytest.raises(ValueError, match=exactly("Unknown expected label yes in fixture two")):
        check_requests(fixtures)


def test_check_requests_names_the_fixture_whose_request_is_refused():
    broken = {"id": "broken", "request": {"state": "x", "questions": {}}}
    # What pydantic says about the request follows the name, over several lines.
    with pytest.raises(ValueError, match=r"(?s)^Fixture broken: .*questions") as refused:
        check_requests([{"id": "fine", "request": VALID_REQUEST}, broken])
    assert isinstance(refused.value.__cause__, ValidationError)
