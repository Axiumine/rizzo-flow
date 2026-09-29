"""Property-based tests (Hypothesis) of the numerical and structural core.

Examples are derandomized and kept few, so the suite is fast and a failure reproduces on every
machine; the properties hold for every input. When hunting, raise `max_examples` or set
`derandomize=False` in `BASE`: `--hypothesis-seed` has no effect on a derandomized test.
"""

import copy
import itertools
import json
import math
import statistics
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError
from test_core_support import parse_question, question_payload, request_payload

from rizzo_flow.calibration import fit_temperature
from rizzo_flow.compat import confidence
from rizzo_flow.decisions import ABOVE, BELOW, UNKNOWN, candidates, decode, softmax
from rizzo_flow.evaluation import evaluate
from rizzo_flow.prompts import canonical
from rizzo_flow.responses import TypedAnswer
from rizzo_flow.schema import Request

# No example database and no deadline (slow CI machines).
BASE = settings(
    deadline=None, database=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow]
)
PROPERTY = settings(BASE, max_examples=100)
HEAVY = settings(BASE, max_examples=40)
MANY = settings(BASE, max_examples=300)
SPECIAL = {UNKNOWN, BELOW, ABOVE}
KINDS = ("boolean", "choice", "score", "numeric")

logits = st.floats(min_value=-40.0, max_value=40.0)
wide_logits = st.floats(min_value=-1e6, max_value=1e6)
temperatures = st.floats(min_value=0.05, max_value=20.0)
any_temperature = st.floats(min_value=0.01, max_value=1000.0)


def sized(elements: st.SearchStrategy[float], low: int = 2, high: int = 26) -> st.SearchStrategy:
    return st.lists(elements, min_size=low, max_size=high)


@st.composite
def separated_logits(draw: st.DrawFn) -> tuple[list[float], int]:
    """Logits with one clear winner: every other logit is at least 0.05 below it.

    A near-tie can resolve the other way once the logits are scaled and rounded, so only a
    margin guarantees the same argmax; the order of the probabilities is checked without one.
    """
    count = draw(st.integers(2, 26))
    winner = draw(st.integers(0, count - 1))
    top = draw(st.floats(min_value=-30.0, max_value=30.0))
    gaps = draw(
        st.lists(st.floats(min_value=0.05, max_value=30.0), min_size=count - 1, max_size=count - 1)
    )
    values = [top - gap for gap in gaps]
    values.insert(winner, top)
    return values, winner


@st.composite
def questions_with_logits(
    draw: st.DrawFn, kinds: tuple[str, ...] = KINDS
) -> tuple[Any, list[float], float]:
    """Any valid question with any policy, logits for all its candidates, and a temperature."""
    kind = draw(st.sampled_from(kinds))
    abstain = draw(st.booleans())
    policy = {
        "allow_abstain": abstain,
        "max_unavailable_probability": draw(st.floats(min_value=0.05, max_value=1.0)),
        "min_top_probability": draw(st.floats(min_value=0.0, max_value=1.0)),
    }
    extra: dict[str, Any] = {}
    if kind == "choice":
        count = draw(st.integers(2, 26 - abstain))
        extra["options"] = [{"id": f"o{i}", "description": f"option {i}"} for i in range(count)]
    elif kind == "score":
        count = draw(st.integers(2, 26 - abstain))
        extra["levels"] = [f"level {i}" for i in range(count)]
    elif kind == "numeric":
        count = draw(st.integers(2, 24 - abstain))
        start = draw(st.floats(min_value=-1e6, max_value=1e6))
        steps = draw(
            st.lists(
                st.floats(min_value=0.5, max_value=1e4), min_size=count - 1, max_size=count - 1
            )
        )
        anchors = [start + sum(steps[:i]) for i in range(count)]
        extra["anchors"] = [
            {"value": v, "description": f"anchor {i}"} for i, v in enumerate(anchors)
        ]
    question = parse_question(kind, policy=policy, **extra)
    values = draw(
        st.lists(logits, min_size=len(candidates(question)), max_size=len(candidates(question)))
    )
    temperature = draw(st.one_of(st.just(1.0), temperatures))
    return question, values, temperature


# softmax ---------------------------------------------------------------------------------------


@PROPERTY
@given(values=sized(wide_logits), temperature=any_temperature)
def test_softmax_is_a_probability_distribution(values, temperature):
    probabilities = softmax(values, temperature)
    assert len(probabilities) == len(values)
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities)
    assert math.fsum(probabilities) == pytest.approx(1.0, abs=1e-12)


@PROPERTY
@given(values=sized(wide_logits), temperature=any_temperature)
def test_softmax_never_reverses_the_order_of_the_logits(values, temperature):
    probabilities = softmax(values, temperature)
    for i, j in itertools.combinations(range(len(values)), 2):
        if values[i] > values[j]:
            assert probabilities[i] + 1e-12 >= probabilities[j]
        elif values[i] < values[j]:
            assert probabilities[j] + 1e-12 >= probabilities[i]
        else:
            assert probabilities[i] == probabilities[j]


@PROPERTY
@given(values=sized(logits), shift=st.floats(min_value=-500.0, max_value=500.0))
def test_softmax_ignores_a_common_shift_of_the_logits(values, shift):
    shifted = softmax([v + shift for v in values])
    assert shifted == pytest.approx(softmax(values), abs=1e-9)


@PROPERTY
@given(case=separated_logits(), temperature=temperatures)
def test_temperature_scaling_keeps_the_argmax(case, temperature):
    values, winner = case
    probabilities = softmax(values, temperature)
    assert probabilities.index(max(probabilities)) == winner
    assert softmax(values).index(max(softmax(values))) == winner


@PROPERTY
@given(case=separated_logits(), temperature=temperatures, abstain=st.booleans())
def test_decode_keeps_the_winning_choice_at_any_temperature(case, temperature, abstain):
    values, winner = case
    assume(len(values) - abstain >= 2)
    options = [{"id": f"o{i}", "description": f"option {i}"} for i in range(len(values) - abstain)]
    question = parse_question("choice", options=options, policy={"allow_abstain": abstain})
    result = decode(question, values, temperature)
    winning = candidates(question)[winner].id
    probabilities = result["probabilities"]
    assert max(probabilities, key=probabilities.__getitem__) == winning
    if winning == UNKNOWN:
        assert (result["status"], result["choice"]) == ("insufficient_evidence", None)
    else:
        assert (result["status"], result["choice"]) == ("ok", winning)


# decode ----------------------------------------------------------------------------------------


@PROPERTY
@given(case=questions_with_logits())
def test_decoded_probabilities_are_a_distribution_over_the_candidates(case):
    question, values, temperature = case
    result = decode(question, values, temperature)
    ids = [c.id for c in candidates(question)]
    probabilities = result["probabilities"]
    assert list(probabilities) == ids
    assert list(result["legend"]) == ids
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities.values())
    assert math.fsum(probabilities.values()) == pytest.approx(1.0, abs=1e-12)
    assert list(result["option_logits"].values()) == values
    assert list(probabilities.values()) == softmax(values, temperature)
    assert result["temperature"] == temperature


@PROPERTY
@given(case=questions_with_logits())
def test_uncertainty_summarizes_the_distribution(case):
    question, values, temperature = case
    result = decode(question, values, temperature)
    probabilities = result["probabilities"]
    uncertainty = result["uncertainty"]
    assert uncertainty["top_probability"] == max(probabilities.values())
    assert 0.0 <= uncertainty["entropy_nats"] <= math.log(len(values)) + 1e-9
    assert 0.0 <= uncertainty["concentration"] <= 1.0
    special = [p for key, p in probabilities.items() if key in SPECIAL]
    assert uncertainty["unavailable_probability"] == pytest.approx(math.fsum(special), abs=1e-15)


@PROPERTY
@given(case=questions_with_logits())
def test_status_follows_the_policy_thresholds(case):
    question, values, temperature = case
    result = decode(question, values, temperature)
    probabilities = result["probabilities"]
    policy = question.policy
    winner = max(probabilities, key=probabilities.__getitem__)
    unavailable = math.fsum(p for key, p in probabilities.items() if key in SPECIAL)
    top = max(probabilities.values())
    status = result["status"]
    if winner in SPECIAL or unavailable >= policy.max_unavailable_probability:
        assert status in {"insufficient_evidence", "out_of_range"}
        range_mass = probabilities.get(BELOW, 0.0) + probabilities.get(ABOVE, 0.0)
        expected = (
            "out_of_range"
            if range_mass > probabilities.get(UNKNOWN, 0.0)
            else "insufficient_evidence"
        )
        assert status == expected
    elif top < policy.min_top_probability:
        assert status == "uncertain"
    else:
        assert status == "ok"
    if question.type != "numeric":
        assert status != "out_of_range"
    if not policy.allow_abstain and question.type != "numeric":
        assert unavailable == 0.0
        assert status in {"ok", "uncertain"}


@PROPERTY
@given(case=questions_with_logits())
def test_only_an_accepted_decision_has_a_primary_value(case):
    question, values, temperature = case
    result = decode(question, values, temperature)
    field = {"boolean": "value", "choice": "choice", "score": "score", "numeric": "value"}
    primary = result[field[question.type]]
    assert (result["status"] == "ok") == (primary is not None)
    if result["status"] == "ok" and question.type == "boolean":
        probabilities = result["probabilities"]
        assert primary is (max(probabilities, key=probabilities.__getitem__) == "true")


@PROPERTY
@given(case=questions_with_logits())
def test_every_decoded_answer_satisfies_the_output_contract(case):
    question, values, temperature = case
    result = decode(question, values, temperature)
    answer = {**result, "prompt_sha256": "0" * 64, "input_tokens": 1}
    validated: Any = TypeAdapter(TypedAnswer).validate_python(answer)
    assert validated.type == question.type


@MANY
@given(case=questions_with_logits(kinds=("score", "numeric")))
def test_weighted_statistics_stay_within_the_support(case):
    question, values, temperature = case
    result = decode(question, values, temperature)
    assume(result["status"] == "ok")
    low, high = result["support"]
    tolerance = 1e-9 * max(1.0, abs(low), abs(high))
    stats = result["statistics_given_available"]
    primary = result["score" if question.type == "score" else "value"]
    assert stats["mean"] == primary
    assert low - tolerance <= primary <= high + tolerance
    assert 0.0 <= stats["stddev"] <= (high - low) / 2 + tolerance  # Popoviciu's inequality
    supported = set(result["values"].values())
    quantiles = stats["anchor_quantiles"]
    assert stats["median"] in supported
    assert quantiles["p10"] in supported
    assert quantiles["p90"] in supported
    assert quantiles["p10"] <= stats["median"] <= quantiles["p90"]
    # The mean is taken over the available candidates only.
    valid = {key: p for key, p in result["probabilities"].items() if key not in SPECIAL}
    total = math.fsum(valid.values())
    expected = math.fsum(result["values"][key] * p for key, p in valid.items()) / total
    assert primary == pytest.approx(expected, rel=1e-9, abs=tolerance)
    if question.type == "score":
        assert 0.0 <= result["normalized_score"] <= 1.0
        assert result["normalized_score"] == pytest.approx(primary / high)
    else:
        assert result["range_probabilities"] == {
            "below": result["probabilities"][BELOW],
            "above": result["probabilities"][ABOVE],
        }


# confidence ------------------------------------------------------------------------------------


@PROPERTY
@given(count=st.integers(2, 26))
def test_confidence_of_a_uniform_distribution_is_zero(count):
    assert confidence([1 / count] * count) == pytest.approx(0.0, abs=1e-12)


@PROPERTY
@given(count=st.integers(2, 26), data=st.data())
def test_confidence_of_a_one_hot_distribution_is_one(count, data):
    index = data.draw(st.integers(0, count - 1))
    one_hot = [1.0 if i == index else 0.0 for i in range(count)]
    assert confidence(one_hot) == 1.0


@PROPERTY
@given(values=sized(wide_logits), temperature=any_temperature)
def test_confidence_is_the_clamped_peak_over_uniform_statistic(values, temperature):
    probabilities = softmax(values, temperature)
    count = len(probabilities)
    score = confidence(probabilities)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx((count * max(probabilities) - 1) / (count - 1), abs=1e-12)
    assert confidence(probabilities[::-1]) == score  # only the peak matters, not its position


@PROPERTY
@given(values=sized(logits), temperature=temperatures)
def test_a_sharper_distribution_is_more_confident(values, temperature):
    flatter = confidence(softmax(values, temperature * 2))
    sharper = confidence(softmax(values, temperature))
    assert sharper + 1e-12 >= flatter


# canonical JSON --------------------------------------------------------------------------------

scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**9), 10**9),
    st.floats(allow_nan=False, allow_infinity=False),
    st.sampled_from(
        ["", "a", "é", 'quote"', "line\nbreak", "tab\t", "日本語", "🙂", "</evidence>"]
    ),
)
json_values = st.recursive(
    scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.sampled_from(["a", "b", "é", "k 1", ""]), children, max_size=4),
    ),
    max_leaves=12,
)


def reordered(value: Any) -> Any:
    """The same JSON value with every object's keys in reverse insertion order."""
    if isinstance(value, dict):
        return {key: reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reordered(item) for item in value]
    return value


@PROPERTY
@given(value=json_values)
def test_canonical_json_round_trips_and_ignores_key_order(value):
    text = canonical(value)
    assert json.loads(text) == value
    assert canonical(reordered(value)) == text
    assert canonical(json.loads(text)) == text
    assert "\n" not in text


# schema: the answer letters ---------------------------------------------------------------------


@PROPERTY
@given(
    kind=st.sampled_from(["choice", "score", "numeric"]),
    count=st.integers(0, 30),
    abstain=st.booleans(),
)
def test_a_question_is_accepted_exactly_when_all_its_candidates_fit_the_letters(
    kind, count, abstain
):
    reserved = 2 if kind == "numeric" else 0
    if kind == "choice":
        extra: dict[str, Any] = {
            "options": [{"id": f"o{i}", "description": f"o{i}"} for i in range(count)]
        }
    elif kind == "score":
        extra = {"levels": [f"level {i}" for i in range(count)]}
    else:
        extra = {"anchors": [{"value": i, "description": f"a{i}"} for i in range(count)]}
    body = request_payload(q=question_payload(kind, policy={"allow_abstain": abstain}, **extra))
    fits = count >= 2 and count + reserved + abstain <= 26
    try:
        request = Request.model_validate(body)
    except ValidationError:
        assert not fits
    else:
        assert fits
        assert len(candidates(request.questions["q"])) == count + reserved + abstain


# calibration -----------------------------------------------------------------------------------


@st.composite
def labeled_rows(draw: st.DrawFn) -> list[dict[str, Any]]:
    kind = draw(st.sampled_from(KINDS))
    rows = []
    for _ in range(draw(st.integers(10, 20))):
        row_logits = draw(st.lists(logits, min_size=2, max_size=5))
        rows.append(
            {
                "type": kind,
                "logits": row_logits,
                "label_index": draw(st.integers(0, len(row_logits) - 1)),
            }
        )
    return rows


@HEAVY
@given(rows=labeled_rows())
def test_the_fitted_temperature_never_does_worse_than_temperature_one(rows):
    fit = fit_temperature(rows, "fp")
    (kind,) = fit.temperatures
    temperature = fit.temperatures[kind]
    assert 0.05 * (1 - 1e-9) <= temperature <= 20.0 * (1 + 1e-9)
    position = (math.log(temperature) - math.log(0.05)) * 240 / math.log(400)
    assert position == pytest.approx(round(position), abs=1e-6)  # a point of the search grid
    metrics = fit.fit_metrics[kind]
    assert metrics["rows"] == len(rows)
    assert metrics["fit_nll_after"] <= metrics["fit_nll_before"] + 1e-12
    assert metrics["fit_nll_before"] >= -1e-12


# evaluation ------------------------------------------------------------------------------------


class FixedEngine:
    """Answers each request from a table keyed by its tag."""

    def __init__(self, table: dict[str, dict[str, Any]]) -> None:
        self.table = table

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(self.table[request["tag"]])


@st.composite
def benchmarks(draw: st.DrawFn) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    fixtures: list[dict[str, Any]] = []
    table: dict[str, dict[str, Any]] = {}
    for index in range(draw(st.integers(1, 5))):
        tag = f"f{index}"
        expected: dict[str, Any] = {}
        answers: dict[str, Any] = {}
        for number in range(draw(st.integers(1, 3))):
            labels = ["a", "b", "c"][: draw(st.integers(2, 3))]
            weights = draw(
                st.lists(
                    st.floats(min_value=0.0, max_value=1.0),
                    min_size=len(labels),
                    max_size=len(labels),
                )
            )
            total = sum(weights)
            shares = [w / total for w in weights] if total > 0 else [1 / len(labels)] * len(labels)
            name = f"q{number}"
            expected[name] = {"label": draw(st.sampled_from(labels))}
            status = draw(st.sampled_from(["ok", "uncertain", "insufficient_evidence"]))
            answers[name] = {
                "type": "choice",
                "status": status,
                "probabilities": dict(zip(labels, shares, strict=True)),
            }
        seconds = draw(st.floats(min_value=0.001, max_value=5.0))
        fixtures.append({"id": tag, "request": {"tag": tag}, "expected": expected})
        table[tag] = {"mode": "shared", "timing": {"total_seconds": seconds}, "answers": answers}
    return fixtures, table


@HEAVY
@given(case=benchmarks())
def test_evaluation_metrics_agree_with_a_plain_recomputation(case):
    fixtures, table = case
    report = evaluate(FixedEngine(table), fixtures)
    summary = report["summary"]
    rows = []
    for fixture in fixtures:
        for name, expected in fixture["expected"].items():
            answer = table[fixture["id"]]["answers"][name]
            probabilities = answer["probabilities"]
            label = expected["label"]
            predicted = max(probabilities, key=probabilities.__getitem__)
            rows.append(
                {
                    "correct": predicted == label,
                    "accepted": answer["status"] == "ok",
                    "nll": -math.log(max(probabilities[label], 1e-300)),
                    "brier": sum((p - (key == label)) ** 2 for key, p in probabilities.items()),
                    "confidence": max(probabilities.values()),
                }
            )
    stats = summary["categorical"]
    assert stats["rows"] == len(rows)
    assert stats["accuracy"] == pytest.approx(statistics.mean(r["correct"] for r in rows))
    assert stats["nll"] == pytest.approx(statistics.mean(r["nll"] for r in rows))
    assert stats["brier"] == pytest.approx(statistics.mean(r["brier"] for r in rows))
    assert summary["labeled_decision_coverage"] == pytest.approx(
        statistics.mean(r["accepted"] for r in rows)
    )
    accepted = [r["correct"] for r in rows if r["accepted"]]
    assert stats["accepted_accuracy"] == (
        pytest.approx(statistics.mean(accepted)) if accepted else None
    )
    bins = stats["reliability_bins"]
    assert sum(b["count"] for b in bins) == len(rows)
    lowers = [b["lower"] for b in bins]
    assert lowers == sorted(set(lowers))
    assert all(0.0 <= lower <= 0.9 for lower in lowers)
    assert 0.0 <= stats["ece_10_bins"] <= 1.0
    assert 0.0 <= stats["accuracy"] <= 1.0
    assert 0.0 <= stats["brier"] <= 2.0 + 1e-12
    assert stats["nll"] >= 0.0
    latency = summary["latency_seconds"]
    seconds = [table[f["id"]]["timing"]["total_seconds"] for f in fixtures]
    assert min(seconds) <= latency["median"] <= latency["p95"] <= max(seconds)
    decisions = sum(len(table[f["id"]]["answers"]) for f in fixtures)
    assert summary["decisions_per_second"] == pytest.approx(decisions / sum(seconds))
