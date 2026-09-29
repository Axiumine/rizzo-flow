"""Directed tests of decisions.py: candidate lists, softmax, summary statistics, decode policy."""

import math
from fractions import Fraction
from typing import Any

import pytest
from test_core_support import exactly, parse_question

from rizzo_flow.decisions import (
    ABOVE,
    BELOW,
    UNKNOWN,
    Candidate,
    candidates,
    decode,
    softmax,
    summarize,
)
from rizzo_flow.schema import BaseQuestion

KINDS = ["boolean", "choice", "score", "numeric"]
NO_ABSTENTION = {"allow_abstain": False}
THREE_OPTIONS = [
    {"id": "x", "description": "first"},
    {"id": "y", "description": "second"},
    {"id": "z", "description": "third"},
]
BASE_KEYS = {
    "type",
    "status",
    "probabilities",
    "option_logits",
    "legend",
    "uncertainty",
    "probability_status",
    "temperature",
}
TYPE_KEYS = {
    "boolean": {"value", "probability_true_given_available"},
    "choice": {"choice"},
    "score": {"score", "statistics_given_available", "values", "support", "normalized_score"},
    "numeric": {
        "value",
        "statistics_given_available",
        "values",
        "support",
        "unit",
        "range_probabilities",
    },
}


def logs(*probabilities: float) -> list[float]:
    """Logits whose softmax is the given distribution."""
    return [math.log(p) for p in probabilities]


def numeric_question(abstain: bool, anchors: Any = (10, 20, 40), **policy: Any) -> Any:
    return parse_question(
        "numeric",
        unit="kg",
        anchors=[{"value": v, "description": f"about {v}"} for v in anchors],
        policy={"allow_abstain": abstain, **policy},
    )


# candidates ------------------------------------------------------------------------------


def test_boolean_candidates_are_false_then_true():
    q = parse_question("boolean", policy=NO_ABSTENTION)
    assert candidates(q) == [
        Candidate("false", q.false_description, 0),
        Candidate("true", q.true_description, 1),
    ]


def test_choice_candidates_keep_the_option_order_and_carry_no_value():
    q = parse_question("choice", options=THREE_OPTIONS, policy=NO_ABSTENTION)
    assert candidates(q) == [
        Candidate("x", "first"),
        Candidate("y", "second"),
        Candidate("z", "third"),
    ]
    assert {c.value for c in candidates(q)} == {None}


def test_score_candidates_are_the_indexed_levels():
    q = parse_question("score", levels=["low", "mid", "high"], policy=NO_ABSTENTION)
    assert candidates(q) == [
        Candidate("0", "low", 0),
        Candidate("1", "mid", 1),
        Candidate("2", "high", 2),
    ]


def test_numeric_candidates_are_anchors_then_the_out_of_range_options():
    q = numeric_question(True, anchors=[0.5, 1_000_000])
    described = candidates(q)
    assert [c.id for c in described] == ["0", "1", BELOW, ABOVE, UNKNOWN]
    assert [c.value for c in described] == [0.5, 1_000_000, None, None, None]
    assert described[0].description == "Approximately 0.5 kg: about 0.5"
    assert described[1].description == "Approximately 1e+06 kg: about 1000000"
    assert described[2].description == "The value is below 0.5 kg."
    assert described[3].description == "The value is above 1e+06 kg."


@pytest.mark.parametrize("kind", KINDS)
def test_abstention_adds_one_trailing_candidate_and_is_the_default(kind):
    without = candidates(parse_question(kind, policy=NO_ABSTENTION))
    with_abstention = candidates(parse_question(kind, policy={"allow_abstain": True}))
    assert UNKNOWN not in [c.id for c in without]
    assert with_abstention[:-1] == without
    last = with_abstention[-1]
    assert (last.id, last.value) == (UNKNOWN, None)
    assert "Cannot determine the answer" in last.description
    assert "outside the numeric range is not missing information" in last.description
    assert candidates(parse_question(kind)) == with_abstention


def test_unsupported_question_types_are_refused():
    unsupported: Any = BaseQuestion(instructions="x")
    with pytest.raises(TypeError, match=exactly("Unsupported question")):
        candidates(unsupported)


# softmax ---------------------------------------------------------------------------------


def test_softmax_known_values():
    assert softmax([0.0, 0.0]) == [0.5, 0.5]
    assert softmax([0.0, math.log(3)]) == pytest.approx([0.25, 0.75])
    assert softmax([1.0, 2.0, 3.0]) == pytest.approx([0.0900305732, 0.2447284711, 0.6652409558])


def test_softmax_temperature_divides_the_logits():
    assert softmax([0.0, 2.0], 2.0) == pytest.approx(softmax([0.0, 1.0]))
    assert softmax([0.0, 2.0], 0.5) == pytest.approx(softmax([0.0, 4.0]))
    plain = softmax([0.0, 1.0])[1]
    assert softmax([0.0, 1.0], 0.1)[1] > plain > softmax([0.0, 1.0], 10.0)[1] > 0.5


def test_softmax_is_stable_for_huge_logits():
    assert softmax([1000.0, 1000.0]) == [0.5, 0.5]
    assert softmax([1000.0, 0.0]) == [1.0, 0.0]
    assert softmax([-1000.0, -999.0]) == pytest.approx(softmax([0.0, 1.0]))


@pytest.mark.parametrize("temperature", [0, -1.0, math.nan, math.inf, -math.inf])
def test_softmax_rejects_a_bad_temperature(temperature):
    with pytest.raises(ValueError, match=exactly("Temperature must be finite and positive")):
        softmax([0.0, 1.0], temperature)


@pytest.mark.parametrize(
    "logits", [[], [1.0], [0.0, math.nan], [math.inf, 0.0], [0.0, -math.inf], [math.nan] * 3]
)
def test_softmax_needs_two_or_more_finite_logits(logits):
    with pytest.raises(ValueError, match=exactly("At least two finite logits are required")):
        softmax(logits)


# summarize -------------------------------------------------------------------------------


def test_summarize_moments_and_quantiles():
    stats = summarize([10, 20, 40], [0.125, 0.625, 0.25])
    assert set(stats) == {"mean", "stddev", "median", "anchor_quantiles"}
    assert stats["mean"] == pytest.approx(23.75)
    assert stats["stddev"] == pytest.approx(math.sqrt(98.4375))
    assert stats["median"] == 20
    assert stats["anchor_quantiles"] == {"p10": 10, "p90": 40}


def test_summarize_takes_the_first_value_whose_cumulative_mass_reaches_the_level():
    stats = summarize([1.0, 2.0], [0.5, 0.5])
    assert stats["median"] == 1.0  # exactly 0.5 of the mass is enough
    assert stats["anchor_quantiles"] == {"p10": 1.0, "p90": 2.0}


def test_summarize_quantile_falls_back_to_the_last_value_when_mass_is_missing():
    # Probabilities that do not reach the level (rounding in practice): the last value answers.
    stats = summarize([1.0, 2.0, 3.0, 4.0], [0.1, 0.1, 0.1, 0.1])
    assert stats["anchor_quantiles"]["p10"] == 1.0
    assert stats["median"] == 4.0
    assert stats["anchor_quantiles"]["p90"] == 4.0


def test_summarize_quantiles_accumulate_the_mass_in_order():
    stats = summarize([1.0, 2.0, 3.0], [0.3, 0.3, 0.4])
    assert stats["median"] == 2.0  # 0.3 + 0.3 crosses a half; no single entry does
    assert stats["anchor_quantiles"] == {"p10": 1.0, "p90": 3.0}


def test_summarize_quantile_levels_are_the_tenth_fiftieth_and_ninetieth_percentiles():
    # Cumulative mass 0.02, 0.12, 0.40, 0.70, 0.95, 1.00: each level lands on a different value.
    stats = summarize([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.02, 0.10, 0.28, 0.30, 0.25, 0.05])
    assert stats["anchor_quantiles"]["p10"] == 2.0
    assert stats["median"] == 4.0
    assert stats["anchor_quantiles"]["p90"] == 5.0


def test_summarize_needs_matching_lengths():
    with pytest.raises(ValueError, match="shorter than argument 1"):
        summarize([1.0, 2.0], [1.0])


# decode: shape and arithmetic --------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_result_has_the_documented_fields_for_every_type(kind):
    q = parse_question(kind, policy=NO_ABSTENTION)
    result = decode(q, [0.0] * len(candidates(q)))
    assert set(result) == BASE_KEYS | TYPE_KEYS[kind]
    assert result["type"] == kind


def test_logit_count_must_match_the_candidates():
    q = parse_question("boolean")
    with pytest.raises(
        ValueError, match=exactly("Logit count does not match the declared candidates")
    ):
        decode(q, [0.0, 1.0])
    with pytest.raises(
        ValueError, match=exactly("Logit count does not match the declared candidates")
    ):
        decode(q, [0.0, 1.0, 2.0, 3.0])


def test_boolean_result_in_detail():
    q = parse_question("boolean")
    logits = logs(0.2, 0.6, 0.2)
    result = decode(q, logits)
    assert result["status"] == "ok"
    assert result["value"] is True
    assert list(result["probabilities"]) == ["false", "true", UNKNOWN]
    assert list(result["probabilities"].values()) == pytest.approx([0.2, 0.6, 0.2])
    assert result["probability_true_given_available"] == pytest.approx(0.75)
    assert result["option_logits"] == dict(zip(["false", "true", UNKNOWN], logits, strict=True))
    assert result["legend"] == {c.id: c.description for c in candidates(q)}
    entropy = -(2 * 0.2 * math.log(0.2) + 0.6 * math.log(0.6))
    assert result["uncertainty"] == pytest.approx(
        {
            "top_probability": 0.6,
            "entropy_nats": entropy,
            "concentration": 1 - entropy / math.log(3),
            "unavailable_probability": 0.2,
        }
    )
    assert result["probability_status"] == "uncalibrated_conditional_option_scores"
    assert result["temperature"] == 1.0


def test_boolean_false_is_reported_as_false():
    result = decode(parse_question("boolean", policy=NO_ABSTENTION), logs(0.7, 0.3))
    assert result["value"] is False
    assert result["probability_true_given_available"] == pytest.approx(0.3)


def test_choice_result_names_the_winning_option():
    q = parse_question("choice", options=THREE_OPTIONS, policy=NO_ABSTENTION)
    result = decode(q, logs(0.2, 0.5, 0.3))
    assert (result["status"], result["choice"]) == ("ok", "y")
    assert list(result["probabilities"]) == ["x", "y", "z"]
    assert result["legend"] == {"x": "first", "y": "second", "z": "third"}


def test_a_tie_goes_to_the_first_candidate():
    q = parse_question("choice", options=THREE_OPTIONS, policy=NO_ABSTENTION)
    assert decode(q, [0.0, 0.0, 0.0])["choice"] == "x"
    boolean = parse_question("boolean", policy=NO_ABSTENTION)
    assert decode(boolean, [0.0, 0.0])["value"] is False


def test_score_statistics_are_conditional_on_the_available_levels():
    q = parse_question("score", levels=["low", "mid", "high"], policy={"allow_abstain": True})
    result = decode(q, logs(0.1, 0.2, 0.4, 0.3))  # the last entry is the abstention
    assert result["status"] == "ok"
    mean = Fraction(10, 7)  # (0*1 + 1*2 + 2*4) / 7
    assert result["score"] == pytest.approx(float(mean))
    assert result["normalized_score"] == pytest.approx(float(mean / 2))
    variance = Fraction(1, 7) * mean**2 + Fraction(2, 7) * (1 - mean) ** 2
    variance += Fraction(4, 7) * (2 - mean) ** 2
    stats = result["statistics_given_available"]
    assert stats["mean"] == result["score"]
    assert stats["stddev"] == pytest.approx(math.sqrt(variance))
    assert stats["median"] == 2
    assert stats["anchor_quantiles"] == {"p10": 0, "p90": 2}
    assert result["values"] == {"0": 0, "1": 1, "2": 2}
    assert result["support"] == [0, 2]
    assert result["uncertainty"]["unavailable_probability"] == pytest.approx(0.3)


def test_numeric_value_is_the_conditional_mean_over_the_anchors():
    q = numeric_question(False)
    result = decode(q, logs(0.1, 0.5, 0.2, 0.1, 0.1))  # anchors 10, 20, 40, then below, above
    assert result["status"] == "ok"
    assert result["value"] == pytest.approx(23.75)  # 10*.125 + 20*.625 + 40*.25
    assert result["statistics_given_available"]["stddev"] == pytest.approx(math.sqrt(98.4375))
    assert result["statistics_given_available"]["median"] == 20
    assert result["values"] == {"0": 10, "1": 20, "2": 40}
    assert result["support"] == [10, 40]
    assert result["unit"] == "kg"
    assert result["range_probabilities"] == pytest.approx({"below": 0.1, "above": 0.1})
    assert result["uncertainty"]["unavailable_probability"] == pytest.approx(0.2)


def test_numeric_support_ignores_the_special_candidates():
    result = decode(numeric_question(True, anchors=[5, 7]), [0.0, 1.0, 0.0, 0.0, 0.0])
    assert result["support"] == [5, 7]
    assert result["values"] == {"0": 5, "1": 7}
    assert list(result["probabilities"]) == ["0", "1", BELOW, ABOVE, UNKNOWN]


def test_score_without_a_primary_value_has_no_statistics():
    q = parse_question(
        "score", levels=["a", "b"], policy=NO_ABSTENTION | {"min_top_probability": 0.9}
    )
    result = decode(q, [0.0, 0.0])
    assert result["status"] == "uncertain"
    assert result["score"] is None
    assert result["normalized_score"] is None
    assert result["statistics_given_available"] is None
    assert result["support"] == [0, 1]
    assert result["values"] == {"0": 0, "1": 1}


# decode: temperature -----------------------------------------------------------------------


def test_temperature_rescales_probabilities_but_reports_the_raw_logits():
    q = parse_question("boolean", policy=NO_ABSTENTION)
    plain = decode(q, [0.0, 2.0])
    cooled = decode(q, [0.0, 2.0], 0.5)
    warmed = decode(q, [0.0, 2.0], 2.0)
    assert cooled["probabilities"]["true"] > plain["probabilities"]["true"]
    assert plain["probabilities"]["true"] > warmed["probabilities"]["true"] > 0.5
    assert warmed["probabilities"]["true"] == pytest.approx(softmax([0.0, 1.0])[1])
    for result in (plain, cooled, warmed):
        assert result["option_logits"] == {"false": 0.0, "true": 2.0}
    assert (plain["temperature"], cooled["temperature"], warmed["temperature"]) == (1.0, 0.5, 2.0)


@pytest.mark.parametrize(
    ("temperature", "label"),
    [
        (1, "uncalibrated_conditional_option_scores"),
        (1.0, "uncalibrated_conditional_option_scores"),
        (0.999, "temperature_scaled_requires_held_out_validation"),
        (2.0, "temperature_scaled_requires_held_out_validation"),
    ],
)
def test_probability_status_says_whether_a_temperature_was_applied(temperature, label):
    q = parse_question("boolean", policy=NO_ABSTENTION)
    assert decode(q, [0.0, 1.0], temperature)["probability_status"] == label


def test_decode_rejects_a_bad_temperature():
    with pytest.raises(ValueError, match=exactly("Temperature must be finite and positive")):
        decode(parse_question("boolean"), [0.0, 1.0, 0.0], 0)


# decode: status policy ---------------------------------------------------------------------


def test_unavailable_mass_at_the_limit_is_rejected():
    q = parse_question(
        "choice", options=THREE_OPTIONS, policy={"max_unavailable_probability": 0.25}
    )
    result = decode(q, [0.0] * 4)  # four equal candidates: the abstention holds exactly 0.25
    assert result["uncertainty"]["unavailable_probability"] == 0.25
    assert result["status"] == "insufficient_evidence"
    assert result["choice"] is None
    lenient = parse_question(
        "choice", options=THREE_OPTIONS, policy={"max_unavailable_probability": 0.26}
    )
    assert decode(lenient, [0.0] * 4)["status"] == "ok"


def test_top_probability_at_the_limit_is_accepted():
    limit = parse_question("boolean", policy=NO_ABSTENTION | {"min_top_probability": 0.5})
    assert decode(limit, [0.0, 0.0])["status"] == "ok"
    above = parse_question("boolean", policy=NO_ABSTENTION | {"min_top_probability": 0.51})
    assert decode(above, [0.0, 0.0])["status"] == "uncertain"


def test_an_unavailable_winner_is_rejected_even_under_a_lenient_limit():
    q = parse_question("boolean", policy={"max_unavailable_probability": 1.0})
    result = decode(q, logs(0.3, 0.3, 0.4))
    assert result["uncertainty"]["unavailable_probability"] == pytest.approx(0.4)
    assert result["status"] == "insufficient_evidence"
    assert result["value"] is None
    assert result["probability_true_given_available"] == pytest.approx(0.5)


def test_unavailability_takes_precedence_over_low_confidence():
    q = parse_question("boolean", policy={"min_top_probability": 0.9})
    assert decode(q, logs(0.3, 0.3, 0.4))["status"] == "insufficient_evidence"


def test_uncertain_answers_have_no_primary_value_in_any_type():
    policy = NO_ABSTENTION | {"min_top_probability": 0.9}
    boolean = decode(parse_question("boolean", policy=policy), [0.0, 0.0])
    choice = decode(parse_question("choice", policy=policy), [0.0, 0.0])
    score = decode(parse_question("score", policy=policy), [0.0, 0.0, 0.0])
    numeric = decode(numeric_question(False, [1, 2, 3, 4], min_top_probability=0.9), [0.0] * 6)
    results = [boolean, choice, score, numeric]
    assert {r["status"] for r in results} == {"uncertain"}
    assert (boolean["value"], choice["choice"], score["score"], numeric["value"]) == (None,) * 4


def test_range_mass_beats_a_larger_single_abstention():
    # Below plus above (0.4) outweigh the abstention (0.35), so the answer is out of range.
    q = numeric_question(True, anchors=[10, 20])
    result = decode(q, logs(0.1, 0.15, 0.2, 0.2, 0.35))
    assert result["status"] == "out_of_range"
    assert result["value"] is None
    assert result["statistics_given_available"] is None


def test_equal_range_and_abstention_mass_counts_as_missing_evidence():
    q = numeric_question(True, anchors=[10, 20])
    # exp(-1000) underflows to zero: 0.5 below, 0.5 abstention, nothing else.
    result = decode(q, [-1000.0, -1000.0, 0.0, -1000.0, 0.0])
    assert result["status"] == "insufficient_evidence"


def test_numeric_out_of_range_reports_which_side_holds_the_mass():
    q = numeric_question(True, anchors=[10, 20])
    result = decode(q, [0.0, 0.0, -1000.0, 30.0, 0.0])
    assert result["status"] == "out_of_range"
    assert result["range_probabilities"]["above"] > 0.99
    assert result["range_probabilities"]["below"] == 0.0


def test_numeric_range_mass_at_the_limit_is_out_of_range():
    q = numeric_question(False, anchors=[10, 20])
    result = decode(q, [0.0] * 4)  # each candidate holds 0.25; below plus above hold 0.5
    assert result["uncertainty"]["unavailable_probability"] == 0.5
    assert result["status"] == "out_of_range"
    lenient = numeric_question(False, anchors=[10, 20], max_unavailable_probability=0.51)
    assert decode(lenient, [0.0] * 4)["status"] == "ok"


def test_when_no_candidate_is_available_the_conditional_statistics_are_absent():
    boolean = decode(parse_question("boolean"), [-1000.0, -1000.0, 0.0])
    assert boolean["status"] == "insufficient_evidence"
    assert boolean["value"] is None
    assert boolean["probability_true_given_available"] is None
    score = decode(parse_question("score", levels=["a", "b"]), [-1000.0, -1000.0, 0.0])
    assert score["score"] is None
    assert score["statistics_given_available"] is None
    assert score["normalized_score"] is None


# decode: uncertainty -----------------------------------------------------------------------


def test_uniform_distribution_has_no_concentration():
    for count in (2, 3, 5):
        q = parse_question(
            "choice",
            options=[{"id": f"o{i}", "description": f"option {i}"} for i in range(count)],
            policy=NO_ABSTENTION,
        )
        uncertainty = decode(q, [0.0] * count)["uncertainty"]
        assert uncertainty["entropy_nats"] == pytest.approx(math.log(count))
        assert uncertainty["top_probability"] == pytest.approx(1 / count)
        assert uncertainty["unavailable_probability"] == 0.0
        assert uncertainty["concentration"] == pytest.approx(0.0, abs=1e-12)


def test_concentration_never_goes_below_zero():
    # In floating point 1 - entropy/ln(n) is slightly negative for some n (-2.2e-16 for n = 5).
    for count in range(2, 27):
        options = [{"id": f"o{i}", "description": f"option {i}"} for i in range(count)]
        q = parse_question("choice", options=options, policy=NO_ABSTENTION)
        concentration = decode(q, [0.0] * count)["uncertainty"]["concentration"]
        assert 0.0 <= concentration <= 1e-12, count


def test_a_one_hot_distribution_is_fully_concentrated():
    q = parse_question("choice", options=THREE_OPTIONS, policy=NO_ABSTENTION)
    uncertainty = decode(q, [-1000.0, 0.0, -1000.0])["uncertainty"]
    assert uncertainty == {
        "top_probability": 1.0,
        "entropy_nats": 0.0,
        "concentration": 1.0,
        "unavailable_probability": 0.0,
    }
