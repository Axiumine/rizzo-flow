"""Mutation tests of `rizzo_flow.compat`."""

from rizzo_flow import compat


def test_the_probability_statuses_of_a_response_are_listed_in_name_order():
    """`from_native` listing the statuses of a set in the order it iterates them (`list` for
    `sorted`). Eight names make that order differ from the sorted one on any hash seed."""
    names = [f"status-{i}" for i in range(8)]
    questions = {f"q{i}": {"type": "noul", "instructions": "Is it?"} for i in range(8)}
    request = compat.SystemOneRequest.model_validate(
        {"state": "Some evidence", "model": "rizzo-latest", "questions": questions}
    )
    response = {
        "answers": {
            f"q{i}": {
                "probabilities": {"true": 0.5, "false": 0.5},
                "input_tokens": 10,
                "probability_status": names[7 - i],  # the reverse of the sorted order
            }
            for i in range(8)
        },
        "timing": {},
        "model": {},
    }
    result = compat.from_native(request, response, {}, "served")
    assert result["x_rizzo"]["probability_status"] == names
