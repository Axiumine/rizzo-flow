"""HTTP API (api.py): the shape of error bodies and what the routes refuse."""

import json
import math
import sys
from typing import Any

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from test_service import FakeBackend

from rizzo_flow import api
from rizzo_flow.api import create_app, jsonable
from rizzo_flow.engine import Engine

JSON = {"content-type": "application/json"}
BOOLEAN_QUESTION = '{"type": "boolean", "instructions": "x"}'
SURROGATE = "\ud800"


class RecordingBackend(FakeBackend):
    """The FakeBackend of test_service, which also keeps what it was asked to score."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def score(self, prefix, jobs, mode):
        self.calls.append([job.id for job in jobs])
        return super().score(prefix, jobs, mode)


def make_client(raise_server_exceptions=True):
    backend = RecordingBackend()
    app = create_app(Engine(backend), api_key="")
    return TestClient(app, raise_server_exceptions=raise_server_exceptions), backend


def boolean(**overrides: Any) -> dict[str, Any]:
    return {"type": "boolean", "instructions": "Evaluate the evidence", **overrides}


def native(state: Any = "Example", **questions: dict[str, Any]) -> dict[str, Any]:
    """A body for /v1/decisions; every keyword argument is one named question."""
    return {"state": state, "questions": questions or {"q": boolean()}}


def wire(**overrides: Any) -> dict[str, Any]:
    """A body for /v1/systemone."""
    body: dict[str, Any] = {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {"urgent": {"type": "noul", "instructions": "Is it urgent?"}},
    }
    body.update(overrides)
    return body


def post_json(client: TestClient, path: str, body: Any):
    """POST a body that holds lone surrogates: `json=` cannot encode them, `dumps` escapes them."""
    return client.post(path, content=json.dumps(body), headers=JSON)


# jsonable ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (math.nan, "<nan>"),
        (math.inf, "<inf>"),
        (-math.inf, "<-inf>"),
        (1.5, 1.5),
        (0.0, 0.0),
        (7, 7),
        (True, True),
        ("text", "text"),
        (None, None),
        (
            {1: math.nan, "b": [math.inf, (2, math.nan)]},
            {"1": "<nan>", "b": ["<inf>", [2, "<nan>"]]},
        ),
        ((1, 2), [1, 2]),
        ([], []),
        ({}, {}),
    ],
)
def test_jsonable_keeps_json_values_and_names_the_rest(value, expected):
    assert jsonable(value) == expected


def nested(levels: int, bottom: Any = "bottom") -> Any:
    value = bottom
    for _ in range(levels):
        value = [value]
    return value


@pytest.mark.parametrize("wrap", [lambda v: [v], lambda v: {"k": v}], ids=["list", "dict"])
def test_jsonable_cuts_nesting_that_the_stack_could_not_walk(wrap):
    value: Any = "x"
    for _ in range(sys.getrecursionlimit() * 2):
        value = wrap(value)
    result = jsonable(value)  # the walk used to end in a RecursionError, at about this depth
    levels = 0
    while not isinstance(result, str):
        result = result[0] if isinstance(result, list) else result["k"]
        levels += 1
    assert (levels, result) == (api.MAX_ECHO_DEPTH, "<nested too deeply>")


def test_jsonable_keeps_as_many_levels_as_the_echo_depth_and_no_more():
    kept = json.dumps(jsonable(nested(api.MAX_ECHO_DEPTH)))
    cut = json.dumps(jsonable(nested(api.MAX_ECHO_DEPTH + 1)))
    assert "bottom" in kept
    assert "<nested too deeply>" not in kept
    assert "bottom" not in cut
    assert "<nested too deeply>" in cut


def test_what_is_cut_is_a_level_of_nesting_never_a_value():
    # Only a list or a dict that lies too deep is named; text, numbers and null are kept at any
    # depth they can be reached at.
    deepest = api.MAX_ECHO_DEPTH - 1
    assert jsonable(nested(deepest, [None, 1, "text", {}])) == nested(
        deepest, [None, 1, "text", "<nested too deeply>"]
    )


# the echo of an error ----------------------------------------------------------------------------


def test_a_validation_error_over_input_nested_beyond_the_stack_is_still_a_422():
    # `json.loads` takes far more levels than the recursion limit, and pydantic echoes the input
    # of its error: the handler used to end in a RecursionError, a 500. Built here, not posted:
    # how deep a body can be parsed depends on the Python version and the OS.
    app = create_app(Engine(RecordingBackend()), api_key="")
    handler: Any = app.exception_handlers[RequestValidationError]
    value: Any = "x"
    for _ in range(sys.getrecursionlimit() * 2):
        value = [{"k": value}]
    error = RequestValidationError(
        [{"type": "string_type", "loc": ("body", "state"), "msg": "bad", "input": value}]
    )
    response = handler(None, error)
    assert response.status_code == 422
    (detail,) = json.loads(response.body)["detail"]
    assert (detail["type"], detail["loc"], detail["msg"]) == (
        "string_type",
        ["body", "state"],
        "bad",
    )
    assert "<nested too deeply>" in json.dumps(detail["input"])


def wire_without_model(**overrides: Any) -> dict[str, Any]:
    body = wire(**overrides)
    del body["model"]
    return body


@pytest.mark.parametrize(
    ("path", "body", "error_type", "loc"),
    [
        # extra="forbid": the field is refused as it is, and the error echoes its content.
        (
            "/v1/decisions",
            {**native(), "deep": nested(api.MAX_ECHO_DEPTH + 20)},
            "extra_forbidden",
            ["body", "deep"],
        ),
        # The wire format ignores unknown fields: the missing model echoes the whole body.
        (
            "/v1/systemone",
            wire_without_model(deep=nested(api.MAX_ECHO_DEPTH + 20)),
            "missing",
            ["body", "model"],
        ),
    ],
    ids=["decisions", "systemone"],
)
def test_a_body_nested_deeper_than_the_echo_gets_the_depth_cut_in_its_422(
    path, body, error_type, loc
):
    # Deep enough to be cut, in a field that no validator has to follow: pydantic reads a `state`
    # nested 254 levels on Linux and macOS but only 98 on Windows.
    client, backend = make_client()
    response = client.post(path, json=body)
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert (error["type"], error["loc"]) == (error_type, loc)
    assert "<nested too deeply>" in response.text
    assert "bottom" not in response.text
    assert backend.calls == []


@pytest.mark.parametrize("path", ["/v1/decisions", "/v1/systemone"])
def test_a_state_nested_beyond_what_pydantic_follows_is_a_422_on_every_platform(path):
    # Below 254 levels on Linux and macOS, 98 on Windows, pydantic gives up (recursion_loop) and
    # its errors echo what it was reading: a 422 with a list of errors, wherever the limit is.
    client, backend = make_client(raise_server_exceptions=False)
    body = {**(native() if path == "/v1/decisions" else wire()), "state": nested(300)}
    response = client.post(path, json=body)
    assert response.status_code == 422
    assert {error["type"] for error in response.json()["detail"]} >= {"recursion_loop"}
    assert backend.calls == []


# lone surrogates ---------------------------------------------------------------------------------
# JSON allows "\ud800" as an escape; it is not text that can be written out as UTF-8.


@pytest.mark.parametrize(
    ("path", "content"),
    [
        pytest.param(
            "/v1/decisions",
            f'{{"state": "\\ud800", "questions": {{"a": {BOOLEAN_QUESTION}}}}}',
            id="native-state",
        ),
        pytest.param(
            "/v1/decisions",
            '{"state": "ok", "questions": {"a": {"type": "boolean", "instructions": "x\\ud800"}}}',
            id="native-instructions",
        ),
        pytest.param(
            "/v1/systemone",
            f'{{"state": "ok", "model": "\\ud800", "questions": {{"a": {BOOLEAN_QUESTION}}}}}',
            id="systemone-model",
        ),
        pytest.param(
            "/v1/systemone",
            '{"state": "\\ud800", "model": "jev-latest", '
            '"questions": {"a": {"type": "noul", "instructions": "x"}}}',
            id="systemone-state",
        ),
    ],
)
def test_a_lone_surrogate_in_the_body_is_a_client_error(path, content):
    client, backend = make_client(raise_server_exceptions=False)
    response = client.post(path, content=content, headers=JSON)
    assert response.status_code == 422
    assert backend.calls == []


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (
            "/v1/systemone",
            (
                '{"state": "ok", "model": "jev-latest", "questions": '
                '{"\\ud800": {"type": "noul", "instructions": "x"}}}'
            ),
        ),
        (
            "/v1/decisions",
            '{"state": "ok", "questions": {"\\ud800": ' + BOOLEAN_QUESTION + "}}",
        ),
    ],
    ids=["systemone", "native"],
)
def test_a_lone_surrogate_in_a_question_id_is_a_client_error(path, content):
    client, _ = make_client(raise_server_exceptions=False)
    assert client.post(path, content=content, headers=JSON).status_code == 422


def test_jsonable_output_can_always_be_written_as_utf_8():
    echoed = jsonable({"input": "a\ud800b", "list": ["\udfff"]})
    assert json.dumps(echoed, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a\ud800b", "a\\ud800b"),
        ("\udfff", "\\udfff"),
        ("plain é 😀", "plain é 😀"),
        ({"k\ud800": ["\udc00", 1]}, {"k\\ud800": ["\\udc00", 1]}),
        (ValueError("bad \ud800"), "bad \\ud800"),
    ],
)
def test_jsonable_writes_a_lone_surrogate_as_its_escape(value, expected):
    assert jsonable(value) == expected


@pytest.mark.parametrize(
    ("path", "body", "field"),
    [
        ("/v1/decisions", native(state=SURROGATE), "state"),
        ("/v1/decisions", native(state={"a": [{SURROGATE: 1}]}), "state"),
        ("/v1/decisions", native(**{SURROGATE: boolean()}), "questions"),
        ("/v1/decisions", native(q=boolean(instructions=SURROGATE)), "questions"),
        (
            "/v1/decisions",
            native(
                q={
                    "type": "choice",
                    "instructions": "Choose",
                    "options": [
                        {"id": "a", "description": "A"},
                        {"id": SURROGATE, "description": "B"},
                    ],
                }
            ),
            "questions",
        ),
        ("/v1/systemone", wire(state=SURROGATE), "state"),
        ("/v1/systemone", wire(model=SURROGATE), "model"),
        (
            "/v1/systemone",
            wire(questions={SURROGATE: {"type": "noul", "instructions": "x"}}),
            "questions",
        ),
        (
            "/v1/systemone",
            wire(questions={"q": {"type": "noul", "instructions": {"k": [SURROGATE]}}}),
            "questions",
        ),
        (
            "/v1/systemone",
            wire(
                questions={
                    "q": {
                        "type": "choice",
                        "instructions": "x",
                        "criteria": {"a": None, SURROGATE: None},
                    }
                }
            ),
            "questions",
        ),
    ],
    ids=[
        "native-state",
        "native-state-key",
        "native-question-id",
        "native-instructions",
        "native-option-id",
        "systemone-state",
        "systemone-model",
        "systemone-question-id",
        "systemone-structured-instructions",
        "systemone-option-key",
    ],
)
def test_both_routes_refuse_a_lone_surrogate_the_same_way(path, body, field):
    client, backend = make_client(raise_server_exceptions=False)
    response = post_json(client, path, body)
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert (error["type"], error["loc"]) == ("value_error", ["body", field])
    assert "lone surrogate" in error["msg"]
    assert backend.calls == []


def test_the_error_echoes_the_offending_text_with_the_surrogate_escaped():
    client, _ = make_client(raise_server_exceptions=False)
    response = post_json(client, "/v1/decisions", native(state=["ok", f"x{SURROGATE}y"]))
    (error,) = response.json()["detail"]
    assert error["input"] == ["ok", "x\\ud800y"]


@pytest.mark.parametrize(
    ("extra", "location", "echoed"),
    [
        ({"mode": SURROGATE}, ["body", "mode"], "\\ud800"),
        ({f"extra{SURROGATE}": 1}, ["body"], "extra\\ud800"),
    ],
    ids=["mode", "unknown-field"],
)
def test_a_lone_surrogate_outside_the_checked_text_is_shown_escaped_in_the_error(
    extra, location, echoed
):
    # Only `state` and `questions` are text the native request carries; the mode and the field
    # names are refused for what they are, and the 422 body still has to be able to show them.
    client, backend = make_client(raise_server_exceptions=False)
    response = post_json(client, "/v1/decisions", {**native(), **extra})
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert (error["loc"], error["input"]) == (location, echoed)
    assert backend.calls == []


def test_a_surrogate_pair_is_an_ordinary_character():
    # JSON spells an emoji as two escapes; only an unpaired one is refused.
    client, _ = make_client()
    pair = "\\ud83d\\ude00"
    native_response = client.post(
        "/v1/decisions",
        content='{"state": "ok", "questions": {"' + pair + '": ' + BOOLEAN_QUESTION + "}}",
        headers=JSON,
    )
    wire_response = client.post(
        "/v1/systemone",
        content=(
            '{"state": "ok", "model": "jev-latest", "questions": '
            '{"' + pair + '": {"type": "noul", "instructions": "x"}}}'
        ),
        headers=JSON,
    )
    assert (native_response.status_code, wire_response.status_code) == (200, 200)
    answers = [list(response.json()["answers"]) for response in (native_response, wire_response)]
    assert answers == [["\U0001f600"], ["\U0001f600"]]


@pytest.mark.parametrize(
    ("path", "content"),
    [
        (
            "/v1/decisions",
            b'{"state": "\xed\xa0\x80", "questions": {"a": {"type": "boolean", "instructions": "x"}}}',
        ),
        (
            "/v1/systemone",
            (
                b'{"state": "ok", "model": "jev-latest", '
                b'"questions": {"\xed\xa0\x80": {"type": "noul", "instructions": "x"}}}'
            ),
        ),
    ],
    ids=["native-state", "systemone-question-id"],
)
def test_a_surrogate_written_as_raw_bytes_is_refused_like_its_escape(path, content):
    # json.loads decodes bytes with `surrogatepass`: the three bytes of U+D800 get in as well.
    client, backend = make_client(raise_server_exceptions=False)
    response = client.post(path, content=content, headers=JSON)
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert "lone surrogate" in error["msg"]
    assert backend.calls == []
