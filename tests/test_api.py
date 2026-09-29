"""HTTP API (api.py): routes, status codes, authorization and the shape of error bodies."""

import json
import math
import re
import sys
from typing import Any

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from test_core_support import StubBackend, exactly, question_payload, request_payload

from rizzo_flow import api
from rizzo_flow.api import API_KEY_ENV, check_api_key, create_app, jsonable
from rizzo_flow.compat import list_models
from rizzo_flow.engine import Engine

JSON = {"content-type": "application/json"}
BOOLEAN = '{"a": {"type": "boolean", "instructions": "x"}}'


def make_client(
    api_key: str | None = "", ctx: int = 8192, raise_server_exceptions: bool = True
) -> tuple[TestClient, StubBackend]:
    backend = StubBackend()
    app = create_app(Engine(backend, ctx=ctx), api_key=api_key)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions), backend


def native_body() -> dict[str, Any]:
    return request_payload(
        state={"ticket": "Cannot log in"},
        supported=question_payload("boolean", instructions="Does the user need login help?"),
        route=question_payload("choice", instructions="Choose a queue"),
    )


def wire_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "state": "Help! My payouts have been failing for 3 days.",
        "model": "jev-latest",
        "questions": {"urgent": {"type": "noul", "instructions": "Is it urgent?"}},
    }
    body.update(overrides)
    return body


def wire_without_model(**overrides: Any) -> dict[str, Any]:
    body = wire_body(**overrides)
    del body["model"]
    return body


# jsonable ---------------------------------------------------------------------------------------


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


def test_jsonable_turns_other_objects_into_their_text():
    assert jsonable(ValueError("boom")) == "boom"
    assert jsonable(
        {
            "error": RuntimeError("bad"),
            "items": {
                3,
            },
        }
    ) == {
        "error": "bad",
        "items": "{3}",
    }
    assert jsonable(b"raw") == "b'raw'"


def test_jsonable_leaves_true_and_false_as_booleans_not_numbers():
    result = jsonable([True, False, 1, 0])
    assert [type(item) for item in result] == [bool, bool, int, int]


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


# /health and the native route --------------------------------------------------------------------


def test_health_reports_ready_with_the_model_metadata():
    client, backend = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model": backend.metadata}


def test_the_native_route_answers_typed_decisions():
    client, backend = make_client()
    response = client.post("/v1/decisions", json=native_body())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model", "mode", "answers", "calibration", "timing"}
    assert body["model"] == backend.metadata
    assert body["answers"]["supported"]["value"] is True
    assert body["answers"]["route"]["choice"] == "b"
    assert body["answers"]["route"]["input_tokens"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", body["answers"]["route"]["prompt_sha256"])
    assert [call["mode"] for call in backend.calls] == ["shared"]


def test_the_native_route_passes_the_mode_on():
    client, backend = make_client()
    assert client.post("/v1/decisions", json={**native_body(), "mode": "direct"}).status_code == 200
    assert backend.calls[0]["mode"] == "direct"


def test_a_prompt_over_the_context_limit_is_a_422_with_the_reason():
    client, backend = make_client(ctx=10)
    response = client.post("/v1/decisions", json=native_body())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert re.fullmatch(
        r"Question \w+: \d+ tokens exceeds the context limit 10 \(--ctx\); no truncation", detail
    )
    assert backend.calls == []  # nothing was sent to the model


def test_schema_violations_are_a_422_with_a_list_of_errors():
    client, _ = make_client()
    body = native_body()
    body["questions"]["route"]["options"][1]["id"] = "a"
    response = client.post("/v1/decisions", json=body)
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert set(error) >= {"type", "loc", "msg"}
    assert error["type"] == "value_error"
    assert "Option IDs must be unique" in error["msg"]
    assert error["loc"][:3] == ["body", "questions", "route"]


@pytest.mark.parametrize(
    ("content", "kind", "location"),
    [
        ('{"state": "x"}', "missing", ["body", "questions"]),
        ("not json", "json_invalid", ["body", 0]),
        ('{"state": "x", "questions": {}}', "too_short", ["body", "questions"]),
        (
            '{"state": "x", "questions": ' + BOOLEAN + ', "extra": 1}',
            "extra_forbidden",
            ["body", "extra"],
        ),
    ],
)
def test_malformed_bodies_are_a_422(content, kind, location):
    client, _ = make_client()
    response = client.post("/v1/decisions", content=content, headers=JSON)
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert (error["type"], error["loc"]) == (kind, location)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_json_numbers_are_reported_not_echoed_as_a_server_error(literal):
    # `json.loads` accepts these literals; serializing them back would have been a 500.
    client, _ = make_client()
    content = f'{{"state": {{"amount": {literal}}}, "questions": {BOOLEAN}}}'
    response = client.post("/v1/decisions", content=content, headers=JSON)
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert "not JSON compliant" in error["msg"]
    assert error["input"]["state"]["amount"] in {"<nan>", "<inf>", "<-inf>"}
    assert error["ctx"]["error"].startswith("Out of range float values")


def test_a_validation_error_over_input_nested_beyond_the_stack_is_still_a_422():
    # `json.loads` takes far more levels than the recursion limit, and pydantic echoes the input
    # of its error: the handler used to end in a RecursionError, a 500. Built here, not posted:
    # how deep a body can be parsed depends on the Python version and the OS.
    app = create_app(Engine(StubBackend()), api_key="")
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


@pytest.mark.parametrize(
    ("path", "body", "error_type", "loc"),
    [
        # extra="forbid": the field is refused as it is, and the error echoes its content.
        (
            "/v1/decisions",
            {**native_body(), "deep": nested(api.MAX_ECHO_DEPTH + 20)},
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
    body = {**(native_body() if path == "/v1/decisions" else wire_body()), "state": nested(300)}
    response = client.post(path, json=body)
    assert response.status_code == 422
    assert {error["type"] for error in response.json()["detail"]} >= {"recursion_loop"}
    assert backend.calls == []


def test_the_native_route_and_health_do_not_ask_for_the_key():
    client, _ = make_client(api_key="secret")
    assert client.get("/health").status_code == 200
    assert client.post("/v1/decisions", json=native_body()).status_code == 200


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [("get", "/v1/decisions", 405), ("post", "/health", 405), ("get", "/nowhere", 404)],
)
def test_unsupported_methods_and_paths(method, path, status):
    client, _ = make_client()
    assert getattr(client, method)(path).status_code == status


# /v1/systemone -----------------------------------------------------------------------------------


def test_systemone_answers_with_the_served_model_id():
    client, _ = make_client()
    for model in ("rizzo-latest", "rizzo-spark-x2.5-4b-q8_0", "jev-latest", "jev-9.9"):
        response = client.post("/v1/systemone", json=wire_body(model=model))
        assert response.status_code == 200, model
        assert response.json()["model"] == "rizzo-spark-x2.5-4b-q8_0"


def test_systemone_returns_the_probability_of_true_for_a_noul_question():
    client, _ = make_client()
    body = client.post("/v1/systemone", json=wire_body()).json()
    assert body["answers"]["urgent"] == {"type": "noul", "noul": pytest.approx(1.0, abs=1e-3)}
    assert body["usage"]["output_tokens"] == 0
    assert body["usage"]["input_tokens"] > 0
    assert "timing" in body["x_rizzo"]


def test_an_unknown_model_is_a_400_with_a_typed_detail():
    client, backend = make_client()
    response = client.post("/v1/systemone", json=wire_body(model="gpt-4"))
    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "error_type": "api_usage_error",
            "message": (
                "Unknown model 'gpt-4'. Use 'rizzo-latest', 'rizzo-spark-x2.5-4b-q8_0' "
                "or a jev-* alias."
            ),
        }
    }
    assert backend.calls == []


def test_the_model_is_checked_before_the_translation():
    client, _ = make_client()
    levels = {"type": "score", "instructions": "i", "criteria": ["a", " a "]}
    response = client.post("/v1/systemone", json=wire_body(model="nope", questions={"q": levels}))
    assert response.status_code == 400


def test_a_question_the_native_schema_refuses_is_a_422_with_the_message():
    client, backend = make_client()
    levels = {"type": "score", "instructions": "i", "criteria": ["a", " a "]}
    response = client.post("/v1/systemone", json=wire_body(questions={"q": levels}))
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)
    assert "Levels must have distinct descriptions" in response.json()["detail"]
    assert backend.calls == []


def test_systemone_reports_a_prompt_over_the_context_limit_as_a_422():
    client, _ = make_client(ctx=10)
    response = client.post("/v1/systemone", json=wire_body())
    assert response.status_code == 422
    assert "no truncation" in response.json()["detail"]


def test_systemone_validation_errors_are_a_list():
    client, _ = make_client()
    body = wire_body()
    del body["model"]
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert (error["type"], error["loc"]) == ("missing", ["body", "model"])


# authorization -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": ""},
        {"Authorization": "secret"},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "Bearer secret "},
        {"Authorization": "bearer secret"},
        {"Authorization": "Basic secret"},
        {"Authorization": "Bearer  secret"},
    ],
)
def test_a_missing_or_wrong_key_is_a_401_on_the_compatible_routes(headers):
    client, backend = make_client(api_key="secret")
    for response in (
        client.get("/v1/models", headers=headers),
        client.post("/v1/systemone", json=wire_body(), headers=headers),
    ):
        assert response.status_code == 401
        assert response.json() == {"detail": "Missing or invalid API key"}
    assert backend.calls == []


def test_the_right_key_opens_the_compatible_routes():
    client, _ = make_client(api_key="secret")
    headers = {"Authorization": "Bearer secret"}
    assert client.get("/v1/models", headers=headers).status_code == 200
    assert client.post("/v1/systemone", json=wire_body(), headers=headers).status_code == 200


def test_authorization_is_checked_before_the_body():
    client, _ = make_client(api_key="secret")
    assert client.post("/v1/systemone", json={}).status_code == 401
    good = {"Authorization": "Bearer secret"}
    assert client.post("/v1/systemone", json={}, headers=good).status_code == 422


def test_without_a_key_the_compatible_routes_are_open():
    client, _ = make_client(api_key="")
    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/models", headers={"Authorization": "Bearer anything"}).status_code == 200


def test_the_key_comes_from_the_environment_unless_given(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "from-env")
    engine = Engine(StubBackend())
    from_env = TestClient(create_app(engine))
    assert from_env.get("/v1/models").status_code == 401
    assert (
        from_env.get("/v1/models", headers={"Authorization": "Bearer from-env"}).status_code == 200
    )
    explicit = TestClient(create_app(engine, api_key="explicit"))
    assert (
        explicit.get("/v1/models", headers={"Authorization": "Bearer from-env"}).status_code == 401
    )
    assert (
        explicit.get("/v1/models", headers={"Authorization": "Bearer explicit"}).status_code == 200
    )
    # An explicitly empty key switches authorization off, whatever the environment says.
    assert TestClient(create_app(engine, api_key="")).get("/v1/models").status_code == 200


def test_no_key_anywhere_means_no_authorization(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    client = TestClient(create_app(Engine(StubBackend())))
    assert client.get("/v1/models").status_code == 200


def test_the_environment_variable_is_named_after_the_project():
    assert API_KEY_ENV == "RIZZO_API_KEY"


@pytest.mark.parametrize("key", ["clé", "密钥"])
def test_a_key_that_is_not_ascii_is_refused_at_startup(key, monkeypatch):
    # The server reads header bytes as Latin-1 and clients write them in other ways: a key
    # with characters above 0x7f would lock out the clients that send it in the other one.
    engine = Engine(StubBackend())
    with pytest.raises(ValueError, match=exactly("RIZZO_API_KEY must be ASCII")):
        create_app(engine, api_key=key)
    monkeypatch.setenv(API_KEY_ENV, key)
    with pytest.raises(ValueError, match=exactly("RIZZO_API_KEY must be ASCII")):
        create_app(engine)
    assert create_app(engine, api_key="")  # an explicit empty key ignores the environment


def test_check_api_key_gives_the_key_that_authorization_will_use(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "from-env")
    assert check_api_key() == "from-env"
    assert check_api_key("explicit") == "explicit"
    assert check_api_key("") == ""  # an explicit empty key ignores the environment, ASCII or not
    monkeypatch.delenv(API_KEY_ENV)
    assert check_api_key() is None


def test_any_ascii_key_works_as_a_bearer_token():
    key = "".join(chr(code) for code in range(0x21, 0x7F))  # every printable ASCII character
    client, _ = make_client(api_key=key)
    assert client.get("/v1/models", headers={"Authorization": f"Bearer {key}"}).status_code == 200
    assert client.get("/v1/models", headers={"Authorization": "Bearer x"}).status_code == 401


# a model that fails ------------------------------------------------------------------------------


class FaultyBackend(StubBackend):
    """A model that fails on requests that are fine; `fault` says how."""

    def __init__(self) -> None:
        super().__init__()
        self.fault: str | None = None

    def score(self, prefix: list[int], jobs: list[Any], mode: str) -> tuple[dict, dict]:
        if self.fault == "decode":
            raise ValueError("llama_decode returned -3: compute error")
        return super().score(prefix, jobs, mode)

    def logits_for(self, job: Any) -> list[float]:
        if self.fault == "nan":
            return [math.nan] * len(job.slots)
        if self.fault == "short":
            return [0.0]
        return super().logits_for(job)


FAULTS = {
    "decode": "llama_decode returned -3: compute error",
    "nan": "At least two finite logits are required",
    "short": "Logit count does not match the declared candidates",
}


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize(
    ("path", "body"),
    [("/v1/decisions", native_body()), ("/v1/systemone", wire_body())],
    ids=["native", "systemone"],
)
def test_a_failure_of_the_model_is_a_503_not_a_client_error(path, body, fault):
    backend = FaultyBackend()
    client = TestClient(create_app(Engine(backend), api_key=""), raise_server_exceptions=False)
    backend.fault = fault
    response = client.post(path, json=body)
    assert (response.status_code, response.json()) == (503, {"detail": FAULTS[fault]})
    backend.fault = None
    assert client.post(path, json=body).status_code == 200  # the failure did not wedge the engine


# /v1/models --------------------------------------------------------------------------------------


def test_models_lists_the_alias_the_served_model_and_the_compatibility_alias():
    client, backend = make_client()
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == list_models(backend.metadata)
    names = [model["name"] for model in response.json()["models"]]
    assert names == ["rizzo-latest", "rizzo-spark-x2.5-4b-q8_0", "jev-latest"]


# pages and assets --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "file", "marker"),
    [("/playground", "PLAYGROUND", "Rizzo Flow"), ("/snake", "SNAKE", "/v1/decisions")],
)
def test_the_pages_are_served_from_the_package(route, file, marker):
    client, _ = make_client()
    response = client.get(route)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == getattr(api, file).read_text(encoding="utf-8")
    assert marker in response.text


def test_the_logo_is_a_png():
    client, _ = make_client()
    response = client.get("/playground/logo.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == api.LOGO.read_bytes()
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_the_root_redirects_to_the_playground():
    client, _ = make_client()
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/playground"
    followed = client.get("/")
    assert followed.status_code == 200
    assert followed.text == api.PLAYGROUND.read_text(encoding="utf-8")


def test_the_pages_are_not_listed_in_the_openapi_document():
    client, _ = make_client()
    document = client.get("/openapi.json").json()
    # The version is the API's own (it advances with the endpoints, not with the package).
    assert document["info"] == {
        "title": "Rizzo Flow",
        "version": "0.2.0",
        "description": "Typed decisions with a local Spark-X2.5 model; no text generation.",
    }
    assert sorted(document["paths"]) == ["/health", "/v1/decisions", "/v1/models", "/v1/systemone"]


# lone surrogates ---------------------------------------------------------------------------------
# JSON allows "\ud800" as an escape; it is not text that can be written out as UTF-8.

BOOLEAN_QUESTION = '{"type": "boolean", "instructions": "x"}'
SURROGATE = "\ud800"


def post_json(client: TestClient, path: str, body: Any):
    """POST a body that holds lone surrogates: `json=` cannot encode them, `dumps` escapes them."""
    return client.post(path, content=json.dumps(body), headers=JSON)


def choice_wire(criteria: dict[str, Any]) -> dict[str, Any]:
    return {"q": {"type": "choice", "instructions": "x", "criteria": criteria}}


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
        ("/v1/decisions", request_payload(state=SURROGATE), "state"),
        ("/v1/decisions", request_payload(state={"a": [{SURROGATE: 1}]}), "state"),
        ("/v1/decisions", request_payload(**{SURROGATE: question_payload()}), "questions"),
        ("/v1/decisions", request_payload(q=question_payload(instructions=SURROGATE)), "questions"),
        (
            "/v1/decisions",
            request_payload(
                q=question_payload(
                    "choice",
                    options=[
                        {"id": "a", "description": "A"},
                        {"id": SURROGATE, "description": "B"},
                    ],
                )
            ),
            "questions",
        ),
        ("/v1/systemone", wire_body(state=SURROGATE), "state"),
        ("/v1/systemone", wire_body(model=SURROGATE), "model"),
        (
            "/v1/systemone",
            wire_body(questions={SURROGATE: {"type": "noul", "instructions": "x"}}),
            "questions",
        ),
        (
            "/v1/systemone",
            wire_body(questions={"q": {"type": "noul", "instructions": {"k": [SURROGATE]}}}),
            "questions",
        ),
        (
            "/v1/systemone",
            wire_body(questions=choice_wire({"a": None, SURROGATE: None})),
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
    response = post_json(client, "/v1/decisions", request_payload(state=["ok", f"x{SURROGATE}y"]))
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
    response = post_json(client, "/v1/decisions", {**native_body(), **extra})
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert (error["loc"], error["input"]) == (location, echoed)
    assert backend.calls == []


def test_a_surrogate_pair_is_an_ordinary_character():
    # JSON spells an emoji as two escapes; only an unpaired one is refused.
    client, _ = make_client()
    pair = "\\ud83d\\ude00"
    native = client.post(
        "/v1/decisions",
        content='{"state": "ok", "questions": {"' + pair + '": ' + BOOLEAN_QUESTION + "}}",
        headers=JSON,
    )
    wire = client.post(
        "/v1/systemone",
        content=(
            '{"state": "ok", "model": "jev-latest", "questions": '
            '{"' + pair + '": {"type": "noul", "instructions": "x"}}}'
        ),
        headers=JSON,
    )
    assert (native.status_code, wire.status_code) == (200, 200)
    assert list(native.json()["answers"]) == list(wire.json()["answers"]) == ["\U0001f600"]


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
