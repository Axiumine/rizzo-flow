import copy
import json

import pytest
from fastapi.testclient import TestClient

from rizzo_flow.api import create_app
from rizzo_flow.calibration import fit_temperature
from rizzo_flow.engine import Engine
from rizzo_flow.evaluation import evaluate
from rizzo_flow.prompts import compile_request
from rizzo_flow.schema import Request


class CharacterTokenizer:
    """Test tokenizer; deliberately distinct from real Spark tokenizer integration tests."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, value, **kwargs):
        return [ord(c) for c in value]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"


class FakeBackend:
    tokenizer = CharacterTokenizer()

    def __init__(self):
        self.metadata = {"fingerprint": "test-only"}

    def score(self, prefix, jobs, mode):
        return {j.id: [0, 10] + [0] * (len(j.slots) - 2) for j in jobs}, {"generated_tokens": 0}


class CountingBackend(FakeBackend):
    """Counts the forward passes: a request that is refused must not cost one."""

    def __init__(self):
        super().__init__()
        self.scored = 0

    def score(self, prefix, jobs, mode):
        self.scored += 1
        return super().score(prefix, jobs, mode)


@pytest.fixture
def payload():
    return {
        "state": {"ticket": "Cannot log in"},
        "questions": {
            "supported": {"type": "boolean", "instructions": "Does the user need login help?"},
            "route": {
                "type": "choice",
                "instructions": "Choose a queue",
                "options": [
                    {"id": "billing", "description": "Payment problem"},
                    {"id": "access", "description": "Login problem"},
                ],
            },
        },
    }


def test_shared_prefix_and_state_mutation(payload):
    request = Request.model_validate(payload)
    prefix, jobs = compile_request(CharacterTokenizer(), request, 8192)
    assert all(job.tokens[: len(prefix)] == prefix for job in jobs)
    payload["state"]["ticket"] = "Changed"
    other, _ = compile_request(CharacterTokenizer(), Request.model_validate(payload), 8192)
    assert other != prefix


def test_limits_reject_without_truncation(payload):
    with pytest.raises(ValueError, match="no truncation"):
        Engine(FakeBackend(), ctx=10).decide(payload)


def test_api_and_all_input_validation(payload):
    with TestClient(create_app(Engine(FakeBackend()))) as client:
        assert client.get("/health").json()["status"] == "ready"
        response = client.post("/v1/decisions", json=payload)
        assert response.status_code == 200
        assert response.json()["answers"]["route"]["choice"] == "access"
        assert response.json()["answers"]["supported"]["value"] is True
        payload["questions"]["route"]["options"][1]["id"] = "billing"
        assert client.post("/v1/decisions", json=payload).status_code == 422


@pytest.mark.parametrize("route", ["/v1/decisions", "/v1/systemone"])
@pytest.mark.parametrize(
    "headers",
    [{}, {"content-type": "text/plain"}, {"content-type": "application/x-www-form-urlencoded"}],
    ids=["no-content-type", "text-plain", "form"],
)
def test_a_body_without_a_json_content_type_is_not_read_as_json(payload, route, headers):
    """FastAPI 0.132 made strict_content_type the default. Before, a body with no Content-Type at
    all was parsed as JSON, and a page on another site can POST one to 127.0.0.1 without a CORS
    preflight (a fetch of an untyped Blob): it ran a decision on the local model."""
    if route == "/v1/systemone":
        payload = {
            "state": "Cannot log in",
            "model": "rizzo-latest",
            "questions": {"q": {"type": "noul", "instructions": "Does the user need login help?"}},
        }
    body = json.dumps(payload).encode()
    backend = CountingBackend()
    with TestClient(create_app(Engine(backend))) as client:
        response = client.post(route, content=body, headers=headers)
        assert response.status_code == 422
        assert backend.scored == 0
        as_json = {"content-type": "application/json"}
        assert client.post(route, content=body, headers=as_json).status_code == 200
        assert backend.scored == 1


def test_non_json_floats_are_a_client_error(payload):
    # `json.loads` accepts NaN and Infinity; echoing them back turned the 422 into a 500 (issue #3).
    with TestClient(create_app(Engine(FakeBackend()))) as client:
        headers = {"content-type": "application/json"}
        for body in (
            '{"state": NaN, "questions": {}}',
            '{"state": {"amount": NaN}, "questions": {"a": {"type": "boolean", "instructions": "x"}}}',
            '{"state": {"amount": Infinity}, "questions": {"a": {"type": "boolean", "instructions": "x"}}}',
        ):
            response = client.post("/v1/decisions", content=body, headers=headers)
            assert response.status_code == 422, body
            detail = response.json()["detail"]
            assert isinstance(detail, list)  # still the usual shape, and serializable
            assert detail


def test_temperature_fit_and_model_binding():
    rows = [{"type": "choice", "logits": [0, 8], "label_index": int(i % 2 == 0)} for i in range(20)]
    calibration = fit_temperature(rows, "test-only")
    assert calibration.temperatures["choice"] > 1
    metric = calibration.fit_metrics["choice"]
    assert metric["fit_nll_after"] < metric["fit_nll_before"]
    Engine(FakeBackend(), calibration=calibration)
    calibration.fingerprint = "different"
    with pytest.raises(ValueError, match="different"):
        Engine(FakeBackend(), calibration=calibration)


def test_evaluation_coverage_raw_evidence(payload):
    fixtures = [
        {
            "id": "sample",
            "request": copy.deepcopy(payload),
            "expected": {
                "route": {"label": "access", "status": "ok"},
                "supported": {"label": "true"},
            },
        }
    ]
    report = evaluate(Engine(FakeBackend()), fixtures, compare_modes=True)
    assert report["summary"]["categorical"]["accuracy"] == 1
    assert report["summary"]["mode_comparison"]["changed_argmaxes"] == 0
    assert report["rows"][0]["response"]["timing"]["generated_tokens"] == 0
