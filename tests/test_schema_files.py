"""The JSON Schemas published at the repository root follow the models they describe."""

import json
from pathlib import Path

import pytest

from rizzo_flow.responses import Response
from rizzo_flow.schema import Request

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "model", "command"),
    [
        ("request.schema.json", Request, "rizzo schema > request.schema.json"),
        ("response.schema.json", Response, "rizzo schema --response > response.schema.json"),
    ],
)
def test_the_published_schema_matches_its_model(name, model, command):
    path = ROOT / name
    assert path.is_file(), f"{name} is missing: generate it with `{command}`"
    published = json.loads(path.read_text(encoding="utf-8"))
    assert published == model.model_json_schema(), (
        f"{name} is out of date with {model.__name__}: regenerate it with `{command}`"
    )


def test_the_two_schemas_describe_different_models():
    request = json.loads((ROOT / "request.schema.json").read_text(encoding="utf-8"))
    response = json.loads((ROOT / "response.schema.json").read_text(encoding="utf-8"))
    assert request["title"] == "Request"
    assert response["title"] == "Response"
    assert request["additionalProperties"] is False
    assert response["additionalProperties"] is False
