"""A chat template that the GGUF carries and cannot run, as a person meets it.

The tests of the llama.cpp backend pin the exception. These pin what reaches the user: one line
and an exit code from the command line, a JSON body with a reason from the API. Never a jinja
traceback, never a 500. The template rides along with a GGUF file that may not be ours, so it
runs in a sandbox and the sandbox may refuse it.

No library and no weights: `Session` is a fake that answers what the loading and the rendering
ask, and the file `--model` names only has to exist.
"""

import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rizzo_flow import backend_llama, cli, llama_release
from rizzo_flow.api import create_app
from rizzo_flow.backend_llama import LlamaBackend, LlamaTokenizer
from rizzo_flow.engine import Engine

META = {
    "general.architecture": "spark2_5",
    "spark2_5.embedding_length": "2560",
    "general.file_type": "7",
}
REQUEST = {
    "state": {"ticket": "Non riesco ad accedere"},
    "questions": {"supported": {"type": "boolean", "instructions": "Does the user need help?"}},
}
# (template, what the person is told). The first cannot even be compiled, so loading stops; the
# others load and fail when a request is rendered. The last one refuses on purpose and says why.
COMPILE = ("{% if %}", "The GGUF chat template does not compile: ")
RENDER = [
    pytest.param("{{ ''.__class__.__mro__ }}", "The GGUF chat template cannot run: ", id="sandbox"),
    pytest.param(
        "{{ strftime_now('%Y') }}", "The GGUF chat template cannot run: ", id="undefined-helper"
    ),
    pytest.param(
        "{{ 1 // 0 }}", "The GGUF chat template cannot run: ZeroDivisionError: ", id="python-error"
    ),
    pytest.param("{{ raise_exception('no system role') }}", "no system role", id="refusal"),
]


class Session:
    """The part of `llama_cpp.Session` that loading, rendering and releasing use."""

    pad_token = None
    eos_token = 2
    device = None
    idle_free = None
    n_ctx = 8192 + 2048

    def __init__(self, template):
        self.template = template
        self.closed = 0

    def meta(self, key):
        return META.get(key)

    def chat_template(self):
        return self.template

    def tokenize(self, text, add_special=False):
        return [ord(character) for character in text]

    def close(self):
        self.closed += 1


class Sessions:
    """Stands in for `llama_cpp.Session.load` and keeps every session it hands out."""

    def __init__(self, template):
        self.template = template
        self.made: list[Session] = []

    def load(self, gguf, **options):
        self.made.append(Session(self.template))
        return self.made[-1]


@pytest.fixture
def weights(tmp_path, monkeypatch):
    """A GGUF file of no content, and the lookup and the opening of a runtime that loading does
    answered."""
    monkeypatch.setattr(llama_release, "locate", lambda family=None: tmp_path)
    monkeypatch.setattr(backend_llama, "Library", SimpleNamespace(open=lambda directory: None))
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF stand-in for the weights")
    return path


@pytest.fixture
def request_file(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(REQUEST), encoding="utf-8")
    return path


def decide(monkeypatch, capsys, request_file, weights, template):
    """`rizzo decide` on a GGUF whose template is `template`: (sessions, stderr, exit code)."""
    sessions = Sessions(template)
    monkeypatch.setattr(backend_llama, "Session", sessions)
    monkeypatch.setattr(
        sys, "argv", ["rizzo", "decide", str(request_file), "--model", str(weights)]
    )
    with pytest.raises(SystemExit) as raised:
        cli.main()
    return sessions.made, capsys.readouterr().err, raised.value.code


@pytest.mark.parametrize(
    ("template", "told"),
    [pytest.param(*COMPILE, id="syntax"), *RENDER],
)
def test_a_chat_template_that_cannot_run_ends_decide_with_one_line_and_frees_the_model(
    monkeypatch, capsys, request_file, weights, template, told
):
    made, err, code = decide(monkeypatch, capsys, request_file, weights, template)
    assert code == 1
    assert err.startswith("rizzo: ")
    assert err.count("\n") == 1  # one line: no traceback
    assert told in err
    assert [session.closed for session in made] == [1]  # the context was given back, once


@pytest.fixture
def serve(monkeypatch):
    """A test client on an engine whose template is the one asked for; the engine is closed."""
    engines = []

    def build(template):
        session = Session(template)
        backend = LlamaBackend(session, LlamaTokenizer(session, template), {"fingerprint": "test"})
        engines.append(Engine(backend))
        return TestClient(create_app(engines[-1]), raise_server_exceptions=False)

    yield build
    for engine in engines:
        engine.close()


@pytest.mark.parametrize(("template", "told"), RENDER)
def test_a_chat_template_that_cannot_run_is_a_422_with_its_reason_and_not_a_500(
    serve, template, told
):
    reply = serve(template).post("/v1/decisions", json=REQUEST)
    assert reply.status_code == 422
    assert reply.json()["detail"].startswith(told)


def test_a_server_whose_template_cannot_run_keeps_answering(serve):
    """Nothing is left in a bad state by a refused request: the next one is refused the same way."""
    client = serve("{{ ''.__class__.__mro__ }}")
    statuses = [client.post("/v1/decisions", json=REQUEST).status_code for _ in range(3)]
    assert statuses == [422, 422, 422]
    assert client.get("/health").status_code == 200
