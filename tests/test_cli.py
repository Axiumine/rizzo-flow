"""The `rizzo` command line, driven through `main()` the way the installed script runs it.

The model loader and the web server are replaced by recording fakes: no network, GPU, real
weights or llama.cpp runtime is needed."""

import json
import sys
from types import SimpleNamespace

import pytest
import uvicorn
from test_service import FakeBackend

from rizzo_flow import cli, loader


class ClosableBackend(FakeBackend):
    """Counts how often its memory was given back."""

    def __init__(self):
        super().__init__()
        self.closed = 0

    def close(self):
        self.closed += 1


def fake_loader(monkeypatch):
    """Replace the model loader; `calls` records (backend name, options) of every load."""
    state = SimpleNamespace(calls=[], model=ClosableBackend())

    def load_backend(backend="llama", **options):
        state.calls.append((backend, options))
        return state.model

    monkeypatch.setattr(loader, "load_backend", load_backend)
    return state


def fake_uvicorn(monkeypatch):
    """Replace the web server; `runs` records (app, options). `hook(app, options)` runs inside
    the call and `error` is raised from it, the way a server that fails to bind would."""
    state = SimpleNamespace(runs=[], hook=None, error=None)

    def run(app, **options):
        state.runs.append((app, options))
        if state.hook:
            state.hook(app, options)
        if state.error:
            raise state.error

    monkeypatch.setattr(uvicorn, "run", run)
    return state


class Rizzo:
    """Runs `main()` with `sys.argv` set as the shell would and captures what it printed."""

    def __init__(self, monkeypatch, capsys):
        self.monkeypatch = monkeypatch
        self.capsys = capsys

    def run(self, *argv):
        self.monkeypatch.setattr(sys, "argv", ["rizzo", *map(str, argv)])
        cli.main()
        return self.capsys.readouterr()

    def fail(self, *argv, code=1):
        with pytest.raises(SystemExit) as raised:
            self.run(*argv)
        assert raised.value.code == code
        return self.capsys.readouterr(), raised.value


@pytest.fixture
def rizzo(monkeypatch, capsys):
    # Python 3.14 colors argparse output when the environment asks for it.
    for name in ("FORCE_COLOR", "PYTHON_COLORS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return Rizzo(monkeypatch, capsys)


@pytest.fixture
def loaded(monkeypatch):
    return fake_loader(monkeypatch)


@pytest.fixture
def served(monkeypatch):
    return fake_uvicorn(monkeypatch)


# serve -----------------------------------------------------------------------------------------


def test_serve_refuses_a_key_that_is_not_ascii_before_it_loads_the_model(
    rizzo, loaded, served, monkeypatch
):
    monkeypatch.setenv("RIZZO_API_KEY", "cl\N{LATIN SMALL LETTER E WITH ACUTE}")
    (out, err), exit_ = rizzo.fail("serve")
    assert (out, err) == ("", "rizzo: RIZZO_API_KEY must be ASCII\n")
    assert isinstance(exit_.__cause__, ValueError)
    assert (loaded.calls, served.runs) == ([], [])  # not after the ~10 s the weights take


def test_a_key_that_is_not_ascii_matters_to_serve_only(rizzo, loaded, tmp_path, monkeypatch):
    monkeypatch.setenv("RIZZO_API_KEY", "cl\N{LATIN SMALL LETTER E WITH ACUTE}")
    request = {"state": {"a": 1}, "questions": {"q": {"type": "boolean", "instructions": "x"}}}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    out, err = rizzo.run("decide", path)
    assert list(json.loads(out)["answers"]) == ["q"]  # a decision was made
    assert (err, loaded.model.closed) == ("", 1)
