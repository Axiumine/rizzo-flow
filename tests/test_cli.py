"""The `rizzo` command line, driven through `main()` the way the installed script runs it.

The model loader and the web server are replaced by recording fakes: no network, GPU, real
weights or llama.cpp runtime is needed."""

import json
import runpy
import sys
from pathlib import Path
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


@pytest.fixture
def payload():
    return {
        "state": {"ticket": "Non riesco ad accedere: l'abbonamento è già pagato ✓"},
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


@pytest.fixture
def request_file(tmp_path, payload):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def fixtures(payload):
    return [
        {
            "id": "sample",
            "request": payload,
            "expected": {
                "route": {"label": "access", "status": "ok"},
                "supported": {"label": "true"},
            },
        }
    ]


def write_jsonl(path, rows):
    lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(lines, encoding="utf-8")
    return path


@pytest.fixture
def fixtures_file(tmp_path, fixtures):
    return write_jsonl(tmp_path / "fixtures.jsonl", fixtures)


@pytest.fixture
def commands(request_file, fixtures_file):
    """The commands that write a result, each with the arguments it cannot do without."""
    return {
        "decide": ["decide", request_file],
        "evaluate": ["evaluate", fixtures_file],
    }


# create-only results ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["decide", "evaluate"])
def test_results_are_written_once_and_never_overwritten(rizzo, loaded, commands, tmp_path, command):
    target = tmp_path / "results" / "run.json"
    out, err = rizzo.run(*commands[command], "--output", target)
    assert (out, err) == ("", "")
    first = target.read_text(encoding="utf-8")
    assert json.loads(first)
    (out, err), exit_ = rizzo.fail(*commands[command], "--output", target)
    assert out == ""
    assert target.name in err
    assert isinstance(exit_.__cause__, FileExistsError)
    assert target.read_text(encoding="utf-8") == first
    assert len(loaded.calls) == 1  # the second run stopped before it loaded the weights
    assert loaded.model.closed == 1


@pytest.mark.parametrize("command", ["decide", "evaluate"])
def test_an_existing_output_is_refused_before_loading_the_weights(
    rizzo, loaded, commands, tmp_path, command
):
    target = tmp_path / "run.json"
    target.write_text("earlier evidence", encoding="utf-8")
    (out, err), exit_ = rizzo.fail(*commands[command], "--output", target)
    with pytest.raises(FileExistsError) as native:
        target.open("x")  # what writing the result would have said
    assert (out, err) == ("", f"rizzo: {native.value}\n")
    assert isinstance(exit_.__cause__, FileExistsError)
    assert loaded.calls == []
    assert target.read_text(encoding="utf-8") == "earlier evidence"


def test_a_taken_output_is_refused_under_its_name_as_text(tmp_path):
    # The error of a `Path` would read `PosixPath('...')` instead of naming the file.
    taken = tmp_path / "report.json"
    taken.write_text("evidence", encoding="utf-8")
    with pytest.raises(FileExistsError) as raised:
        cli.refuse_existing(taken)
    assert raised.value.filename == str(taken)
    assert str(raised.value) == f"[Errno 17] File exists: {str(taken)!r}"
    cli.refuse_existing(None)
    cli.refuse_existing(str(tmp_path / "later.json"))  # nothing there: nothing to refuse


@pytest.mark.parametrize("command", ["decide", "evaluate"])
def test_the_folder_of_the_output_is_made_before_the_weights_are_loaded(
    rizzo, monkeypatch, commands, tmp_path, command
):
    target = tmp_path / "reports" / "deep" / "run.json"
    seen = []
    model = ClosableBackend()

    def load_backend(backend="llama", **options):
        seen.append(target.parent.is_dir())
        return model

    monkeypatch.setattr(loader, "load_backend", load_backend)
    out, err = rizzo.run(*commands[command], "--output", target)
    assert (out, err) == ("", "")
    assert seen == [True]
    assert json.loads(target.read_text(encoding="utf-8"))


@pytest.mark.parametrize("command", ["decide", "evaluate"])
def test_an_output_that_cannot_be_created_is_refused_before_loading_the_weights(
    rizzo, loaded, commands, tmp_path, command
):
    # A file where the folder of the report should be: the run used to end in this error, after
    # the last decision, with nothing to show for it.
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder", encoding="utf-8")
    (out, err), exit_ = rizzo.fail(*commands[command], "--output", blocker / "run.json")
    assert out == ""
    assert err.startswith("rizzo: ")
    assert blocker.name in err
    assert isinstance(exit_.__cause__, OSError)
    assert loaded.calls == []
    assert blocker.read_text(encoding="utf-8") == "a file, not a folder"


@pytest.mark.parametrize("command", ["decide", "evaluate"])
def test_an_output_that_appears_while_the_weights_load_is_still_refused(
    rizzo, monkeypatch, commands, tmp_path, command
):
    # The early check is a courtesy: the file is still created exclusively when it is written.
    target = tmp_path / "run.json"
    model = ClosableBackend()

    def load_backend(backend="llama", **options):
        target.write_text("evidence of another run", encoding="utf-8")
        return model

    monkeypatch.setattr(loader, "load_backend", load_backend)
    (out, err), exit_ = rizzo.fail(*commands[command], "--output", target)
    assert out == ""
    assert target.name in err
    assert isinstance(exit_.__cause__, FileExistsError)
    assert target.read_text(encoding="utf-8") == "evidence of another run"
    assert model.closed == 1


def test_a_dangling_symlink_is_an_existing_output_too(rizzo, loaded, request_file, tmp_path):
    target = tmp_path / "run.json"
    aimed_at = tmp_path / "nowhere.json"
    try:
        target.symlink_to(aimed_at)
    except (OSError, NotImplementedError):  # Windows without the right to make symlinks
        pytest.skip("cannot create symlinks here")
    assert not target.exists()  # exists() follows the link to nothing; creating the file fails
    (out, err), exit_ = rizzo.fail("decide", request_file, "--output", target)
    assert out == ""
    assert target.name in err
    assert isinstance(exit_.__cause__, FileExistsError)
    assert loaded.calls == []
    assert not aimed_at.exists()


def test_the_typed_decisions_script_refuses_a_taken_output_before_it_reads_its_data(
    monkeypatch, tmp_path
):
    taken = tmp_path / "report.json"
    taken.write_text("earlier evidence", encoding="utf-8")
    script = Path(__file__).parent.parent / "scripts" / "typed_decisions.py"
    argv = [script.name, str(tmp_path / "no-such-data.jsonl"), "--output", str(taken)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError):  # not FileNotFoundError, which reading the data would say
        runpy.run_path(str(script), run_name="__main__")
    assert taken.read_text(encoding="utf-8") == "earlier evidence"


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
