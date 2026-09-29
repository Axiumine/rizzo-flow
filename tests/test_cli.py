"""The `rizzo` command line, driven through `main()` the way the installed script runs it.

The model loader and the web server are replaced by recording fakes: no network, GPU, real
weights or llama.cpp runtime is needed."""

import json
import math
import re
import runpy
import sys
import threading
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi.testclient import TestClient
from test_service import FakeBackend

from rizzo_flow import cli, loader
from rizzo_flow.calibration import Calibration


class ClosableBackend(FakeBackend):
    """Counts how often its memory was given back."""

    def __init__(self):
        super().__init__()
        self.closed = 0

    def close(self):
        self.closed += 1


class BusyBackend(ClosableBackend):
    """A decision that takes as long as the test lets it, with a journal of what happened."""

    def __init__(self):
        super().__init__()
        self.journal = []
        self.started = threading.Event()
        self.finish = threading.Event()

    def score(self, prefix, jobs, mode):
        self.journal.append("score:start")
        self.started.set()
        self.finish.wait(5)
        self.journal.append("score:end")
        return super().score(prefix, jobs, mode)

    def close(self):
        self.journal.append("close")
        super().close()


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
    """The three commands that load a model, each with the arguments it cannot do without."""
    return {
        "decide": ["decide", request_file],
        "evaluate": ["evaluate", fixtures_file],
        "serve": ["serve"],
    }


def write_calibration(path, fingerprint="test-only", boolean=2.0, choice=4.0):
    calibration = Calibration(
        fingerprint=fingerprint,
        dataset_sha256="0" * 64,
        temperatures={"boolean": boolean, "choice": choice},
        fit_metrics={},
    )
    path.write_text(calibration.model_dump_json(), encoding="utf-8")
    return path


# decide, evaluate, serve: loading the model ----------------------------------------------------


@pytest.fixture(params=["mismatch", "ctx"])
def bad_setup(request, tmp_path):
    """Flags that parse but fail once the model is loaded, while the engine is being set up,
    with the error each one raises."""
    if request.param == "mismatch":
        calibration = write_calibration(tmp_path / "calibration.json", fingerprint="another-model")
        return ["--calibration", calibration], "fitted for a different model/runtime/prompt"
    return ["--ctx", "0"], "ctx must be positive"


@pytest.fixture(params=["missing", "invalid"])
def bad_calibration(request, tmp_path):
    """A --calibration file that cannot be used, with the error it raises."""
    if request.param == "missing":
        return tmp_path / "missing-calibration.json", "missing-calibration.json"
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"fingerprint": 3}', encoding="utf-8")
    return calibration, "validation error"


def test_setup_errors_after_the_load_are_one_line(rizzo, loaded, request_file, bad_setup):
    flags, message = bad_setup
    (out, err), _ = rizzo.fail("decide", request_file, *flags)
    assert out == ""
    assert err.startswith("rizzo: ")
    assert message in err


@pytest.mark.parametrize("command", ["decide", "evaluate", "serve"])
def test_the_model_is_released_when_the_setup_after_the_load_fails(
    rizzo, loaded, served, commands, bad_setup, command
):
    flags, _ = bad_setup
    rizzo.fail(*commands[command], *flags)
    assert len(loaded.calls) == 1
    assert loaded.model.closed == 1  # Metal aborts at exit when the context outlives Python
    assert served.runs == []


@pytest.mark.parametrize("command", ["decide", "evaluate", "serve"])
def test_a_calibration_file_that_cannot_be_read_fails_before_the_model_is_loaded(
    rizzo, loaded, served, commands, bad_calibration, command
):
    path, message = bad_calibration
    (out, err), _ = rizzo.fail(*commands[command], "--calibration", path)
    assert out == ""
    assert err.startswith("rizzo: ")
    assert message in err
    assert loaded.calls == []
    assert served.runs == []


# evaluate --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "content", "message"),
    [
        (["--repeats", "0"], None, "Provide fixtures and at least one repetition"),
        (["--repeats", "-2"], None, "Provide fixtures and at least one repetition"),
        ([], "", "Provide fixtures and at least one repetition"),
        ([], "\n  \n", "Provide fixtures and at least one repetition"),
        ([], "not json\n", "Expecting value"),
    ],
)
def test_evaluate_reports_unusable_fixtures_before_loading_the_weights(
    rizzo, loaded, fixtures_file, tmp_path, argv, content, message
):
    path = fixtures_file
    if content is not None:
        path = tmp_path / "other.jsonl"
        path.write_text(content, encoding="utf-8")
    (out, err), _ = rizzo.fail("evaluate", path, *argv)
    assert out == ""
    assert err.startswith(f"rizzo: {message}")
    assert err.endswith("\n")
    assert loaded.calls == []


NOT_AN_OBJECT = "is not a JSON object"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"id": "x"}, 'Fixture 2 has no "request"'),
        ({"request": {}}, 'Fixture 2 has no "id"'),
        ({}, 'Fixture 2 has no "id"'),
        ([1], f"Fixture 2 {NOT_AN_OBJECT}"),
        ("text", f"Fixture 2 {NOT_AN_OBJECT}"),
        (None, f"Fixture 2 {NOT_AN_OBJECT}"),
        ({"id": "x", "request": "text"}, f'Fixture 2: "request" {NOT_AN_OBJECT}'),
        ({"id": "x", "request": None}, f'Fixture 2: "request" {NOT_AN_OBJECT}'),
        ({"id": "x", "request": {}, "expected": None}, f'Fixture 2: "expected" {NOT_AN_OBJECT}'),
        ({"id": "x", "request": {}, "expected": [1]}, f'Fixture 2: "expected" {NOT_AN_OBJECT}'),
        (
            {"id": "x", "request": {}, "expected": {"route": "access"}},
            f"Fixture 2: expected route {NOT_AN_OBJECT}",
        ),
        (
            {"id": "x", "request": {}, "expected": {"route": {"label": "access"}, "next": []}},
            f"Fixture 2: expected next {NOT_AN_OBJECT}",
        ),
    ],
)
def test_evaluate_reports_a_malformed_fixture_as_an_error(
    rizzo, loaded, tmp_path, fixtures, row, message
):
    # The second record is the bad one: every fixture is checked, not only the first.
    path = write_jsonl(tmp_path / "bad.jsonl", [fixtures[0], row])
    (out, err), exit_ = rizzo.fail("evaluate", path)
    assert (out, err) == ("", f"rizzo: {message}\n")
    assert isinstance(exit_.__cause__, ValueError)
    assert loaded.calls == []


def test_evaluate_validates_every_request_before_loading_the_weights(
    rizzo, loaded, tmp_path, fixtures
):
    broken = {"id": "broken", "request": {"state": "x", "questions": {}}, "expected": {}}
    path = write_jsonl(tmp_path / "bad.jsonl", [fixtures[0], broken])
    (out, err), exit_ = rizzo.fail("evaluate", path)
    assert out == ""
    assert err.startswith("rizzo: Fixture broken: ")
    assert "questions" in err
    assert isinstance(exit_.__cause__, ValueError)
    assert loaded.calls == []


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        ({"route": {"label": "nope"}}, "Unknown expected label nope in fixture odd"),
        ({"route": {"label": 3}}, "Unknown expected label 3 in fixture odd"),
        # A JSON list or object is not hashable: still an unknown label, not a TypeError.
        ({"route": {"label": ["access"]}}, "Unknown expected label ['access'] in fixture odd"),
        ({"route": {"label": {"a": 1}}}, "Unknown expected label {'a': 1} in fixture odd"),
        ({"billing": {"label": "true"}}, "Unknown expected question billing in fixture odd"),
        ({"billing": {"status": "ok"}}, "Unknown expected question billing in fixture odd"),
        ({"route": {"value": "12"}}, "Expected numeric targets must be finite"),
        ({"route": {"value": [1]}}, "Expected numeric targets must be finite"),
        ({"route": {"value": 10**400}}, "Expected numeric targets must be finite"),
        ({"route": {"value": 1e101}}, "Expected numeric targets must be finite"),
        ({"route": {"value": math.nan}}, "Expected numeric targets must be finite"),
    ],
)
def test_evaluate_refuses_expectations_it_could_not_score_before_loading_the_weights(
    rizzo, loaded, tmp_path, fixtures, expected, message
):
    # The last fixture is the bad one: it used to fail after every decision of the others.
    odd = {"id": "odd", "request": fixtures[0]["request"], "expected": expected}
    path = write_jsonl(tmp_path / "odd.jsonl", [fixtures[0], odd])
    (out, err), exit_ = rizzo.fail("evaluate", path)
    assert (out, err) == ("", f"rizzo: {message}\n")
    assert isinstance(exit_.__cause__, ValueError)
    assert loaded.calls == []


def fixtures_as_ascii(path, rows):
    """A JSON Lines file whose lone surrogates are written as escapes, as JSON allows."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "spoil",
    [
        lambda row: {**row, "id": "odd\ud800"},
        lambda row: {**row, "expected": {"route": {"label": "access\ud800"}}},
        lambda row: {**row, "expected": {"route\ud800": {"label": "access"}}},
        lambda row: {**row, "request": {**row["request"], "state": "odd\ud800"}},
    ],
    ids=["id", "expected-label", "expected-question", "request-state"],
)
def test_evaluate_refuses_text_utf8_cannot_encode_before_loading_the_weights(
    rizzo, loaded, tmp_path, fixtures, spoil
):
    # It used to fail after the last decision, when the report was hashed: nothing was kept.
    path = fixtures_as_ascii(tmp_path / "odd.jsonl", [fixtures[0], spoil(fixtures[0])])
    (out, err), exit_ = rizzo.fail("evaluate", path)
    assert out == ""
    assert err.startswith("rizzo: Fixture 2: Strings must be valid Unicode text")
    assert isinstance(exit_.__cause__, ValueError)
    assert loaded.calls == []


def test_evaluate_refuses_nan_in_a_fixture_before_loading_the_weights(
    rizzo, loaded, tmp_path, fixtures
):
    # json.loads takes NaN and Infinity, and the hash of the report cannot: the run used to end
    # there, after the last decision.
    odd = {
        "id": "odd",
        "request": fixtures[0]["request"],
        "expected": {"route": {"note": math.nan}},
    }
    path = write_jsonl(tmp_path / "odd.jsonl", [fixtures[0], odd])
    (out, err), exit_ = rizzo.fail("evaluate", path)
    assert out == ""
    assert re.match(r"rizzo: Fixture 2: .*not JSON compliant", err)
    assert isinstance(exit_.__cause__, ValueError)
    assert loaded.calls == []


@pytest.mark.parametrize("content", [None, "not json\n"])
def test_evaluate_reads_its_fixtures_before_loading_the_weights(rizzo, loaded, tmp_path, content):
    path = tmp_path / "fixtures.jsonl"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    (out, err), exit_ = rizzo.fail("evaluate", path)
    assert out == ""
    assert err.startswith("rizzo: ")
    assert isinstance(exit_.__cause__, (FileNotFoundError, json.JSONDecodeError))
    assert loaded.calls == []


# Ctrl-C ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["decide", "evaluate"])
def test_ctrl_c_frees_the_model_only_after_the_decode_in_flight_returned(
    rizzo, loaded, commands, monkeypatch, command
):
    busy = loaded.model = BusyBackend()
    real_result = Future.result
    interrupts: list[bool] = []

    def interrupted(self, timeout=None):
        if interrupts:  # Ctrl-C comes once; releasing the engine waits on a future of its own
            return real_result(self, timeout)
        interrupts.append(True)
        # What Future.result() raises in the main thread on Ctrl-C, while the decision is running.
        busy.started.wait(5)
        threading.Timer(0.3, busy.finish.set).start()
        raise KeyboardInterrupt

    monkeypatch.setattr(Future, "result", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            rizzo.run(*commands[command])
        # Freeing the context of a llama.cpp model under a running llama_decode is a use after free.
        assert busy.journal == ["score:start", "score:end", "close"]
    finally:
        busy.finish.set()


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


def test_the_engine_of_the_served_app_is_closed_when_the_server_stops(
    rizzo, loaded, served, payload
):
    rizzo.run("serve")
    [(app, _)] = served.runs
    with TestClient(app) as client, pytest.raises(RuntimeError, match="The engine is closed"):
        client.post("/v1/decisions", json=payload)


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
