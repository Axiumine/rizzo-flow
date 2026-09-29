"""The `rizzo` command line, driven through `main()` the way the installed script runs it.

The model loader and the web server are replaced by recording fakes: no network, GPU, real
weights or llama.cpp runtime is needed."""

import argparse
import importlib
import io
import json
import math
import os
import re
import runpy
import sys
import threading
from concurrent.futures import Future
from contextlib import redirect_stdout
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


# write_json ------------------------------------------------------------------------------------


def test_write_json_leaves_nothing_behind_when_the_text_cannot_be_encoded(tmp_path):
    target = tmp_path / "later" / "out.json"
    with pytest.raises(UnicodeEncodeError):
        cli.write_json({"id": "\ud800"}, target)  # JSON can spell a lone surrogate, UTF-8 cannot
    assert not target.parent.exists()


def ansi_console():
    """A stdout as a Windows pipe presents it: text in the ANSI code page, over bytes."""
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding="cp1252", newline="\n")


def test_write_json_on_a_console_that_cannot_encode_the_text():
    # A Windows pipe uses the ANSI code page, and the JSON keeps non-ASCII text as it is.
    value = {"description": "\N{CHECK MARK} \N{CYRILLIC SMALL LETTER ZHE}"}
    raw, console = ansi_console()
    with redirect_stdout(console):
        cli.write_json(value, None)
    console.flush()
    assert json.loads(raw.getvalue().decode("utf-8")) == value


def test_write_json_emits_utf8_even_where_the_console_could_encode_the_text():
    # cp1252 holds "è", but a reader of JSON expects UTF-8: the bytes must not be cp1252 ones.
    raw, console = ansi_console()
    with redirect_stdout(console):
        cli.write_json({"name": "è già"}, None)
    console.flush()
    assert raw.getvalue() == '{\n  "name": "è già"\n}\n'.encode()


def test_write_json_writes_the_same_bytes_to_a_file_where_text_files_get_crlf(
    tmp_path, monkeypatch
):
    # Without newline=, a text file gets os.linesep for "\n": "\r\n" on Windows. Emulate that here,
    # since nothing else in this test tells the platforms apart.
    real_open = Path.open

    def open_as_on_windows(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode and newline is None:
            newline = "\r\n"
        return real_open(self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", open_as_on_windows)
    value = {"name": "è ✓", "list": [1, 2]}
    raw, console = ansi_console()
    with redirect_stdout(console):
        cli.write_json(value, None)
    console.flush()
    target = tmp_path / "report.json"
    cli.write_json(value, target)
    assert target.read_bytes() == raw.getvalue()  # the digest of a report is the same everywhere
    assert b"\r" not in target.read_bytes()


def test_write_json_writes_lf_whatever_the_platform_newline(tmp_path, monkeypatch):
    """`write_json` opening the file without `newline="\\n"`, or with `newline=None`.

    A text file then gets os.linesep for every "\\n": "\\r\\n" on Windows, where the report would
    not hash as the LF blob that git keeps, so its line in results/SHA256SUMS could not be
    checked. Linux and macOS cannot tell, so this makes the platform one that can: the C
    implementation of `io` has the newline of the platform built in, the Python one asks os.
    """
    monkeypatch.setattr(os, "linesep", "\r\n")
    monkeypatch.setattr(io, "open", importlib.import_module("_pyio").open)  # no stub for it
    target = tmp_path / "report.json"
    cli.write_json({"n": 1}, target)
    assert target.read_bytes() == b'{\n  "n": 1\n}\n'


def test_write_json_keeps_the_order_of_what_was_printed_around_it():
    raw, console = ansi_console()  # buffers text until it is flushed
    with redirect_stdout(console):
        print("before")
        cli.write_json({"n": 1}, None)
        print("after")
    console.flush()
    assert raw.getvalue().decode("utf-8") == 'before\n{\n  "n": 1\n}\nafter\n'


def test_write_json_flushes_the_result_at_once():
    # A CUDA or Metal runtime can abort the process on its way out; the result must be out by then.
    raw = io.BytesIO()
    console = io.TextIOWrapper(io.BufferedWriter(raw, buffer_size=1 << 16), encoding="cp1252")
    with redirect_stdout(console):
        cli.write_json({"n": 1}, None)
        assert raw.getvalue() == b'{\n  "n": 1\n}\n'


def test_write_json_writes_text_to_a_stream_that_has_no_binary_side():
    stream = io.StringIO()  # no `buffer`: there is no encoding to get wrong
    with redirect_stdout(stream):
        cli.write_json({"name": "è ✓"}, None)
    assert stream.getvalue() == '{\n  "name": "è ✓"\n}\n'


# read_jsonl ------------------------------------------------------------------------------------


def test_read_jsonl_splits_records_on_newlines_only(tmp_path):
    # `json.dumps(ensure_ascii=False)` leaves these characters in a string as they are, and
    # JSON Lines ends a record at "\n" only; str.splitlines() also breaks at all three.
    record = {
        "state": "line\N{LINE SEPARATOR}break, next\N{PARAGRAPH SEPARATOR}one, nel\N{NEXT LINE}end"
    }
    path = tmp_path / "rows.jsonl"
    path.write_bytes((json.dumps(record, ensure_ascii=False) + "\n").encode())
    assert cli.read_jsonl(path) == [record]


def test_read_jsonl_keeps_every_record_when_the_strings_hold_separators(tmp_path):
    rows = [
        {"text": f"a{separator}b"}
        for separator in "\N{LINE SEPARATOR}\N{PARAGRAPH SEPARATOR}\N{NEXT LINE}"
    ]
    path = tmp_path / "rows.jsonl"
    path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\r\n" for row in rows).encode())
    assert cli.read_jsonl(path) == rows


def with_bom(path):
    """The file with the byte order mark that Windows PowerShell 5.1 (`Out-File -Encoding utf8`)
    and old versions of Notepad put at the start of a UTF-8 file."""
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    return path


def test_read_jsonl_ignores_a_byte_order_mark_at_the_start_of_the_file(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes('{"a": "è ✓"}\n[1, 2]\n'.encode())
    assert cli.read_jsonl(with_bom(path)) == [{"a": "è ✓"}, [1, 2]]


@pytest.mark.parametrize(
    "values",
    [
        ["\x85"],
        ["\u2028", "\u2029"],
        ["", " ", "a \x85\u2028\u2029 b"],
        ["\n\r\x0b\x0c\x1c\x1d\x1e", "\x85\n\u2028"],  # json.dumps escapes the control ones
    ],
)
def test_read_jsonl_keeps_strings_made_of_line_breaking_characters_whole(tmp_path, values):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values).encode()
    )
    assert cli.read_jsonl(path) == values


# calibrate -------------------------------------------------------------------------------------


def calibration_rows(count=12):
    return [
        {"type": "boolean", "logits": [0, 8], "label_index": int(i % 2 == 0)} for i in range(count)
    ]


def test_calibrate_reads_rows_that_start_with_a_byte_order_mark(rizzo, tmp_path):
    plain = write_jsonl(tmp_path / "plain.jsonl", calibration_rows())
    marked = with_bom(write_jsonl(tmp_path / "marked.jsonl", calibration_rows()))
    for source, name in ((plain, "plain"), (marked, "marked")):
        out, err = rizzo.run(
            "calibrate", source, "--fingerprint", "fp", "--output", tmp_path / f"{name}.json"
        )
        assert (out, err) == ("", "")
    # The digest is that of the rows, not of the bytes of the file that held them.
    assert (tmp_path / "marked.json").read_bytes() == (tmp_path / "plain.json").read_bytes()


def test_a_calibration_file_may_start_with_a_byte_order_mark(tmp_path):
    plain = write_calibration(tmp_path / "plain.json")
    marked = with_bom(write_calibration(tmp_path / "marked.json"))
    assert Calibration.from_file(marked) == Calibration.from_file(plain)


def test_calibrate_leaves_no_file_behind_for_a_fingerprint_utf8_cannot_encode(rizzo, tmp_path):
    # Command-line bytes that are not UTF-8 reach Python as lone surrogates (surrogateescape).
    source = write_jsonl(tmp_path / "rows.jsonl", calibration_rows())
    target = tmp_path / "later" / "calibration.json"
    (out, err), exit_ = rizzo.fail(
        "calibrate", source, "--fingerprint", "fp\udcff", "--output", target
    )
    assert out == ""
    assert err.startswith("rizzo: ")
    assert isinstance(exit_.__cause__, UnicodeEncodeError)
    assert not target.parent.exists()
    rizzo.run("calibrate", source, "--fingerprint", "fp", "--output", target)  # still free
    assert Calibration.from_file(target).fingerprint == "fp"


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


def test_decide_reads_a_request_file_that_starts_with_a_byte_order_mark(
    rizzo, loaded, request_file
):
    plain, _ = rizzo.run("decide", request_file)
    marked, err = rizzo.run("decide", with_bom(request_file))
    assert err == ""
    assert json.loads(marked)["answers"] == json.loads(plain)["answers"]


def test_evaluate_reads_fixtures_that_start_with_a_byte_order_mark(rizzo, loaded, fixtures_file):
    plain = json.loads(rizzo.run("evaluate", fixtures_file).out)
    marked = json.loads(rizzo.run("evaluate", with_bom(fixtures_file)).out)
    assert marked["dataset_sha256"] == plain["dataset_sha256"]
    assert marked["summary"]["categorical"] == plain["summary"]["categorical"]


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


@pytest.mark.parametrize("value", ["65536", "70000", "-1", "99999999999999999999"])
def test_an_out_of_range_port_is_an_error_message_not_a_traceback(rizzo, loaded, served, value):
    def bind(app, options):
        raise OverflowError("bind(): port must be 0-65535.")  # what uvicorn.run raises

    served.hook = bind
    (out, err), _ = rizzo.fail("serve", f"--port={value}", code=2)
    assert out == ""
    assert err.startswith("usage: rizzo serve")
    assert f"argument --port: {value} is not a port, use 0-65535" in err
    assert (loaded.calls, served.runs) == ([], [])  # refused before the weights were loaded


def test_a_port_that_is_not_a_number_is_a_usage_error_too(rizzo, loaded, served):
    (out, err), _ = rizzo.fail("serve", "--port", "http", code=2)
    assert out == ""
    assert err.startswith("usage: rizzo serve")
    assert "argument --port: invalid int value: 'http'" in err
    assert (loaded.calls, served.runs) == ([], [])


def test_usage_and_errors_name_the_program_rizzo_however_python_was_started(
    rizzo, loaded, monkeypatch, tmp_path
):
    """Python 3.14 names a script that runs from an archive `python.exe <path>`: that is what
    rizzo.exe is on Windows, and its usage line read "usage: python.exe C:\\...\\rizzo.exe serve"."""
    archive = ModuleType("__main__")
    archive.__spec__ = ModuleSpec("__main__", None)
    monkeypatch.setitem(sys.modules, "__main__", archive)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    (out, err), _ = rizzo.fail("serve", "--port", "http", code=2)
    assert out == ""
    assert err.startswith("usage: rizzo serve [-h]")
    assert "\nrizzo serve: error: argument --port: invalid int value: 'http'\n" in err


@pytest.mark.parametrize("value", [0, 1, 65535])
def test_the_ends_of_the_port_range_reach_uvicorn(rizzo, loaded, served, value):
    rizzo.run("serve", "--port", value)  # 0: the system picks a free port, which uvicorn reports
    assert [options["port"] for _, options in served.runs] == [value]


@pytest.mark.parametrize(
    "number", [-(10**30), -65536, -1, 0, 1, 1023, 8017, 65535, 65536, 65537, 10**30]
)
def test_port_accepts_exactly_the_numbers_a_socket_binds(number):
    if 0 <= number <= 65535:
        assert cli.port(str(number)) == number
    else:
        with pytest.raises(argparse.ArgumentTypeError, match="is not a port"):
            cli.port(str(number))


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
