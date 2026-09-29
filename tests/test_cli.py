"""The `rizzo` command line, driven through `main()` the way the installed script runs it.

The model loader, the installers and downloads and the web server are replaced by recording
fakes: no network, GPU, real weights or llama.cpp runtime is needed."""

import argparse
import hashlib
import io
import json
import math
import platform
import re
import runpy
import sys
import tempfile
import threading
from concurrent.futures import Future
from contextlib import redirect_stderr, redirect_stdout
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from rizzo_flow import cli, config, llama_release, loader
from rizzo_flow.calibration import Calibration
from rizzo_flow.prompts import canonical, compile_request
from rizzo_flow.responses import Response
from rizzo_flow.schema import Request

PROPERTY = settings(max_examples=40, deadline=None, database=None, derandomize=True)
JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(10**15), 10**15)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(), children, max_size=4)
    ),
    max_leaves=10,
)
COMMANDS = ("download", "schema", "devices", "calibrate", "decide", "serve", "evaluate")
# What `load_backend` receives when a command line names no model option at all.
DEFAULT_OPTIONS = {
    "size": "4b",
    "model": None,
    "quant": None,
    "weights": None,
    "bits": None,
    "device": "auto",
    "ctx": 8192,
    "batch_size": 4,
    "threads": None,
    "kv_type": None,
}
# Every model option set to a value that is not its default.
ALL_MODEL_FLAGS = [
    "--backend=mlx",
    "--size=1.7b",
    "--model=weights.gguf",
    "--quant=q4_k_m",
    "--weights=base",
    "--bits=4",
    "--device=cuda",
    "--threads=3",
    "--batch-size=7",
    "--kv-type=q8_0",
]
ROSETTA_WARNING = (
    r"rizzo: this Python is an x86_64 build running under Rosetta on an Apple Silicon Mac, "
    r"so the only package it can load is the Intel CPU one: no Metal, and decisions take "
    r"seconds instead of milliseconds\. For the GPU, recreate the environment with a native "
    r"interpreter \N{EM DASH} uv python install (\d+\.\d+) && uv sync --locked --python \1 "
    r"\N{EM DASH} and run rizzo download --only runtime again\.\n"
)


class CharacterTokenizer:
    """Test tokenizer, one token per character; deliberately not the real Spark tokenizer."""

    pad_token_id = 0
    eos_token_id = 1

    def encode(self, value, **kwargs):
        return [ord(c) for c in value]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"


class FakeBackend:
    """A loaded model that favors the second candidate of every question; it has no `close`."""

    tokenizer = CharacterTokenizer()

    def __init__(self):
        self.metadata = {"fingerprint": "test-only"}

    def score(self, prefix, jobs, mode):
        return {j.id: [0, 10] + [0] * (len(j.slots) - 2) for j in jobs}, {"generated_tokens": 0}


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


class Terminal(io.StringIO):
    """A stream that claims to be a terminal, as stderr is when a person runs `rizzo download`."""

    def isatty(self):
        return True


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
def downloads(monkeypatch):
    """Replace every installer and download; `calls` records them in order, `errors` maps a step
    to the exception it raises, `results` holds what each step returns."""
    state = SimpleNamespace(
        calls=[],
        errors={},
        results={
            "install": Path("runtimes") / "llama-fake",
            "download_gguf": Path("models") / "fake.gguf",
            "download_model": "models/fake-checkpoint",
        },
        translated=llama_release.translated,  # the real detector, for the Rosetta tests
    )

    def record(name, *arguments):
        state.calls.append((name, *arguments))
        if name in state.errors:
            raise state.errors[name]
        return state.results[name]

    def translated():
        state.calls.append(("translated",))
        return False

    def install(accelerator="auto", progress=None):
        return record("install", accelerator, progress)

    def download_gguf(size="4b", quant=None, destination=None, progress=None, variant=None):
        return record("download_gguf", size, quant, destination, progress, variant)

    def download_model(destination=None, size="4b", variant=None):
        return record("download_model", destination, size, variant)

    monkeypatch.setattr(llama_release, "install", install)
    monkeypatch.setattr(llama_release, "translated", translated)
    monkeypatch.setattr(cli, "download_gguf", download_gguf)
    monkeypatch.setattr(cli, "download_model", download_model)
    return state


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


def test_write_json_prints_indented_readable_json(capsys):
    cli.write_json({"name": "è già", "list": [1, 2], "none": None}, None)
    assert capsys.readouterr().out == (
        '{\n  "name": "è già",\n  "list": [\n    1,\n    2\n  ],\n  "none": null\n}\n'
    )


def test_write_json_creates_missing_directories_and_writes_utf8(tmp_path, capsys):
    target = tmp_path / "reports" / "nested" / "out.json"
    cli.write_json({"name": "è ✓"}, str(target))
    assert target.read_text(encoding="utf-8") == '{\n  "name": "è ✓"\n}\n'
    assert "è ✓".encode() in target.read_bytes()  # kept as text, not as an escape sequence
    assert capsys.readouterr().out == ""


def test_write_json_never_overwrites_a_file(tmp_path, capsys):
    target = tmp_path / "evidence.json"
    target.write_text("original evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        cli.write_json({"replacement": True}, target)
    assert target.read_text(encoding="utf-8") == "original evidence"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_write_json_refuses_non_json_numbers_before_touching_anything(tmp_path, capsys, number):
    target = tmp_path / "later" / "out.json"
    with pytest.raises(ValueError, match="not JSON compliant"):
        cli.write_json({"x": number}, target)
    assert not target.parent.exists()
    with pytest.raises(ValueError, match="not JSON compliant"):
        cli.write_json({"x": number}, None)
    assert capsys.readouterr().out == ""


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


@PROPERTY
@given(value=JSON_VALUES)
def test_write_json_output_parses_back_and_is_the_same_on_stdout_and_in_a_file(value):
    stream = io.StringIO()
    with redirect_stdout(stream):
        cli.write_json(value, None)
    printed = stream.getvalue()
    assert json.loads(printed) == value
    assert printed.endswith("\n")
    assert not printed.endswith("\n\n")
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "out.json"
        cli.write_json(value, target)
        assert target.read_text(encoding="utf-8") == printed


# read_jsonl ------------------------------------------------------------------------------------


def test_read_jsonl_reads_utf8_records_and_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes('{"a": "è ✓"}\n\n   \n\t\n[1, 2]\r\n"x"'.encode())
    assert cli.read_jsonl(path) == [{"a": "è ✓"}, [1, 2], "x"]
    assert cli.read_jsonl(str(path)) == [{"a": "è ✓"}, [1, 2], "x"]


def test_read_jsonl_of_an_empty_file_is_an_empty_list(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    assert cli.read_jsonl(path) == []


def test_read_jsonl_reports_a_malformed_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError, match="Expecting value"):
        cli.read_jsonl(path)


def test_read_jsonl_of_a_missing_file_is_an_os_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        cli.read_jsonl(tmp_path / "missing.jsonl")


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


# Every character that str.splitlines() breaks a line at, and two that it does not.
LINE_BREAKERS = "a \n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"


@PROPERTY
@given(values=st.lists(st.text(alphabet=LINE_BREAKERS, max_size=8), max_size=5))
def test_read_jsonl_keeps_strings_made_of_line_breaking_characters_whole(values):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.jsonl"
        lines = "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values)
        path.write_bytes(lines.encode())
        assert cli.read_jsonl(path) == values


@PROPERTY
@given(
    rows=st.lists(st.tuples(st.sampled_from(["", " ", "\t", "  \t "]), JSON_VALUES), max_size=6),
    trailer=st.sampled_from(["", " ", "\t"]),
)
def test_read_jsonl_returns_every_record_in_order_whatever_the_blank_lines(rows, trailer):
    lines = []
    for blank, value in rows:
        lines += [blank, blank + json.dumps(value) + blank]
    lines.append(trailer)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.jsonl"
        path.write_bytes("\n".join(lines).encode())
        assert cli.read_jsonl(path) == [value for _, value in rows]


# progress --------------------------------------------------------------------------------------


def shown(name, done, total, terminal=True):
    stream = Terminal() if terminal else io.StringIO()
    with redirect_stderr(stream):
        cli.progress(name, done, total)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("done", "total", "expected"),
    [
        (0, 100, "\rmodel.gguf:   0.0%"),
        (1, 3, "\rmodel.gguf:  33.3%"),
        (512, 1024, "\rmodel.gguf:  50.0%"),
        (1023, 1024, "\rmodel.gguf:  99.9%"),
        (1024, 1024, "\rmodel.gguf: 100.0%\n"),  # the file is complete: end the line
        (2048, 1024, "\rmodel.gguf: 200.0%\n"),
        (0, 0, "\rmodel.gguf: 0 MiB"),  # no Content-Length: count megabytes, never end the line
        ((1 << 20) - 1, 0, "\rmodel.gguf: 0 MiB"),
        (5 << 20, 0, "\rmodel.gguf: 5 MiB"),
        ((5 << 20) + 1000, 0, "\rmodel.gguf: 5 MiB"),
        (3 << 30, 0, "\rmodel.gguf: 3072 MiB"),
    ],
)
def test_progress_rewrites_one_line_per_file_on_a_terminal(done, total, expected):
    assert shown("model.gguf", done, total) == expected


@pytest.mark.parametrize(("done", "total"), [(1, 2), (2, 2), (5 << 20, 0)])
def test_progress_is_silent_when_stderr_is_not_a_terminal(done, total):
    assert shown("model.gguf", done, total, terminal=False) == ""


def test_progress_updates_accumulate_on_the_same_line():
    stream = Terminal()
    with redirect_stderr(stream):
        cli.progress("a.zip", 1, 4)
        cli.progress("a.zip", 4, 4)
    assert stream.getvalue() == "\ra.zip:  25.0%\ra.zip: 100.0%\n"


@PROPERTY
@given(total=st.integers(1, 10**12), share=st.floats(0, 2))
def test_progress_percentage_tracks_the_bytes_done(total, share):
    done = int(total * share)
    text = shown("f", done, total)
    assert text.startswith("\rf: ")
    assert text.endswith("\n") == (done >= total)
    percent = float(text.removeprefix("\rf: ").rstrip("\n").removesuffix("%"))
    assert abs(percent - 100 * done / total) <= 0.05 + 1e-9


# argument errors -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ((), "the following arguments are required: command"),
        (("frobnicate",), "invalid choice: 'frobnicate'"),
        (("download", "--size", "3b"), "argument --size: invalid choice: '3b'"),
        (("download", "--backend", "cuda"), "argument --backend: invalid choice: 'cuda'"),
        (("download", "--quant", "q5"), "argument --quant: invalid choice: 'q5'"),
        (("download", "--weights", "mine"), "argument --weights: invalid choice: 'mine'"),
        (("download", "--runtime", "tpu"), "argument --runtime: invalid choice: 'tpu'"),
        (("download", "--only", "all"), "argument --only: invalid choice: 'all'"),
        (("download", "--destination"), "argument --destination: expected one argument"),
        (("devices", "now"), "unrecognized arguments: now"),
        (("schema", "--output"), "argument --output: expected one argument"),
        (("calibrate", "rows.jsonl"), "required: --fingerprint, --output"),
        (("calibrate", "--fingerprint", "f", "--output", "o"), "required: input"),
        (("decide",), "the following arguments are required: input"),
        (("evaluate",), "the following arguments are required: input"),
        (("decide", "r.json", "--size", "3b"), "argument --size: invalid choice: '3b'"),
        (("serve", "--backend", "cuda"), "argument --backend: invalid choice: 'cuda'"),
        (("evaluate", "f.jsonl", "--quant", "q5"), "argument --quant: invalid choice: 'q5'"),
        (("decide", "r.json", "--weights", "mine"), "argument --weights: invalid choice: 'mine'"),
        (("decide", "r.json", "--bits", "5"), "argument --bits: invalid choice:"),
        (("decide", "r.json", "--device", "tpu"), "argument --device: invalid choice: 'tpu'"),
        (("decide", "r.json", "--kv-type", "f32"), "argument --kv-type: invalid choice: 'f32'"),
        (("decide", "r.json", "--ctx", "big"), "argument --ctx/--max-tokens: invalid int value"),
        (("decide", "r.json", "--threads", "x"), "argument --threads: invalid int value: 'x'"),
        (("decide", "r.json", "--batch-size", "x"), "argument --batch-size: invalid int value"),
        (("decide", "r.json", "--repeats", "2"), "unrecognized arguments: --repeats 2"),
        (("decide", "r.json", "--compare-modes"), "unrecognized arguments: --compare-modes"),
        (("decide", "r.json", "--port", "9"), "unrecognized arguments: --port 9"),
        (("evaluate", "f.jsonl", "--repeats", "x"), "argument --repeats: invalid int value: 'x'"),
        (("evaluate", "f.jsonl", "--host", "h"), "unrecognized arguments: --host h"),
        (("serve", "r.json"), "unrecognized arguments: r.json"),
        (("serve", "--output", "o"), "unrecognized arguments: --output o"),
        (("serve", "--port", "http"), "argument --port: invalid int value: 'http'"),
    ],
)
def test_bad_command_lines_are_usage_errors_that_do_nothing(
    rizzo, loaded, downloads, served, argv, message
):
    (out, err), _ = rizzo.fail(*argv, code=2)
    assert out == ""
    assert err.startswith("usage: rizzo")
    assert message in err
    assert loaded.calls == []
    assert downloads.calls == []
    assert served.runs == []


def test_help_lists_exactly_the_subcommands(rizzo):
    (out, err), _ = rizzo.fail("--help", code=0)
    assert err == ""
    listed = re.search(r"\{([a-z,]+)\}", out)
    assert listed
    assert listed.group(1).split(",") == list(COMMANDS)


# Every help text of the command line, as `--help` prints it (line breaks aside).
HELP = {
    (): [
        r"Rizzo Flow \N{EM DASH} local Spark typed decisions",
        "Download the pinned llama.cpp runtime for this machine and the weights",
        "Print the JSON Schema for requests",
        "Show which compute backends this install can use",
        "Fit temperatures on separate labeled logit rows",
    ],
    ("download",): [
        "GGUF file",
        (
            "flow = our fine-tune for typed decisions (default), "
            "base = the original Spark-X2.5 GGUF files"
        ),
        (
            "llama.cpp build: auto = Metal on a Mac, CUDA with an NVIDIA driver, else Vulkan "
            "(AMD, Intel and NVIDIA GPUs)"
        ),
        "Weights; default: under models/",
    ],
    ("schema",): ["Print the output schema"],
    ("decide",): [
        "GGUF file (MLX: checkpoint directory)",
        "Pinned GGUF file; default q8_0",
        (
            "Pinned weights: flow = our fine-tune for typed decisions (default), "
            "base = the original Spark-X2.5 GGUF files"
        ),
        "MLX backend only",
        "auto = best GPU of the installed runtime, else CPU; a family name requires it",
        "CPU threads (llama backend)",
        (
            "KV cache precision (llama backend): q8_0 halves it, q4_0 quarters it; "
            "default is llama.cpp's own (f16). Use it when a long --ctx does not fit"
        ),
        "Context limit in tokens per question (state + question); longer inputs are rejected",
    ],
}


@pytest.mark.parametrize("argv", list(HELP), ids=[" ".join(argv) or "rizzo" for argv in HELP])
def test_help_documents_the_commands_and_their_options(rizzo, argv):
    (out, err), _ = rizzo.fail(*argv, "--help", code=0)
    assert err == ""
    text = " ".join(out.split())
    for phrase in HELP[argv]:
        pattern = phrase if phrase.startswith("Rizzo Flow") else re.escape(phrase)
        assert re.search(rf"(?<!\w){pattern}(?!\w)", text), phrase


# download --------------------------------------------------------------------------------------


def outputs(downloads, calls):
    """What `download` prints for these calls: the path each step returned."""
    return "".join(f"{downloads.results[call[0]]}\n" for call in calls if call[0] != "translated")


def test_download_fetches_the_runtime_and_then_the_weights(rizzo, downloads):
    out, err = rizzo.run("download")
    assert downloads.calls == [
        ("translated",),
        ("install", "auto", cli.progress),
        ("download_gguf", "4b", "q8_0", None, cli.progress, None),
    ]
    assert out == f"{downloads.results['install']}\n{downloads.results['download_gguf']}\n"
    assert err == ""


@pytest.mark.parametrize(
    ("argv", "calls"),
    [
        (
            ["--only", "runtime"],
            [("translated",), ("install", "auto", cli.progress)],
        ),
        (
            ["--only=runtime", "--runtime=vulkan"],
            [("translated",), ("install", "vulkan", cli.progress)],
        ),
        (
            ["--only", "weights", "--destination", "here/x.gguf"],
            [("download_gguf", "4b", "q8_0", Path("here/x.gguf"), cli.progress, None)],
        ),
        (
            ["--backend=mlx"],
            [("download_model", None, "4b", None)],
        ),
        (
            ["--backend=mlx", "--only=weights"],
            [("download_model", None, "4b", None)],
        ),
        (
            ["--backend=mlx", "--weights=flow"],
            [("download_model", None, "4b", "flow")],
        ),
        (
            ["--backend", "mlx", "--size", "1.7b", "--weights", "base", "--destination", "ckpt"],
            [("download_model", Path("ckpt"), "1.7b", "base")],
        ),
    ],
)
def test_download_does_only_what_was_asked(rizzo, downloads, argv, calls):
    out, err = rizzo.run("download", *argv)
    assert downloads.calls == calls
    assert out == outputs(downloads, calls)
    assert err == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["--backend=mlx", "--only=runtime"],
        ["--only", "runtime", "--backend", "mlx"],
        ["--backend", "mlx", "--only", "runtime", "--size", "1.7b", "--weights", "base"],
        ["--backend=mlx", "--only=runtime", "--runtime=vulkan", "--destination=ckpt"],
    ],
)
def test_download_of_the_runtime_alone_is_refused_for_mlx_before_anything_is_fetched(
    rizzo, downloads, argv
):
    """MLX does not use the llama.cpp runtime: `--only runtime` used to fetch its weights."""
    (out, err), exit_ = rizzo.fail("download", *argv)
    assert err == (
        "rizzo: --only runtime installs the llama.cpp runtime, which --backend mlx does not use "
        "(MLX comes with the Python environment: uv sync --extra mlx|cuda|cpu); "
        "drop --only or use --only weights\n"
    )
    assert out == ""
    assert isinstance(exit_.__cause__, ValueError)
    assert downloads.calls == []  # no weights, no runtime, not even the Rosetta check


@pytest.mark.parametrize("runtime", llama_release.ACCELERATORS)
def test_download_passes_the_runtime_family_on(rizzo, downloads, runtime):
    rizzo.run("download", "--only", "runtime", "--runtime", runtime)
    assert downloads.calls == [("translated",), ("install", runtime, cli.progress)]


@pytest.mark.parametrize("size", tuple(config.MODELS))
@pytest.mark.parametrize("quant", config.QUANTS)
@pytest.mark.parametrize("weights", config.VARIANTS)
def test_download_passes_every_weights_choice_on(rizzo, downloads, size, quant, weights):
    rizzo.run(
        "download", "--only", "weights", "--size", size, "--quant", quant, "--weights", weights
    )
    assert downloads.calls == [("download_gguf", size, quant, None, cli.progress, weights)]


def test_download_warns_under_rosetta_before_the_runtime_arrives(
    rizzo, downloads, monkeypatch, capsys
):
    seen = []

    def translated():
        return True

    def install(accelerator="auto", progress=None):
        seen.append(capsys.readouterr().err)  # what the person had read when the download began
        return downloads.results["install"]

    monkeypatch.setattr(llama_release, "translated", translated)
    monkeypatch.setattr(llama_release, "install", install)
    out, err = rizzo.run("download", "--only", "runtime")
    assert len(seen) == 1
    assert re.fullmatch(ROSETTA_WARNING, seen[0])
    assert out == f"{downloads.results['install']}\n"  # the warning goes to stderr only
    assert err == ""


@pytest.mark.parametrize(
    ("system", "machine", "sysctl", "warned"),
    [
        ("darwin", "x86_64", "1\n", True),  # an Intel Python on Apple Silicon: the kernel says so
        ("darwin", "x86_64", "0\n", False),  # a real Intel Mac
        ("darwin", "x86_64", None, False),  # sysctl cannot be started
        ("darwin", "arm64", "1\n", False),  # a native interpreter: nothing to ask
        ("linux", "x86_64", "1\n", False),
        ("win32", "AMD64", "1\n", False),
    ],
)
def test_the_rosetta_warning_follows_the_platform(
    rizzo, downloads, monkeypatch, system, machine, sysctl, warned
):
    asked = []

    def run(command, **options):
        asked.append(command)
        if sysctl is None:
            raise FileNotFoundError("sysctl")
        return SimpleNamespace(stdout=sysctl)

    monkeypatch.setattr(llama_release, "translated", downloads.translated)
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr("subprocess.run", run)
    out, err = rizzo.run("download", "--only", "runtime")
    assert out == f"{downloads.results['install']}\n"
    assert downloads.calls == [("install", "auto", cli.progress)]
    if warned:
        assert re.fullmatch(ROSETTA_WARNING, err)
    else:
        assert err == ""
    asks_the_kernel = system == "darwin" and machine == "x86_64"
    assert asked == ([["sysctl", "-n", "hw.optional.arm64"]] if asks_the_kernel else [])


def test_download_of_weights_alone_does_not_look_for_rosetta(rizzo, downloads, monkeypatch):
    monkeypatch.setattr(llama_release, "translated", downloads.translated)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr("subprocess.run", lambda *args, **options: pytest.fail("asked sysctl"))
    out, err = rizzo.run("download", "--only", "weights")
    assert err == ""
    assert out == f"{downloads.results['download_gguf']}\n"


@pytest.mark.parametrize(
    ("argv", "step", "failure"),
    [
        (["download"], "install", ValueError("No prebuilt llama.cpp package for freebsd/riscv64")),
        (["download"], "download_gguf", OSError("[Errno 28] No space left on device")),
        (["download", "--only=weights"], "download_gguf", ValueError("x.gguf: sha256 mismatch")),
        (["download", "--backend=mlx"], "download_model", ImportError("No module named 'hf'")),
    ],
)
def test_download_failures_are_one_line_and_stop_the_command(rizzo, downloads, argv, step, failure):
    downloads.errors[step] = failure
    (out, err), exit_ = rizzo.fail(*argv)
    assert err == f"rizzo: {failure}\n"
    assert exit_.__cause__ is failure
    assert downloads.calls[-1][0] == step  # nothing was tried after the failing step
    assert out == outputs(downloads, downloads.calls[:-1])  # what came before was printed


def test_download_lets_unexpected_errors_through_with_their_traceback(rizzo, downloads):
    downloads.errors["install"] = RuntimeError("a bug, not a user error")
    with pytest.raises(RuntimeError, match="a bug"):
        rizzo.run("download")


# devices and schema ----------------------------------------------------------------------------


def test_devices_prints_the_report_of_the_loader(rizzo, loaded, monkeypatch):
    report = {"llama.cpp": {"release": "b1", "installed": []}, "mlx": {"installed": False}}
    monkeypatch.setattr(loader, "describe", lambda: report)
    out, err = rizzo.run("devices")
    assert out == json.dumps(report, indent=2) + "\n"
    assert err == ""
    assert loaded.calls == []


def test_a_failing_device_probe_is_a_one_line_error(rizzo, monkeypatch):
    def describe():
        raise OSError("driver unavailable")

    monkeypatch.setattr(loader, "describe", describe)
    (out, err), _ = rizzo.fail("devices")
    assert (out, err) == ("", "rizzo: driver unavailable\n")


@pytest.mark.parametrize(
    ("flags", "model", "title"), [((), Request, "Request"), (("--response",), Response, "Response")]
)
def test_schema_prints_the_request_or_the_response_schema(rizzo, loaded, flags, model, title):
    out, err = rizzo.run("schema", *flags)
    assert json.loads(out)["title"] == title
    assert out == json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
    assert err == ""
    assert loaded.calls == []


@pytest.mark.parametrize(("flags", "model"), [((), Request), (("--response",), Response)])
def test_schema_can_be_written_to_a_new_file_but_never_over_one(rizzo, tmp_path, flags, model):
    target = tmp_path / "out" / "schema.json"
    out, err = rizzo.run("schema", *flags, "--output", target)
    assert (out, err) == ("", "")
    written = target.read_text(encoding="utf-8")
    assert json.loads(written) == model.model_json_schema()
    (out, err), exit_ = rizzo.fail("schema", *flags, "--output", target)
    assert out == ""
    assert err.startswith("rizzo: ")
    assert target.name in err
    assert isinstance(exit_.__cause__, FileExistsError)
    assert target.read_text(encoding="utf-8") == written


# calibrate -------------------------------------------------------------------------------------


def calibration_rows(count=12):
    return [
        {"type": "boolean", "logits": [0, 8], "label_index": int(i % 2 == 0)} for i in range(count)
    ]


def test_calibrate_fits_temperatures_and_writes_a_file_the_other_commands_load(
    rizzo, loaded, tmp_path
):
    rows = calibration_rows()
    source = tmp_path / "rows.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")
    target = tmp_path / "fits" / "calibration.json"
    out, err = rizzo.run("calibrate", source, "--fingerprint", "fp-42", "--output", target)
    assert (out, err) == ("", "")
    fitted = Calibration.from_file(target)
    assert fitted.fingerprint == "fp-42"
    assert fitted.dataset_sha256 == hashlib.sha256(canonical(rows).encode()).hexdigest()
    assert list(fitted.temperatures) == ["boolean"]
    assert fitted.temperatures["boolean"] > 1  # half right on confident logits: soften them
    assert fitted.fit_metrics["boolean"]["rows"] == 12
    assert loaded.calls == []


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


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "Calibration requires labeled rows"),
        (
            "".join(json.dumps(row) + "\n" for row in calibration_rows(9)),
            "Provide at least 10 calibration examples for boolean",
        ),
        (
            json.dumps({"type": "boolean", "logits": [0, 8], "label_index": 2}) + "\n",
            "Label index is outside the candidate list",
        ),
        ('{"type": "boolean"}\n', "validation error"),
        ("not json\n", "Expecting value"),
    ],
)
def test_calibrate_rejects_unusable_rows(rizzo, tmp_path, content, message):
    source = tmp_path / "rows.jsonl"
    source.write_text(content, encoding="utf-8")
    target = tmp_path / "calibration.json"
    (out, err), exit_ = rizzo.fail("calibrate", source, "--fingerprint", "fp", "--output", target)
    assert out == ""
    assert err.startswith("rizzo: ")
    assert message in err
    assert isinstance(exit_.__cause__, ValueError)
    assert not target.exists()


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


def test_calibrate_reports_a_missing_input_file(rizzo, tmp_path):
    target = tmp_path / "calibration.json"
    missing = tmp_path / "missing.jsonl"
    (out, err), exit_ = rizzo.fail("calibrate", missing, "--fingerprint", "f", "--output", target)
    assert out == ""
    assert err.startswith("rizzo: ")
    assert missing.name in err
    assert isinstance(exit_.__cause__, FileNotFoundError)
    assert not target.exists()


def test_calibrate_never_overwrites_a_calibration(rizzo, tmp_path):
    source = write_jsonl(tmp_path / "rows.jsonl", calibration_rows())
    target = tmp_path / "calibration.json"
    target.write_text("previous fit", encoding="utf-8")
    (out, err), exit_ = rizzo.fail("calibrate", source, "--fingerprint", "f", "--output", target)
    assert out == ""
    assert target.name in err
    assert isinstance(exit_.__cause__, FileExistsError)
    assert target.read_text(encoding="utf-8") == "previous fit"


# decide, evaluate, serve: loading the model ----------------------------------------------------


@pytest.mark.parametrize("command", ["decide", "evaluate", "serve"])
def test_model_options_default_to_the_pinned_fine_tune(rizzo, loaded, commands, served, command):
    rizzo.run(*commands[command])
    assert loaded.calls == [("llama", DEFAULT_OPTIONS)]


@pytest.mark.parametrize("command", ["decide", "evaluate", "serve"])
@pytest.mark.parametrize("ctx_flag", ["--ctx", "--max-tokens"])
def test_every_model_option_reaches_the_loader(rizzo, loaded, commands, served, command, ctx_flag):
    rizzo.run(*commands[command], *ALL_MODEL_FLAGS, ctx_flag, "4096")
    assert loaded.calls == [
        (
            "mlx",
            {
                "size": "1.7b",
                "model": Path("weights.gguf"),
                "quant": "q4_k_m",
                "weights": "base",
                "bits": 4,
                "device": "cuda",
                "ctx": 4096,
                "batch_size": 7,
                "threads": 3,
                "kv_type": "q8_0",
            },
        )
    ]


@pytest.mark.parametrize(
    ("flag", "value", "option", "expected"),
    [
        *[("--size", value, "size", value) for value in config.MODELS],
        *[("--quant", value, "quant", value) for value in config.QUANTS],
        *[("--weights", value, "weights", value) for value in config.VARIANTS],
        *[("--device", value, "device", value) for value in loader.DEVICES],
        *[("--bits", str(value), "bits", value) for value in (4, 8)],
        *[("--kv-type", value, "kv_type", value) for value in ("f16", "q8_0", "q4_0")],
        *[("--backend", value, None, value) for value in loader.BACKENDS],
    ],
)
def test_every_documented_choice_is_accepted_and_forwarded(
    rizzo, loaded, commands, flag, value, option, expected
):
    rizzo.run(*commands["decide"], flag, value)
    [(backend, options)] = loaded.calls
    assert (backend if option is None else options[option]) == expected


def test_the_engine_gets_the_context_limit_of_the_command_line(rizzo, loaded, commands):
    limit = r"Question supported: \d+ tokens exceeds the context limit 10 \(--ctx\); no truncation"
    for flag in ("--ctx", "--max-tokens"):
        (out, err), _ = rizzo.fail(*commands["decide"], flag, "10")
        assert out == ""
        assert re.fullmatch(f"rizzo: {limit}\n", err)


def test_the_served_engine_gets_the_context_limit_too(rizzo, loaded, served, payload):
    answers = []

    def ask(app, options):  # the engine stops with the server, so ask while it is running
        with TestClient(app) as client:
            answers.append(client.post("/v1/decisions", json=payload))

    served.hook = ask
    rizzo.run("serve", "--ctx", "10")
    [response] = answers
    assert response.status_code == 422
    assert "context limit 10" in response.json()["detail"]


def test_a_backend_without_a_close_method_is_fine(rizzo, request_file, monkeypatch):
    monkeypatch.setattr(loader, "load_backend", lambda backend="llama", **options: FakeBackend())
    out, err = rizzo.run("decide", request_file)
    assert json.loads(out)["answers"]["route"]["choice"] == "access"
    assert err == ""


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


# decide ----------------------------------------------------------------------------------------


def test_decide_prints_the_typed_answers_and_releases_the_model(rizzo, loaded, request_file):
    out, err = rizzo.run("decide", request_file)
    response = json.loads(out)
    Response.model_validate(response)
    assert response["model"] == {"fingerprint": "test-only"}
    assert response["mode"] == "shared"
    assert response["calibration"] is None
    assert response["answers"]["route"]["choice"] == "access"
    assert response["answers"]["supported"]["value"] is True
    assert out == json.dumps(response, ensure_ascii=False, indent=2) + "\n"
    assert err == ""
    assert loaded.model.closed == 1


def test_decide_reads_the_request_file_as_utf8(rizzo, loaded, request_file, payload):
    assert payload["state"]["ticket"].encode() in request_file.read_bytes()
    out, _ = rizzo.run("decide", request_file)
    _, jobs = compile_request(CharacterTokenizer(), Request.model_validate(payload), 8192)
    answers = json.loads(out)["answers"]
    assert {job.id: len(job.tokens) for job in jobs} == {
        key: answer["input_tokens"] for key, answer in answers.items()
    }


def test_decide_applies_the_calibration_file(rizzo, loaded, request_file, tmp_path):
    calibration = write_calibration(tmp_path / "calibration.json")
    out, _ = rizzo.run("decide", request_file, "--calibration", calibration)
    response = json.loads(out)
    assert response["calibration"]["fingerprint"] == "test-only"
    assert response["answers"]["supported"]["temperature"] == 2.0
    assert response["answers"]["route"]["temperature"] == 4.0
    assert loaded.model.closed == 1


def test_decide_reads_a_request_file_that_starts_with_a_byte_order_mark(
    rizzo, loaded, request_file
):
    plain, _ = rizzo.run("decide", request_file)
    marked, err = rizzo.run("decide", with_bom(request_file))
    assert err == ""
    assert json.loads(marked)["answers"] == json.loads(plain)["answers"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"{", "Invalid JSON"),
        (b'{"state": "x", "questions": {}}', "questions"),
        (
            b'{"state": "x", "extra": 1, "questions": {"q": {"type": "boolean", "instructions": "?"}}}',
            "Extra inputs are not permitted",
        ),
        (b"\xff\xfe", "utf-8"),
    ],
)
def test_decide_validates_the_request_before_loading_the_weights(
    rizzo, loaded, tmp_path, content, message
):
    request = tmp_path / "request.json"
    request.write_bytes(content)
    (out, err), exit_ = rizzo.fail("decide", request)
    assert out == ""
    assert err.startswith("rizzo: ")
    assert message in err
    assert isinstance(exit_.__cause__, ValueError)
    assert loaded.calls == []


def test_decide_reports_a_missing_request_file_before_loading_the_weights(rizzo, loaded, tmp_path):
    (out, err), exit_ = rizzo.fail("decide", tmp_path / "nothing.json")
    assert out == ""
    assert "nothing.json" in err
    assert isinstance(exit_.__cause__, FileNotFoundError)
    assert loaded.calls == []


def test_decide_releases_the_model_when_the_decision_fails(rizzo, loaded, request_file):
    (out, err), _ = rizzo.fail("decide", request_file, "--ctx", "10")
    assert out == ""
    assert "context limit 10" in err
    assert loaded.model.closed == 1


# evaluate --------------------------------------------------------------------------------------


def test_evaluate_prints_the_report(rizzo, loaded, fixtures_file, fixtures):
    out, err = rizzo.run("evaluate", fixtures_file)
    report = json.loads(out)
    summary = report["summary"]
    assert err == ""
    assert report["dataset_sha256"] == hashlib.sha256(canonical(fixtures).encode()).hexdigest()
    assert [row["id"] for row in report["rows"]] == ["sample"]
    assert summary["requests"] == 1
    assert summary["repeats"] == 1
    assert summary["categorical"]["rows"] == 2
    assert summary["categorical"]["accuracy"] == 1
    assert summary["mode_comparison"] is None
    assert loaded.model.closed == 1


def test_evaluate_passes_repeats_and_mode_comparison_on(rizzo, loaded, fixtures_file, tmp_path):
    target = tmp_path / "reports" / "evaluation.json"
    out, err = rizzo.run(
        "evaluate", fixtures_file, "--repeats", "3", "--compare-modes", "--output", target
    )
    assert (out, err) == ("", "")
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["summary"]["repeats"] == 3
    assert len(report["rows"][0]["repeat_timings"]) == 3
    comparison = report["summary"]["mode_comparison"]
    assert comparison["decisions"] == 2
    assert comparison["changed_argmaxes"] == 0
    assert report["rows"][0]["alternate_mode_response"]["mode"] == "direct"
    assert loaded.model.closed == 1


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


def test_evaluate_reads_fixtures_that_start_with_a_byte_order_mark(rizzo, loaded, fixtures_file):
    plain = json.loads(rizzo.run("evaluate", fixtures_file).out)
    marked = json.loads(rizzo.run("evaluate", with_bom(fixtures_file)).out)
    assert marked["dataset_sha256"] == plain["dataset_sha256"]
    assert marked["summary"]["categorical"] == plain["summary"]["categorical"]


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


def test_serve_runs_uvicorn_on_the_local_address_by_default(rizzo, loaded, served):
    out, err = rizzo.run("serve")
    [(app, options)] = served.runs
    assert options == {"host": "127.0.0.1", "port": 8017}
    assert (out, err) == ("", "")
    assert isinstance(app, FastAPI)
    with TestClient(app) as client:
        health = client.get("/health").json()
    assert health == {"status": "ready", "model": {"fingerprint": "test-only"}}


def test_serve_takes_host_and_port_and_releases_the_model_after_the_server_stops(
    rizzo, loaded, served
):
    seen = []
    served.hook = lambda app, options: seen.append(loaded.model.closed)
    rizzo.run("serve", "--host", "192.0.2.7", "--port", "9123")
    assert [options for _, options in served.runs] == [{"host": "192.0.2.7", "port": 9123}]
    assert seen == [0]  # still loaded while serving
    assert loaded.model.closed == 1


def test_serve_answers_with_the_calibrated_engine(rizzo, loaded, served, payload, tmp_path):
    calibration = write_calibration(tmp_path / "calibration.json")
    answers = []

    def ask(app, options):  # the engine stops with the server, so ask while it is running
        with TestClient(app) as client:
            answers.append(client.post("/v1/decisions", json=payload).json())

    served.hook = ask
    rizzo.run("serve", "--calibration", calibration)
    [response] = answers
    assert response["calibration"]["fingerprint"] == "test-only"
    assert response["answers"]["route"]["choice"] == "access"


def test_serve_reports_a_server_that_cannot_start_and_releases_the_model(rizzo, loaded, served):
    served.error = OSError("[Errno 98] Address already in use")
    (out, err), _ = rizzo.fail("serve")
    assert (out, err) == ("", "rizzo: [Errno 98] Address already in use\n")
    assert loaded.model.closed == 1


def test_the_engine_of_the_served_app_is_closed_when_the_server_stops(
    rizzo, loaded, served, payload
):
    rizzo.run("serve")
    [(app, _)] = served.runs
    with TestClient(app) as client, pytest.raises(RuntimeError, match="The engine is closed"):
        client.post("/v1/decisions", json=payload)


def test_serve_releases_the_model_when_it_is_interrupted(rizzo, loaded, served):
    served.error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        rizzo.run("serve")
    assert loaded.model.closed == 1


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


def test_serve_releases_the_model_when_uvicorn_exits_the_process(rizzo, loaded, served):
    served.error = SystemExit(1)  # what uvicorn does when it cannot bind
    (out, err), _ = rizzo.fail("serve")
    assert (out, err) == ("", "")
    assert loaded.model.closed == 1


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


@pytest.mark.parametrize("program", ["rizzo", "rizzo.exe", "__main__.py"])
def test_usage_and_errors_name_the_program_rizzo_however_python_was_started(
    rizzo, loaded, monkeypatch, tmp_path, program
):
    """Python 3.14 names a script that runs from an archive `python.exe <path>`: that is what
    rizzo.exe is on Windows, and its usage line read "usage: python.exe C:\\...\\rizzo.exe serve".
    Before 3.14 the name is argv[0], whatever it is: the parser has to say "rizzo" itself."""
    archive = ModuleType("__main__")
    archive.__spec__ = ModuleSpec("__main__", None)
    monkeypatch.setitem(sys.modules, "__main__", archive)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(sys, "argv", [program, "serve", "--port", "http"])
    with pytest.raises(SystemExit):
        cli.main()
    err = rizzo.capsys.readouterr().err
    assert err.startswith("usage: rizzo serve [-h]")
    assert "\nrizzo serve: error: argument --port: invalid int value: 'http'\n" in err


@pytest.mark.parametrize("value", [0, 1, 65535])
def test_the_ends_of_the_port_range_reach_uvicorn(rizzo, loaded, served, value):
    rizzo.run("serve", "--port", value)  # 0: the system picks a free port, which uvicorn reports
    assert [options["port"] for _, options in served.runs] == [value]


@PROPERTY
@given(number=st.integers(-(10**30), 10**30))
def test_port_accepts_exactly_the_numbers_a_socket_binds(number):
    if 0 <= number <= 65535:
        assert cli.port(str(number)) == number
    else:
        with pytest.raises(argparse.ArgumentTypeError, match="is not a port"):
            cli.port(str(number))


def test_serve_without_uvicorn_installed_is_a_one_line_error(rizzo, loaded, monkeypatch):
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    (out, err), exit_ = rizzo.fail("serve")
    assert out == ""
    assert err.startswith("rizzo: ")
    assert "uvicorn" in err
    assert isinstance(exit_.__cause__, ImportError)
    assert loaded.model.closed == 1


@PROPERTY
@given(
    ctx=st.integers(1, 10**6),
    batch=st.integers(1, 64),
    threads=st.integers(1, 512),
    port=st.integers(0, 65535),
)
def test_numeric_options_reach_their_collaborators_unchanged(ctx, batch, threads, port):
    with pytest.MonkeyPatch.context() as patch:
        model = fake_loader(patch)
        server = fake_uvicorn(patch)
        flags = ["--ctx", ctx, "--batch-size", batch, "--threads", threads, "--port", port]
        patch.setattr(sys, "argv", ["rizzo", "serve", *map(str, flags)])
        cli.main()
    [(_, options)] = model.calls
    assert (options["ctx"], options["batch_size"], options["threads"]) == (ctx, batch, threads)
    assert [options for _, options in server.runs] == [{"host": "127.0.0.1", "port": port}]


# the script entry point ------------------------------------------------------------------------


def test_running_the_module_as_a_script_calls_main(monkeypatch, capsys):
    report = {"llama.cpp": {"release": "b1"}}
    monkeypatch.setattr(loader, "describe", lambda: report)
    monkeypatch.setattr(sys, "argv", ["rizzo", "devices"])
    # Without this, runpy warns that the module was imported before it is executed as __main__.
    monkeypatch.delitem(sys.modules, "rizzo_flow.cli")
    runpy.run_module("rizzo_flow.cli", run_name="__main__")
    assert json.loads(capsys.readouterr().out) == report
