"""Mutation tests of `rizzo_flow.cli`."""

import importlib
import io
import json
import os
import sys
from pathlib import Path

import pytest
from test_mutation_support import use_ascii_as_default_encoding

from rizzo_flow import cli, loader

TEXT = "\N{LATIN SMALL LETTER E WITH GRAVE} \N{CHECK MARK}"


@pytest.fixture
def ascii_default(monkeypatch, tmp_path_factory):
    use_ascii_as_default_encoding(monkeypatch, tmp_path_factory.mktemp("probe"))


def run_main(monkeypatch, *argv):
    """`main()` with the command line of a shell; returns the exit status it ended with."""
    monkeypatch.setattr(sys, "argv", ["rizzo", *map(str, argv)])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    return raised.value.code


def test_write_json_writes_a_file_as_utf8_whatever_the_platform_default(tmp_path, ascii_default):
    """`write_json` opening the file without `encoding="utf-8"`, or with `encoding=None`."""
    target = tmp_path / "report.json"
    cli.write_json({"name": TEXT}, target)
    assert target.read_text(encoding="utf-8") == f'{{\n  "name": "{TEXT}"\n}}\n'
    assert TEXT.encode("utf-8") in target.read_bytes()  # kept as text, not as an escape


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


def test_read_jsonl_reads_a_file_as_utf8_whatever_the_platform_default(tmp_path, ascii_default):
    """`read_jsonl` reading without `encoding="utf-8"`, or with `encoding=None`."""
    path = tmp_path / "rows.jsonl"
    path.write_bytes(f'{{"text": "{TEXT}"}}\n[1, 2]\n'.encode())
    assert cli.read_jsonl(path) == [{"text": TEXT}, [1, 2]]


def test_decide_reads_the_request_file_as_utf8_whatever_the_platform_default(
    tmp_path, monkeypatch, capsys, ascii_default
):
    """`decide` reading the request without `encoding="utf-8"`, or with `encoding=None`.

    The model loader stops the command with a message of its own: reaching it means the request
    was read and validated, and an unreadable request would put another message there.
    """
    request = {"state": TEXT, "questions": {"q": {"type": "boolean", "instructions": "Is it?"}}}
    path = tmp_path / "request.json"
    path.write_bytes(json.dumps(request, ensure_ascii=False).encode("utf-8"))

    def load_backend(*args, **kwargs):
        raise ValueError("the request was accepted")

    monkeypatch.setattr(loader, "load_backend", load_backend)
    assert run_main(monkeypatch, "decide", path) == 1
    assert capsys.readouterr().err == "rizzo: the request was accepted\n"


def test_calibrate_hands_the_rows_file_over_as_a_path(tmp_path, monkeypatch, capsys):
    """The `input` of `calibrate` without `type=Path`: `read_jsonl` would get a string."""
    received = []

    def read_jsonl(path):
        received.append(path)
        raise ValueError("the rows were requested")

    monkeypatch.setattr(cli, "read_jsonl", read_jsonl)
    argv = ["calibrate", "rows.jsonl", "--fingerprint", "f", "--output", tmp_path / "fit.json"]
    assert run_main(monkeypatch, *argv) == 1
    assert capsys.readouterr().err == "rizzo: the rows were requested\n"
    assert received == [Path("rows.jsonl")]
    assert isinstance(received[0], Path)


def test_a_taken_output_is_refused_under_its_name_as_text(tmp_path):
    """`refuse_existing` handing the path over as it was given: the error of a `Path` would read
    `PosixPath('...')` instead of naming the file."""
    taken = tmp_path / "report.json"
    taken.write_text("evidence", encoding="utf-8")
    with pytest.raises(FileExistsError) as raised:
        cli.refuse_existing(taken)
    assert raised.value.filename == str(taken)
    assert str(raised.value) == f"[Errno 17] File exists: {str(taken)!r}"
    cli.refuse_existing(None)
    cli.refuse_existing(str(tmp_path / "later.json"))  # nothing there: nothing to refuse
