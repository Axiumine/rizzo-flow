"""Helpers of the mutation tests, `test_mutation_<module>.py`.

Not a test module: pytest collects it because of its name and finds nothing to run.

`mutmut run` edits the code one place at a time and reruns the tests that reach it (see
[tool.mutmut] in pyproject.toml). A test in one of these files exists because such an edit got
through the others, and its docstring says which edit.
"""

import io
import re
from pathlib import Path

import pytest


def exactly(message: str) -> str:
    """A `match` pattern for pytest.raises that accepts the message and nothing around it."""
    return f"^{re.escape(message)}$"


def use_ascii_as_default_encoding(monkeypatch: pytest.MonkeyPatch, probe_dir: Path) -> None:
    """Make ASCII the encoding that a text file gets when the caller names none.

    Linux and macOS default to UTF-8, so code that forgets `encoding="utf-8"` still round-trips
    the text of a test there, and only Windows (the ANSI code page) tells. With this in place the
    code has to name its encoding, or the text of the test fails to decode or encode, on every
    platform. `pathlib` asks `io.text_encoding` for the default, which is what this replaces.

    The last two lines prove the replacement reaches `pathlib` (`Path.read_text` and
    `Path.open`), so that a Python that stops asking `io.text_encoding` fails here, loudly,
    instead of letting the tests that use this pass for nothing.
    """

    def default(encoding: str | None, stacklevel: int = 2) -> str:
        return "ascii" if encoding is None else encoding

    monkeypatch.setattr(io, "text_encoding", default)
    probe = probe_dir / "probe.txt"
    probe.write_bytes("\N{LATIN SMALL LETTER E WITH GRAVE}".encode())
    with pytest.raises(UnicodeDecodeError):
        probe.read_text()
    with pytest.raises(UnicodeDecodeError), probe.open() as stream:
        stream.read()
