"""The reports that scripts/ create are LF on every OS.

Text mode writes "\\r\\n" for each "\\n" on Windows unless the file is opened with newline="\\n",
and the bytes of a report are hashed (results/SHA256SUMS): the same report with other line
endings has another hash. So every text-mode `open(..., "x")` in scripts/ has to say so; this
looks at the scripts that open their own file for creation.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def is_open(call: ast.Call) -> bool:
    """`open(...)`, `Path(...).open(...)`, `io.open(...)`."""
    function = call.func
    return (isinstance(function, ast.Name) and function.id == "open") or (
        isinstance(function, ast.Attribute) and function.attr == "open"
    )


def creates_a_text_file(call: ast.Call) -> bool:
    """A literal mode with `x` and without `b`, positional or by keyword."""
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    given = [*call.args, keywords.get("mode")]
    modes = [
        arg.value for arg in given if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]
    return any("x" in mode and "b" not in mode for mode in modes)


def has_newline_lf(call: ast.Call) -> bool:
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    newline = keywords.get("newline")
    return isinstance(newline, ast.Constant) and newline.value == "\n"


def creations_without_lf(source: str) -> list[int]:
    """The lines of `source` that open a file for creation in text mode without newline="\\n":
    `open(path, "x")`, `Path.open("x")`, `io.open(path, mode="xt")`, and so on."""
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    return sorted(
        call.lineno
        for call in calls
        if is_open(call) and creates_a_text_file(call) and not has_newline_lf(call)
    )


@pytest.mark.parametrize(
    ("source", "lines"),
    [
        ('open(path, "x", encoding="utf-8")', [1]),
        ('open(path, mode="x")', [1]),
        ('Path(path).open("xt", encoding="utf-8")', [1]),
        ('io.open(path, "x+")', [1]),
        ('open(path, "x", encoding="utf-8", newline="\\r\\n")', [1]),
        ('open(path, "x", encoding="utf-8", newline=None)', [1]),
        ('x = 1\nwith open(path, "x") as stream:\n    pass', [2]),
        ('open(path, "x", encoding="utf-8", newline="\\n")', []),
        ('Path(path).open("x", newline="\\n")', []),
        ('open(path, mode="x", newline="\\n")', []),
        ('open(path, "xb")', []),  # binary: nothing to translate
        ('open(path, "rb")', []),
        ('open(path, encoding="utf-8")', []),
        ('Path(path).open(encoding="utf-8")', []),
        ("open(path, mode)", []),  # not a literal: nothing to say about it
        ('write("x")', []),
    ],
)
def test_the_check_names_the_text_files_created_without_lf(source, lines):
    assert creations_without_lf(source) == lines


@pytest.mark.parametrize("script", sorted(SCRIPTS.glob("*.py")), ids=lambda script: script.name)
def test_a_script_creates_its_reports_with_lf_line_endings(script):
    assert creations_without_lf(script.read_text(encoding="utf-8")) == []
