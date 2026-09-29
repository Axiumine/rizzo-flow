"""scripts/semif_compare.py: how it reads the fixtures and writes its reports."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "semif_compare.py"
# What str.splitlines() breaks a line at besides "\n" and "\r", and json.dumps(ensure_ascii=False)
# does not escape. Written by name: ruff format turns a "\u2028" escape into the bare character.
SEPARATORS = {
    "nel": "\N{NEXT LINE}",
    "ls": "\N{LINE SEPARATOR}",
    "ps": "\N{PARAGRAPH SEPARATOR}",
}


@pytest.fixture(scope="module")
def semif_compare() -> ModuleType:
    spec = importlib.util.spec_from_file_location("semif_compare", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("separator", SEPARATORS.values(), ids=SEPARATORS.keys())
def test_a_record_holding_a_unicode_line_separator_is_read_whole(
    semif_compare, tmp_path, separator
):
    """str.splitlines() cuts a line at these, and a producer that keeps characters as they are
    (ensure_ascii=False) leaves them inside a string: half a record is not JSON."""
    rows = [{"id": "a", "text": f"before{separator}after"}, {"id": "b", "text": "plain"}]
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    assert semif_compare.read(path) == rows


def test_blank_lines_and_a_last_record_without_a_newline_are_fine(semif_compare, tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n\n  \r\n{"id": 2}', encoding="utf-8")
    assert semif_compare.read(path) == [{"id": 1}, {"id": 2}]


def test_a_report_is_created_once_with_lf_line_endings(semif_compare, tmp_path):
    rows = tmp_path / "rows.jsonl"
    semif_compare.write(rows, [{"id": 1}, {"id": 2}])
    assert rows.read_bytes() == b'{"id": 1}\n{"id": 2}\n'
    report = tmp_path / "report.json"
    semif_compare.write(report, {"a": [1]})
    assert report.read_bytes() == b'{\n  "a": [\n    1\n  ]\n}\n'
    with pytest.raises(FileExistsError):  # results are create-only
        semif_compare.write(report, {"a": [2]})
    assert report.read_bytes() == b'{\n  "a": [\n    1\n  ]\n}\n'
