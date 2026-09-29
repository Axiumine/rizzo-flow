"""pyproject.toml: the versions the code cannot run without."""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def floor(specifier: str) -> tuple[int, ...]:
    """The version after `>=` in a specifier: ">=3.11.4" -> (3, 11, 4)."""
    found = re.search(r">=\s*(\d+(?:\.\d+)*)", specifier)
    assert found, f"no lower bound in {specifier!r}"
    return tuple(int(part) for part in found.group(1).split("."))


def test_python_is_at_least_3_11_4_where_tarfile_has_extraction_filters():
    """llama_release.unpack extracts with filter="data" (PEP 706), which 3.11.0 to 3.11.3 lack:
    they would pass the installer's check and fail in `rizzo download` with a TypeError."""
    assert floor(project()["requires-python"]) >= (3, 11, 4)
