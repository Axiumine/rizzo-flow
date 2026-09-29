"""Mutation tests of `rizzo_flow.calibration`."""

import json

import pytest
from test_mutation_support import use_ascii_as_default_encoding

from rizzo_flow.calibration import Calibration


@pytest.fixture
def ascii_default(monkeypatch, tmp_path_factory):
    use_ascii_as_default_encoding(monkeypatch, tmp_path_factory.mktemp("probe"))


def test_a_calibration_file_is_read_as_utf8_whatever_the_platform_default(tmp_path, ascii_default):
    """`Calibration.from_file` without `encoding="utf-8"`, or with `encoding=None`."""
    fingerprint = "mod\N{LATIN SMALL LETTER E WITH GRAVE}le-\N{CHECK MARK}"
    path = tmp_path / "calibration.json"
    text = json.dumps(
        {
            "fingerprint": fingerprint,
            "dataset_sha256": "0" * 64,
            "temperatures": {"boolean": 2.0},
            "fit_metrics": {},
        },
        ensure_ascii=False,
    )
    path.write_bytes(text.encode("utf-8"))
    assert fingerprint.encode("utf-8") in path.read_bytes()  # kept as text, not as an escape
    assert Calibration.from_file(path).fingerprint == fingerprint
