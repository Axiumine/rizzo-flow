"""Mutation tests of `rizzo_flow.runtime`."""

import os

from rizzo_flow import runtime


def test_with_no_path_at_all_no_folder_is_on_it(monkeypatch):
    """`on_path` falling back to "XXXX" for an unset PATH: a folder of that name would be on it."""
    monkeypatch.delenv("PATH", raising=False)
    assert runtime.on_path("XXXX") is False


def test_an_empty_entry_of_path_is_no_folder(monkeypatch):
    """`on_path` counting the empty entries of PATH, which normalize to the current folder."""
    monkeypatch.setenv("PATH", os.pathsep.join(["", "/opt/other", ""]))
    assert runtime.on_path(".") is False
    assert runtime.on_path("/opt/other") is True
