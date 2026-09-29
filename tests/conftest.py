"""Test-wide setup: the environment before any test imports MLX, and fixtures for every test."""

import os
import urllib.request

import pytest

# TF32 rounding on NVIDIA GPUs exceeds the exact-equivalence tolerances of test_mlx.py.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

import rizzo_flow  # noqa: F401  (prepares the MLX stack on Windows)


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    """Local servers are reached directly: `urlopen` would send them to the proxy of the
    environment or of the system settings (WinINET, macOS) even for 127.0.0.1."""
    handler = urllib.request.ProxyHandler({})
    monkeypatch.setattr(urllib.request, "_opener", urllib.request.build_opener(handler))


@pytest.fixture(autouse=True)
def caller_environment(monkeypatch):
    """What the shell exports must not change a result."""
    monkeypatch.delenv("RIZZO_LLAMA_LOG", raising=False)  # would let llama.cpp's info lines through
    monkeypatch.setenv("COLUMNS", "200")  # argparse wraps --help at the terminal width
