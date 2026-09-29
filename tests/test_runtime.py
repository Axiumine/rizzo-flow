"""Device selection logic with a fake MLX: no backend package or GPU required."""

import sys
from types import SimpleNamespace

import pytest

from rizzo_flow import runtime


def fake_mlx(monkeypatch, gpu, platform):
    mx = SimpleNamespace(cpu="CPU", gpu="GPU", is_available=lambda device: device == "CPU" or gpu)
    monkeypatch.setattr(runtime, "import_mlx", lambda: mx)
    monkeypatch.setattr(sys, "platform", platform)


@pytest.mark.parametrize(
    ("gpu", "platform", "device", "expected"),
    [
        (True, "darwin", "auto", ("GPU", "mlx")),
        (True, "darwin", "mlx", ("GPU", "mlx")),
        (True, "win32", "auto", ("GPU", "cuda")),
        (True, "linux", "cuda", ("GPU", "cuda")),
        (True, "linux", "gpu", ("GPU", "cuda")),
        (True, "win32", "cpu", ("CPU", "cpu")),
        (False, "win32", "auto", ("CPU", "cpu")),
    ],
)
def test_resolve(monkeypatch, gpu, platform, device, expected):
    fake_mlx(monkeypatch, gpu, platform)
    assert runtime.resolve(device) == expected


@pytest.mark.parametrize(
    ("gpu", "platform", "device"),
    [
        (False, "linux", "cuda"),
        (False, "win32", "gpu"),
        (True, "win32", "mlx"),
        (True, "darwin", "cuda"),
    ],
)
def test_resolve_rejects_missing_or_wrong_gpu(monkeypatch, gpu, platform, device):
    fake_mlx(monkeypatch, gpu, platform)
    with pytest.raises(ValueError, match="--device"):
        runtime.resolve(device)


def test_resolve_rejects_unknown_name():
    with pytest.raises(ValueError, match="one of"):
        runtime.resolve("tpu")
