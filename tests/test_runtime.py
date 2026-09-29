"""Device selection logic with a fake MLX: no backend package or GPU required."""

import ntpath
import os
import posixpath
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


@pytest.fixture
def no_resource_module():
    """`resource` absent from sys.modules, as on Windows; whatever was there is put back."""
    saved = sys.modules.pop("resource", None)
    yield
    if saved is None:
        sys.modules.pop("resource", None)
    else:
        sys.modules["resource"] = saved


@pytest.fixture
def windows(monkeypatch, no_resource_module):
    monkeypatch.setattr(sys, "platform", "win32")


@pytest.fixture
def wheels(tmp_path, monkeypatch):
    """The `nvidia` folder of a site-packages directory that does not exist yet, and a bare PATH."""
    monkeypatch.setattr(runtime.sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    monkeypatch.setenv("PATH", "original")
    return tmp_path / "nvidia"


def folders(nvidia):
    """The two folders `prepare` looks for, in the order it adds them to PATH."""
    return nvidia / "cu13" / "bin" / "x86_64", nvidia / "cudnn" / "bin"


@pytest.mark.usefixtures("windows")
def test_prepare_adds_a_folder_when_only_a_longer_entry_starts_with_it(monkeypatch, wheels):
    _, cudnn = folders(wheels)
    cudnn.mkdir(parents=True)
    monkeypatch.setenv("PATH", os.pathsep.join([f"{cudnn}-old", "original"]))
    runtime.prepare()
    assert str(cudnn) in os.environ["PATH"].split(os.pathsep)


@pytest.mark.usefixtures("windows")
def test_prepare_counts_an_entry_that_is_the_folder_however_it_is_written(monkeypatch, wheels):
    cu13, cudnn = folders(wheels)
    cu13.mkdir(parents=True)
    cudnn.mkdir(parents=True)
    written = [str(cu13) + os.sep, os.sep.join([str(cudnn.parent), ".", cudnn.name])]  # not normal
    monkeypatch.setenv("PATH", os.pathsep.join([*written, "original"]))
    runtime.prepare()
    assert os.environ["PATH"] == os.pathsep.join([*written, "original"])  # nothing to add


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        (["/opt/cuda"], True),
        (["/opt/other", "/opt/cuda", "/opt/more"], True),
        (["/opt/cuda/"], True),  # a trailing separator
        (["/opt//cuda"], True),
        (["/opt/./cuda"], True),
        (["/opt/other/../cuda"], True),
        (["/opt/cuda-old"], False),  # starts with the folder, is not the folder
        (["/opt/cuda/bin"], False),  # inside the folder
        (["/opt"], False),  # holds the folder
        (["", "/opt/other"], False),  # an empty entry names no folder
        ([], False),
    ],
)
def test_on_path_looks_for_a_whole_entry(monkeypatch, entries, expected):
    monkeypatch.setenv("PATH", os.pathsep.join(entries))
    assert runtime.on_path("/opt/cuda") is expected


def test_on_path_takes_a_path_object_and_a_missing_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", os.pathsep.join(["original", str(tmp_path)]))
    assert runtime.on_path(tmp_path) is True
    assert runtime.on_path(tmp_path / "bin") is False
    monkeypatch.delenv("PATH")
    assert runtime.on_path(tmp_path) is False


def test_with_no_path_at_all_no_folder_is_on_it(monkeypatch):
    monkeypatch.delenv("PATH", raising=False)
    assert runtime.on_path("XXXX") is False


def test_an_empty_entry_of_path_is_no_folder(monkeypatch):
    # An empty entry would normalize to the current folder.
    monkeypatch.setenv("PATH", os.pathsep.join(["", "/opt/other", ""]))
    assert runtime.on_path(".") is False
    assert runtime.on_path("/opt/other") is True


def test_on_path_ignores_case_where_paths_do(monkeypatch):
    monkeypatch.setenv("PATH", os.pathsep.join(["/OPT/Cuda/BIN", "/other"]))
    with monkeypatch.context() as nt:
        nt.setattr(os.path, "normcase", ntpath.normcase)  # what `os.path` is on Windows
        assert runtime.on_path("/opt/cuda/bin") is True
        assert runtime.on_path("/opt/cuda/bin2") is False
    with monkeypatch.context() as posix:
        posix.setattr(os.path, "normcase", posixpath.normcase)  # and on Linux and macOS
        assert runtime.on_path("/opt/cuda/bin") is False
        assert runtime.on_path("/OPT/Cuda/BIN") is True
