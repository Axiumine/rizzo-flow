"""MLX runtime preparation, import and reporting on each platform, with fake platforms and a fake MLX.

`test_runtime.py` covers the choice of device; this file covers the rest of `runtime.py`. Nothing
here needs MLX, a GPU or Windows: `sys.platform`, the wheel folders and `mlx.core` are replaced.
"""

import importlib
import ntpath
import os
import posixpath
import re
import sys
import types
from types import SimpleNamespace

import pytest

import rizzo_flow
from rizzo_flow import runtime


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
def wheels(tmp_path, monkeypatch):
    """The `nvidia` folder of a site-packages directory that does not exist yet, and a bare PATH."""
    monkeypatch.setattr(runtime.sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    monkeypatch.setenv("PATH", "original")
    return tmp_path / "nvidia"


def folders(nvidia):
    """The two folders `prepare` looks for, in the order it adds them to PATH."""
    return nvidia / "cu13" / "bin" / "x86_64", nvidia / "cudnn" / "bin"


# --- prepare --------------------------------------------------------------------------------


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_prepare_leaves_linux_and_macos_alone(monkeypatch, wheels, no_resource_module, platform):
    for folder in folders(wheels):
        folder.mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", platform)
    runtime.prepare()
    assert "resource" not in sys.modules
    assert os.environ["PATH"] == "original"


def test_prepare_stubs_the_unix_only_resource_module_on_windows(
    monkeypatch, wheels, no_resource_module
):
    monkeypatch.setattr(sys, "platform", "win32")
    runtime.prepare()
    stub = sys.modules["resource"]
    assert stub.__name__ == "resource"
    assert stub.RLIMIT_NOFILE == 0
    # mlx-lm raises the limit of open files at import time; here that is a no-op.
    assert stub.setrlimit(stub.RLIMIT_NOFILE, (2048, 4096)) is None
    runtime.prepare()  # safe to call twice: the stub is not replaced
    assert sys.modules["resource"] is stub


def test_prepare_keeps_a_resource_module_that_is_already_loaded(monkeypatch, wheels):
    existing = types.ModuleType("resource")
    monkeypatch.setitem(sys.modules, "resource", existing)
    monkeypatch.setattr(sys, "platform", "win32")
    runtime.prepare()
    assert sys.modules["resource"] is existing
    assert "setrlimit" not in existing.__dict__


@pytest.fixture
def windows(monkeypatch, no_resource_module):
    monkeypatch.setattr(sys, "platform", "win32")


@pytest.mark.usefixtures("windows")
def test_prepare_puts_the_cuda_dlls_of_the_nvidia_wheels_on_the_path(wheels):
    cu13, cudnn = folders(wheels)
    cu13.mkdir(parents=True)
    cudnn.mkdir(parents=True)
    runtime.prepare()
    assert os.environ["PATH"] == os.pathsep.join([str(cu13), str(cudnn), "original"])
    runtime.prepare()  # safe to call twice: nothing is added again
    assert os.environ["PATH"] == os.pathsep.join([str(cu13), str(cudnn), "original"])


@pytest.mark.usefixtures("windows")
@pytest.mark.parametrize("present", [0, 1])
def test_prepare_adds_only_the_folders_that_exist(wheels, present):
    folder = folders(wheels)[present]
    folder.mkdir(parents=True)
    runtime.prepare()
    assert os.environ["PATH"] == os.pathsep.join([str(folder), "original"])


@pytest.mark.usefixtures("windows")
def test_prepare_leaves_the_path_alone_without_the_wheels(wheels):
    runtime.prepare()
    assert os.environ["PATH"] == "original"


@pytest.mark.usefixtures("windows")
def test_prepare_ignores_files_that_look_like_the_folders(wheels):
    _, cudnn = folders(wheels)
    cudnn.parent.mkdir(parents=True)
    cudnn.write_bytes(b"not a directory")
    runtime.prepare()
    assert os.environ["PATH"] == "original"


@pytest.mark.usefixtures("windows")
def test_prepare_does_not_repeat_a_folder_that_is_already_on_the_path(monkeypatch, wheels):
    cu13, cudnn = folders(wheels)
    cu13.mkdir(parents=True)
    cudnn.mkdir(parents=True)
    monkeypatch.setenv("PATH", os.pathsep.join([str(cudnn), "original"]))
    runtime.prepare()
    assert os.environ["PATH"] == os.pathsep.join([str(cu13), str(cudnn), "original"])


@pytest.mark.usefixtures("windows")
def test_prepare_starts_a_path_when_there_is_none(monkeypatch, wheels):
    cu13, cudnn = folders(wheels)
    cu13.mkdir(parents=True)
    cudnn.mkdir(parents=True)
    monkeypatch.delenv("PATH")
    runtime.prepare()
    entries = [entry for entry in os.environ["PATH"].split(os.pathsep) if entry]
    assert entries == [str(cu13), str(cudnn)]


@pytest.mark.usefixtures("windows")
def test_prepare_does_not_invent_a_path_when_there_is_nothing_to_add(monkeypatch, wheels):
    monkeypatch.delenv("PATH")
    runtime.prepare()
    assert "PATH" not in os.environ


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


def test_importing_the_package_prepares_the_stack(monkeypatch):
    """Tests and scripts that import MLX or mlx-lm straight after `import rizzo_flow` rely on it."""
    calls = []
    monkeypatch.setattr(runtime, "prepare", lambda: calls.append("prepare"))
    try:
        importlib.reload(rizzo_flow)  # runs the package's `__init__` again, with the spy in place
        assert calls == ["prepare"]
    finally:
        monkeypatch.undo()
        importlib.reload(rizzo_flow)  # the package namespace gets the real `prepare` back


# --- on_path --------------------------------------------------------------------------------


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


# --- import_mlx -----------------------------------------------------------------------------


def test_import_mlx_prepares_the_stack_before_importing_and_returns_mlx_core(monkeypatch):
    core = types.ModuleType("mlx.core")
    package = types.ModuleType("mlx")
    package.__dict__["core"] = core
    calls = []

    def prepare():
        calls.append("prepare")
        sys.modules["mlx"], sys.modules["mlx.core"] = package, core  # importable from now on

    for name in ("mlx", "mlx.core"):
        monkeypatch.setitem(sys.modules, name, None)  # not importable until `prepare` ran
    monkeypatch.setattr(runtime, "prepare", prepare)
    assert runtime.import_mlx() is core
    assert calls == ["prepare"]


def test_import_mlx_explains_a_missing_compute_backend(monkeypatch):
    monkeypatch.setattr(runtime, "prepare", lambda: None)
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    with pytest.raises(ImportError, match=r"^MLX has no compute backend here \(.+\)\. ") as failure:
        runtime.import_mlx()
    message = str(failure.value)
    assert message.endswith(runtime.INSTALL_HINT)
    assert isinstance(failure.value.__cause__, ImportError)  # the original failure stays chained
    assert str(failure.value.__cause__) in message


def test_install_hint_names_the_three_runtimes():
    for extra in ("mlx", "cuda", "cpu"):
        assert f"`uv sync --extra {extra}`" in runtime.INSTALL_HINT


# --- accelerator and describe ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("platform", "expected"), [("darwin", "mlx"), ("linux", "cuda"), ("win32", "cuda")]
)
def test_accelerator_is_metal_on_macos_and_cuda_elsewhere(monkeypatch, platform, expected):
    asked = []

    def is_available(device):
        asked.append(device)
        return True

    monkeypatch.setattr(sys, "platform", platform)
    mx = SimpleNamespace(gpu="GPU", cpu="CPU", is_available=is_available)
    assert runtime.accelerator(mx) == expected
    assert asked == ["GPU"]  # the question is about the GPU, not the CPU


def test_accelerator_is_none_without_a_usable_gpu():
    mx = SimpleNamespace(gpu="GPU", cpu="CPU", is_available=lambda device: device == "CPU")
    assert runtime.accelerator(mx) is None


@pytest.mark.parametrize(
    ("gpu", "platform", "available", "auto"),
    [
        (True, "darwin", ["mlx", "cpu"], "mlx"),
        (True, "linux", ["cuda", "cpu"], "cuda"),
        (True, "win32", ["cuda", "cpu"], "cuda"),
        (False, "linux", ["cpu"], "cpu"),
        (False, "darwin", ["cpu"], "cpu"),
    ],
)
def test_describe_lists_the_backends_this_install_can_use(
    monkeypatch, gpu, platform, available, auto
):
    mx = SimpleNamespace(gpu="GPU", is_available=lambda device: gpu, __version__="0.32.2")
    monkeypatch.setattr(runtime, "import_mlx", lambda: mx)
    monkeypatch.setattr(sys, "platform", platform)
    assert runtime.describe() == {
        "mlx_version": "0.32.2",
        "platform": platform,
        "available": available,
        "auto_selects": auto,
    }


# --- resolve, every combination -------------------------------------------------------------


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
@pytest.mark.parametrize("gpu", [True, False])
@pytest.mark.parametrize("device", runtime.DEVICES)
def test_resolve_for_every_device_platform_and_install(monkeypatch, device, gpu, platform):
    mx = SimpleNamespace(cpu="CPU", gpu="GPU", is_available=lambda name: name == "CPU" or gpu)
    monkeypatch.setattr(runtime, "import_mlx", lambda: mx)
    monkeypatch.setattr(sys, "platform", platform)
    native = "mlx" if platform == "darwin" else "cuda"
    if device == "cpu" or (device == "auto" and not gpu):
        assert runtime.resolve(device) == ("CPU", "cpu")
    elif not gpu:
        expected = f"--device {device}: this install has no usable GPU. {runtime.INSTALL_HINT}"
        with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
            runtime.resolve(device)
    elif device in ("mlx", "cuda") and device != native:
        expected = f"--device {device}: the GPU backend of this install is `{native}`."
        with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
            runtime.resolve(device)
    else:
        assert runtime.resolve(device) == ("GPU", native)


def test_resolve_names_the_valid_devices_and_does_not_import_mlx_for_a_typo(monkeypatch):
    def import_mlx():
        raise AssertionError("a bad device name is refused before MLX is imported")

    monkeypatch.setattr(runtime, "import_mlx", import_mlx)
    with pytest.raises(ValueError, match=r"^Device must be one of: auto, gpu, mlx, cuda, cpu$"):
        runtime.resolve("tpu")
