"""Backend selection (`loader.py`): option checks, what each backend is asked to load, `rizzo devices`.

Both backends are replaced by recorders, and the llama.cpp runtime and MLX by stand-ins, so that
what is tested is the decision `loader.py` takes and the report it writes.
"""

import re

import pytest

from rizzo_flow import backend, backend_llama, config, llama_cpp, llama_release, loader, runtime
from rizzo_flow.llama_cpp import Device


@pytest.fixture
def loaded(monkeypatch):
    """Both backends replaced by recorders: (backend, path, options) for every load asked for."""
    calls: list[tuple] = []

    def spark(path, **options):
        calls.append(("mlx", path, options))
        return "spark-backend"

    def llama(path, **options):
        calls.append(("llama", path, options))
        return "llama-backend"

    monkeypatch.setattr(backend.SparkBackend, "load", spark)
    monkeypatch.setattr(backend_llama.LlamaBackend, "load", llama)
    return calls


def test_the_choices_offered_on_the_command_line():
    assert loader.BACKENDS == ("llama", "mlx")
    # MLX takes the devices it can name; llama.cpp adds its own families to them.
    assert set(loader.MLX_DEVICES) == set(runtime.DEVICES)
    assert set(loader.MLX_DEVICES) < set(loader.DEVICES)
    assert {"metal", "vulkan", "rocm", "sycl"} <= set(loader.DEVICES) - set(loader.MLX_DEVICES)


# --- llama.cpp ------------------------------------------------------------------------------


def test_llama_is_the_default_and_loads_the_pinned_fine_tune(loaded):
    assert loader.load_backend() == "llama-backend"
    assert loaded == [
        (
            "llama",
            config.gguf_spec().path,
            {"device": "auto", "ctx": 8192, "batch_size": 4, "threads": None, "kv_type": None},
        )
    ]


def test_llama_options_reach_the_backend(loaded):
    loader.load_backend(
        "llama",
        size="1.7b",
        quant="q4_k_m",
        weights="base",
        device="vulkan",
        ctx=4096,
        batch_size=8,
        threads=6,
        kv_type="q8_0",
    )
    assert loaded == [
        (
            "llama",
            config.GGUF[("1.7b", "q4_k_m", "base")].path,
            {"device": "vulkan", "ctx": 4096, "batch_size": 8, "threads": 6, "kv_type": "q8_0"},
        )
    ]


def test_llama_takes_a_gguf_file_as_it_is_given(loaded):
    loader.load_backend("llama", model="my/own.gguf", size="1.7b", quant="bf16")
    ((_, path, _),) = loaded
    assert path == "my/own.gguf"  # the size and quantization only choose pinned files


@pytest.mark.parametrize("device", [name for name in loader.DEVICES if name != "mlx"])
def test_llama_accepts_every_device_but_mlx(loaded, device):
    loader.load_backend("llama", device=device)
    assert loaded[0][2]["device"] == device


def test_llama_refuses_options_of_the_other_runtime(loaded, tmp_path):
    refused: list[tuple[dict, str]] = [
        (
            {"bits": 8},
            (
                "--bits quantizes MLX weights in memory (--backend mlx); llama.cpp loads a "
                "quantized file instead: --quant q8_0 (default), q4_k_m or bf16"
            ),
        ),
        (
            {"device": "mlx"},
            "--device mlx needs --backend mlx; with llama.cpp use --device metal",
        ),
        (
            {"model": "mine.gguf", "weights": "base"},
            "--weights picks a pinned file; with --model the file is yours",
        ),
        (
            {"model": str(tmp_path)},
            (
                f"{tmp_path} is a checkpoint directory (MLX); llama.cpp needs a .gguf file. "
                "Pass --backend mlx, or a GGUF path, or drop --model to use the pinned file."
            ),
        ),
    ]
    for options, message in refused:
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            loader.load_backend("llama", **options)
    assert loaded == []


def test_llama_refuses_an_unknown_backend_and_a_missing_combination(loaded):
    with pytest.raises(ValueError, match=r"^Backend must be one of: llama, mlx$"):
        loader.load_backend("onnx")
    with pytest.raises(ValueError, match=r"^No flow GGUF for 4b q2_k;"):
        loader.load_backend("llama", quant="q2_k")
    assert loaded == []


# --- MLX ------------------------------------------------------------------------------------


def test_mlx_loads_the_checkpoint_directory_of_the_chosen_weights(loaded):
    assert loader.load_backend("mlx") == "spark-backend"
    loader.load_backend("mlx", size="1.7b", weights="base", bits=4, device="cuda", batch_size=8)
    loader.load_backend("mlx", model="my/checkpoint", ctx=2048, threads=3)
    assert loaded == [
        (
            "mlx",
            config.checkpoint_path("4b", None),
            {"bits": None, "device": "auto", "batch_size": 4},
        ),
        (
            "mlx",
            config.checkpoint_path("1.7b", "base"),
            {"bits": 4, "device": "cuda", "batch_size": 8},
        ),
        # The context limit and the thread count belong to llama.cpp; MLX is not given them.
        ("mlx", "my/checkpoint", {"bits": None, "device": "auto", "batch_size": 4}),
    ]


@pytest.mark.parametrize("device", loader.MLX_DEVICES)
def test_mlx_accepts_the_devices_it_can_name(loaded, device):
    loader.load_backend("mlx", device=device)
    assert loaded[0][2]["device"] == device


@pytest.mark.parametrize(
    "device", [name for name in loader.DEVICES if name not in loader.MLX_DEVICES]
)
def test_mlx_refuses_the_devices_only_llama_cpp_has(loaded, device):
    with pytest.raises(ValueError, match=rf"^--device {device} exists only in the llama backend$"):
        loader.load_backend("mlx", device=device)
    assert loaded == []


def test_mlx_refuses_options_of_the_other_runtime(loaded):
    refused: list[tuple[dict, str]] = [
        (
            {"quant": "q8_0"},
            "--quant selects a GGUF file (llama backend); with MLX use --bits 4|8",
        ),
        ({"kv_type": "q8_0"}, "--kv-type exists only in the llama backend"),
        (
            {"model": "mine", "weights": "flow"},
            "--weights picks a pinned checkpoint; with --model the files are yours",
        ),
    ]
    for options, message in refused:
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            loader.load_backend("mlx", **options)
    assert loaded == []


# --- rizzo devices --------------------------------------------------------------------------

CPU = Device(1, "CPU", "Some CPU", "cpu", "CPU", 64 << 30)
RADEON = Device(3, "Vulkan1", "AMD Radeon RX 7900 XTX", "gpu", "Vulkan", 24 << 30)
MLX_REPORT = {
    "mlx_version": "0.32.2",
    "platform": "linux",
    "available": ["cpu"],
    "auto_selects": "cpu",
}


class FakeLibrary:
    def __init__(self, devices):
        self._devices = devices

    def devices(self):
        return self._devices


def recommend(accelerator):
    """`pick` as `describe` has to call it: asking for the recommendation, by name."""
    assert accelerator == "auto"
    return "vulkan"


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """A Linux x86-64 host with a Vulkan runtime installed, a Radeon and MLX on the CPU."""
    monkeypatch.setattr(llama_release, "host", lambda: ("linux", "x64"))
    monkeypatch.setattr(llama_release, "supported", lambda: ["cuda", "vulkan", "cpu"])
    monkeypatch.setattr(llama_release, "installed", lambda: ["vulkan"])
    monkeypatch.setattr(llama_release, "pick", recommend)
    monkeypatch.setattr(llama_release, "locate", lambda family=None: tmp_path / "runtime")
    monkeypatch.setattr(llama_cpp.Library, "open", lambda: FakeLibrary([CPU, RADEON]))
    monkeypatch.setattr(runtime, "describe", lambda: MLX_REPORT)
    return tmp_path / "runtime"


def test_describe_reports_the_runtime_its_devices_and_mlx(machine):
    assert loader.describe() == {
        "llama.cpp": {
            "release": llama_release.RELEASE,
            "host": "linux/x64",
            "packages": ["cuda", "vulkan", "cpu"],
            "recommended": "vulkan",
            "installed": ["vulkan"],
            "directory": str(machine),
            "devices": [CPU.public(), RADEON.public()],
            "auto_selects": "Vulkan1",
        },
        "mlx": MLX_REPORT,
    }


def test_describe_says_cpu_when_no_gpu_is_visible(machine, monkeypatch):
    monkeypatch.setattr(llama_cpp.Library, "open", lambda: FakeLibrary([CPU]))
    section = loader.describe()["llama.cpp"]
    assert section["devices"] == [CPU.public()]
    assert section["auto_selects"] == "CPU"
    assert "error" not in section


def test_describe_turns_a_missing_runtime_into_an_error_entry(machine, monkeypatch):
    def not_installed(family=None):
        raise ValueError("llama.cpp runtime not installed")

    monkeypatch.setattr(llama_release, "locate", not_installed)
    report = loader.describe()
    section = report["llama.cpp"]
    assert section["error"] == "llama.cpp runtime not installed"
    assert section["recommended"] == "vulkan"  # known before the runtime was looked for
    assert section["installed"] == ["vulkan"]
    assert not {"directory", "devices", "auto_selects"} & set(section)
    assert report["mlx"] == MLX_REPORT  # the second runtime is reported all the same


def test_describe_turns_an_unsupported_machine_into_an_error_entry(machine, monkeypatch):
    def nothing_for_this_machine(accelerator="auto"):
        raise ValueError("No prebuilt llama.cpp package for this machine")

    monkeypatch.setattr(llama_release, "supported", list)
    monkeypatch.setattr(llama_release, "installed", list)
    monkeypatch.setattr(llama_release, "pick", nothing_for_this_machine)
    section = loader.describe()["llama.cpp"]
    assert section["error"] == "No prebuilt llama.cpp package for this machine"
    assert section["recommended"] is None
    assert section["packages"] == section["installed"] == []
    assert not {"directory", "devices", "auto_selects"} & set(section)


@pytest.mark.parametrize("failure", [OSError("cannot open libllama"), ValueError("symbol missing")])
def test_describe_turns_a_library_that_will_not_load_into_an_error_entry(
    machine, monkeypatch, failure
):
    def open_library():
        raise failure

    monkeypatch.setattr(llama_cpp.Library, "open", open_library)
    section = loader.describe()["llama.cpp"]
    assert section["error"] == str(failure)
    assert section["directory"] == str(machine)  # found, but not loadable
    assert not {"devices", "auto_selects"} & set(section)


def test_describe_reports_mlx_as_not_installed_when_it_cannot_be_imported(machine, monkeypatch):
    def no_mlx():
        raise ImportError("MLX has no compute backend here")

    monkeypatch.setattr(runtime, "describe", no_mlx)
    report = loader.describe()
    assert report["mlx"] == {"installed": False}
    assert report["llama.cpp"]["auto_selects"] == "Vulkan1"
