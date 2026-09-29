"""Runtime packages, second part: host detection, driver probe, lookup, retries, archives, installs.

No network and no native library: HTTP is a scripted stand-in for `urlopen`, ctypes is replaced,
and every directory is a temporary one. `test_llama_release.py` holds the first part.
"""

import email.message
import hashlib
import http.client
import io
import re
import tarfile
import tempfile
import types
import urllib.error
import zipfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rizzo_flow import llama_release as release

LIBRARY = "llama.testlib"  # any name will do: the platform's real one is not what is tested
HUGGING_FACE = "https://huggingface.co/rizzoaiacademy/rizzo-flow/resolve/abc/model.gguf"
GITHUB = "https://github.com/ggml-org/llama.cpp/releases/download/b1/runtime.zip"
PAYLOAD = bytes(range(10))
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def at(monkeypatch, system, machine, nvidia=False):
    monkeypatch.setattr(release, "host", lambda: (system, machine))
    monkeypatch.setattr(release, "nvidia_driver", lambda: nvidia)


@pytest.fixture
def runtimes(tmp_path, monkeypatch):
    """Runtimes are installed under a temporary directory, on a Linux x86-64 box without NVIDIA."""
    monkeypatch.setattr(release, "RUNTIMES", tmp_path / "runtimes")
    monkeypatch.setattr(release, "library_name", lambda: LIBRARY)
    monkeypatch.delenv(release.RUNTIME_DIR_ENV, raising=False)
    at(monkeypatch, "linux", "x64")
    return tmp_path


def make_runtime(family, nested=None):
    """An installed runtime of `family`, with libllama one level down when `nested` is given."""
    folder = release.install_dir(family)
    folder = folder / nested if nested else folder
    folder.mkdir(parents=True)
    (folder / LIBRARY).write_bytes(b"library")
    return folder


def archive_zip(path, members):
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)


def archive_tar(path, members):
    """`members` maps a name to bytes (a file) or to a ready-made TarInfo (anything else)."""
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in members.items():
            if isinstance(data, tarfile.TarInfo):
                bundle.addfile(data)
                continue
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))


# --- host, translation, library name --------------------------------------------------------


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("linux", "linux"),
        ("linux2", "linux"),
        ("win32", "win32"),
        ("darwin", "darwin"),
        ("freebsd14", "freebsd14"),
    ],
)
def test_host_names_the_system(monkeypatch, platform, expected):
    monkeypatch.setattr(release.sys, "platform", platform)
    monkeypatch.setattr(release.platform, "machine", lambda: "x86_64")
    assert release.host() == (expected, "x64")


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("AMD64", "x64"),
        ("x86_64", "x64"),
        ("X86_64", "x64"),
        ("aarch64", "arm64"),
        ("arm64", "arm64"),
        ("ARM64", "arm64"),  # what Windows on ARM reports
        ("riscv64", "riscv64"),
        ("i386", "i386"),
    ],
)
def test_host_names_the_machine_in_the_vocabulary_of_the_packages(monkeypatch, machine, expected):
    monkeypatch.setattr(release.sys, "platform", "linux")
    monkeypatch.setattr(release.platform, "machine", lambda: machine)
    assert release.host() == ("linux", expected)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("1\n", True), (" 1 ", True), ("0\n", False), ("", False), ("11", False)],
)
def test_rosetta_is_what_the_kernel_says_about_arm64(monkeypatch, answer, expected):
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return types.SimpleNamespace(stdout=answer)

    monkeypatch.setattr(release.sys, "platform", "darwin")
    monkeypatch.setattr(release, "host", lambda: ("darwin", "x64"))
    monkeypatch.setattr(release.subprocess, "run", run)
    assert release.translated() is expected
    assert calls == [
        (
            ["sysctl", "-n", "hw.optional.arm64"],
            {"capture_output": True, "text": True, "check": False},
        )
    ]


def test_rosetta_without_sysctl_is_not_rosetta(monkeypatch):
    def run(*args, **options):
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(release.sys, "platform", "darwin")
    monkeypatch.setattr(release, "host", lambda: ("darwin", "x64"))
    monkeypatch.setattr(release.subprocess, "run", run)
    assert release.translated() is False


@pytest.mark.parametrize(("platform", "machine"), [("darwin", "arm64"), ("win32", "x64")])
def test_rosetta_is_not_asked_where_it_cannot_apply(monkeypatch, platform, machine):
    def run(*args, **options):
        raise AssertionError("a native Apple Silicon or a non-Apple machine is not translated")

    monkeypatch.setattr(release.sys, "platform", platform)
    monkeypatch.setattr(release, "host", lambda: (platform, machine))
    monkeypatch.setattr(release.subprocess, "run", run)
    assert release.translated() is False


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("win32", "llama.dll"),
        ("darwin", "libllama.dylib"),
        ("linux", "libllama.so"),
        ("freebsd14", "libllama.so"),
    ],
)
def test_library_name_per_platform(monkeypatch, platform, expected):
    monkeypatch.setattr(release.sys, "platform", platform)
    assert release.library_name() == expected


# --- NVIDIA driver probe --------------------------------------------------------------------


class Driver:
    """A CUDA driver: `cuInit` answers `init`; `cuDeviceGetCount` answers `status` and reports
    `count` (None: it leaves the number alone)."""

    def __init__(self, init=0, count=1, status=0):
        self.init, self.count, self.status = init, count, status
        self.calls = []

    def cuInit(self, flags):
        self.calls.append(("cuInit", flags))
        return self.init

    def cuDeviceGetCount(self, pointer):
        self.calls.append(("cuDeviceGetCount",))
        if self.count is not None:
            pointer._obj.value = self.count
        return self.status


def load_driver(monkeypatch, driver):
    names = []

    def cdll(name):
        names.append(name)
        return driver

    monkeypatch.setattr(release.ctypes, "CDLL", cdll)
    return names


@pytest.mark.parametrize(
    ("platform", "name"),
    [("win32", "nvcuda.dll"), ("linux", "libcuda.so.1"), ("darwin", "libcuda.so.1")],
)
def test_the_driver_library_is_named_per_platform(monkeypatch, platform, name):
    monkeypatch.setattr(release.sys, "platform", platform)
    names = load_driver(monkeypatch, Driver())
    assert release.nvidia_driver() is True
    assert names == [name]


@pytest.mark.parametrize(("count", "expected"), [(0, False), (1, True), (4, True)])
def test_a_driver_needs_at_least_one_device(monkeypatch, count, expected):
    driver = Driver(count=count)
    load_driver(monkeypatch, driver)
    assert release.nvidia_driver() is expected
    assert driver.calls == [("cuInit", 0), ("cuDeviceGetCount",)]


def test_a_driver_that_cannot_initialize_is_not_asked_for_devices(monkeypatch):
    driver = Driver(init=100, count=2)  # CUDA_ERROR_NO_DEVICE
    load_driver(monkeypatch, driver)
    assert release.nvidia_driver() is False
    assert driver.calls == [("cuInit", 0)]


def test_a_failing_device_count_is_no_gpu_whatever_it_reports(monkeypatch):
    load_driver(monkeypatch, Driver(count=2, status=101))
    assert release.nvidia_driver() is False


def test_a_driver_that_reports_no_count_at_all_has_no_device(monkeypatch):
    load_driver(monkeypatch, Driver(count=None))
    assert release.nvidia_driver() is False


def test_a_library_with_only_cuinit_is_not_a_driver(monkeypatch):
    class HalfDriver:
        def cuInit(self, flags):
            return 0

    load_driver(monkeypatch, HalfDriver())
    assert release.nvidia_driver() is False


# --- choosing and finding a package ---------------------------------------------------------


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("darwin", "arm64", ["metal"]),
        ("darwin", "x64", ["cpu"]),
        ("win32", "x64", ["cuda", "rocm", "sycl", "vulkan", "cpu"]),
        ("win32", "arm64", ["cpu"]),
        ("linux", "x64", ["cuda", "rocm", "sycl", "vulkan", "cpu"]),
        ("linux", "arm64", ["cuda", "vulkan", "cpu"]),
        ("freebsd", "x64", []),
    ],
)
def test_supported_families_come_in_order_of_preference(system, machine, expected):
    assert release.supported(system, machine) == expected


def test_supported_defaults_each_missing_half_to_this_machine(monkeypatch):
    monkeypatch.setattr(release, "host", lambda: ("linux", "arm64"))
    assert release.supported() == ["cuda", "vulkan", "cpu"]
    assert release.supported("darwin") == ["metal"]  # this machine's arm64
    assert release.supported(machine="x64") == ["cuda", "rocm", "sycl", "vulkan", "cpu"]


def test_pick_error_messages_name_the_machine_and_what_exists(monkeypatch):
    at(monkeypatch, "darwin", "arm64")
    with pytest.raises(ValueError, match=r"^No cuda package for darwin/arm64; available: metal$"):
        release.pick("cuda")
    at(monkeypatch, "linux", "x64")
    unknown = "^Runtime must be one of: auto, metal, cuda, vulkan, rocm, sycl, cpu$"
    with pytest.raises(ValueError, match=unknown):
        release.pick("tpu")
    several = r"^No metal package for linux/x64; available: cuda, rocm, sycl, vulkan, cpu$"
    with pytest.raises(ValueError, match=several):
        release.pick("metal")
    at(monkeypatch, "freebsd", "riscv64")
    text = (
        f"No prebuilt llama.cpp {release.RELEASE} package for freebsd/riscv64. Build commit "
        f"{release.COMMIT[:7]} with -DBUILD_SHARED_LIBS=ON and point RIZZO_LLAMA_DIR at it."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(text)}$"):
        release.pick("auto")


def test_pick_does_not_probe_the_driver_when_the_answer_does_not_depend_on_it(monkeypatch):
    def probe():
        raise AssertionError("the driver is probed for `auto` on a machine with a CUDA build only")

    monkeypatch.setattr(release, "nvidia_driver", probe)
    monkeypatch.setattr(release, "host", lambda: ("darwin", "arm64"))
    assert release.pick("auto") == "metal"  # before CUDA is even considered
    monkeypatch.setattr(release, "host", lambda: ("linux", "x64"))
    assert release.pick("cuda") == "cuda"  # an explicit request needs no probe
    monkeypatch.setattr(release, "supported", lambda: ["vulkan", "cpu"])
    assert release.pick("auto") == "vulkan"  # no CUDA build: nothing to probe for
    monkeypatch.setattr(release, "supported", lambda: ["metal", "vulkan", "cpu"])
    assert release.pick("auto") == "metal"  # Metal comes before anything else, probe or not


def test_pick_recommends_for_auto_when_it_is_not_told_what_to_pick(monkeypatch):
    at(monkeypatch, "linux", "x64", nvidia=False)
    assert release.pick() == "vulkan"
    at(monkeypatch, "linux", "x64", nvidia=True)
    assert release.pick() == "cuda"


def test_install_dir_names_release_system_machine_and_family(monkeypatch):
    at(monkeypatch, "linux", "arm64")
    assert release.install_dir("vulkan") == Path("runtimes/llama-b11081-linux-arm64-vulkan")
    assert (
        release.install_dir("cpu") == release.RUNTIMES / f"llama-{release.RELEASE}-linux-arm64-cpu"
    )


def test_release_constants_agree_with_each_other():
    page = f"https://github.com/ggml-org/llama.cpp/releases/download/{release.RELEASE}"
    assert page == release.BASE_URL
    assert re.fullmatch(r"b\d+", release.RELEASE)
    assert re.fullmatch(r"[0-9a-f]{40}", release.COMMIT)
    assert release.RUNTIME_DIR_ENV == "RIZZO_LLAMA_DIR"
    assert Path("runtimes") == release.RUNTIMES
    assert set(release.ACCELERATORS) == {"auto", *release.PREFERENCE}


def test_find_library_prefers_the_directory_itself_then_sorted_subdirectories(runtimes):
    root = runtimes / "root"
    (root / "b").mkdir(parents=True)
    (root / "a").mkdir()
    (root / "b" / LIBRARY).write_bytes(b"b")
    assert release.find_library(root) == root / "b" / LIBRARY
    (root / "a" / LIBRARY).write_bytes(b"a")
    assert release.find_library(root) == root / "a" / LIBRARY  # sorted, not directory order
    (root / LIBRARY).write_bytes(b"top")
    assert release.find_library(root) == root / LIBRARY


def test_find_library_gives_up_on_look_alikes(runtimes):
    root = runtimes / "root"
    (root / "deep" / "deeper").mkdir(parents=True)
    (root / "deep" / "deeper" / LIBRARY).write_bytes(b"too deep")
    (root / "folder").mkdir()
    (root / "folder" / LIBRARY).mkdir()  # a directory with the right name is not a library
    assert release.find_library(root) is None
    assert release.find_library(runtimes / "missing") is None


def test_installed_lists_the_families_that_have_a_library(runtimes):
    assert release.installed() == []
    make_runtime("cpu")
    make_runtime("cuda", nested="build")
    release.install_dir("vulkan").mkdir(parents=True)  # a folder without the library
    assert release.installed() == ["cuda", "cpu"]


def test_locate_takes_the_recommended_family_first_then_the_preference_order(runtimes, monkeypatch):
    make_runtime("cuda")
    make_runtime("vulkan")
    make_runtime("cpu")
    assert release.locate() == release.install_dir("vulkan")  # no NVIDIA driver: Vulkan
    at(monkeypatch, "linux", "x64", nvidia=True)
    assert release.locate() == release.install_dir("cuda")
    # Only builds that are not recommended are left: the best of them, not the plainest.
    at(monkeypatch, "linux", "x64")
    for family in ("cuda", "vulkan"):
        (release.install_dir(family) / LIBRARY).unlink()
    make_runtime("sycl")
    assert release.locate() == release.install_dir("sycl")


def test_locate_with_a_family_puts_it_first_and_never_asks_for_a_recommendation(
    runtimes, monkeypatch
):
    def recommend(accelerator="auto"):
        raise AssertionError("a named family is not a question for pick()")

    monkeypatch.setattr(release, "pick", recommend)
    make_runtime("cpu")
    make_runtime("vulkan")
    make_runtime("sycl")
    assert release.locate("cpu") == release.install_dir("cpu")
    assert release.locate("sycl") == release.install_dir("sycl")
    # `rocm` is not installed: the rest keep the order of preference (sycl before vulkan).
    assert release.locate("rocm") == release.install_dir("sycl")
    assert release.locate("metal") == release.install_dir("sycl")  # not even a family here


@pytest.mark.parametrize(
    "device",
    ["auto", "gpu", "AUTO", "Vulkan1", "radeon", "mlx", pytest.param("", id="empty")],
)
def test_locate_follows_the_recommendation_unless_a_family_is_named(runtimes, monkeypatch, device):
    """`--device` reaches locate() as it was typed, and only a family name asks for a runtime:
    `auto`, `gpu` or the name of a device get what `pick("auto")` recommends, as with no argument
    at all. An installed CUDA build must not shadow Vulkan on a machine without an NVIDIA driver."""
    for family in ("cuda", "sycl", "vulkan", "cpu"):
        make_runtime(family)
    assert release.locate(device) == release.install_dir("vulkan")  # no NVIDIA driver
    at(monkeypatch, "linux", "x64", nvidia=True)
    assert release.locate(device) == release.install_dir("cuda")
    # The recommendation is not installed: the rest keep the order of preference (sycl first).
    (release.install_dir("cuda") / LIBRARY).unlink()
    (release.install_dir("vulkan") / LIBRARY).unlink()
    assert release.locate(device) == release.install_dir("sycl")


def test_locate_recommends_for_auto_and_gpu_exactly_as_it_does_without_a_family(
    runtimes, monkeypatch
):
    for family in ("cuda", "sycl", "cpu"):
        make_runtime(family)
    asked = []

    def recommend(accelerator="auto"):
        asked.append(accelerator)
        return "cpu"

    monkeypatch.setattr(release, "pick", recommend)
    picked = [release.locate(device) for device in (None, "auto", "gpu")]
    assert picked == [release.install_dir("cpu")] * 3  # the recommendation, not cuda
    assert asked == ["auto"] * 3


@pytest.mark.parametrize("family", ["cuda", "CUDA", "Cuda"])
def test_a_family_is_a_request_whatever_its_case(runtimes, family):
    make_runtime("cuda")
    make_runtime("vulkan")  # what `auto` would take here: no NVIDIA driver
    assert release.locate(family) == release.install_dir("cuda")


def test_locate_reports_a_missing_runtime_with_the_way_out(runtimes):
    text = (
        "llama.cpp runtime not installed. Run `rizzo download` (runtime + weights) or "
        f"`rizzo download --only runtime`; or set RIZZO_LLAMA_DIR to a build of {release.COMMIT[:7]}."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(text)}$"):
        release.locate()
    with pytest.raises(ValueError, match="runtime not installed"):
        release.locate("vulkan")


def test_locate_on_a_machine_without_packages_says_not_installed_instead_of_asking_pick(
    runtimes, monkeypatch
):
    def recommend(accelerator="auto"):
        raise AssertionError("nothing is supported, so there is nothing to recommend")

    monkeypatch.setattr(release, "supported", list)
    monkeypatch.setattr(release, "pick", recommend)
    with pytest.raises(ValueError, match="runtime not installed"):
        release.locate()
    with pytest.raises(ValueError, match="runtime not installed"):
        release.locate("cuda")


def test_the_directory_of_the_environment_replaces_every_installed_runtime(runtimes, monkeypatch):
    make_runtime("vulkan")
    own = runtimes / "own"
    own.mkdir()
    (own / LIBRARY).write_bytes(b"mine")
    monkeypatch.setenv(release.RUNTIME_DIR_ENV, str(own))
    assert release.locate() == own
    assert release.locate("cpu") == own
    empty = runtimes / "empty"
    empty.mkdir()
    monkeypatch.setenv(release.RUNTIME_DIR_ENV, str(empty))
    with pytest.raises(
        ValueError, match=re.escape(f"RIZZO_LLAMA_DIR={empty}: {LIBRARY} not found there")
    ):
        release.locate()
    monkeypatch.setenv(release.RUNTIME_DIR_ENV, "")  # empty means unset
    assert release.locate() == release.install_dir("vulkan")


# --- checksums ------------------------------------------------------------------------------


@settings(
    max_examples=40,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.binary(max_size=4096))
def test_sha256_file_is_the_sha256_of_the_bytes(tmp_path, data):
    path = tmp_path / "blob"
    path.write_bytes(data)
    assert release.sha256_file(path) == hashlib.sha256(data).hexdigest()


# --- fetch: a scripted urlopen --------------------------------------------------------------


class Response:
    """What `urlopen` returns: a context manager with a status, headers and a body read in pieces.

    `promised` is the Content-Length the server announces (None: no header at all)."""

    def __init__(self, body, *, status=200, promised="all", piece=1 << 20):
        self.status = status
        self.headers = {}
        if promised is not None:
            self.headers["Content-Length"] = str(len(body) if promised == "all" else promised)
        self._stream = io.BytesIO(body)
        self._piece = piece

    def read(self, size=-1):
        return self._stream.read(min(size, self._piece) if size >= 0 else self._piece)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._stream.close()


def http_error(code, reason="Nope", body=None):
    # With a body: older Pythons leave an HTTPError without one half initialized.
    return urllib.error.HTTPError(
        "https://example.test/file",
        code,
        reason,
        email.message.Message(),
        io.BytesIO() if body is None else body,
    )


def serve(monkeypatch, handler):
    """Replace `urlopen`: every request goes to `handler(request)`; returns (request, timeout)s."""
    seen = []

    def urlopen(request, timeout=None):
        seen.append((request, timeout))
        return handler(request)

    monkeypatch.setattr(release.urllib.request, "urlopen", urlopen)
    return seen


def script(monkeypatch, *steps):
    """One step per request: an exception to raise, a Response, or a callable(request) -> either."""
    remaining = list(steps)

    def handler(request):
        step = remaining.pop(0)  # a request beyond the script is a failure of the test
        if callable(step):
            step = step(request)
        if isinstance(step, BaseException):
            raise step
        return step

    return serve(monkeypatch, handler)


def range_server(payload):
    """A server that honours `Range: bytes=N-` and answers 416 past the end, like a CDN does."""

    def handler(request):
        header = request.get_header("Range")
        if header is None:
            return Response(payload)
        start = int(header.removeprefix("bytes=").rstrip("-"))
        if start >= len(payload):
            raise http_error(416, "Range Not Satisfiable")
        return Response(payload[start:], status=206)

    return handler


def test_fetch_sends_its_headers_reports_progress_and_leaves_no_partial_file(tmp_path, monkeypatch):
    seen = script(monkeypatch, Response(PAYLOAD, piece=4))
    progress = []
    target = tmp_path / "deep" / "er" / "model.gguf"
    result = release.fetch(HUGGING_FACE, target, DIGEST, lambda *call: progress.append(call))
    assert result == target
    assert target.read_bytes() == PAYLOAD
    assert progress == [("model.gguf", 4, 10), ("model.gguf", 8, 10), ("model.gguf", 10, 10)]
    assert not target.with_name("model.gguf.part").exists()
    ((request, timeout),) = seen
    assert request.full_url == HUGGING_FACE
    assert request.get_header("User-agent") == "rizzo-flow"
    assert request.get_header("Range") is None
    assert request.get_header("Authorization") is None
    assert timeout == 120


def test_the_token_is_bound_to_the_first_host_and_not_forwarded_by_redirects(tmp_path, monkeypatch):
    seen = script(monkeypatch, Response(PAYLOAD))
    release.fetch(HUGGING_FACE, tmp_path / "model.gguf", DIGEST, token="hf_secret")
    ((request, _),) = seen
    assert request.unredirected_hdrs == {"Authorization": "Bearer hf_secret"}
    assert "Authorization" not in request.headers  # urllib copies `headers` to a redirect


def test_a_verified_copy_is_never_downloaded_again_and_a_wrong_one_is_replaced(
    tmp_path, monkeypatch
):
    seen = script(monkeypatch, Response(PAYLOAD))
    target = tmp_path / "model.gguf"
    target.write_bytes(b"something else")
    release.fetch(GITHUB, target, DIGEST)
    assert target.read_bytes() == PAYLOAD
    assert len(seen) == 1
    release.fetch(GITHUB, target, DIGEST)  # verified now: no second request (the script is spent)
    assert len(seen) == 1


FLAKY = [
    pytest.param(lambda: http_error(500, "Internal Server Error"), id="500"),
    pytest.param(lambda: http_error(503, "Service Unavailable"), id="503"),
    pytest.param(lambda: http_error(429, "Too Many Requests"), id="429"),
    pytest.param(lambda: http_error(416, "Range Not Satisfiable"), id="416-without-a-range"),
    pytest.param(lambda: urllib.error.URLError("no route to host"), id="url-error"),
    pytest.param(lambda: TimeoutError("timed out"), id="timeout"),
    pytest.param(lambda: ConnectionResetError("reset by peer"), id="reset"),
    pytest.param(lambda: http.client.IncompleteRead(b"ab", 8), id="incomplete-read"),
    pytest.param(lambda: http.client.RemoteDisconnected("closed"), id="remote-disconnected"),
    pytest.param(lambda: http.client.BadStatusLine("garbage"), id="bad-status-line"),
]


@pytest.mark.parametrize("failure", FLAKY)
def test_transient_failures_are_retried(tmp_path, monkeypatch, failure):
    seen = script(monkeypatch, failure(), failure(), Response(PAYLOAD))
    target = release.fetch(GITHUB, tmp_path / "runtime.zip", DIGEST)
    assert target.read_bytes() == PAYLOAD
    assert len(seen) == 3


def test_giving_up_names_the_attempts_and_the_last_reason(tmp_path, monkeypatch):
    seen = script(
        monkeypatch,
        urllib.error.URLError("first"),
        TimeoutError("second"),
        http_error(503, "Service Unavailable"),
    )
    expected = (
        r"^model.gguf: download failed after 3 attempts \(HTTP Error 503: Service Unavailable\)$"
    )
    with pytest.raises(ValueError, match=expected):
        release.fetch(GITHUB, tmp_path / "model.gguf", DIGEST, attempts=3)
    assert len(seen) == 3
    seen = script(monkeypatch, *[TimeoutError("slow")] * 5)  # five is the default
    with pytest.raises(
        ValueError, match=r"^model.gguf: download failed after 5 attempts \(slow\)$"
    ):
        release.fetch(GITHUB, tmp_path / "model.gguf", DIGEST)
    assert len(seen) == 5


@pytest.mark.parametrize(("code", "denied"), [(503, False), (416, False), (404, True)])
def test_the_connection_of_an_error_response_is_let_go_whatever_fetch_does_next(
    tmp_path, monkeypatch, code, denied
):
    body = io.BytesIO(b"<html>an error page nobody reads</html>")
    script(monkeypatch, http_error(code, body=body), Response(PAYLOAD))
    if denied:
        with pytest.raises(ValueError, match=rf"HTTP {code}"):
            release.fetch(GITHUB, tmp_path / "model.gguf", DIGEST)
    else:  # retried
        release.fetch(GITHUB, tmp_path / "model.gguf", DIGEST)
    assert body.closed


@pytest.mark.parametrize("code", [401, 403, 404])
@pytest.mark.parametrize(
    ("url", "hint"),
    [
        (
            HUGGING_FACE,
            (
                "the repository is private or gated: set HF_TOKEN (environment or .env) "
                "to a token that can read it"
            ),
        ),
        (GITHUB, "access denied or not found"),
    ],
)
def test_denials_fail_at_once_with_a_hint_for_the_source(tmp_path, monkeypatch, code, url, hint):
    seen = script(monkeypatch, http_error(code))  # a second request would run past the script
    message = f"^{re.escape(f'model.gguf: HTTP {code}, {hint}')}$"
    with pytest.raises(ValueError, match=message) as failure:
        release.fetch(url, tmp_path / "model.gguf", DIGEST)
    assert len(seen) == 1
    assert failure.value.__suppress_context__ is True  # the transport error is noise for the user


def test_a_body_cut_short_resumes_from_the_bytes_on_disk(tmp_path, monkeypatch):
    def rest(request):
        assert request.get_header("Range") == "bytes=3-"
        return Response(PAYLOAD[3:], status=206)

    seen = script(monkeypatch, Response(PAYLOAD[:3], promised=10), rest)
    progress = []
    target = release.fetch(
        GITHUB, tmp_path / "model.gguf", DIGEST, lambda *call: progress.append(call)
    )
    assert target.read_bytes() == PAYLOAD
    assert progress == [("model.gguf", 3, 10), ("model.gguf", 10, 10)]
    assert len(seen) == 2


def test_running_out_of_attempts_keeps_the_partial_file_for_the_next_run(tmp_path, monkeypatch):
    def more(request):
        return Response(PAYLOAD[3:5], promised=7, status=206)

    script(monkeypatch, Response(PAYLOAD[:3], promised=10), more)
    target = tmp_path / "model.gguf"
    with pytest.raises(
        ValueError, match=r"failed after 2 attempts \(connection closed at 5 of 10 bytes\)"
    ):
        release.fetch(GITHUB, target, DIGEST, attempts=2)
    assert not target.exists()
    assert target.with_name("model.gguf.part").read_bytes() == PAYLOAD[:5]


def test_a_partial_file_is_extended_when_the_server_honours_the_range(tmp_path, monkeypatch):
    partial = tmp_path / "model.gguf.part"
    partial.write_bytes(PAYLOAD[:4])
    seen = serve(monkeypatch, range_server(PAYLOAD))
    progress = []
    target = release.fetch(
        GITHUB, tmp_path / "model.gguf", DIGEST, lambda *call: progress.append(call)
    )
    assert target.read_bytes() == PAYLOAD
    assert [request.get_header("Range") for request, _ in seen] == ["bytes=4-"]
    assert progress == [("model.gguf", 10, 10)]  # counted from what was already there
    assert not partial.exists()


def test_a_server_that_ignores_the_range_restarts_the_file_from_scratch(tmp_path, monkeypatch):
    (tmp_path / "model.gguf.part").write_bytes(b"stale")
    progress = []
    seen = script(monkeypatch, Response(PAYLOAD))  # a plain 200, Range or not
    target = release.fetch(
        GITHUB, tmp_path / "model.gguf", DIGEST, lambda *call: progress.append(call)
    )
    assert seen[0][0].get_header("Range") == "bytes=5-"
    assert target.read_bytes() == PAYLOAD  # the stale bytes were not kept
    assert progress == [("model.gguf", 10, 10)]


def test_without_a_content_length_only_the_hash_decides(tmp_path, monkeypatch):
    progress = []
    script(monkeypatch, Response(PAYLOAD, promised=None))
    release.fetch(GITHUB, tmp_path / "ok.bin", DIGEST, lambda *call: progress.append(call))
    assert progress == [("ok.bin", 10, 0)]  # no total to report
    script(monkeypatch, Response(PAYLOAD[:6], promised=None))  # cut short, and nobody can tell
    with pytest.raises(ValueError, match="sha256 mismatch"):
        release.fetch(GITHUB, tmp_path / "cut.bin", DIGEST)
    assert not (tmp_path / "cut.bin.part").exists()  # never kept


def test_a_hash_mismatch_reports_both_digests(tmp_path, monkeypatch):
    script(monkeypatch, Response(PAYLOAD))
    wrong = "0" * 64
    expected = rf"^model.gguf: sha256 mismatch \(expected {wrong}, got {DIGEST}\)$"
    with pytest.raises(ValueError, match=expected):
        release.fetch(GITHUB, tmp_path / "model.gguf", wrong)
    assert not (tmp_path / "model.gguf").exists()
    assert not (tmp_path / "model.gguf.part").exists()


def test_a_complete_partial_file_left_by_an_interrupted_run_is_accepted(tmp_path, monkeypatch):
    """The previous run was stopped after the last byte arrived and before the checksum and the
    rename. A CDN answers `Range: bytes=<size>-` with 416: the bytes on disk are checked then, and
    they are the file, so there is nothing left to download."""
    partial = tmp_path / "model.gguf.part"
    partial.write_bytes(PAYLOAD)
    (tmp_path / "model.gguf").write_bytes(b"an older copy")
    seen = serve(monkeypatch, range_server(PAYLOAD))
    progress = []
    target = release.fetch(
        GITHUB, tmp_path / "model.gguf", DIGEST, lambda *call: progress.append(call)
    )
    assert target.read_bytes() == PAYLOAD
    assert [request.get_header("Range") for request, _ in seen] == ["bytes=10-"]  # asked once
    assert not partial.exists()
    assert progress == []  # nothing was transferred


@pytest.mark.parametrize(
    "leftover",
    [
        pytest.param(bytes(reversed(PAYLOAD)), id="same-size-other-bytes"),
        pytest.param(PAYLOAD + b"more", id="longer-than-the-file"),
    ],
)
def test_a_partial_file_that_is_not_the_file_is_dropped_when_the_range_is_refused(
    tmp_path, monkeypatch, leftover
):
    """416 says the partial file is as long as the file, or longer; only the checksum can say it is
    the file. When it is not, the next request starts from the first byte, not the same one again."""
    partial = tmp_path / "model.gguf.part"
    partial.write_bytes(leftover)
    seen = serve(monkeypatch, range_server(PAYLOAD))
    target = release.fetch(GITHUB, tmp_path / "model.gguf", DIGEST)
    assert target.read_bytes() == PAYLOAD
    assert [request.get_header("Range") for request, _ in seen] == [f"bytes={len(leftover)}-", None]
    assert not partial.exists()


def test_a_refused_range_never_lets_an_unverified_partial_file_through(tmp_path, monkeypatch):
    partial = tmp_path / "model.gguf.part"
    partial.write_bytes(bytes(reversed(PAYLOAD)))  # as long as the file, and not the file
    serve(monkeypatch, range_server(PAYLOAD))
    with pytest.raises(ValueError, match=r"^model.gguf: download failed after 1 attempts \(HTTP"):
        release.fetch(GITHUB, tmp_path / "model.gguf", DIGEST, attempts=1)
    assert not (tmp_path / "model.gguf").exists()  # never renamed on the strength of a 416
    assert not partial.exists()  # dropped: the next run starts from the first byte


@settings(max_examples=60, deadline=None, database=None, derandomize=True)
@given(
    payload=st.binary(max_size=200),
    cuts=st.lists(st.integers(0, 60), max_size=4),
    partial=st.integers(0, 60),
)
def test_a_download_cut_anywhere_resumes_from_exactly_the_bytes_received(payload, cuts, partial):
    """Whatever an interrupted run left behind and wherever each attempt is cut, the next request
    asks for the first byte that is missing, and the file that comes out is the payload."""
    have = min(
        partial, max(len(payload) - 1, 0)
    )  # a leftover .part is always shorter than the file
    starts, delivered, pending = [], [], list(cuts)

    def handler(request):
        header = request.get_header("Range")
        start = 0 if header is None else int(header.removeprefix("bytes=").rstrip("-"))
        starts.append(start)
        body = payload[start:]
        cut = pending.pop(0) if pending else len(body)
        delivered.append(min(cut, len(body)))
        return Response(body[:cut], status=200 if header is None else 206, promised=len(body))

    with tempfile.TemporaryDirectory() as folder, pytest.MonkeyPatch.context() as patch:
        serve(patch, handler)
        (Path(folder) / "model.gguf.part").write_bytes(payload[:have])
        progress = []
        target = release.fetch(
            GITHUB,
            Path(folder) / "model.gguf",
            hashlib.sha256(payload).hexdigest(),
            lambda *call: progress.append(call),
            attempts=len(cuts) + 1,
        )
        assert target.read_bytes() == payload
        assert not target.with_name("model.gguf.part").exists()
    received = have
    for start, size in zip(starts, delivered, strict=True):
        assert start == received  # the first byte that is not on disk yet
        received += size
    assert received == len(payload)
    reported = [done for _, done, _ in progress]
    assert reported == sorted(reported)  # the count only grows, and ends at the size of the file
    assert reported[-1:] == ([len(payload)] if payload else [])


# --- unpack ---------------------------------------------------------------------------------


def test_zip_members_land_under_the_destination_and_a_missing_destination_is_created(
    tmp_path, monkeypatch
):
    archive_zip(tmp_path / "ok.zip", {"llama.dll": b"1", "bin/ggml.dll": b"2", "a/../b.txt": b"3"})
    monkeypatch.chdir(tmp_path)
    release.unpack(Path("ok.zip"), Path("made/on/demand"))
    out = tmp_path / "made" / "on" / "demand"
    assert (out / "llama.dll").read_bytes() == b"1"
    assert (out / "bin" / "ggml.dll").read_bytes() == b"2"
    inside = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert all(name == "ok.zip" or name.startswith("made/on/demand/") for name in inside)


def test_one_unsafe_zip_member_refuses_the_whole_archive_before_extracting_anything(tmp_path):
    archive_zip(tmp_path / "bad.zip", {"fine.dll": b"1", "../evil.dll": b"2"})
    with pytest.raises(ValueError, match=r"^bad.zip: unsafe member \.\./evil.dll$"):
        release.unpack(tmp_path / "bad.zip", tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []
    assert not (tmp_path / "evil.dll").exists()


def test_absolute_zip_member_names_are_refused(tmp_path):
    archive_zip(tmp_path / "abs.zip", {"/absolute/evil.dll": b"x"})
    with pytest.raises(ValueError, match=r"^abs.zip: unsafe member /absolute/evil.dll$"):
        release.unpack(tmp_path / "abs.zip", tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def escapes(parts):
    """Whether a path made of these components ever climbs above the directory it starts in."""
    depth = 0
    for part in parts:
        depth += {"..": -1, ".": 0}.get(part, 1)
        if depth < 0:
            return True
    return False


@settings(max_examples=60, deadline=None, database=None, derandomize=True)
@given(parts=st.lists(st.sampled_from(["a", "b", "..", "."]), max_size=4))
def test_no_zip_member_ever_lands_outside_the_destination(parts):
    name = "/".join([*parts, "payload.bin"])
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        archive_zip(root / "in.zip", {name: b"x"})
        if escapes(parts):
            with pytest.raises(ValueError, match="unsafe member"):
                release.unpack(root / "in.zip", root / "out")
        else:
            release.unpack(root / "in.zip", root / "out")
        landed = [p for p in root.rglob("*") if p.is_file() and p != root / "in.zip"]
        assert all(p.is_relative_to(root / "out") for p in landed)
        assert len(landed) == (0 if escapes(parts) else 1)


@settings(max_examples=50, deadline=None, database=None, derandomize=True)
@given(
    paths=st.lists(
        # Folders are upper case and files lower case, so that a file never shares a name with a
        # folder.
        st.tuples(st.lists(st.sampled_from("AB"), max_size=2), st.sampled_from("fg")).map(
            lambda pair: "/".join([*pair[0], pair[1]])
        ),
        unique=True,
        max_size=6,
    )
)
def test_a_tarball_unpacks_to_the_tree_inside_its_wrapper_folder(paths):
    files = {path: path.encode() for path in paths}
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        archive_tar(root / "in.tar.gz", {f"llama-bX/{name}": data for name, data in files.items()})
        release.unpack(root / "in.tar.gz", root / "out")
        tree = {
            p.relative_to(root / "out").as_posix(): p.read_bytes()
            for p in (root / "out").rglob("*")
            if p.is_file()
        }
    assert tree == files


def test_tarball_folders_are_flattened_and_wrappers_and_odd_members_are_skipped(tmp_path):
    wrapper = tarfile.TarInfo("llama-b1")
    wrapper.type = tarfile.DIRTYPE
    nested = tarfile.TarInfo("llama-b1/sub")
    nested.type = tarfile.DIRTYPE
    odd = tarfile.TarInfo("llama-b1/")  # not a directory entry, and nothing to extract
    archive_tar(
        tmp_path / "in.tar.gz",
        {
            "llama-b1": wrapper,
            "llama-b1/": odd,
            "llama-b1/sub": nested,
            "llama-b1/libllama.so": b"lib",
            "llama-b1/sub/deep.so": b"deep",
        },
    )
    release.unpack(tmp_path / "in.tar.gz", tmp_path / "out")
    found = sorted(
        p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*")
    )
    assert found == ["libllama.so", "sub", "sub/deep.so"]
    assert (tmp_path / "out" / "sub" / "deep.so").read_bytes() == b"deep"


def test_tarball_links_leaving_the_destination_are_refused(tmp_path):
    link = tarfile.TarInfo("llama-b1/libllama.so")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/hostname"
    archive_tar(tmp_path / "links.tar.gz", {"llama-b1/libllama.so": link})
    with pytest.raises(tarfile.FilterError):
        release.unpack(tmp_path / "links.tar.gz", tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_any_name_that_is_not_a_zip_is_read_as_a_tarball(tmp_path):
    archive_tar(tmp_path / "release.tgz", {"top/file": b"x"})
    release.unpack(tmp_path / "release.tgz", tmp_path / "out")
    assert (tmp_path / "out" / "file").read_bytes() == b"x"


# --- install --------------------------------------------------------------------------------


@pytest.fixture
def served(runtimes, monkeypatch):
    """Two archives per family in a folder that stands in for the release page."""
    folder = runtimes / "served"
    folder.mkdir()
    monkeypatch.setattr(release, "BASE_URL", folder.as_uri())
    at(monkeypatch, "win32", "x64")

    def publish(family, *archives):
        packages = []
        for name, members in archives:
            archive_zip(folder / name, members)
            packages.append((name, release.sha256_file(folder / name)))
        release.PACKAGES[("win32", "x64", family)] = packages

    monkeypatch.setattr(release, "PACKAGES", {})
    return publish


def test_a_package_with_several_archives_lands_in_one_directory(served, runtimes):
    served(
        "cuda",
        ("main.zip", {LIBRARY: b"lib", "ggml.dll": b"ggml"}),
        ("cudart.zip", {"cudart.dll": b"rt"}),
    )
    progress = []
    directory = release.install("cuda", lambda *call: progress.append(call))
    assert directory == release.install_dir("cuda")
    assert sorted(p.name for p in directory.iterdir()) == ["cudart.dll", "ggml.dll", LIBRARY]
    assert not directory.with_name(directory.name + ".partial").exists()
    downloads = runtimes / "runtimes" / "downloads"
    assert sorted(p.name for p in downloads.iterdir()) == ["cudart.zip", "main.zip"]
    assert {name for name, _, _ in progress} == {"main.zip", "cudart.zip"}


def test_install_downloads_from_the_release_page_with_the_pinned_checksum(
    served, runtimes, monkeypatch
):
    served("cpu", ("cpu.zip", {LIBRARY: b"lib"}))
    calls = []
    original = release.fetch

    def recording(url, target, sha256, progress=None, attempts=5, token=None):
        calls.append((url, target, sha256, progress, token))
        return original(url, target, sha256, progress, attempts, token)

    monkeypatch.setattr(release, "fetch", recording)
    reported = []

    def progress(*call):
        reported.append(call)

    release.install("cpu", progress)
    (call,) = calls
    assert reported  # the callback given to install is the one fetch reports to
    assert call == (
        f"{release.BASE_URL}/cpu.zip",
        release.RUNTIMES / "downloads" / "cpu.zip",
        release.PACKAGES[("win32", "x64", "cpu")][0][1],
        progress,
        None,
    )


def test_installing_twice_downloads_once(served, monkeypatch):
    served("cpu", ("cpu.zip", {LIBRARY: b"lib"}))
    calls = []
    original = release.fetch

    def counting(*args, **options):
        calls.append(args)
        return original(*args, **options)

    monkeypatch.setattr(release, "fetch", counting)
    first = release.install("cpu")
    assert release.install("cpu") == first
    assert len(calls) == 1


def test_install_without_a_family_installs_the_recommended_one(served):
    served("cpu", ("cpu.zip", {LIBRARY: b"cpu"}))
    served("vulkan", ("vulkan.zip", {LIBRARY: b"vulkan"}))
    assert release.install() == release.install_dir("vulkan")  # any GPU vendor, else the CPU
    assert not release.install_dir("cpu").exists()


def test_install_returns_the_directory_that_holds_the_library(served):
    served("cpu", ("cpu.zip", {f"bin/{LIBRARY}": b"lib", "bin/ggml.dll": b"g"}))
    assert release.install("cpu") == release.install_dir("cpu") / "bin"
    assert release.install("cpu") == release.install_dir("cpu") / "bin"  # and when it is found


def test_a_package_without_the_library_is_refused_and_the_next_run_starts_clean(served):
    served("cpu", ("empty.zip", {"README.txt": b"nothing useful"}))
    with pytest.raises(ValueError, match=rf"^{re.escape(LIBRARY)} not found in the cpu package$"):
        release.install("cpu")
    directory = release.install_dir("cpu")
    assert not directory.exists()  # nothing half-installed under the final name
    staging = directory.with_name(directory.name + ".partial")
    assert (staging / "README.txt").is_file()
    served("cpu", ("fixed.zip", {LIBRARY: b"lib"}))
    assert release.install("cpu") == directory
    assert sorted(p.name for p in directory.iterdir()) == [LIBRARY]  # the stale file is gone
    assert not staging.exists()


def test_a_runtime_that_lost_its_library_is_replaced_by_the_next_install(served):
    served("cpu", ("v1.zip", {LIBRARY: b"lib-v1", "ggml.dll": b"ggml-v1"}))
    directory = release.install("cpu")
    (directory / LIBRARY).unlink()  # antivirus quarantine, partial delete: the rest stays
    assert release.installed() == []
    served("cpu", ("v2.zip", {LIBRARY: b"lib-v2", "ggml.dll": b"ggml-v2"}))
    assert release.install("cpu") == directory
    assert release.installed() == ["cpu"]
    assert (directory / LIBRARY).read_bytes() == b"lib-v2"
    assert (directory / "ggml.dll").read_bytes() == b"ggml-v2"
    assert not directory.with_name(directory.name + ".partial").exists()


def test_a_bad_package_never_destroys_the_damaged_runtime_it_was_meant_to_repair(served):
    served("cpu", ("v1.zip", {LIBRARY: b"lib-v1", "ggml.dll": b"ggml-v1"}))
    directory = release.install("cpu")
    (directory / LIBRARY).unlink()
    served("cpu", ("empty.zip", {"README.txt": b"nothing useful"}))
    with pytest.raises(ValueError, match=rf"^{re.escape(LIBRARY)} not found in the cpu package$"):
        release.install("cpu")
    # Refused before anything was removed.
    assert (directory / "ggml.dll").read_bytes() == b"ggml-v1"


def test_install_refuses_an_unknown_family_before_downloading_anything(
    served, runtimes, monkeypatch
):
    served("cpu", ("cpu.zip", {LIBRARY: b"lib"}))

    def fetch(*args, **options):
        raise AssertionError("nothing may be downloaded for a family that does not exist")

    monkeypatch.setattr(release, "fetch", fetch)
    with pytest.raises(ValueError, match=r"^Runtime must be one of: auto, "):
        release.install("tpu")
    with pytest.raises(ValueError, match=r"^No metal package for win32/x64; available: "):
        release.install("metal")
    assert not (runtimes / "runtimes").exists()
