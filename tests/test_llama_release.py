"""Runtime packages: choice per machine, verified download, safe unpacking. No network."""

import email.message
import hashlib
import http.server
import io
import re
import tarfile
import threading
import types
import urllib.error
import zipfile

import pytest

from rizzo_flow import llama_release as release

LIBRARY = "llama.testlib"  # any name will do: the platform's real one is not what is tested
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


def make_runtime(family):
    """An installed runtime of `family`."""
    folder = release.install_dir(family)
    folder.mkdir(parents=True)
    (folder / LIBRARY).write_bytes(b"library")
    return folder


@pytest.mark.parametrize(
    ("system", "machine", "nvidia", "expected"),
    [
        ("darwin", "arm64", False, "metal"),
        ("win32", "x64", True, "cuda"),
        ("win32", "x64", False, "vulkan"),  # AMD, Intel: one build for every GPU vendor
        ("linux", "x64", True, "cuda"),
        ("linux", "x64", False, "vulkan"),
        ("linux", "arm64", False, "vulkan"),
        ("win32", "arm64", False, "cpu"),
        ("darwin", "x64", False, "cpu"),
    ],
)
def test_auto_picks_the_package_for_the_machine(monkeypatch, system, machine, nvidia, expected):
    at(monkeypatch, system, machine, nvidia)
    assert release.pick("auto") == expected


def test_explicit_family_must_exist_for_the_machine(monkeypatch):
    at(monkeypatch, "darwin", "arm64")
    with pytest.raises(ValueError, match="No cuda package"):
        release.pick("cuda")
    at(monkeypatch, "linux", "x64")
    assert release.pick("rocm") == "rocm"
    with pytest.raises(ValueError, match="one of"):
        release.pick("tpu")


def test_unknown_machine_is_told_how_to_bring_a_build(monkeypatch):
    at(monkeypatch, "freebsd", "riscv64")
    with pytest.raises(ValueError, match=release.RUNTIME_DIR_ENV):
        release.pick("auto")


def test_nvidia_driver_is_false_when_the_library_has_no_device(monkeypatch):
    """A leftover libcuda.so.1 dlopens fine on an AMD box but cuInit says 100."""

    class Driver:
        def cuInit(self, flags):
            return 100  # CUDA_ERROR_NO_DEVICE

        def cuDeviceGetCount(self, pointer):
            raise AssertionError("no device count to read after cuInit failed")

    monkeypatch.setattr(release.ctypes, "CDLL", lambda name: Driver())
    assert release.nvidia_driver() is False


def test_nvidia_driver_is_true_when_a_device_is_present(monkeypatch):
    class Driver:
        def cuInit(self, flags):
            assert flags == 0
            return 0

        def cuDeviceGetCount(self, pointer):
            pointer._obj.value = 1
            return 0

    monkeypatch.setattr(release.ctypes, "CDLL", lambda name: Driver())
    assert release.nvidia_driver() is True


def test_nvidia_driver_is_false_without_the_library(monkeypatch):
    def missing(name):
        raise OSError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(release.ctypes, "CDLL", missing)
    assert release.nvidia_driver() is False


def test_nvidia_driver_is_false_for_a_library_that_is_not_cuda(monkeypatch):
    monkeypatch.setattr(release.ctypes, "CDLL", lambda name: object())
    assert release.nvidia_driver() is False


@pytest.mark.parametrize(("nvidia", "expected"), [(False, "vulkan"), (True, "cuda")])
def test_auto_prefers_cuda_only_with_a_usable_nvidia_gpu(monkeypatch, nvidia, expected):
    monkeypatch.setattr(release, "supported", lambda: ["cuda", "vulkan", "cpu"])
    monkeypatch.setattr(release, "nvidia_driver", lambda: nvidia)
    assert release.pick("auto") == expected


def test_every_package_is_pinned_by_a_sha256():
    for (system, machine, family), archives in release.PACKAGES.items():
        assert family in release.PREFERENCE, (system, machine, family)
        for name, sha256 in archives:
            assert re.fullmatch(r"[0-9a-f]{64}", sha256), name
            assert name.endswith((".zip", ".tar.gz"))
            # Only the Windows CUDA runtime archive is shared between releases.
            assert release.RELEASE in name or name.startswith("cudart-llama-bin-win")


def archive_zip(path, members):
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)


def archive_tar(path, members):
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))


def test_tarballs_lose_their_wrapping_folder(tmp_path):
    archive_tar(tmp_path / "a.tar.gz", {"llama-bX/libllama.so": b"1", "llama-bX/sub/x": b"2"})
    archive_tar(tmp_path / "b.tar.gz", {"cudart-bX/libcudart.so.13": b"3"})
    for name in ("a.tar.gz", "b.tar.gz"):
        release.unpack(tmp_path / name, tmp_path / "out")
    found = sorted(
        p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*")
    )
    assert found == ["libcudart.so.13", "libllama.so", "sub", "sub/x"]


def test_archives_cannot_write_outside_the_destination(tmp_path):
    archive_zip(tmp_path / "bad.zip", {"../escaped.dll": b"x"})
    with pytest.raises(ValueError, match="unsafe"):
        release.unpack(tmp_path / "bad.zip", tmp_path / "out")
    archive_tar(tmp_path / "bad.tar.gz", {"top/../../escaped.so": b"x"})
    with pytest.raises(tarfile.FilterError):
        release.unpack(tmp_path / "bad.tar.gz", tmp_path / "out")
    assert not (tmp_path / "escaped.dll").exists() and not (tmp_path / "escaped.so").exists()


def test_fetch_verifies_and_never_keeps_a_bad_file(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    target = tmp_path / "cache" / "file.bin"
    seen = []
    release.fetch(source.as_uri(), target, good, lambda *call: seen.append(call))
    assert target.read_bytes() == b"weights" and seen[-1][1] == 7
    with pytest.raises(ValueError, match="sha256 mismatch"):
        release.fetch(source.as_uri(), tmp_path / "other.bin", "0" * 64)
    assert not (tmp_path / "other.bin").exists() and not (tmp_path / "other.bin.part").exists()
    source.unlink()  # a verified copy is reused without touching the source
    assert release.fetch(source.as_uri(), target, good) == target


def test_interrupted_download_resumes_where_it_stopped(tmp_path):
    payload = bytes(range(256)) * 64
    requests = []

    class Flaky(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.headers.get("Range"))
            if self.headers.get("Range"):
                start = int(self.headers["Range"].removeprefix("bytes=").rstrip("-"))
                self.send_response(206)
                body = payload[start:]
            else:
                self.send_response(200)
                body = payload
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            # The first answer promises everything and delivers a third, then hangs up.
            self.wfile.write(body[: len(body) // 3] if len(requests) == 1 else body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/weights.gguf"
        target = release.fetch(url, tmp_path / "weights.gguf", hashlib.sha256(payload).hexdigest())
    finally:
        server.shutdown()
    assert target.read_bytes() == payload
    assert requests == [None, f"bytes={len(payload) // 3}-"]


@pytest.mark.parametrize(
    ("extra", "then"),
    [(b"", []), (b"x", [None])],
    ids=["complete", "longer-than-the-file"],
)
def test_a_range_past_the_end_is_answered_by_checking_the_partial_file(tmp_path, extra, then):
    """A CDN answers `Range: bytes=<size>-` with a real 416. The interrupted run may have stopped
    after the last byte: then the partial file is the download. If it is longer than the file, it
    cannot be, and the next request starts from the first byte."""
    payload = bytes(range(256)) * 64
    partial = payload + extra
    requests = []

    class Cdn(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            header = self.headers.get("Range")
            requests.append(header)
            start = int(header.removeprefix("bytes=").rstrip("-")) if header else 0
            if start >= len(payload):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(payload)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = payload[start:]
            self.send_response(206 if header else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Cdn)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (tmp_path / "weights.gguf.part").write_bytes(partial)
        url = f"http://127.0.0.1:{server.server_port}/weights.gguf"
        target = release.fetch(url, tmp_path / "weights.gguf", hashlib.sha256(payload).hexdigest())
    finally:
        server.shutdown()
        server.server_close()
    assert target.read_bytes() == payload
    assert requests == [f"bytes={len(partial)}-", *then]
    assert not (tmp_path / "weights.gguf.part").exists()


class Response:
    """What `urlopen` returns: a context manager with a status, headers and a body."""

    def __init__(self, body, status=200):
        self.status = status
        self.headers = {"Content-Length": str(len(body))}
        self._stream = io.BytesIO(body)

    def read(self, size=-1):
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._stream.close()


def http_error(code, reason="Nope"):
    return urllib.error.HTTPError(
        "https://example.test/file", code, reason, email.message.Message(), io.BytesIO()
    )


def range_server(monkeypatch, payload):
    """Replace `urlopen` by a server that honours `Range: bytes=N-` and answers 416 past the end,
    as a CDN does; the list it returns holds the Range header of every request."""
    ranges = []

    def urlopen(request, timeout=None):
        header = request.get_header("Range")
        ranges.append(header)
        if header is None:
            return Response(payload)
        start = int(header.removeprefix("bytes=").rstrip("-"))
        if start >= len(payload):
            raise http_error(416, "Range Not Satisfiable")
        return Response(payload[start:], status=206)

    monkeypatch.setattr(release.urllib.request, "urlopen", urlopen)
    return ranges


def test_a_complete_partial_file_left_by_an_interrupted_run_is_accepted(tmp_path, monkeypatch):
    """The previous run was stopped after the last byte arrived and before the checksum and the
    rename. A CDN answers `Range: bytes=<size>-` with 416: the bytes on disk are checked then, and
    they are the file, so there is nothing left to download."""
    partial = tmp_path / "model.gguf.part"
    partial.write_bytes(PAYLOAD)
    (tmp_path / "model.gguf").write_bytes(b"an older copy")
    ranges = range_server(monkeypatch, PAYLOAD)
    progress = []
    target = release.fetch(
        GITHUB, tmp_path / "model.gguf", DIGEST, lambda *call: progress.append(call)
    )
    assert target.read_bytes() == PAYLOAD
    assert ranges == ["bytes=10-"]  # asked once
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
    ranges = range_server(monkeypatch, PAYLOAD)
    target = release.fetch(GITHUB, tmp_path / "model.gguf", DIGEST)
    assert target.read_bytes() == PAYLOAD
    assert ranges == [f"bytes={len(leftover)}-", None]
    assert not partial.exists()


def test_a_refused_range_never_lets_an_unverified_partial_file_through(tmp_path, monkeypatch):
    partial = tmp_path / "model.gguf.part"
    partial.write_bytes(bytes(reversed(PAYLOAD)))  # as long as the file, and not the file
    range_server(monkeypatch, PAYLOAD)
    with pytest.raises(ValueError, match=r"^model.gguf: download failed after 1 attempts \(HTTP"):
        release.fetch(GITHUB, tmp_path / "model.gguf", DIGEST, attempts=1)
    assert not (tmp_path / "model.gguf").exists()  # never renamed on the strength of a 416
    assert not partial.exists()  # dropped: the next run starts from the first byte


def test_token_goes_to_the_first_host_only_and_denials_fail_fast(tmp_path):
    payload = b"private weights"
    seen = []

    class Hub(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            if self.path == "/resolve/weights.gguf":  # the hub redirects to its CDN
                self.send_response(302)
                self.send_header("Location", "/cdn/weights.gguf")
                self.end_headers()
                return
            if self.path == "/denied.gguf":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Hub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        digest = hashlib.sha256(payload).hexdigest()
        target = release.fetch(
            f"{base}/resolve/weights.gguf", tmp_path / "w.gguf", digest, token="t"
        )
        with pytest.raises(ValueError, match="HTTP 401"):
            release.fetch(f"{base}/denied.gguf", tmp_path / "d.gguf", digest)
    finally:
        server.shutdown()
    assert target.read_bytes() == payload
    assert seen[:2] == [("/resolve/weights.gguf", "Bearer t"), ("/cdn/weights.gguf", None)]
    assert len(seen) == 3  # a denial is not retried


def test_install_and_locate(tmp_path, monkeypatch):
    at(monkeypatch, "win32", "x64", nvidia=False)
    monkeypatch.setattr(release, "library_name", lambda: "llama.dll")
    monkeypatch.setattr(release, "RUNTIMES", tmp_path / "runtimes")
    monkeypatch.delenv(release.RUNTIME_DIR_ENV, raising=False)
    with pytest.raises(ValueError, match="rizzo download"):
        release.locate()
    served = tmp_path / "served"
    served.mkdir()
    packages = {}
    for family in ("vulkan", "cpu"):
        name = f"llama-{family}.zip"
        archive_zip(served / name, {"llama.dll": family.encode(), "ggml.dll": b"g"})
        packages[("win32", "x64", family)] = [(name, release.sha256_file(served / name))]
    monkeypatch.setattr(release, "PACKAGES", packages)
    monkeypatch.setattr(release, "BASE_URL", served.as_uri())
    directory = release.install("auto")
    assert directory == release.install_dir("vulkan") and (directory / "ggml.dll").is_file()
    assert release.install("auto") == directory  # idempotent
    release.install("cpu")
    assert release.installed() == ["vulkan", "cpu"]
    assert release.locate() == release.install_dir("vulkan")
    assert release.locate("cpu") == release.install_dir("cpu")
    assert release.locate("cuda") == release.install_dir("vulkan")  # not installed: best one
    own = tmp_path / "own" / "build" / "bin"
    own.mkdir(parents=True)
    (own / "llama.dll").write_bytes(b"x")
    monkeypatch.setenv(release.RUNTIME_DIR_ENV, str(own.parent))
    assert release.locate() == own  # one level of nesting is searched
    monkeypatch.setenv(release.RUNTIME_DIR_ENV, str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="not found"):
        release.locate()


def test_locate_orders_by_pick_not_by_preference(monkeypatch, tmp_path):
    """The ordering is the point, not the filesystem: with both runtimes installed, an
    installed CUDA runtime must not shadow the Vulkan one `pick("auto")` recommends on a
    machine with no NVIDIA GPU (PREFERENCE alone would return cuda)."""
    monkeypatch.setattr(release, "supported", lambda: ["cuda", "vulkan", "cpu"])
    monkeypatch.setattr(release, "install_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(release, "pick", lambda accelerator="auto": "vulkan")
    monkeypatch.setattr(release, "find_library", lambda directory: directory / "libllama.so")
    monkeypatch.delenv(release.RUNTIME_DIR_ENV, raising=False)
    assert release.locate() == tmp_path / "vulkan"


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


def test_rosetta_is_reported_so_the_cpu_package_is_not_a_surprise(monkeypatch):
    """An Intel interpreter on Apple Silicon can only load the Intel build: `host()` sees x64 and
    the GPU stays out of reach, so `download` has something to warn about."""
    monkeypatch.setattr(release.sys, "platform", "darwin")
    monkeypatch.setattr(release, "host", lambda: ("darwin", "x64"))
    monkeypatch.setattr(
        release.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="1\n")
    )
    assert release.translated() is True


def test_a_real_intel_mac_is_not_mistaken_for_rosetta(monkeypatch):
    monkeypatch.setattr(release.sys, "platform", "darwin")
    monkeypatch.setattr(release, "host", lambda: ("darwin", "x64"))
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))
    assert release.translated() is False


def test_translation_is_a_macos_question_only(monkeypatch):
    monkeypatch.setattr(release.sys, "platform", "linux")
    monkeypatch.setattr(release, "host", lambda: ("linux", "x64"))
    assert release.translated() is False
