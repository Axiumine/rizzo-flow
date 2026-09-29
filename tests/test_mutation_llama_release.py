"""Mutation tests of `rizzo_flow.llama_release`."""

import hashlib
import http.server
import tarfile
import threading
from pathlib import Path

import pytest
from test_mutation_support import exactly

from rizzo_flow import llama_release as release

MIB = 1 << 20


def test_a_download_that_was_never_attempted_reports_no_failure(tmp_path):
    """`fetch` starting from `failure = ""`: the only way to see it is to allow no attempt."""
    with pytest.raises(
        ValueError, match=exactly("weights.gguf: download failed after 0 attempts (None)")
    ):
        release.fetch(
            "https://example.invalid/weights.gguf", tmp_path / "weights.gguf", "0" * 64, attempts=0
        )


def test_a_download_is_read_and_reported_one_mebibyte_at_a_time(tmp_path):
    """`fetch` reading blocks of 2 MiB (`2 << 20`, `1 << 21`): fewer, coarser progress reports."""
    payload = bytes(range(256)) * 10240  # 2.5 MiB
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    progress = []
    release.fetch(
        source.as_uri(),
        tmp_path / "out.bin",
        hashlib.sha256(payload).hexdigest(),
        lambda *call: progress.append(call),
    )
    total = len(payload)
    assert progress == [
        ("out.bin", MIB, total),
        ("out.bin", 2 * MIB, total),
        ("out.bin", total, total),
    ]


def test_a_tarball_is_unpacked_with_the_data_filter_whatever_the_default(tmp_path, monkeypatch):
    """`unpack` extracting with `filter=None` or without a filter: the default of the interpreter
    would decide, and before Python 3.14 that means no filter at all (and here a trusting one)."""
    trusting = staticmethod(tarfile.fully_trusted_filter)
    monkeypatch.setattr(tarfile.TarFile, "extraction_filter", trusting, raising=False)
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        link = tarfile.TarInfo("llama-bX/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"  # leaves the destination
        bundle.addfile(link)
    with pytest.raises(tarfile.LinkOutsideDestinationError):
        release.unpack(archive, tmp_path / "out")
    assert not (tmp_path / "out" / "escape").is_symlink()


def test_a_download_target_may_be_given_as_text(tmp_path):
    """`fetch` without `target = Path(target)`."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"weights")
    as_text = str(tmp_path / "cache" / "file.bin")
    digest = hashlib.sha256(b"weights").hexdigest()
    target = release.fetch(source.as_uri(), as_text, digest)  # type: ignore[arg-type]
    assert target == tmp_path / "cache" / "file.bin"
    assert target.read_bytes() == b"weights"


def test_a_server_error_keeps_the_partial_file_the_next_attempt_resumes_from(tmp_path):
    """`fetch` treating every HTTP error as a 416 when a partial file exists (`if have`): the
    bytes it could resume from would be deleted, and the retry would start from the first."""
    payload = bytes(range(256)) * 64
    requests = []

    class Overloaded(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            header = self.headers.get("Range")
            requests.append(header)
            if len(requests) == 1:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start = int(header.removeprefix("bytes=").rstrip("-")) if header else 0
            self.send_response(206 if header else 200)
            self.send_header("Content-Length", str(len(payload) - start))
            self.end_headers()
            self.wfile.write(payload[start:])

        def log_message(self, *args):
            pass

    (tmp_path / "weights.gguf.part").write_bytes(payload[:1000])
    server = http.server.HTTPServer(("127.0.0.1", 0), Overloaded)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/weights.gguf"
        target = release.fetch(url, tmp_path / "weights.gguf", hashlib.sha256(payload).hexdigest())
    finally:
        server.shutdown()
        server.server_close()
    assert target.read_bytes() == payload
    assert requests == ["bytes=1000-", "bytes=1000-"]


def test_find_library_takes_a_folder_given_as_text(tmp_path):
    """`find_library` without `directory = Path(directory)`."""
    (tmp_path / release.library_name()).touch()
    found = release.find_library(str(tmp_path))  # type: ignore[arg-type]
    assert found == tmp_path / release.library_name()


def test_find_library_tries_the_nested_folders_in_name_order_whatever_the_listing(
    tmp_path, monkeypatch
):
    """`find_library` taking the nested folders in listing order (`list` for `sorted`)."""
    for name in ("build-a", "build-b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / release.library_name()).touch()
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [self / "build-b", self / "build-a"])
    assert release.find_library(tmp_path) == tmp_path / "build-a" / release.library_name()
