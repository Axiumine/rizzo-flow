"""skills/rizzo-flow/scripts/rizzo_client.py: it speaks http(s) only, and the API key goes to the
server it was set for, and to no other."""

import http.server
import importlib.util
import json
import threading
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "skills" / "rizzo-flow" / "scripts" / "rizzo_client.py"
)
KEY = "rzk-not-a-real-key"
ANSWER = {"status": "ok"}
BODY = {"state": "x"}

# What a test server records about each request: method, path, Authorization header, body.
Request = tuple[str, str, str | None, bytes]


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    """Local servers are reached directly: `urlopen` would send them to the proxy of the
    environment or of the system settings (WinINET, macOS) even for 127.0.0.1."""
    handler = urllib.request.ProxyHandler({})
    monkeypatch.setattr(urllib.request, "_opener", urllib.request.build_opener(handler))


@pytest.fixture(scope="module")
def client() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rizzo_client", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def serve(redirect_to: str | None = None) -> Iterator[tuple[str, list[Request]]]:
    """A server on a free local port: yields its base URL and the requests it gets. It answers
    with ANSWER, or with a 302 to the same path under `redirect_to` (another server, so another
    host as far as a client is concerned)."""
    seen: list[Request] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def handle_any(self) -> None:
            size = int(self.headers.get("Content-Length") or 0)
            seen.append(
                (self.command, self.path, self.headers.get("Authorization"), self.rfile.read(size))
            )
            if redirect_to is None:
                payload = json.dumps(ANSWER).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(302)
                self.send_header("Location", redirect_to + self.path)
                self.send_header("Content-Length", "0")
                self.end_headers()

        do_GET = do_POST = handle_any

        def log_message(self, *args: object, **kwargs: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # A short poll interval: shutdown() waits for the next poll, half a second by default.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_the_key_goes_out_as_a_bearer_token(client):
    with serve() as (base, seen):
        assert client.call(f"{base}/v1/decisions", BODY, KEY) == ANSWER
    assert seen == [("POST", "/v1/decisions", f"Bearer {KEY}", json.dumps(BODY).encode())]


def test_without_a_key_no_authorization_header_goes_out(client):
    with serve() as (base, seen):
        assert client.call(f"{base}/health") == ANSWER
    assert seen == [("GET", "/health", None, b"")]


@pytest.mark.parametrize("body", [None, BODY], ids=["get", "post"])
def test_a_redirect_to_another_host_does_not_carry_the_key_along(client, body):
    """urllib repeats the headers of a request on a redirect, the Authorization header included,
    unless it was added as an unredirected one. A POST that is redirected turns into a GET."""
    with serve() as (elsewhere, landed), serve(redirect_to=elsewhere) as (base, seen):
        assert client.call(f"{base}/v1/decisions", body, KEY) == ANSWER
    assert [request[2] for request in seen] == [f"Bearer {KEY}"]  # the server it was set for
    assert landed == [("GET", "/v1/decisions", None, b"")]  # the redirect was followed, keyless


def test_a_file_url_is_refused_and_not_read(client, tmp_path):
    """$RIZZO_URL may name anything: urllib opens a file:// URL as readily as an http one, and the
    client would print the content of the file as if it were the answer of the server."""
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(ANSWER), encoding="utf-8")
    with pytest.raises(SystemExit, match="http:// or https://"):
        client.call(answer.as_uri())


@pytest.mark.parametrize("url", ["ftp://127.0.0.1:1/health", "localhost:8017/health"])
def test_any_other_url_than_http_or_https_is_refused_with_the_reason(client, url):
    """Not "Rizzo Flow is not reachable": the URL itself is wrong, and starting the server would
    not help."""
    with pytest.raises(SystemExit, match="http:// or https://"):
        client.call(url, key=KEY)
