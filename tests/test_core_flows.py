"""The default runtime end to end: cli or loader -> LlamaBackend -> Session -> Engine -> Response.

Every module has tests of its own, each against its own fake. Here only the native library is
replaced (the `Native` of test_llama_cpp), so what one module promises and what the next one
expects have to agree: the cells and sequences `Session.load` reserves against the ones `score`
uses, the exceptions `Session` raises against the ones `api` maps, the folder `install` makes
against the one `locate` finds.
"""

import hashlib
import io
import json
import sys
import tarfile
import threading
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_llama_cpp import Native

from rizzo_flow import backend_llama, cli, llama_release, runtime
from rizzo_flow.api import create_app
from rizzo_flow.backend_llama import LlamaBackend
from rizzo_flow.engine import Engine
from rizzo_flow.llama_cpp import Library
from rizzo_flow.prompts import compile_request
from rizzo_flow.schema import Request

REAL_PLATFORM = sys.platform  # `Native.install` pretends to be Linux; a web server should not
TEMPLATE = b"{% for m in messages %}{{ m.content }}\n{% endfor %}ASSISTANT:"
NOUL = {  # the same kind of question in the format of the hosted API
    "state": "s",
    "model": "rizzo-latest",
    "questions": {"q": {"type": "noul", "instructions": "i"}},
}
REQUEST = {
    "state": {"message": "Cannot log in after a password reset", "account": {"active": True}},
    "questions": {
        "access": {"type": "boolean", "instructions": "Does the customer need login help?"},
        "queue": {
            "type": "choice",
            "instructions": "Which team handles this?",
            "options": [
                {"id": "access", "description": "Login, password or account access."},
                {"id": "billing", "description": "Invoices, charges or payments."},
                {"id": "sales", "description": "Purchases of new products."},
            ],
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is it?",
            "levels": ["No deadline.", "Work is slowed down.", "Work is blocked today."],
        },
        "hours": {
            "type": "numeric",
            "instructions": "How many hours are left?",
            "unit": "hours",
            "anchors": [
                {"value": 1, "description": "One hour."},
                {"value": 8, "description": "Eight hours."},
            ],
        },
    },
}


@pytest.fixture
def stack(tmp_path, monkeypatch):
    """A runtime folder, a GGUF file and a fake libllama behind ctypes, found the way `rizzo`
    finds them: through RIZZO_LLAMA_DIR."""
    monkeypatch.setattr(Library, "_loaded", {})
    native = Native().install(tmp_path, monkeypatch)
    native.metadata = {
        b"general.architecture": b"spark2_5",
        b"spark2_5.embedding_length": b"2560",
        b"general.file_type": b"7",
    }
    native.template = TEMPLATE
    native.pad, native.eos = 0, 1
    monkeypatch.setenv("RIZZO_LLAMA_DIR", str(native.directory))
    return native


@pytest.fixture
def run(monkeypatch, capsys):
    """`rizzo <argv>` in this process; the JSON it prints."""

    def command(*argv):
        monkeypatch.setattr(sys, "argv", ["rizzo", *map(str, argv)])
        cli.main()
        out = capsys.readouterr().out
        return json.loads(out) if out else None  # --output: nothing on the console

    return command


def logit(tokens, slot):
    """What the fake model answers for a slot after `tokens`: exact in float32."""
    digest = hashlib.sha256(repr((tokens, slot)).encode()).digest()
    return float(int.from_bytes(digest[:2], "big") % 2001 - 1000)


class Cache:
    """The unified KV cache of libllama, in Python. The logits of a row depend on the tokens of
    its sequence and on nothing else: an answer read from the wrong row, or from a sequence that
    was copied, cut or overwritten wrongly, cannot come out right. It also holds the caller to
    what `Session.load` reserved: sequences, output rows per call, and cells (a cell is shared by
    the sequences that were branched from it)."""

    def __init__(self, native):
        self.native = native
        self.tokens = {}  # sequence -> the tokens it holds
        self.cells = {}  # sequence -> the cells it holds
        self.owners = {}  # cell -> the sequences that hold it
        self.made = 0  # cells ever taken: a number is never used twice
        native.functions["llama_memory_clear"].implementation = self.clear
        native.functions["llama_memory_seq_cp"].implementation = self.copy
        native.functions["llama_memory_seq_rm"].implementation = self.remove
        native.functions["llama_decode"].implementation = self.decode

    def clear(self, memory, data):
        self.tokens, self.cells, self.owners = {}, {}, {}

    def copy(self, memory, source, target, start, end):
        assert not self.cells.get(target), "a branch starts from an empty sequence"
        self.tokens[target] = list(self.tokens.get(source, []))
        self.cells[target] = list(self.cells.get(source, []))
        for cell in self.cells[target]:
            self.owners[cell].add(target)

    def remove(self, memory, sequence, start, end):
        for cell in self.cells.pop(sequence, []):
            self.owners[cell].discard(sequence)
            if not self.owners[cell]:
                del self.owners[cell]
        self.tokens.pop(sequence, None)
        return True

    def decode(self, context, batch):
        reserved = self.native.context_params
        rows = {}
        for i in range(batch.n_tokens):
            sequence, position = batch.seq_id[i][0], batch.pos[i]
            assert sequence < reserved.n_seq_max, "more sequences than were reserved"
            held = self.tokens.setdefault(sequence, [])
            assert position == len(held), "positions continue a sequence without gaps"
            held.append(batch.token[i])
            self.cells.setdefault(sequence, []).append(self.made)
            self.owners[self.made] = {sequence}
            self.made += 1
            if batch.logits[i]:
                rows[i] = [logit(held, slot) for slot in range(self.native.ROWS)]
        assert len(self.owners) <= reserved.n_ctx, "the reserved cells ran out: llama_decode = 1"
        assert len(rows) <= reserved.n_outputs_max, "more output rows than were reserved"
        self.native.rows = rows
        return 0


# limit: tokens per llama_decode call. 8 slices every prompt, 300 splits some microbatches, and
# the default leaves the split to the number of questions.
@pytest.mark.parametrize("limit", [8, 300, backend_llama.N_BATCH])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("mode", ["shared", "direct"])
def test_the_engine_answers_from_the_logits_of_the_llama_backend(
    stack, monkeypatch, mode, batch_size, limit
):
    Cache(stack)
    monkeypatch.setattr(backend_llama, "N_BATCH", limit)
    with closing(LlamaBackend.load(stack.gguf)) as probe:
        prefix, jobs = compile_request(probe.tokenizer, Request.model_validate(REQUEST), 8192)
    ctx = max(len(job.tokens) for job in jobs)  # the least the engine accepts for this request
    with closing(LlamaBackend.load(stack.gguf, ctx=ctx, batch_size=batch_size)) as backend:
        response = Engine(backend, ctx=ctx).decide({**REQUEST, "mode": mode})
    assert len(jobs) == 4
    for job in jobs:
        answer = response["answers"][job.id]
        assert list(answer["option_logits"].values()) == [logit(job.tokens, s) for s in job.slots]
        assert answer["input_tokens"] == len(job.tokens)
    assert response["timing"]["shared_prefix_tokens"] == (len(prefix) if mode == "shared" else 0)
    assert response["timing"]["generated_tokens"] == 0
    assert not stack.model_alive


def test_devices_reports_the_runtime_the_session_opens(stack, run, monkeypatch):
    monkeypatch.setattr(llama_release, "host", lambda: ("linux", "x64"))
    monkeypatch.setattr(runtime, "describe", lambda: {"installed": False})  # MLX is not under test
    section = run("devices")["llama.cpp"]
    assert "error" not in section
    assert Path(section["directory"]).resolve() == stack.directory.resolve()
    assert [device["name"] for device in section["devices"]] == ["CPU", "CUDA0"]
    assert section["auto_selects"] == "CUDA0"


def test_a_runtime_laid_out_by_install_is_the_one_a_session_opens(stack, tmp_path, monkeypatch):
    served = tmp_path / "served"
    served.mkdir()
    archive = served / "llama-b11081-bin-ubuntu-x64.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("libggml-base.so", "libggml.so", "libllama.so"):
            tar.addfile(tarfile.TarInfo(f"llama-b11081/{name}"), io.BytesIO(b""))
    monkeypatch.delenv("RIZZO_LLAMA_DIR")
    monkeypatch.setattr(llama_release, "RUNTIMES", tmp_path / "runtimes")
    monkeypatch.setattr(llama_release, "BASE_URL", served.as_uri())
    monkeypatch.setattr(llama_release, "host", lambda: ("linux", "x64"))
    monkeypatch.setattr(
        llama_release,
        "PACKAGES",
        {("linux", "x64", "cpu"): [(archive.name, llama_release.sha256_file(archive))]},
    )
    installed = llama_release.install("cpu")
    LlamaBackend.load(stack.gguf, device="cpu").close()  # locate() finds what install() made
    assert {path.parent for path, _ in stack.loads} == {installed.resolve()}


def test_a_calibration_fitted_on_this_stack_applies_to_it_and_to_no_other_run(
    stack, tmp_path, run, capsys
):
    request = tmp_path / "request.json"
    request.write_text(json.dumps(REQUEST), encoding="utf-8")
    decided = run("decide", request, "--model", stack.gguf)
    fingerprint = decided["model"]["fingerprint"]
    logits = list(decided["answers"]["queue"]["option_logits"].values())
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        "".join(
            json.dumps({"type": "choice", "logits": logits, "label_index": i % 3}) + "\n"
            for i in range(12)
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.json"
    run("calibrate", rows, "--fingerprint", fingerprint, "--output", calibration)
    command = ("decide", request, "--model", stack.gguf, "--calibration", calibration)
    again = run(*command)
    assert again["calibration"]["fingerprint"] == fingerprint
    assert (
        again["answers"]["queue"]["temperature"] == again["calibration"]["temperatures"]["choice"]
    )
    with pytest.raises(SystemExit):  # a quantized KV cache moves the logits: another fingerprint
        run(*command, "--kv-type", "q8_0")
    assert "different model/runtime/prompt configuration" in capsys.readouterr().err


def test_the_api_serves_both_interfaces_from_the_inference_thread(stack, monkeypatch):
    threads = set()
    decode = stack.functions["llama_decode"].implementation

    def watched(context, batch):
        threads.add(threading.current_thread().name)
        return decode(context, batch)

    stack.functions["llama_decode"].implementation = watched
    with closing(LlamaBackend.load(stack.gguf)) as backend:
        monkeypatch.setattr(sys, "platform", REAL_PLATFORM)
        with TestClient(create_app(Engine(backend))) as client:
            native = client.post("/v1/decisions", json=REQUEST)
            compat = client.post("/v1/systemone", json=NOUL)
    assert (native.status_code, compat.status_code) == (200, 200)
    assert compat.json()["model"] == "rizzo-spark-x2.5-4b-q8_0"  # named after the loaded file
    assert {name.rsplit("_", 1)[0] for name in threads} == {"rizzo-inference"}
    assert len(threads) == 1


@pytest.mark.parametrize(
    ("status", "reason"), [(1, "the context is full"), (-3, "compute error")], ids=["full", "error"]
)
def test_a_llama_decode_that_fails_is_a_503_and_the_next_request_is_served(
    stack, monkeypatch, status, reason
):
    # The session says ValueError, the engine turns it into BackendError, the API into a 503: a
    # client that sent a good request is not told that it sent a bad one.
    with closing(LlamaBackend.load(stack.gguf)) as backend:
        monkeypatch.setattr(sys, "platform", REAL_PLATFORM)
        with TestClient(create_app(Engine(backend))) as client:
            stack.decode_status = status
            failed = [
                client.post("/v1/decisions", json=REQUEST),
                client.post("/v1/systemone", json=NOUL),
            ]
            stack.decode_status = 0
            served = [
                client.post("/v1/decisions", json=REQUEST),
                client.post("/v1/systemone", json=NOUL),
            ]
    detail = {"detail": f"llama_decode returned {status}: {reason}"}
    assert [(response.status_code, response.json()) for response in failed] == [(503, detail)] * 2
    assert [response.status_code for response in served] == [200, 200]
