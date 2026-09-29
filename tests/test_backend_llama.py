"""llama.cpp backend logic against a recording fake session: no library, no weights."""

import hashlib
import re
import sys

import jinja2
import pytest
from fastapi.testclient import TestClient
from test_llama_cpp import Native
from test_service import CharacterTokenizer

from rizzo_flow import backend_llama, config, llama_release, loader
from rizzo_flow.api import create_app
from rizzo_flow.backend_llama import LlamaBackend, LlamaTokenizer
from rizzo_flow.engine import Engine
from rizzo_flow.llama_cpp import Device, Library, choose_device
from rizzo_flow.prompts import Compiled


class FakeSession:
    """Records every call; the logit of a slot is `position of the row + slot / 1000`."""

    device = None
    idle_free = None
    pad_token = None
    eos_token = 2

    def __init__(self):
        self.calls = []
        self.cells = {}  # sequence -> positions held
        self.last = None

    def clear(self):
        self.calls.append(("clear",))
        self.cells = {}

    def branch(self, source, target):
        self.calls.append(("branch", source, target))
        self.cells[target] = list(self.cells[source])

    def drop(self, sequence):
        self.calls.append(("drop", sequence))
        del self.cells[sequence]

    def decode(self, tokens, positions, sequences, outputs=()):
        tokens, positions, sequences = list(tokens), list(positions), list(sequences)
        assert len(tokens) == len(positions) == len(sequences) <= backend_llama.N_BATCH
        for position, sequence in zip(positions, sequences, strict=True):
            held = self.cells.setdefault(sequence, [])
            assert position == len(held), "positions must continue each sequence without gaps"
            held.append(position)
        self.calls.append(("decode", tokens, positions, sequences, list(outputs)))
        self.last = (positions, list(outputs))

    def logits(self, index, slots):
        positions, outputs = self.last
        assert index in outputs, "logits were not requested at this row"
        return [positions[index] + slot / 1000 for slot in slots]

    def synchronize(self):
        pass

    def close(self):
        self.calls.append(("close",))

    def free_bytes(self):
        return None

    def tokenize(self, text, add_special=False):
        return [ord(c) for c in text]


def backend(batch_size=4):
    return LlamaBackend(FakeSession(), None, {"fingerprint": "test"}, batch_size=batch_size)


def job(name, prefix, suffix, slots=(65, 66)):
    return Compiled(name, prefix + suffix, list(slots), "hash")


def test_shared_prefills_once_and_packs_suffixes_without_padding():
    prefix = list(range(100, 110))
    jobs = [
        job("long", prefix, [1, 2, 3, 4]),
        job("short", prefix, [5, 6]),
        job("mid", prefix, [7, 8, 9]),
    ]
    engine = backend(batch_size=2)
    logits, timing = engine.score(prefix, jobs, "shared")
    calls = engine.session.calls
    assert calls[0] == ("clear",)
    assert calls[1] == ("decode", prefix, list(range(10)), [0] * 10, [])
    # Shortest first, two per microbatch, laid end to end on sequences 1 and 2.
    assert calls[2:4] == [("branch", 0, 1), ("branch", 0, 2)]
    assert calls[4] == ("decode", [5, 6, 7, 8, 9], [10, 11, 10, 11, 12], [1, 1, 2, 2, 2], [1, 4])
    assert calls[5:7] == [("drop", 1), ("drop", 2)]
    assert calls[7:] == [
        ("branch", 0, 1),
        ("decode", [1, 2, 3, 4], [10, 11, 12, 13], [1] * 4, [3]),
        ("drop", 1),
    ]
    # Each answer is read at the last position of its own question.
    assert logits == {"short": [11.065, 11.066], "mid": [12.065, 12.066], "long": [13.065, 13.066]}
    assert engine.session.cells == {0: list(range(10))}  # the prefix survives every microbatch
    assert timing["batches"] == 2 and timing["shared_prefix_tokens"] == 10
    assert timing["evaluated_tokens_including_padding"] == 10 + 5 + 4
    assert timing["logical_input_tokens"] == 14 + 12 + 13
    assert timing["generated_tokens"] == 0 and "peak_device_bytes" not in timing


def test_direct_recomputes_every_question_from_an_empty_cache():
    prefix = [100, 101]
    jobs = [job("a", prefix, [1]), job("b", prefix, [2, 3])]
    engine = backend()
    logits, timing = engine.score(prefix, jobs, "direct")
    assert [c[0] for c in engine.session.calls] == ["clear", "decode", "clear", "decode"]
    assert engine.session.calls[3] == ("decode", [100, 101, 2, 3], [0, 1, 2, 3], [0] * 4, [3])
    assert logits["a"][0] == pytest.approx(2.065) and logits["b"][0] == pytest.approx(3.065)
    assert timing["shared_prefix_tokens"] == 0 and timing["prefill_seconds"] == 0.0


def test_a_single_question_takes_one_pass_even_in_shared_mode():
    engine = backend()
    logits, timing = engine.score([100, 101], [job("only", [100, 101], [1, 2])], "shared")
    assert engine.session.calls == [
        ("clear",),
        ("decode", [100, 101, 1, 2], [0, 1, 2, 3], [0] * 4, [3]),
    ]
    assert logits["only"][0] == pytest.approx(3.065)
    assert timing["shared_prefix_tokens"] == 0 and timing["batches"] == 1


def test_inputs_longer_than_one_call_are_fed_in_slices(monkeypatch):
    monkeypatch.setattr(backend_llama, "N_BATCH", 4)
    prefix = list(range(10))
    jobs = [job("huge", prefix, list(range(50, 59))), job("tiny", prefix, [1])]
    engine = backend()
    logits, _ = engine.score(prefix, jobs, "shared")
    decodes = [c for c in engine.session.calls if c[0] == "decode"]
    assert [len(c[1]) for c in decodes] == [4, 4, 2, 1, 4, 4, 1]  # prefix, tiny, huge in slices
    assert [c[4] for c in decodes] == [[], [], [], [0], [], [], [0]]  # logits at the very end only
    assert logits["huge"][0] == pytest.approx(18.065)
    # A microbatch never exceeds one call: 3 + 3 tokens do not fit in 4.
    engine = backend()
    engine.score(prefix, [job("x", prefix, [1, 2, 3]), job("y", prefix, [4, 5, 6])], "shared")
    assert [len(c[1]) for c in engine.session.calls if c[0] == "decode"] == [4, 4, 2, 3, 3]


def test_score_rejects_bad_input():
    engine = backend()
    with pytest.raises(ValueError, match="mode"):
        engine.score([1], [job("a", [1], [2])], "batched")
    with pytest.raises(ValueError, match="No decisions"):
        engine.score([1], [], "shared")
    with pytest.raises(ValueError, match="prefix"):
        engine.score([1, 2], [job("a", [1, 3], [4])], "shared")
    with pytest.raises(ValueError, match="prefix"):
        engine.score([1, 2], [job("a", [1, 2], [])], "shared")
    with pytest.raises(ValueError, match="batch_size"):
        backend(batch_size=17)


def test_tokenizer_renders_like_transformers_and_encodes_with_the_gguf():
    template = (
        "{%- if not messages %}{{ raise_exception('No messages provided.') }}{%- endif %}\n"
        "{%- for m in messages %}\n"
        "    {{- '<' + m.role + '>' + m.content }}\n"
        "{%- endfor %}\n"
        "{%- if add_generation_prompt %}{{ '<bot>' }}{%- if not enable_thinking %}{{ '</think>' }}{%- endif %}{%- endif %}"
    )
    tokenizer = LlamaTokenizer(FakeSession(), template)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert text == "<system>S<user>U<bot></think>"  # block lines and indentation leave no trace
    assert tokenizer.encode("AB", add_special_tokens=False) == [65, 66]
    assert tokenizer.pad_token_id is None and tokenizer.eos_token_id == 2
    with pytest.raises(ValueError, match="No messages"):
        tokenizer.apply_chat_template([], tokenize=False)
    with pytest.raises(ValueError, match="encode"):
        tokenizer.apply_chat_template([{"role": "user", "content": "U"}], tokenize=True)


@pytest.mark.parametrize(
    "template",
    [
        "{{ ''.__class__.__mro__ }}",  # a way out of the template into the interpreter
        "{% set _ = messages.append(1) %}",  # the messages belong to the caller
    ],
)
def test_a_template_downloaded_with_the_weights_runs_in_a_sandbox(template):
    tokenizer = LlamaTokenizer(FakeSession(), template)
    with pytest.raises(ValueError, match=r"^The GGUF chat template cannot run: ") as raised:
        tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])
    assert isinstance(raised.value.__cause__, jinja2.exceptions.SecurityError)


def test_a_template_that_needs_what_the_environment_lacks_cannot_run_either():
    template = "{{ strftime_now('%Y') }}"  # a helper of transformers that this renderer has not
    tokenizer = LlamaTokenizer(FakeSession(), template)
    with pytest.raises(ValueError, match=r"^The GGUF chat template cannot run: .*strftime_now"):
        tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize(
    ("template", "error"),
    [
        ("{{ 1 // 0 }}", ZeroDivisionError),  # jinja2 leaves the division to Python
        ("{{ 1 + 'a' }}", TypeError),
        ("{% macro f() %}{{ f() }}{% endmacro %}{{ f() }}", RecursionError),
        ("{{ 'abc'.index('z') }}", ValueError),  # Python's, not the template's own refusal
    ],
)
def test_a_template_that_fails_in_python_cannot_run_and_says_which_exception(template, error):
    tokenizer = LlamaTokenizer(FakeSession(), template)
    message = rf"^The GGUF chat template cannot run: {error.__name__}: "
    with pytest.raises(ValueError, match=message) as raised:
        tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])
    assert type(raised.value) is ValueError  # wrapped: not the class of the template's refusal
    assert type(raised.value.__cause__) is error


def test_a_template_the_parser_cannot_finish_does_not_compile_and_says_so():
    depth = sys.getrecursionlimit()  # a level of parentheses takes several frames of the parser
    template = "{{ " + "(" * depth + "1" + ")" * depth + " }}"
    message = r"^The GGUF chat template does not compile: RecursionError: "
    with pytest.raises(ValueError, match=message) as raised:
        LlamaTokenizer(FakeSession(), template)
    assert type(raised.value.__cause__) is RecursionError


def test_a_template_that_refuses_the_conversation_keeps_its_own_message():
    tokenizer = LlamaTokenizer(FakeSession(), "{{ raise_exception('no system role') }}")
    with pytest.raises(ValueError, match=r"^no system role$") as raised:
        tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])
    assert raised.value.__cause__ is None  # not wrapped: it was ours to begin with


CPU = Device(1, "CPU", "Some CPU", "cpu", "CPU", 64 << 30)
IGPU = Device(2, "Vulkan0", "Intel(R) UHD Graphics", "igpu", "Vulkan", 32 << 30)
RADEON = Device(3, "Vulkan1", "AMD Radeon RX 7900 XTX", "gpu", "Vulkan", 24 << 30)
SMALL = Device(4, "Vulkan2", "AMD Radeon RX 6600", "gpu", "Vulkan", 8 << 30)
GEFORCE = Device(5, "CUDA0", "NVIDIA GeForce RTX 5060 Ti", "gpu", "CUDA", 16 << 30)
METAL = Device(6, "MTL0", "Apple M4", "gpu", "MTL", 16 << 30)  # Apple registers itself as MTL


def test_device_choice():
    machine = [CPU, IGPU, SMALL, RADEON]
    assert choose_device(machine, "auto") is RADEON  # discrete before integrated, then memory
    assert choose_device(machine, "gpu") is RADEON
    assert choose_device(machine, "cpu") is None
    assert choose_device([CPU, IGPU], "auto") is IGPU
    assert choose_device([CPU], "auto") is None  # no GPU: the CPU, without an error
    assert choose_device(machine, "vulkan") is RADEON
    assert choose_device([CPU, GEFORCE], "cuda") is GEFORCE
    with pytest.raises(ValueError, match="no such GPU"):
        choose_device([CPU], "gpu")  # an explicit request is never downgraded
    with pytest.raises(ValueError, match="rizzo download"):
        choose_device(machine, "cuda")


def test_metal_is_found_under_its_registry_name():
    # Apple's backend calls itself MTL: `--device metal` must still find it (issue #9).
    assert choose_device([CPU, METAL], "metal") is METAL
    assert choose_device([CPU, METAL], "Metal") is METAL
    assert choose_device([CPU, METAL], "mtl") is METAL
    assert choose_device([CPU, METAL], "MTL0") is METAL  # one named device, as before
    assert choose_device([CPU, METAL], "auto") is METAL
    assert choose_device([CPU, METAL], "m4") is METAL  # description still matches
    with pytest.raises(ValueError, match="no such GPU"):
        choose_device([CPU, METAL], "cuda")  # an alias never widens another family
    assert choose_device([CPU, IGPU, RADEON], "vulkan") is RADEON  # numbered names unaffected


def test_loader_rejects_options_of_the_other_backend(tmp_path):
    with pytest.raises(ValueError, match="--quant"):
        loader.load_backend("llama", bits=8)
    with pytest.raises(ValueError, match="--bits"):
        loader.load_backend("mlx", quant="q8_0")
    with pytest.raises(ValueError, match="--backend mlx"):
        loader.load_backend("llama", device="mlx")
    with pytest.raises(ValueError, match="llama backend"):
        loader.load_backend("mlx", device="vulkan")
    with pytest.raises(ValueError, match="--kv-type"):
        loader.load_backend("mlx", kv_type="q8_0")
    with pytest.raises(ValueError, match=".gguf"):
        loader.load_backend("llama", model=tmp_path)
    with pytest.raises(ValueError, match="rizzo download"):
        loader.load_backend("llama", model=tmp_path / "missing.gguf")
    with pytest.raises(ValueError, match="one of"):
        loader.load_backend("onnx")
    with pytest.raises(ValueError, match="--weights"):
        loader.load_backend("mlx", model=tmp_path, weights="flow")
    with pytest.raises(ValueError, match="--weights"):
        loader.load_backend("llama", model=tmp_path / "mine.gguf", weights="base")
    with pytest.raises(ValueError, match="No flow GGUF"):
        config.gguf_spec("4b", "q2_k")


def test_fine_tuned_weights_are_the_default_and_pinned():
    spec = config.gguf_spec()
    assert (spec.size, spec.quant, spec.variant) == ("4b", "q8_0", "flow")
    assert spec.repo == "rizzoaiacademy/rizzo-flow" and len(spec.revision) == 40
    assert config.gguf_spec("1.7b").repo == "rizzoaiacademy/rizzo-flow-1.7b"
    assert config.gguf_spec("4b", "q4_k_m", "base").repo == "XHToken/Spark-X2.5-4B-GGUF"
    assert len({spec.sha256 for spec in config.GGUF.values()}) == len(config.GGUF)
    assert len({spec.path for spec in config.GGUF.values()}) == len(config.GGUF)
    # MLX: the merged checkpoint sits next to the GGUF files; base keeps XHToken's directory.
    assert config.checkpoint_path() == config.FLOW_CHECKPOINTS["4b"].path == spec.path.parent
    assert config.checkpoint_path("1.7b", "base") == config.MODELS["1.7b"].path
    for flow in config.FLOW_CHECKPOINTS.values():
        assert flow.weights and all(len(sha) == 64 for sha in flow.weights.values())
        assert not set(flow.weights) & set(config.CHECKPOINT_FILES)


def test_hf_token_comes_from_the_environment_then_dotenv_then_login(tmp_path, monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert config.hf_token() is None
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "token").write_text("hf_saved\n", encoding="utf-8")
    assert config.hf_token() == "hf_saved"
    (tmp_path / ".env").write_text('# local\nOTHER=1\nHF_TOKEN="hf_dotenv"\n', encoding="utf-8")
    assert config.hf_token() == "hf_dotenv"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert config.hf_token() == "hf_env"


def test_close_releases_the_session_before_the_interpreter_goes_away():
    """The Metal device is torn down by a static destructor at exit and aborts if a buffer is
    still registered, so the context has to be freed while Python is still running."""
    engine = backend()
    engine.close()
    assert ("close",) in engine.session.calls


class FailingSession(FakeSession):
    """A runtime whose decode fails the way `Session.decode` does, while `status` is not 0."""

    status = 0

    def decode(self, tokens, positions, sequences, outputs=()):
        if self.status:
            reason = "the context is full" if self.status == 1 else "compute error"
            raise ValueError(f"llama_decode returned {self.status}: {reason}")
        super().decode(tokens, positions, sequences, outputs)


@pytest.mark.parametrize(
    ("status", "reason"), [(1, "the context is full"), (-3, "compute error")], ids=["full", "error"]
)
def test_a_llama_decode_that_fails_is_a_503_and_the_next_request_is_served(status, reason):
    # The session says ValueError, the engine turns it into BackendError, the API into a 503: a
    # client that sent a good request is not told that it sent a bad one.
    session = FailingSession()
    model = LlamaBackend(session, CharacterTokenizer(), {"fingerprint": "test"})
    native = {"state": "ticket", "questions": {"q": {"type": "boolean", "instructions": "Urgent?"}}}
    wire = {
        "state": "ticket",
        "model": "rizzo-latest",
        "questions": {"q": {"type": "noul", "instructions": "Urgent?"}},
    }
    with TestClient(create_app(Engine(model), api_key="")) as client:
        session.status = status
        failed = [
            client.post("/v1/decisions", json=native),
            client.post("/v1/systemone", json=wire),
        ]
        session.status = 0
        served = [
            client.post("/v1/decisions", json=native),
            client.post("/v1/systemone", json=wire),
        ]
    detail = {"detail": f"llama_decode returned {status}: {reason}"}
    assert [(response.status_code, response.json()) for response in failed] == [(503, detail)] * 2
    assert [response.status_code for response in served] == [200, 200]


# --- load: the order of its steps ------------------------------------------------------------

WEIGHTS = b"GGUF stand-in for the weights"
TEMPLATE = "{{ messages[0].content }}!"
META = {
    "general.architecture": "spark2_5",
    "spark2_5.embedding_length": "2560",
    "general.file_type": "7",
}


class LoadedSession:
    """What `Session.load` hands back for a GGUF file: its metadata, and how often it was closed."""

    pad_token = 7
    eos_token = 2
    device = None
    n_ctx = 8192 + 2048

    def __init__(self, template):
        self.template = template
        self.closed = 0

    def meta(self, key):
        return META.get(key)

    def chat_template(self):
        return self.template

    def tokenize(self, text, add_special=False):
        return [ord(character) for character in text]

    def close(self):
        self.closed += 1


class Runtime:
    """Stands in for `llama_cpp.Session`, for `llama_cpp.Library` and for the lookup of the runtime
    directory, and notes the order in which they are used."""

    def __init__(self, directory):
        self.directory = directory
        self.template = TEMPLATE
        self.located = []  # devices `llama_release.locate` was asked about
        self.opened = []  # directories `Library.open` was asked for
        self.calls = []  # (gguf, options) of every `Session.load`
        self.events = []  # "locate", "open", "hash", "load", in the order they happened
        self.sessions = []

    def locate(self, family=None):
        self.located.append(family)
        self.events.append("locate")
        return self.directory

    def open(self, directory):
        self.opened.append(directory)
        self.events.append("open")

    def load(self, gguf, **options):
        self.calls.append((gguf, options))
        self.events.append("load")
        self.sessions.append(LoadedSession(self.template))
        return self.sessions[-1]


@pytest.fixture
def native(monkeypatch, tmp_path):
    runtime = Runtime(tmp_path / "runtime")
    monkeypatch.setattr(backend_llama, "Session", runtime)
    monkeypatch.setattr(backend_llama, "Library", runtime)
    monkeypatch.setattr(llama_release, "locate", runtime.locate)
    return runtime


@pytest.fixture
def gguf(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(WEIGHTS)
    return path


@pytest.fixture
def no_hashing(monkeypatch):
    """Reading the weights to hash them fails the test."""

    def hashed(*args, **kwargs):
        raise AssertionError("the weights were read")

    monkeypatch.setattr(hashlib, "file_digest", hashed)


@pytest.fixture
def nothing_installed(tmp_path, monkeypatch):
    """The lookup of a runtime is the real one, on a Linux x86-64 box without NVIDIA that has no
    runtime installed."""
    monkeypatch.setattr(llama_release, "RUNTIMES", tmp_path / "runtimes")
    monkeypatch.setattr(llama_release, "host", lambda: ("linux", "x64"))
    monkeypatch.setattr(llama_release, "nvidia_driver", lambda: False)
    monkeypatch.delenv(llama_release.RUNTIME_DIR_ENV, raising=False)


WRONG_SIZES = [{"batch_size": 17}, {"batch_size": 0}, {"prefill_chunk": 2049}, {"prefill_chunk": 0}]


@pytest.mark.parametrize("options", WRONG_SIZES)
def test_wrong_batch_sizes_are_refused_before_the_weights_are_hashed_or_loaded(
    native, gguf, no_hashing, options
):
    with pytest.raises(ValueError, match=r"^batch_size must be 1–16 and prefill_chunk 1–2048$"):
        LlamaBackend.load(gguf, **options)
    assert native.located == []
    assert native.opened == []
    assert native.calls == []
    assert native.sessions == []


@pytest.mark.parametrize("options", WRONG_SIZES)
def test_wrong_batch_sizes_do_not_leave_a_loaded_model_behind(native, gguf, options):
    with pytest.raises(ValueError, match="batch_size must be"):
        LlamaBackend.load(gguf, **options)
    assert all(session.closed for session in native.sessions)


def test_a_backend_that_cannot_be_constructed_releases_the_session(native, gguf):
    class Unwilling(LlamaBackend):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no backend today")

    with pytest.raises(RuntimeError, match=r"^no backend today$"):
        Unwilling.load(gguf)
    assert [session.closed for session in native.sessions] == [1]


def test_a_broken_chat_template_is_reported_and_the_session_released(native, gguf):
    native.template = "{% if %}"
    with pytest.raises(ValueError, match=r"^The GGUF chat template does not compile: ") as raised:
        LlamaBackend.load(gguf)
    assert isinstance(raised.value.__cause__, jinja2.TemplateSyntaxError)
    assert [session.closed for session in native.sessions] == [1]


def test_the_runtime_is_found_and_opened_before_the_weights_are_hashed(native, gguf, monkeypatch):
    sha256_file = llama_release.sha256_file

    def hashed(path):
        native.events.append("hash")
        return sha256_file(path)

    monkeypatch.setattr(llama_release, "sha256_file", hashed)
    LlamaBackend.load(gguf, device="cuda")
    assert native.events == ["locate", "open", "hash", "load"]
    native.events.clear()
    LlamaBackend.load(gguf, runtime_dir=native.directory)  # a directory that is given: no lookup
    assert native.events == ["open", "hash", "load"]


def test_load_looks_for_the_runtime_of_the_device_unless_a_directory_is_given(
    native, gguf, tmp_path
):
    LlamaBackend.load(gguf, device="cuda")
    assert native.located == ["cuda"]
    assert native.calls[0][1]["directory"] == native.directory
    assert native.opened == [native.directory]
    LlamaBackend.load(gguf, device="cuda", runtime_dir=tmp_path)
    assert native.located == ["cuda"]  # not asked again
    assert native.calls[1][1]["directory"] == tmp_path
    assert native.opened == [native.directory, tmp_path]


def test_a_runtime_that_is_not_installed_is_refused_before_the_weights_are_hashed(
    gguf, no_hashing, nothing_installed
):
    message = (
        "llama.cpp runtime not installed. Run `rizzo download` (runtime + weights) or "
        "`rizzo download --only runtime`; or set RIZZO_LLAMA_DIR to a build of "
        f"{llama_release.COMMIT[:7]}."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        LlamaBackend.load(gguf)


def test_a_directory_of_the_environment_without_the_library_is_refused_before_the_hash(
    gguf, tmp_path, monkeypatch, no_hashing
):
    monkeypatch.setenv(llama_release.RUNTIME_DIR_ENV, str(tmp_path))
    message = f"RIZZO_LLAMA_DIR={tmp_path}: {llama_release.library_name()} not found there"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        LlamaBackend.load(gguf)


def test_a_directory_that_is_given_without_the_library_is_refused_before_the_hash(
    gguf, tmp_path, monkeypatch, no_hashing
):
    monkeypatch.setattr(Library, "_loaded", {})
    message = f"{llama_release.library_name()} not found in {tmp_path.resolve()}"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        LlamaBackend.load(gguf, runtime_dir=tmp_path)
    assert Library._loaded == {}


def test_a_runtime_of_another_build_is_refused_before_the_weights_are_hashed(
    tmp_path, monkeypatch, no_hashing
):
    monkeypatch.setattr(Library, "_loaded", {})
    build = Native().install(tmp_path, monkeypatch)
    build.libraries["llama"].exports.pop("llama_decode")  # a function the binding calls
    message = (
        f"{build.directory.resolve()}: symbol llama_decode is missing; the runtime must be "
        f"llama.cpp {llama_release.RELEASE} ({llama_release.COMMIT[:7]})"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        LlamaBackend.load(build.gguf, runtime_dir=build.directory)
    assert build.model_loads == []  # the weights were never asked for either


def test_the_session_uses_the_library_that_was_opened_before_the_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(Library, "_loaded", {})
    build = Native().install(tmp_path, monkeypatch)
    build.metadata = {
        b"general.architecture": b"spark2_5",
        b"spark2_5.embedding_length": b"2560",
        b"general.file_type": b"7",
    }
    build.template = TEMPLATE.encode()
    LlamaBackend.load(build.gguf, runtime_dir=build.directory).close()
    # Backends register globally in ggml: a second instance of the runtime is never made.
    assert len(build.loads) == 3
    assert build.calls("ggml_backend_load_all_from_path") == [
        (str(build.directory.resolve()).encode(),)
    ]
    assert build.calls("llama_backend_init") == [()]
    assert len(build.model_loads) == 1
