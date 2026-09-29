"""`LlamaBackend.load` and the bookkeeping of `score` on fake sessions: no runtime, no weights.

`test_backend_llama.py` drives `score` with a recording session; this file covers what the load
puts into the identity of a backend, the order of its steps (the runtime is found and opened
before the weights are hashed: a few tests run the real `Library` on the fake libllama of
`test_llama_cpp`), how it cleans up after itself, the memory statistics, and the shape of the
microbatches for any mix of sizes.
"""

import hashlib
import itertools
import re
import sys
from types import SimpleNamespace

import jinja2
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_llama_cpp import Native

from rizzo_flow import backend_llama, llama_release
from rizzo_flow.backend_llama import LlamaBackend, LlamaTokenizer
from rizzo_flow.config import MODELS, GgufSpec
from rizzo_flow.llama_cpp import Device, Library
from rizzo_flow.prompts import PROMPT_VERSION, Compiled, canonical

WEIGHTS = b"GGUF stand-in for the weights"
TEMPLATE = "{{ messages[0].content }}!"
META = {
    "general.architecture": "spark2_5",
    "spark2_5.embedding_length": "2560",
    "general.file_type": "7",
}
# Not part of the identity a fingerprint is computed from.
NOT_IDENTITY = {"fingerprint", "device_name", "context_cells", "load_seconds"}
GEFORCE = Device(5, "CUDA0", "NVIDIA GeForce RTX 5060 Ti", "gpu", "CUDA", 16 << 30)
APPLE = Device(6, "MTL0", "Apple M4", "gpu", "MTL", 16 << 30)


class LoadedSession:
    """What `Session.load` hands back: the metadata of a GGUF file, without a native library."""

    pad_token = 7
    eos_token = 2
    bos_token = 1

    def __init__(self, meta, template, device, n_ctx):
        self.meta_values = meta
        self.template = template
        self.device = device
        self.n_ctx = n_ctx
        self.closed = 0

    def meta(self, key):
        return self.meta_values.get(key)

    def chat_template(self):
        return self.template

    def tokenize(self, text, add_special=False):
        return [*([self.bos_token] if add_special else []), *map(ord, text)]

    def close(self):
        self.closed += 1


class Runtime:
    """Stands in for `llama_cpp.Session`, for `llama_cpp.Library` and for the lookup of the runtime
    directory."""

    def __init__(self, directory):
        self.meta = dict(META)
        self.template = TEMPLATE
        self.device = None
        self.directory = directory
        self.located = []  # devices `llama_release.locate` was asked about
        self.opened = []  # directories `Library.open` was asked for
        self.calls = []  # (gguf, options) of every `Session.load`
        self.events = []  # "locate", "open", "load", in the order they happened
        self.sessions = []
        self.on_load = None  # called after the model is "loaded"

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
        # llama.cpp rounds the context up: what it reports is not what was asked for.
        session = LoadedSession(self.meta, self.template, self.device, options["n_ctx"] + 64)
        self.sessions.append(session)
        if self.on_load:
            self.on_load()
        return session


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
def pin(monkeypatch):
    """Registers `gguf` as one of the pinned files."""

    def pinned(path, variant="flow", quant="q8_0"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        spec = GgufSpec("4b", quant, variant, "some/repo", "a" * 40, path.name, digest)
        monkeypatch.setattr(backend_llama, "GGUF", {("4b", quant, variant): spec})
        return spec

    return pinned


@pytest.fixture
def no_hashing(monkeypatch):
    """Reading the weights fails the test: `sha256_file` hashes them with `hashlib.file_digest`."""

    def hashed(*args, **kwargs):
        raise AssertionError("the weights were read")

    monkeypatch.setattr(llama_release, "hashlib", SimpleNamespace(file_digest=hashed))


@pytest.fixture
def nothing_installed(tmp_path, monkeypatch):
    """The lookup of a runtime is the real one, on a Linux x86-64 box without NVIDIA that has no
    runtime installed."""
    monkeypatch.setattr(llama_release, "RUNTIMES", tmp_path / "runtimes")
    monkeypatch.setattr(llama_release, "host", lambda: ("linux", "x64"))
    monkeypatch.setattr(llama_release, "nvidia_driver", lambda: False)
    monkeypatch.delenv(llama_release.RUNTIME_DIR_ENV, raising=False)


# --- load: arguments ------------------------------------------------------------------------


def test_load_sizes_the_context_for_the_longest_question_plus_one_microbatch(
    native, gguf, tmp_path
):
    backend = LlamaBackend.load(
        gguf,
        device="vulkan",
        ctx=4096,
        batch_size=3,
        prefill_chunk=256,
        threads=6,
        runtime_dir=tmp_path,
        kv_type="q8_0",
    )
    ((path, options),) = native.calls
    assert path == gguf.resolve()
    assert options == {
        "directory": tmp_path,
        "device": "vulkan",
        "n_ctx": 4096 + 2048,
        "n_batch": 2048,
        "n_ubatch": 256,
        "n_seq_max": 4,  # the prefix and one branch per question of a microbatch
        "threads": 6,
        "kv_type": "q8_0",
    }
    assert (backend.batch_size, backend.prefill_chunk) == (3, 256)


def test_load_defaults(native, gguf):
    LlamaBackend.load(str(gguf))  # a string is a path too
    ((_, options),) = native.calls
    assert options == {
        "directory": native.directory,
        "device": "auto",
        "n_ctx": 8192 + 2048,
        "n_batch": 2048,
        "n_ubatch": 512,
        "n_seq_max": 5,
        "threads": None,
        "kv_type": None,
    }


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


def test_load_refuses_bad_arguments_before_touching_the_runtime(native, gguf, tmp_path):
    missing = tmp_path / "missing.gguf"
    message = f"GGUF file not found at {missing.resolve()}. Run `rizzo download` first."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        LlamaBackend.load(missing)
    with pytest.raises(ValueError, match=r"^GGUF file not found at "):
        LlamaBackend.load(tmp_path)  # a directory is not a file
    for ctx in (0, -1):
        with pytest.raises(ValueError, match=r"^ctx must be positive$"):
            LlamaBackend.load(gguf, ctx=ctx)
    assert native.calls == []
    assert native.located == []
    assert native.opened == []
    LlamaBackend.load(gguf, ctx=1)  # the smallest context is fine
    assert native.calls[0][1]["n_ctx"] == 1 + 2048


# --- load: identity -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_type", "precision"),
    [
        ("1", "f16"),
        ("7", "q8_0"),
        ("15", "q4_k_m"),
        ("32", "bf16"),
        ("99", "ftype99"),
        (None, "ftypeNone"),
    ],
)
def test_a_file_that_is_not_pinned_is_described_by_its_hash_and_its_file_type(
    native, gguf, file_type, precision
):
    native.meta["general.file_type"] = file_type
    metadata = LlamaBackend.load(gguf).metadata
    assert {key: value for key, value in metadata.items() if key not in NOT_IDENTITY} == {
        "source": "XHToken/Spark-X2.5-4B",
        "requested_revision": MODELS["4b"].revision,
        "gguf_source": None,
        "gguf_revision": None,
        "source_files": {"model.gguf": hashlib.sha256(WEIGHTS).hexdigest()},
        "precision": precision,
        "device": "cpu",
        "backend": "cpu",
        "runtime": "llama.cpp",
        "llama_cpp_release": llama_release.RELEASE,
        "llama_cpp_commit": llama_release.COMMIT,
        "prompt_version": PROMPT_VERSION,
    }


def test_the_small_checkpoint_is_recognized_by_its_embedding_length(native, gguf):
    native.meta["spark2_5.embedding_length"] = "2048"
    metadata = LlamaBackend.load(gguf).metadata
    assert metadata["source"] == "XHToken/Spark-X2.5-1.7B"
    assert metadata["requested_revision"] == MODELS["1.7b"].revision


def test_a_pinned_fine_tuned_file_is_recognized_by_its_hash_and_says_so(native, gguf, pin):
    spec = pin(gguf, "flow", "q4_k_m")
    native.meta["general.file_type"] = "7"  # the pin, not the file type, tells the precision
    metadata = LlamaBackend.load(gguf).metadata
    assert metadata["gguf_source"] == "some/repo"
    assert metadata["gguf_revision"] == spec.revision
    assert metadata["precision"] == "q4_k_m"
    assert metadata["weights"] == "flow"


def test_a_pinned_original_file_adds_no_weights_key_so_old_fingerprints_hold(native, gguf, pin):
    pin(gguf, "base", "bf16")
    metadata = LlamaBackend.load(gguf).metadata
    assert (metadata["gguf_source"], metadata["precision"]) == ("some/repo", "bf16")
    assert "weights" not in metadata


def test_a_pin_of_another_file_does_not_apply(native, gguf, tmp_path, pin):
    other = tmp_path / "other.gguf"
    other.write_bytes(b"not the pinned bytes")
    pin(other)
    metadata = LlamaBackend.load(gguf).metadata
    assert metadata["gguf_source"] is None
    assert "weights" not in metadata


def test_metadata_names_the_device_that_runs_the_model(native, gguf):
    native.device = GEFORCE
    gpu = LlamaBackend.load(gguf).metadata
    assert (gpu["device"], gpu["backend"], gpu["device_name"]) == (
        "gpu",
        "cuda",
        "NVIDIA GeForce RTX 5060 Ti",
    )
    native.device = APPLE
    assert LlamaBackend.load(gguf).metadata["backend"] == "mtl"  # Apple's registry name
    native.device = None
    cpu = LlamaBackend.load(gguf).metadata
    assert (cpu["device"], cpu["backend"], cpu["device_name"]) == ("cpu", "cpu", None)


def test_the_kv_cache_type_is_part_of_the_identity_unless_it_is_the_default(native, gguf):
    plain = LlamaBackend.load(gguf).metadata
    default = LlamaBackend.load(gguf, kv_type="f16").metadata
    q8 = LlamaBackend.load(gguf, kv_type="q8_0").metadata
    q4 = LlamaBackend.load(gguf, kv_type="q4_0").metadata
    assert "kv_cache" not in plain
    assert "kv_cache" not in default
    assert default["fingerprint"] == plain["fingerprint"]  # recorded before the option existed
    assert (q8["kv_cache"], q4["kv_cache"]) == ("q8_0", "q4_0")
    assert len({plain["fingerprint"], q8["fingerprint"], q4["fingerprint"]}) == 3


def test_the_fingerprint_hashes_the_identity_and_not_how_the_model_is_run(native, gguf, tmp_path):
    small = LlamaBackend.load(gguf, ctx=1024, batch_size=2, threads=1).metadata
    large = LlamaBackend.load(gguf, ctx=8192, batch_size=8, prefill_chunk=128).metadata
    assert small["fingerprint"] == large["fingerprint"]
    assert small["context_cells"] != large["context_cells"]
    identity = {key: value for key, value in small.items() if key not in NOT_IDENTITY}
    assert small["fingerprint"] == hashlib.sha256(canonical(identity).encode()).hexdigest()
    other = tmp_path / "other.gguf"
    other.write_bytes(b"other weights")
    assert LlamaBackend.load(other).metadata["fingerprint"] != small["fingerprint"]
    native.meta["general.file_type"] = "15"  # unpinned: the file type is part of the identity
    assert LlamaBackend.load(gguf).metadata["fingerprint"] != small["fingerprint"]


def test_load_reports_the_cells_the_runtime_gave_and_how_long_the_load_took(
    native, gguf, monkeypatch
):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(backend_llama, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    def spend_the_time_loading():
        clock.now += 2.5

    native.on_load = spend_the_time_loading
    metadata = LlamaBackend.load(gguf, ctx=1000).metadata
    assert metadata["load_seconds"] == 2.5
    assert metadata["context_cells"] == 1000 + 2048 + 64  # the session's answer, not the request


def test_the_backend_serves_the_session_it_loaded(native, gguf):
    backend = LlamaBackend.load(gguf)
    (session,) = native.sessions
    assert backend.session is session
    assert isinstance(backend.tokenizer, LlamaTokenizer)
    assert backend.tokenizer.session is session
    text = backend.tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])
    assert text == "hi!"  # the template of the file
    assert (backend.tokenizer.pad_token_id, backend.tokenizer.eos_token_id) == (7, 2)
    assert session.closed == 0


def test_the_tokenizer_adds_special_tokens_only_when_asked_to():
    tokenizer = LlamaTokenizer(LoadedSession({}, TEMPLATE, None, 0), TEMPLATE)
    assert tokenizer.encode("AB") == [65, 66]  # the template already spells them out
    assert tokenizer.encode("AB", add_special_tokens=False) == [65, 66]
    assert tokenizer.encode("AB", add_special_tokens=True) == [1, 65, 66]
    message = r"^Render the text, then call encode\(\)$"
    with pytest.raises(ValueError, match=message):
        tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True)


def test_the_template_is_rendered_with_the_block_rules_of_transformers():
    template = (
        "{% for m in messages %}\n"
        "    {% if m.role == 'user' %}\n"
        "[{{ m.content }}]\n"
        "    {% endif %}\n"
        "{% endfor %}\n"
    )
    tokenizer = LlamaTokenizer(LoadedSession({}, template, None, 0), template)
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    # No stray newline after a block tag, no indentation before one.
    assert tokenizer.apply_chat_template(messages) == "[a]\n[c]\n"


@pytest.mark.parametrize(
    "template",
    [
        "{{ ''.__class__.__mro__ }}",  # a way out of the template into the interpreter
        "{% set _ = messages.append(1) %}",  # the messages belong to the caller
    ],
)
def test_a_template_downloaded_with_the_weights_runs_in_a_sandbox(template):
    tokenizer = LlamaTokenizer(LoadedSession({}, template, None, 0), template)
    with pytest.raises(ValueError, match=r"^The GGUF chat template cannot run: ") as raised:
        tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])
    assert isinstance(raised.value.__cause__, jinja2.exceptions.SecurityError)


def test_a_template_that_needs_what_the_environment_lacks_cannot_run_either():
    template = "{{ strftime_now('%Y') }}"  # a helper of transformers that this renderer has not
    tokenizer = LlamaTokenizer(LoadedSession({}, template, None, 0), template)
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
    tokenizer = LlamaTokenizer(LoadedSession({}, template, None, 0), template)
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
        LlamaTokenizer(LoadedSession({}, template, None, 0), template)
    assert type(raised.value.__cause__) is RecursionError


def test_a_template_that_refuses_the_conversation_keeps_its_own_message():
    template = "{{ raise_exception('no system role') }}"
    tokenizer = LlamaTokenizer(LoadedSession({}, template, None, 0), template)
    with pytest.raises(ValueError, match=r"^no system role$") as raised:
        tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])
    assert raised.value.__cause__ is None  # not wrapped: it was ours to begin with


# --- load: refusals -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("meta", "message"),
    [
        (
            {"general.architecture": "llama"},
            "Only the Spark2.5 architecture is supported, not llama",
        ),
        ({"general.architecture": None}, "Only the Spark2.5 architecture is supported, not None"),
        (
            {"spark2_5.embedding_length": "4096"},
            "Unrecognized Spark2.5 checkpoint (hidden_size=4096); supported sizes: 4b, 1.7b",
        ),
        (
            {"spark2_5.embedding_length": None},
            "Unrecognized Spark2.5 checkpoint (hidden_size=0); supported sizes: 4b, 1.7b",
        ),
        (
            {"spark2_5.embedding_length": ""},
            "Unrecognized Spark2.5 checkpoint (hidden_size=0); supported sizes: 4b, 1.7b",
        ),
    ],
)
def test_files_of_another_architecture_or_size_are_refused_and_the_session_released(
    native, gguf, meta, message
):
    native.meta.update(meta)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        LlamaBackend.load(gguf)
    assert [session.closed for session in native.sessions] == [1]


@pytest.mark.parametrize("template", [None, ""])
def test_a_file_without_a_chat_template_is_refused_and_the_session_released(native, gguf, template):
    native.template = template
    with pytest.raises(ValueError, match=r"^The GGUF file carries no chat template$"):
        LlamaBackend.load(gguf)
    assert [session.closed for session in native.sessions] == [1]


def test_a_broken_chat_template_is_reported_and_the_session_released(native, gguf):
    native.template = "{% if %}"
    with pytest.raises(ValueError, match=r"^The GGUF chat template does not compile: ") as raised:
        LlamaBackend.load(gguf)
    assert isinstance(raised.value.__cause__, jinja2.TemplateSyntaxError)
    assert [session.closed for session in native.sessions] == [1]


WRONG_SIZES = [{"batch_size": 17}, {"batch_size": 0}, {"prefill_chunk": 2049}, {"prefill_chunk": 0}]


@pytest.mark.parametrize("options", WRONG_SIZES)
def test_wrong_batch_sizes_do_not_leave_a_loaded_model_behind(native, gguf, options):
    with pytest.raises(ValueError, match="batch_size must be"):
        LlamaBackend.load(gguf, **options)
    assert all(session.closed for session in native.sessions)


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


# --- load: the runtime comes before the weights -----------------------------------------------


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


def test_a_backend_that_cannot_be_constructed_releases_the_session(native, gguf):
    class Unwilling(LlamaBackend):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no backend today")

    with pytest.raises(RuntimeError, match=r"^no backend today$"):
        Unwilling.load(gguf)
    assert [session.closed for session in native.sessions] == [1]


def test_the_defaults_are_four_questions_per_microbatch_and_chunks_of_512_tokens():
    backend = LlamaBackend(object(), None, {})
    assert (backend.batch_size, backend.prefill_chunk) == (4, 512)


@pytest.mark.parametrize(("batch_size", "prefill_chunk"), [(1, 1), (4, 512), (16, 2048)])
def test_the_documented_ranges_of_batch_size_and_prefill_chunk_are_accepted(
    batch_size, prefill_chunk
):
    backend = LlamaBackend(object(), None, {}, batch_size, prefill_chunk)
    assert (backend.batch_size, backend.prefill_chunk) == (batch_size, prefill_chunk)


@pytest.mark.parametrize(
    ("batch_size", "prefill_chunk"), [(0, 512), (-1, 512), (17, 512), (4, 0), (4, -1), (4, 2049)]
)
def test_sizes_outside_the_documented_ranges_are_refused(batch_size, prefill_chunk):
    with pytest.raises(ValueError, match=r"^batch_size must be 1.16 and prefill_chunk 1.2048$"):
        LlamaBackend(object(), None, {}, batch_size, prefill_chunk)


# --- score: memory statistics ---------------------------------------------------------------


class Gpu:
    """A session that only answers what the memory bookkeeping asks: one free-memory value per
    call (None: there is no device to ask), and the same fixed logits for every question."""

    def __init__(self, free, idle_free=1000):
        self.free = iter(free)
        self.idle_free = idle_free

    def clear(self):
        pass

    def decode(self, tokens, positions, sequences, outputs=()):
        pass

    def logits(self, index, slots):
        return [0.0] * len(slots)

    def synchronize(self):
        pass

    def free_bytes(self):
        return next(self.free)


def one_question():
    return [1], [Compiled("q", [1, 2], [65, 66], "hash")]


def test_the_peak_is_the_deepest_drop_of_free_memory_since_before_the_load():
    engine = LlamaBackend(Gpu([900, 600, 800]), None, {})
    assert engine.peak_device_bytes() is None  # nothing has been measured yet
    peaks = [engine.score(*one_question())[1]["peak_device_bytes"] for _ in range(3)]
    assert peaks == [100, 400, 400]  # a later call with more free memory does not lower it
    assert engine.peak_device_bytes() == 400


def test_memory_freed_by_other_processes_never_makes_a_negative_peak():
    engine = LlamaBackend(Gpu([1500]), None, {})
    assert engine.score(*one_question())[1]["peak_device_bytes"] == 0


def test_without_a_device_there_is_no_peak_at_all():
    engine = LlamaBackend(Gpu([None, None]), None, {})
    _, timing = engine.score(*one_question())
    assert "peak_device_bytes" not in timing
    assert engine.peak_device_bytes() is None


# --- score: microbatches --------------------------------------------------------------------


def suffixed(sizes, prefix_length=2):
    prefix = list(range(prefix_length))
    return [Compiled(str(i), [*prefix, *[7] * size], [65], "h") for i, size in enumerate(sizes)]


def test_microbatches_split_on_the_batch_size_and_on_the_limit_of_one_call(monkeypatch):
    monkeypatch.setattr(backend_llama, "N_BATCH", 6)
    engine = LlamaBackend(object(), None, {}, batch_size=3)
    groups = engine._groups(suffixed([3, 3, 1, 1, 1, 1, 1]), 2)
    # 3 + 3 fills a call exactly and stays together; three per group at most.
    assert [[job.id for job in group] for group in groups] == [
        ["0", "1"],
        ["2", "3", "4"],
        ["5", "6"],
    ]
    # A question longer than one call is alone in its group; `_feed` slices it.
    assert [[job.id for job in g] for g in engine._groups(suffixed([10, 1]), 2)] == [["0"], ["1"]]


def test_no_questions_make_no_microbatches():
    engine = LlamaBackend(object(), None, {})
    assert list(engine._groups([], 5)) == []


@settings(max_examples=100, deadline=None, database=None, derandomize=True)
@given(
    sizes=st.lists(st.integers(1, 12), max_size=25),
    batch_size=st.integers(1, 16),
    limit=st.integers(1, 30),
)
def test_microbatches_are_ordered_bounded_and_as_full_as_they_can_be(sizes, batch_size, limit):
    jobs = suffixed(sizes)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend_llama, "N_BATCH", limit)
        groups = list(LlamaBackend(object(), None, {}, batch_size=batch_size)._groups(jobs, 2))
    assert [job for group in groups for job in group] == jobs  # each once, in order
    assert all(groups)  # never an empty microbatch

    def used(group):
        return sum(len(job.tokens) - 2 for job in group)

    for group in groups:
        assert len(group) <= batch_size
        if len(group) > 1:
            assert used(group) <= limit
    for group, following in itertools.pairwise(groups):
        # Greedy: the next question was left out because it would not have fitted.
        assert len(group) == batch_size or used(group) + used(following[:1]) > limit


class Ledger:
    """A session that checks how it is driven. Positions must continue each sequence without gaps,
    a call carries at most N_BATCH tokens and only sequences below `n_seq_max`, and the logit of a
    slot is the position it was read at plus slot / 1000: every answer says which token it is."""

    idle_free = None

    def __init__(self, n_seq_max, on_decode=None):
        self.n_seq_max = n_seq_max
        self.on_decode = on_decode
        self.cells = {}
        self.last = None
        self.calls = []  # (tokens, positions, sequences, outputs) of every decode

    def clear(self):
        self.cells = {}

    def branch(self, source, target):
        assert target < self.n_seq_max
        self.cells[target] = list(self.cells[source])

    def drop(self, sequence):
        del self.cells[sequence]

    def decode(self, tokens, positions, sequences, outputs=()):
        tokens, positions, sequences = list(tokens), list(positions), list(sequences)
        assert len(tokens) == len(positions) == len(sequences) <= backend_llama.N_BATCH
        for position, sequence in zip(positions, sequences, strict=True):
            assert sequence < self.n_seq_max
            held = self.cells.setdefault(sequence, [])
            assert position == len(held), "positions must continue each sequence without gaps"
            held.append(position)
        self.last = (positions, list(outputs))
        self.calls.append((tokens, positions, sequences, list(outputs)))
        if self.on_decode:
            self.on_decode(tokens)

    def logits(self, index, slots):
        assert self.last is not None
        positions, outputs = self.last
        assert index in outputs, "logits were not requested at this row"
        return [positions[index] + slot / 1000 for slot in slots]

    def synchronize(self):
        pass

    def free_bytes(self):
        return None


def test_feed_slices_the_tokens_and_wants_logits_only_at_the_very_end(monkeypatch):
    monkeypatch.setattr(backend_llama, "N_BATCH", 4)
    session = Ledger(2)
    engine = LlamaBackend(session, None, {})
    tokens = list(range(10))
    assert engine._feed(tokens, 0, 0, True) == 1  # the last token is row 1 of the last call
    assert [(len(call[0]), call[3]) for call in session.calls] == [(4, []), (4, []), (2, [1])]
    session.calls.clear()
    assert engine._feed(tokens, 0, 1, False) is None  # nobody reads a row: none is reported
    assert [(len(call[0]), call[3]) for call in session.calls] == [(4, []), (4, []), (2, [])]


def test_feeding_nothing_decodes_nothing():
    session = Ledger(2)
    assert LlamaBackend(session, None, {})._feed([], 0, 0, True) is None
    assert session.calls == []


def test_score_refuses_bad_requests_before_running_anything():
    session = Ledger(5)
    engine = LlamaBackend(session, None, {})
    job = Compiled("a", [1, 2], [65], "h")
    with pytest.raises(ValueError, match=r"^Unknown execution mode$"):
        engine.score([1], [job], "batched")
    with pytest.raises(ValueError, match=r"^No decisions supplied$"):
        engine.score([1], [], "shared")
    with pytest.raises(ValueError, match=r"^Invalid shared prefix$"):
        engine.score([9], [job], "shared")  # the question does not start with the prefix
    with pytest.raises(ValueError, match=r"^Invalid shared prefix$"):
        engine.score([1, 2], [job], "shared")  # and one that is only the prefix has no answer
    assert session.calls == []


@settings(max_examples=80, deadline=None, database=None, derandomize=True)
@given(
    prefix_length=st.integers(1, 12),
    suffixes=st.lists(st.integers(1, 10), min_size=1, max_size=7),
    batch_size=st.integers(1, 4),
    limit=st.integers(1, 15),
    mode=st.sampled_from(["shared", "direct"]),
)
def test_every_question_is_answered_from_its_own_last_token_in_either_mode(
    prefix_length, suffixes, batch_size, limit, mode
):
    prefix = list(range(prefix_length))
    jobs = [
        Compiled(f"q{i}", [*prefix, *[100 + i] * size], [65, 66], "h")
        for i, size in enumerate(suffixes)
    ]
    session = Ledger(batch_size + 1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend_llama, "N_BATCH", limit)
        engine = LlamaBackend(session, None, {}, batch_size=batch_size)
        logits, timing = engine.score(prefix, jobs, mode)
        by_length = sorted(jobs, key=lambda job: len(job.tokens))
        groups = len(list(engine._groups(by_length, prefix_length)))
    assert set(logits) == {job.id for job in jobs}
    for job in jobs:
        last = len(job.tokens) - 1
        assert logits[job.id] == pytest.approx([last + 0.065, last + 0.066])
    total = sum(len(job.tokens) for job in jobs)
    assert timing["logical_input_tokens"] == total
    assert timing["generated_tokens"] == 0
    if mode == "shared" and len(jobs) > 1:
        assert timing["shared_prefix_tokens"] == prefix_length
        assert timing["evaluated_tokens_including_padding"] == prefix_length + sum(suffixes)
        assert timing["batches"] == groups
        assert session.cells == {0: list(range(prefix_length))}  # the prefix outlives every branch
    else:
        assert timing["shared_prefix_tokens"] == 0
        assert timing["evaluated_tokens_including_padding"] == total
        assert timing["batches"] == len(jobs)


def test_timing_tells_the_prefix_pass_from_the_whole_request(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(backend_llama, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    def half_a_second_per_token(tokens):
        clock.now += 0.5 * len(tokens)

    prefix = [1, 2, 3, 4]
    jobs = [
        Compiled("a", [*prefix, 9], [65], "h"),
        Compiled("b", [*prefix, 8, 9], [65], "h"),
    ]
    _, shared = LlamaBackend(Ledger(5, half_a_second_per_token), None, {}).score(
        prefix, jobs, "shared"
    )
    assert shared["prefill_seconds"] == 2.0  # four tokens of prefix
    assert shared["inference_seconds"] == 3.5  # plus one microbatch of three suffix tokens
    _, direct = LlamaBackend(Ledger(5, half_a_second_per_token), None, {}).score(
        prefix, jobs, "direct"
    )
    assert direct["prefill_seconds"] == 0.0
    assert direct["inference_seconds"] == 5.5  # 5 + 6 tokens, each question from scratch
