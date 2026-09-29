"""The ctypes binding to libllama, run against a fake native library: no runtime, no weights.

`Native` plays libllama and ggml in Python and takes the place of `ctypes.CDLL`, so `Library`
and `Session` run unchanged, platform branches included. Every symbol is a callable that
converts its arguments with the prototype the binding declared (as ctypes does), records the
call, and behaves like the C function: handles have to come back to the right function, a model
outlives its context, and a batch is read the way the C side reads it.
"""

import ctypes
import os
import re
import sys
import types
import weakref
from ctypes import POINTER, c_float, c_int
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rizzo_flow import llama_cpp, llama_release
from rizzo_flow.llama_cpp import (
    LOG_CALLBACK,
    SIGNATURES,
    ContextParams,
    Library,
    ModelParams,
    Session,
)

GIB = 1 << 30
MODEL_BYTES = 4 * GIB
CPU, GPU, IGPU, ACCEL, META = range(5)  # enum ggml_backend_dev_type


# --- the fake native library ------------------------------------------------------------------


class Function:
    """A native symbol: callable like a ctypes function, with a settable prototype."""

    def __init__(self, name, implementation, journal):
        self.name = name
        self.implementation = implementation
        self.journal = journal
        self.restype = c_int  # what ctypes assumes until told otherwise
        self.argtypes = None

    def __call__(self, *args):
        assert self.argtypes is not None, f"{self.name}: the prototype was never declared"
        assert len(args) == len(self.argtypes), f"{self.name}: {len(args)} arguments"
        for kind, value in zip(self.argtypes, args, strict=True):
            kind.from_param(value)  # what ctypes does: a value of the wrong type raises
        # C does not own a callback, so neither does the record of the call.
        kept = tuple(weakref.ref(a) if isinstance(a, LOG_CALLBACK) else a for a in args)
        self.journal.append((self.name, kept))
        return self.implementation(*args)


class Handle:
    """A loaded shared library: an attribute is a symbol, and a missing one raises."""

    def __init__(self, exports):
        self.exports = exports

    def __getattr__(self, name):
        try:
            return self.exports[name]
        except KeyError:
            raise AttributeError(name) from None


@dataclass
class Hardware:
    """One device of the fake ggml registry."""

    name: str
    description: str
    kind: int
    backend: str
    total: int
    free: int
    handle: int


def stem(file_name):
    """libggml-base.so, libggml-base.dylib, ggml-base.dll -> ggml-base"""
    return file_name.split(".")[0].removeprefix("lib")


def shared_name(name, platform):
    if platform == "win32":
        return f"{name}.dll"
    return f"lib{name}.dylib" if platform == "darwin" else f"lib{name}.so"


# ggml keeps the backend registry in libggml and the device accessors in libggml-base.
GGML_EXPORTS = (
    "ggml_backend_load_all_from_path",
    "ggml_backend_dev_count",
    "ggml_backend_dev_get",
    "ggml_backend_reg_name",
)
BASE_EXPORTS = tuple(n for n in SIGNATURES if n.startswith("ggml_") and n not in GGML_EXPORTS)


class Native:
    """libllama and ggml in Python. Tests set what it reports and read what it was asked."""

    MODEL, CONTEXT, VOCAB, MEMORY = 0x1000, 0x2000, 0x3000, 0x4000
    BOS, EOS = 1000, 1001  # ids of the special tokens the fake vocabulary adds
    ROWS = 300  # entries of every logits row

    def __init__(self):
        self.journal: list[tuple[str, tuple]] = []
        self.functions = {
            name: Function(name, getattr(self, name), self.journal) for name in SIGNATURES
        }
        self.libraries = {
            "llama": Handle({n: f for n, f in self.functions.items() if n.startswith("llama_")}),
            "ggml": Handle({n: self.functions[n] for n in GGML_EXPORTS}),
            "ggml-base": Handle({n: self.functions[n] for n in BASE_EXPORTS}),
        }
        self.directory = Path()
        self.gguf = Path()
        self.loads: list[tuple[Path, Any]] = []  # every ctypes.CDLL(path, mode)
        self.unloadable: set[str] = set()
        self.log_callback: Any = None
        self.backends_loaded = False
        self.hardware: list[Hardware] = []
        self.registries: dict[int, str] = {}
        self.model_ok = self.context_ok = True
        self.model_alive = self.context_alive = False
        self.model_loads: list[Any] = []
        self.context_inits: list[Any] = []
        self.context_params = ContextParams()
        self.granted: dict[str, int] = {}  # n_ctx, n_batch as llama.cpp reports them
        self.metadata: dict[bytes, bytes] = {}
        self.template: bytes | None = None
        self.pad = self.eos = -1
        self.extra_tokens = 0  # special tokens the vocabulary adds to every text
        self.decode_status = 0
        self.decodes: list[Any] = []
        self.rows: dict[int, list[float]] = {}
        self.kept: list[Any] = []  # native memory that outlives the call that returned it
        self.add_device("CPU", "Fake CPU", CPU, "CPU", 64 * GIB)
        self.gpu = self.add_device(
            "CUDA0", "NVIDIA GeForce RTX 5060 Ti", GPU, "CUDA", 16 * GIB, 15 * GIB
        )

    # --- the test side ---

    def add_device(self, name, description, kind, backend, total, free=None):
        handle = 0x5000 + len(self.hardware)
        free = total if free is None else free
        self.hardware.append(Hardware(name, description, kind, backend, total, free, handle))
        return self.hardware[-1]

    def lay_out(self, directory, platform="linux"):
        """The three libraries of a runtime, as empty files named the way `platform` names them."""
        directory.mkdir(parents=True)
        for name in ("ggml-base", "ggml", "llama"):
            (directory / shared_name(name, platform)).touch()

    def install(self, tmp_path, monkeypatch, platform="linux"):
        """A runtime on disk for `platform`, with this fake behind ctypes.CDLL."""
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(ctypes, "CDLL", self.cdll)
        self.directory = tmp_path / "runtime"
        self.lay_out(self.directory, platform)
        self.gguf = tmp_path / "model.gguf"
        self.gguf.touch()
        return self

    def cdll(self, path, mode=None):
        self.loads.append((Path(path), mode))
        if Path(path).name in self.unloadable:
            raise OSError(f"{path}: cannot open shared object file")
        return self.libraries.get(stem(Path(path).name), Handle({}))  # the CUDA runtime: no symbols

    def calls(self, name):
        return [args for called, args in self.journal if called == name]

    def log(self, level, text):
        """What llama.cpp does: call the registered callback, if its memory is still there."""
        callback = self.log_callback()
        assert callback is not None, (
            "the callback was freed: llama.cpp would call a dangling pointer"
        )
        callback(level, text, None)

    # --- ggml-backend.h ---

    def registry(self, backend):
        for handle, name in self.registries.items():
            if name == backend:
                return handle
        handle = 0x6000 + len(self.registries)
        self.registries[handle] = backend
        return handle

    def device(self, handle):
        found = [d for d in self.hardware if d.handle == handle]
        assert len(found) == 1, f"{handle!r} is not a device handle"
        return found[0]

    def need_model(self, model):
        assert model == self.MODEL, f"{model!r} is not the model handle"
        assert self.model_alive, "the model was freed"

    def need_context(self, context):
        assert context == self.CONTEXT, f"{context!r} is not the context handle"
        assert self.context_alive, "the context was freed"

    def ggml_backend_load_all_from_path(self, path):
        self.backends_loaded = True  # plug-ins register their devices only now

    def ggml_backend_dev_count(self):
        return len(self.hardware) if self.backends_loaded else 0

    def ggml_backend_dev_get(self, index):
        return self.hardware[index].handle

    def ggml_backend_dev_name(self, handle):
        return self.device(handle).name.encode()

    def ggml_backend_dev_description(self, handle):
        return self.device(handle).description.encode()

    def ggml_backend_dev_memory(self, handle, free, total):
        free._obj.value = self.device(handle).free
        total._obj.value = self.device(handle).total

    def ggml_backend_dev_type(self, handle):
        return self.device(handle).kind

    def ggml_backend_dev_backend_reg(self, handle):
        return self.registry(self.device(handle).backend)

    def ggml_backend_reg_name(self, registry):
        return self.registries[registry].encode()

    # --- llama.h: the runtime and the model ---

    def llama_log_set(self, callback, user_data):
        self.log_callback = weakref.ref(callback)  # C keeps a bare pointer, not the Python object

    def llama_backend_init(self):
        pass

    def llama_model_default_params(self):
        params = ModelParams()
        params.n_gpu_layers = -1
        params.split_mode = 1  # LLAMA_SPLIT_MODE_LAYER: the binding has to ask for NONE
        params.use_extra_bufts = True
        return params

    def llama_model_load_from_file(self, path, params):
        devices: list[int] = []
        while params.devices[len(devices)] is not None:  # the list ends with NULL
            devices.append(params.devices[len(devices)])
            assert len(devices) < 8, "the device list is not NULL-terminated"
        self.model_loads.append(
            types.SimpleNamespace(
                path=path, params=ModelParams.from_buffer_copy(params), devices=devices
            )
        )
        if not self.model_ok:
            return None
        self.model_alive = True
        for hardware in self.hardware:
            if hardware.handle in devices:
                hardware.free -= MODEL_BYTES  # the weights now live there
        return self.MODEL

    def llama_model_free(self, model):
        self.need_model(model)
        assert not self.context_alive, "the context has to be freed before its model"
        self.model_alive = False

    def llama_model_get_vocab(self, model):
        self.need_model(model)
        return self.VOCAB

    def llama_model_meta_val_str(self, model, key, buffer, size):
        self.need_model(model)
        assert ctypes.sizeof(buffer) == size, "the announced room is not the buffer's"
        value = self.metadata.get(key)
        if value is None:
            return -1  # a failure: the buffer is cleared
        kept = value[: size - 1] + b"\0"  # always terminated, cut to the room there is
        ctypes.memmove(buffer, kept, len(kept))
        return len(value)

    def llama_model_chat_template(self, model, name):
        self.need_model(model)
        assert name is None, "the default template is the one with no name"
        return self.template

    def llama_vocab_pad(self, vocab):
        assert vocab == self.VOCAB
        return self.pad

    def llama_vocab_eos(self, vocab):
        assert vocab == self.VOCAB
        return self.eos

    def llama_tokenize(self, vocab, text, text_len, tokens, capacity, add_special, parse_special):
        assert vocab == self.VOCAB
        assert len(tokens) == capacity, "the announced room is not the array's"
        made = list(text[:text_len])  # a byte is a token; only text_len bytes are read
        if add_special:
            made.insert(0, self.BOS)
        made += [self.EOS] * self.extra_tokens
        if len(made) > capacity:
            return -len(made)  # the number that would have been needed
        for index, token in enumerate(made):
            tokens[index] = token
        return len(made)

    # --- llama.h: the context ---

    def llama_context_default_params(self):
        params = ContextParams()
        params.n_ctx = 512
        params.n_batch = 2048
        params.n_ubatch = 512
        params.n_seq_max = 1
        params.n_threads = 4
        params.n_threads_batch = 8
        params.type_k = params.type_v = 1  # GGML_TYPE_F16
        params.offload_kqv = True
        params.op_offload = True
        return params

    def llama_init_from_model(self, model, params):
        self.need_model(model)
        self.context_inits.append(ContextParams.from_buffer_copy(params))
        if not self.context_ok:
            return None
        self.context_alive = True
        self.context_params = ContextParams.from_buffer_copy(params)
        return self.CONTEXT

    def llama_free(self, context):
        self.need_context(context)
        self.context_alive = False

    def llama_n_ctx(self, context):
        self.need_context(context)
        padded = -(-self.context_params.n_ctx // 256) * 256  # llama.cpp rounds up to 256
        return self.granted.get("n_ctx", padded)

    def llama_n_batch(self, context):
        self.need_context(context)
        return self.granted.get("n_batch", self.context_params.n_batch)

    def llama_get_memory(self, context):
        self.need_context(context)
        return self.MEMORY

    def llama_memory_clear(self, memory, data):
        assert memory == self.MEMORY

    def llama_memory_seq_cp(self, memory, source, target, start, end):
        assert memory == self.MEMORY

    def llama_memory_seq_rm(self, memory, sequence, start, end):
        assert memory == self.MEMORY
        return True

    def llama_decode(self, context, batch):
        self.need_context(context)
        count = batch.n_tokens
        seen = types.SimpleNamespace(
            tokens=[batch.token[i] for i in range(count)],
            positions=[batch.pos[i] for i in range(count)],
            n_seq_ids=[batch.n_seq_id[i] for i in range(count)],
            sequences=[batch.seq_id[i][0] for i in range(count)],
            outputs=[i for i in range(count) if batch.logits[i]],
            embeddings=bool(batch.embd),
        )
        self.decodes.append(seen)
        self.rows = {}
        if self.decode_status == 0:
            # Row `i` is the logits of batch position `i`: entry `slot` is `i * 1000 + slot`.
            self.rows = {i: [i * 1000.0 + slot for slot in range(self.ROWS)] for i in seen.outputs}
        return self.decode_status

    def llama_synchronize(self, context):
        assert context == self.CONTEXT

    def llama_get_logits_ith(self, context, index):
        self.need_context(context)
        row = self.rows.get(index)
        if row is None:
            return POINTER(c_float)()  # NULL: no logits at that position
        values = (c_float * len(row))(*row)
        self.kept.append(values)
        return ctypes.cast(values, POINTER(c_float))


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    """`Library.open` keeps one instance per directory for the whole process: start empty."""
    monkeypatch.setattr(Library, "_loaded", {})


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    return Native().install(tmp_path, monkeypatch)


def exactly(message):
    """A `match=` that has to be the whole message, not a piece of it."""
    return f"^{re.escape(message)}$"


def load(native, **options):
    """A session on the fake runtime, on the device `auto` picks unless told otherwise."""
    return Session.load(native.gguf, directory=native.directory, **options)


# --- the library ----------------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["llama_log_set", "llama_decode", "ggml_backend_dev_memory"])
def test_a_runtime_without_a_symbol_is_refused_naming_the_expected_release(runtime, missing):
    for handle in runtime.libraries.values():
        handle.exports.pop(missing, None)
    message = (
        f"{runtime.directory.resolve()}: symbol {missing} is missing; the runtime must be "
        f"llama.cpp {llama_release.RELEASE} ({llama_release.COMMIT[:7]})"
    )
    with pytest.raises(ValueError, match=exactly(message)):
        Library.open(runtime.directory)
    assert Library._loaded == {}


def test_a_library_resolves_the_directory_it_is_given(runtime):
    library = Library(runtime.directory / ".." / "runtime")
    assert library.directory == runtime.directory.resolve()


def test_a_runtime_one_folder_down_is_opened_from_there(runtime, tmp_path):
    root = tmp_path / "unpacked"
    runtime.lay_out(root / "llama-b11081")
    Library.open(root)
    assert {path.parent for path, _ in runtime.loads} == {(root / "llama-b11081").resolve()}


def test_a_runtime_reached_through_its_parent_folder_is_one_library(runtime, tmp_path):
    root, inner = tmp_path / "unpacked", tmp_path / "unpacked" / "llama-b11081"
    runtime.lay_out(inner)
    (inner / "libcudart.so.13").touch()  # the CUDA runtime beside libllama is found there as well
    first = Library.open(root)
    assert first.directory == inner.resolve()
    assert Library.open(inner) is first  # backends register globally: never a second instance
    assert Library.open(root / ".." / "unpacked") is first
    assert Library._loaded == {inner.resolve(): first}
    assert runtime.calls("ggml_backend_load_all_from_path") == [(str(inner.resolve()).encode(),)]
    assert runtime.calls("llama_backend_init") == [()]
    assert [path.name for path, _ in runtime.loads] == [
        "libcudart.so.13",
        "libggml-base.so",
        "libggml.so",
        "libllama.so",
    ]
    assert {path.parent for path, _ in runtime.loads} == {inner.resolve()}


def test_a_library_made_from_a_parent_folder_reports_the_folder_that_holds_libllama(
    runtime, tmp_path
):
    runtime.lay_out(tmp_path / "unpacked" / "b")
    assert Library(tmp_path / "unpacked").directory == (tmp_path / "unpacked" / "b").resolve()


def test_libllama_in_the_folder_itself_wins_over_one_below_it(runtime, tmp_path):
    root = tmp_path / "unpacked"
    runtime.lay_out(root)
    runtime.lay_out(root / "older")
    assert Library.open(root).directory == root.resolve()
    assert {path.parent for path, _ in runtime.loads} == {root.resolve()}


def test_the_runtime_folder_is_the_resolved_one_even_when_the_library_sits_behind_a_link(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / llama_release.library_name()).touch()
    outer = tmp_path / "runtime"
    outer.mkdir()
    try:
        (outer / "build").symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("this account cannot create symbolic links")
    assert llama_cpp._runtime_folder(outer) == real.resolve()


def test_an_unknown_kv_type_does_not_leave_a_model_loaded(runtime):
    with pytest.raises(ValueError, match=exactly("KV cache type must be one of: f16, q8_0, q4_0")):
        load(runtime, kv_type="f32")
    assert runtime.model_loads == []  # refused before the weights were touched
    assert runtime.model_alive is False


@pytest.mark.parametrize(
    "options",
    [
        {"threads": "x"},
        {"threads": 2.5},
        {"n_ctx": 1.5},
        {"n_batch": 1.5},
        {"n_seq_max": "5"},
    ],
)
def test_an_option_that_cannot_be_set_is_refused_before_the_model_is_loaded(runtime, options):
    with pytest.raises(TypeError):
        load(runtime, **options)
    assert runtime.model_loads == []  # the ~10 s load of the weights was never started
    assert runtime.model_alive is False


def test_windows_puts_the_folder_on_path_unless_an_entry_is_exactly_it(tmp_path, monkeypatch):
    native = Native().install(tmp_path, monkeypatch, "win32")
    monkeypatch.setattr(os, "add_dll_directory", lambda folder: None, raising=False)
    folder = str(native.directory.resolve())
    monkeypatch.setenv("PATH", os.pathsep.join([folder + "-old", "C:\\Windows"]))
    Library(native.directory)
    assert os.environ["PATH"].split(os.pathsep)[0] == folder


def test_windows_takes_another_spelling_of_the_folder_for_the_folder(tmp_path, monkeypatch):
    native = Native().install(tmp_path, monkeypatch, "win32")
    monkeypatch.setattr(os, "add_dll_directory", lambda folder: None, raising=False)
    present = os.pathsep.join(["C:\\Tools", str(native.directory.resolve()) + os.sep])
    monkeypatch.setenv("PATH", present)
    Library(native.directory)
    assert os.environ["PATH"] == present
