"""The ctypes binding to libllama, run against a fake native library: no runtime, no weights.

`Native` plays libllama and ggml in Python and takes the place of `ctypes.CDLL`, so `Library`
and `Session` run unchanged, platform branches included. Every symbol is a callable that
converts its arguments with the prototype the binding declared (as ctypes does), records the
call, and behaves like the C function: handles have to come back to the right function, a model
outlives its context, and a batch is read the way the C side reads it.
"""

import ctypes
import gc
import os
import re
import sys
import types
import weakref
from ctypes import (
    POINTER,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_int8,
    c_int32,
    c_size_t,
    c_uint32,
    c_void_p,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rizzo_flow import llama_cpp, llama_release
from rizzo_flow.llama_cpp import (
    LOG_CALLBACK,
    SIGNATURES,
    Batch,
    ContextParams,
    Device,
    Library,
    ModelParams,
    Session,
    choose_device,
)

PROPERTY = settings(
    max_examples=40,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

GIB = 1 << 30
MODEL_BYTES = 4 * GIB
CPU, GPU, IGPU, ACCEL, META = range(5)  # enum ggml_backend_dev_type
LOG_NONE, LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CONT = range(6)  # enum ggml_log_level


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


@pytest.fixture
def library(runtime):
    """A library kept alive by the test, as `Library.open` keeps one for the whole process."""
    return Library(runtime.directory)


def exactly(message):
    """A `match=` that has to be the whole message, not a piece of it."""
    return f"^{re.escape(message)}$"


def load(native, **options):
    """A session on the fake runtime, on the device `auto` picks unless told otherwise."""
    return Session.load(native.gguf, directory=native.directory, **options)


def changed(struct, defaults):
    """Names of the fields whose bytes differ between two instances of a struct."""
    mine, theirs = bytes(struct), bytes(defaults)
    names = []
    for name, _ in type(struct)._fields_:
        field = getattr(type(struct), name)
        span = slice(field.offset, field.offset + field.size)
        if mine[span] != theirs[span]:
            names.append(name)
    return set(names)


# --- layouts and prototypes against the header of the pinned commit -----------------------------

HEADER_COMMIT = "161755f29e415e2c33efe906e91843c068efd664"

# The structs of include/llama.h at HEADER_COMMIT, comments dropped. When `llama_release.COMMIT`
# moves, compare every line here and in PROTOTYPES with the headers of the new commit.
MODEL_PARAMS_H = """
    ggml_backend_dev_t * devices;
    const struct llama_model_tensor_buft_override * tensor_buft_overrides;
    int32_t n_gpu_layers;
    enum llama_split_mode split_mode;
    enum llama_load_mode load_mode;
    enum llama_lazy_mode lazy_mode;
    int32_t main_gpu;
    const float * tensor_split;
    llama_progress_callback progress_callback;
    void * progress_callback_user_data;
    const struct llama_model_kv_override * kv_overrides;
    bool vocab_only;
    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;
    bool load_mtp;
"""
CONTEXT_PARAMS_H = """
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;
    uint32_t n_outputs_max;
    uint32_t n_outputs_max_per_seq;
    int32_t n_threads;
    int32_t n_threads_batch;
    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;
    enum llama_attention_type attention_type;
    enum llama_flash_attn_type flash_attn_type;
    float rope_freq_base;
    float rope_freq_scale;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float defrag_thold;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    enum ggml_type type_k;
    enum ggml_type type_v;
    ggml_abort_callback abort_callback;
    void * abort_callback_data;
    bool embeddings;
    bool offload_kqv;
    bool no_perf;
    bool op_offload;
    bool swa_full;
    bool kv_unified;
    struct llama_sampler_seq_config * samplers;
    size_t n_samplers;
    struct llama_context * ctx_other;
"""
BATCH_H = """
    int32_t n_tokens;
    llama_token * token;
    float * embd;
    llama_pos * pos;
    int32_t * n_seq_id;
    llama_seq_id ** seq_id;
    int8_t * logits;
"""
PROTOTYPES = """
    void llama_log_set(ggml_log_callback log_callback, void * user_data);
    void llama_backend_init(void);
    struct llama_model_params llama_model_default_params(void);
    struct llama_context_params llama_context_default_params(void);
    struct llama_model * llama_model_load_from_file(
        const char * path_model, struct llama_model_params params);
    void llama_model_free(struct llama_model * model);
    struct llama_context * llama_init_from_model(
        struct llama_model * model, struct llama_context_params params);
    void llama_free(struct llama_context * ctx);
    const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
    int32_t llama_model_meta_val_str(
        const struct llama_model * model, const char * key, char * buf, size_t buf_size);
    const char * llama_model_chat_template(const struct llama_model * model, const char * name);
    llama_token llama_vocab_pad(const struct llama_vocab * vocab);
    llama_token llama_vocab_eos(const struct llama_vocab * vocab);
    int32_t llama_tokenize(
        const struct llama_vocab * vocab, const char * text, int32_t text_len,
        llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special);
    uint32_t llama_n_ctx(const struct llama_context * ctx);
    uint32_t llama_n_batch(const struct llama_context * ctx);
    llama_memory_t llama_get_memory(const struct llama_context * ctx);
    void llama_memory_clear(llama_memory_t mem, bool data);
    void llama_memory_seq_cp(
        llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
        llama_pos p0, llama_pos p1);
    bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
    int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch);
    void llama_synchronize(struct llama_context * ctx);
    float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);
    void ggml_backend_load_all_from_path(const char * dir_path);
    size_t ggml_backend_dev_count(void);
    ggml_backend_dev_t ggml_backend_dev_get(size_t index);
    const char * ggml_backend_dev_name(ggml_backend_dev_t device);
    const char * ggml_backend_dev_description(ggml_backend_dev_t device);
    void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);
    enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device);
    ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device);
    const char * ggml_backend_reg_name(ggml_backend_reg_t reg);
"""

# C type -> ctypes type, for what the binding reads or writes. Every other pointer, callback and
# handle is only ever NULL or passed back untouched, so it is a plain c_void_p.
TYPED = {
    "void": None,
    "bool": c_bool,
    "int8_t": c_int8,
    "int32_t": c_int32,
    "uint32_t": c_uint32,
    "size_t": c_size_t,
    "float": c_float,
    "llama_token": c_int32,
    "llama_pos": c_int32,
    "llama_seq_id": c_int32,
    "int8_t *": POINTER(c_int8),
    "int32_t *": POINTER(c_int32),
    "float *": POINTER(c_float),
    "size_t *": POINTER(c_size_t),
    "llama_token *": POINTER(c_int32),
    "llama_pos *": POINTER(c_int32),
    "llama_seq_id **": POINTER(POINTER(c_int32)),
    "ggml_backend_dev_t *": POINTER(c_void_p),
    "const char *": c_char_p,
    "char *": c_char_p,
    "ggml_log_callback": LOG_CALLBACK,
    "struct llama_model_params": ModelParams,
    "struct llama_context_params": ContextParams,
    "struct llama_batch": Batch,
}


OPAQUE = {"llama_memory_t", "ggml_backend_dev_t", "ggml_backend_reg_t"}  # typedefs of pointers


def c_type_of(c_type):
    if c_type.startswith("enum "):
        return c_int
    if c_type in TYPED:
        return TYPED[c_type]
    assert c_type.endswith(("*", "_callback")) or c_type in OPAQUE, f"unexpected C type {c_type!r}"
    return c_void_p


def header_fields(text):
    """[(name, c type)] of a struct body written one field per line."""
    fields = re.findall(r"^\s*([^;]+?)\s*\b(\w+);$", text, re.MULTILINE)
    return [(name, c_type) for c_type, name in fields]


def header_prototypes(text):
    """[(name, result C type, [argument C types])] of declarations that may span lines."""
    found = []
    for declaration in text.split(";")[:-1]:
        flat = " ".join(declaration.split())
        match = re.fullmatch(r"(?P<result>.+?)\s*\b(?P<name>\w+)\((?P<args>.*)\)", flat)
        assert match, flat
        args = [] if match["args"] == "void" else match["args"].split(",")
        found.append(
            (match["name"], match["result"], [re.sub(r"\s*\b\w+$", "", a.strip()) for a in args])
        )
    return found


def test_header_transcriptions_are_for_the_pinned_commit():
    assert llama_release.COMMIT == HEADER_COMMIT, (
        "the pinned llama.cpp commit moved: check MODEL_PARAMS_H, CONTEXT_PARAMS_H, BATCH_H and "
        "PROTOTYPES against include/llama.h and ggml/include/ggml-backend.h of the new commit"
    )


@pytest.mark.parametrize(
    ("struct", "header", "size"),
    [
        (ModelParams, MODEL_PARAMS_H, 80),
        (ContextParams, CONTEXT_PARAMS_H, 160),
        (Batch, BATCH_H, 56),
    ],
)
def test_struct_layouts_follow_the_header(struct, header, size):
    expected = [(name, c_type_of(c_type)) for name, c_type in header_fields(header)]
    assert list(struct._fields_) == expected
    if ctypes.sizeof(c_void_p) == 8:  # sizes as gcc lays these structs out on a 64-bit target
        assert ctypes.sizeof(struct) == size


def test_signatures_follow_the_header():
    declared = {
        name: (c_type_of(result), [c_type_of(a) for a in arguments])
        for name, result, arguments in header_prototypes(PROTOTYPES)
    }
    assert set(SIGNATURES) == set(declared)
    for name, prototype in declared.items():
        assert SIGNATURES[name] == prototype, name


def test_log_callback_follows_the_header():
    # typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * data)
    assert LOG_CALLBACK._restype_ is None
    assert list(LOG_CALLBACK._argtypes_) == [c_int, c_char_p, c_void_p]


def test_constants_follow_the_header():
    # enum ggml_backend_dev_type: CPU, GPU, IGPU, ACCEL, META
    assert llama_cpp.DEVICE_KINDS == {0: "cpu", 1: "gpu", 2: "igpu", 3: "accel", 4: "meta"}
    # enum ggml_type: GGML_TYPE_F16 = 1, GGML_TYPE_Q4_0 = 2, GGML_TYPE_Q8_0 = 8
    assert llama_cpp.KV_TYPES == {"f16": 1, "q8_0": 8, "q4_0": 2}
    # LLAMA_SPLIT_MODE_NONE = 0; GGML_LOG_LEVEL_WARN = 3, GGML_LOG_LEVEL_ERROR = 4
    assert llama_cpp.SPLIT_MODE_NONE == 0
    assert (llama_cpp.LOG_LEVEL_WARN, llama_cpp.LOG_LEVEL_ERROR) == (LOG_WARN, LOG_ERROR)
    assert llama_cpp.LOG_ENV == "RIZZO_LLAMA_LOG"
    assert llama_cpp.GPU_LAYERS == 999  # more layers than any model has: offload everything


# --- devices --------------------------------------------------------------------------------------


def device(name, kind, total, backend="Vulkan", description="Some device") -> Device:
    return Device(1, name, description, kind, backend, total)


def test_public_description_leaves_the_handle_out():
    found = Device(0x5001, "CUDA0", "NVIDIA GeForce RTX 5060 Ti", "gpu", "CUDA", 16 * GIB)
    assert found.public() == {
        "name": "CUDA0",
        "description": "NVIDIA GeForce RTX 5060 Ti",
        "kind": "gpu",
        "backend": "CUDA",
        "total_bytes": 16 * GIB,
    }


def test_auto_takes_a_discrete_gpu_before_a_larger_integrated_one():
    cpu = device("CPU", "cpu", 64 * GIB, "CPU")
    igpu = device("Vulkan0", "igpu", 48 * GIB)
    small = device("Vulkan1", "gpu", 8 * GIB)
    big = device("CUDA0", "gpu", 24 * GIB, "CUDA")
    assert choose_device([cpu, igpu, small, big], "auto") is big  # most memory among discrete
    assert choose_device([big, small, igpu, cpu], "auto") is big  # whatever the order
    assert choose_device([cpu, igpu, small], "auto") is small  # discrete beats more memory
    other = device("Vulkan2", "igpu", 16 * GIB)
    assert choose_device([cpu, other, igpu], "auto") is igpu  # most memory among integrated
    assert choose_device([other, igpu], "gpu") is igpu


def test_only_gpus_are_candidates():
    cpu = device("CPU", "cpu", 64 * GIB, "CPU")
    blas = device("BLAS", "accel", 96 * GIB, "BLAS")
    meta = device("Meta", "meta", 96 * GIB, "Meta")
    assert choose_device([cpu, blas, meta], "auto") is None
    for wanted in ("gpu", "blas", "meta"):
        with pytest.raises(ValueError, match="no such GPU"):
            choose_device([cpu, blas, meta], wanted)
    assert choose_device([cpu, device("CUDA0", "gpu", GIB, "CUDA")], "cpu") is None


def test_a_request_that_cannot_be_met_lists_the_devices_and_never_falls_back():
    cpu = device("CPU", "cpu", 64 * GIB, "CPU", "AMD Ryzen 9")
    igpu = device("Vulkan0", "igpu", 8 * GIB, "Vulkan", "Intel(R) UHD Graphics")
    message = (
        "--device cuda: no such GPU in this llama.cpp runtime. Devices: CPU (AMD Ryzen 9), "
        "Vulkan0 (Intel(R) UHD Graphics). "
        "Install another runtime with `rizzo download --only runtime --runtime ...`."
    )
    with pytest.raises(ValueError, match=exactly(message)):
        choose_device([cpu, igpu], "cuda")
    with pytest.raises(ValueError, match=re.escape("--device gpu: no such GPU")):
        choose_device([cpu], "gpu")
    with pytest.raises(ValueError, match=re.escape("Devices: none.")):
        choose_device([], "gpu")


@pytest.mark.parametrize(
    ("wanted", "matches"),
    [
        ("backend", True),  # the registry name of the backend
        ("BACKEND", True),  # in any case
        ("xyz0", True),  # the device name
        ("Xyz0", True),
        ("xyz", True),  # a name without its number reaches the numbered devices of the family
        ("xy", False),  # ... but only when what is left is a number
        ("xyz0x", False),
        ("xyz1", False),
        ("shiny", True),  # text in the description
        ("SHINY GRAPHICS", True),
        ("other", False),
    ],
)
def test_a_device_matches_by_backend_name_numbered_name_or_description(wanted, matches):
    found = device("Xyz0", "gpu", GIB, "Backend", "A Shiny Graphics card")
    try:
        chosen = choose_device([found], wanted)
    except ValueError:
        chosen = None
    assert (chosen is found) == matches


def test_family_aliases_reach_the_spelling_ggml_registers_the_backend_under():
    metal = device("MTL0", "gpu", 16 * GIB, "MTL", "Apple M4")
    hip = device("HIP0", "gpu", 24 * GIB, "HIP", "AMD Radeon RX 7900 XTX")
    cuda = device("CUDA0", "gpu", 16 * GIB, "CUDA")
    assert choose_device([cuda, metal], "metal") is metal
    assert choose_device([cuda, hip], "rocm") is hip
    assert choose_device([cuda, hip], "ROCm") is hip
    assert choose_device([cuda, hip], "hip") is hip
    for machine, wanted in (([cuda, metal], "rocm"), ([cuda, hip], "metal")):
        with pytest.raises(ValueError, match="no such GPU"):  # an alias never widens a family
            choose_device(machine, wanted)


DEVICES = st.lists(
    st.builds(
        Device,
        handle=st.integers(1, 99),
        name=st.sampled_from(["CPU", "CUDA0", "Vulkan0", "Vulkan1"]),
        description=st.sampled_from(["a", "b"]),
        kind=st.sampled_from(["cpu", "gpu", "igpu", "accel", "meta"]),
        backend=st.sampled_from(["CPU", "CUDA", "Vulkan"]),
        total_bytes=st.integers(0, 64 * GIB),
    ),
    max_size=6,
)


@PROPERTY
@given(devices=DEVICES)
def test_auto_and_gpu_choose_the_discrete_gpu_with_most_memory(devices):
    gpus = [d for d in devices if d.kind in ("gpu", "igpu")]
    discrete = [d for d in gpus if d.kind == "gpu"]
    best = max(discrete or gpus, key=lambda d: d.total_bytes, default=None)
    assert choose_device(devices, "auto") is best
    assert choose_device(devices, "cpu") is None
    if gpus:
        assert choose_device(devices, "gpu") is best
    else:
        with pytest.raises(ValueError, match="no such GPU"):
            choose_device(devices, "gpu")


# --- the library ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("platform", "library"),
    [("linux", "libllama.so"), ("darwin", "libllama.dylib"), ("win32", "llama.dll")],
)
def test_a_directory_without_libllama_is_refused_by_the_name_of_the_platform(
    tmp_path, monkeypatch, platform, library
):
    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(ValueError, match=exactly(f"{library} not found in {tmp_path.resolve()}")):
        Library.open(tmp_path)
    assert Library._loaded == {}


def test_every_symbol_is_bound_with_its_prototype(runtime):
    library = Library(runtime.directory)
    for name, (result, arguments) in SIGNATURES.items():
        function = runtime.functions[name]
        assert function.restype is result, name
        assert function.argtypes == arguments, name
        assert getattr(library, name) is function, name


def test_symbols_are_taken_from_libllama_before_the_ggml_libraries(runtime):
    decoy = Function("decoy", lambda *args: None, runtime.journal)
    runtime.libraries["ggml"].exports["llama_free"] = decoy
    runtime.libraries["ggml-base"].exports["llama_free"] = decoy
    runtime.libraries["ggml-base"].exports["ggml_backend_dev_count"] = decoy
    library = Library(runtime.directory)
    assert library.llama_free is runtime.functions["llama_free"]
    assert library.ggml_backend_dev_count is runtime.functions["ggml_backend_dev_count"]
    assert decoy.argtypes is None  # never bound


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


def test_a_library_that_cannot_be_opened_is_not_kept(runtime):
    runtime.unloadable = {"libggml.so"}
    with pytest.raises(OSError, match=re.escape("libggml.so: cannot open")):
        Library.open(runtime.directory)
    assert Library._loaded == {}


def test_only_the_libraries_that_exist_are_opened(runtime):
    (runtime.directory / "libggml.so").unlink()
    (runtime.directory / "libggml-base.so").unlink()
    with pytest.raises(ValueError, match="symbol ggml_backend_load_all_from_path is missing"):
        Library(runtime.directory)
    assert [path.name for path, _ in runtime.loads] == ["libllama.so"]


def test_logging_is_routed_before_the_backends_are_loaded_and_initialized(runtime):
    library = Library(runtime.directory)
    assert [name for name, _ in runtime.journal] == [
        "llama_log_set",
        "ggml_backend_load_all_from_path",
        "llama_backend_init",
    ]
    ((callback, user_data),) = runtime.calls("llama_log_set")
    assert callback() is library._log
    assert user_data is None
    directory = str(runtime.directory.resolve()).encode()
    assert runtime.calls("ggml_backend_load_all_from_path") == [(directory,)]
    assert runtime.calls("llama_backend_init") == [()]


def test_the_log_callback_lives_as_long_as_the_library(runtime, capsys):
    library = Library(runtime.directory)
    gc.collect()
    callback = runtime.log_callback  # a weak reference: llama.cpp holds a bare pointer
    assert callback() is library._log
    runtime.log(LOG_ERROR, b"still there\n")
    assert capsys.readouterr().err == "still there\n"
    del library
    gc.collect()
    assert callback() is None  # nothing but the library kept it


SWA_NOTICE = (
    b"llama_kv_cache_iswa: using full-size SWA cache "
    b"(ref: https://github.com/ggml-org/llama.cpp/pull/13194)\n"
)


@pytest.mark.parametrize(
    ("level", "text", "shown"),
    [
        (LOG_NONE, b"none\n", False),
        (LOG_DEBUG, b"debug\n", False),
        (LOG_INFO, b"info\n", False),
        (LOG_WARN, b"warning\n", True),
        (LOG_ERROR, b"error\n", True),
        (LOG_CONT, b"continued\n", False),
        (LOG_WARN, SWA_NOTICE, False),  # the full-size window cache is our own choice, not news
    ],
)
def test_only_warnings_and_errors_reach_stderr(runtime, library, capsys, level, text, shown):
    runtime.log(level, text)
    captured = capsys.readouterr()
    assert captured.err == (text.decode() if shown else "")
    assert captured.out == ""


def test_undecodable_bytes_in_a_log_line_are_replaced(runtime, library, capsys):
    runtime.log(LOG_ERROR, b"bad \xff byte\n")
    assert capsys.readouterr().err == "bad \ufffd byte\n"


@pytest.mark.parametrize("level", [LOG_NONE, LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CONT])
def test_the_environment_shows_everything_llama_cpp_says(runtime, monkeypatch, capsys, level):
    monkeypatch.setenv("RIZZO_LLAMA_LOG", "1")
    library = Library(runtime.directory)
    runtime.log(level, b"line\n")
    runtime.log(LOG_WARN, SWA_NOTICE)
    assert capsys.readouterr().err == "line\n" + SWA_NOTICE.decode()
    assert runtime.log_callback() is library._log


@pytest.mark.parametrize("value", ["0", "", "true", "yes", "11"])
def test_only_the_value_1_turns_the_full_log_on(runtime, monkeypatch, capsys, value):
    monkeypatch.setenv("RIZZO_LLAMA_LOG", value)
    library = Library(runtime.directory)
    runtime.log(LOG_INFO, b"info\n")
    assert capsys.readouterr().err == ""
    assert runtime.log_callback() is library._log


def test_open_keeps_one_library_per_directory(runtime, tmp_path):
    first = Library.open(runtime.directory)
    same_folder = runtime.directory / ".." / "runtime"
    assert Library.open(same_folder) is first
    assert len(runtime.loads) == 3  # the libraries were not opened again
    assert runtime.calls("llama_backend_init") == [()]
    elsewhere = tmp_path / "second"
    runtime.lay_out(elsewhere)
    second = Library.open(elsewhere)
    assert second is not first
    assert Library.open(elsewhere) is second
    assert second.directory == elsewhere.resolve()
    assert len(runtime.loads) == 6


def test_open_without_a_directory_uses_the_installed_runtime(runtime, monkeypatch):
    monkeypatch.setattr(llama_release, "locate", lambda family=None: runtime.directory)
    library = Library.open()
    assert library.directory == runtime.directory.resolve()
    assert Library.open(None) is library
    assert Session.load(runtime.gguf).library is library


def test_a_library_resolves_the_directory_it_is_given(runtime):
    library = Library(runtime.directory / ".." / "runtime")
    assert library.directory == runtime.directory.resolve()


def test_open_does_not_look_for_a_runtime_when_it_is_given_one(runtime, monkeypatch):
    monkeypatch.setattr(llama_release, "locate", lambda family=None: pytest.fail("locate()"))
    assert Library.open(runtime.directory).directory == runtime.directory.resolve()


def test_linux_loads_the_cuda_runtime_beside_libllama_first(runtime):
    for name in (
        "libcudart.so.13.0.96",
        "libcudart.so.13",
        "libcublasLt.so.13",
        "libcublas.so.13",
        "libcudnn.so.9",
        "cublas64_13.dll",
    ):
        (runtime.directory / name).touch()
    Library(runtime.directory)
    # One file per pattern (the first by name), Lt before the library that needs it, then ggml
    # from the bottom up: what llama depends on is already there when it is opened.
    assert [(path.name, mode) for path, mode in runtime.loads] == [
        ("libcudart.so.13", ctypes.RTLD_GLOBAL),
        ("libcublasLt.so.13", ctypes.RTLD_GLOBAL),
        ("libcublas.so.13", ctypes.RTLD_GLOBAL),
        ("libggml-base.so", ctypes.RTLD_GLOBAL),
        ("libggml.so", ctypes.RTLD_GLOBAL),
        ("libllama.so", ctypes.RTLD_GLOBAL),
    ]
    assert {path.parent for path, _ in runtime.loads} == {runtime.directory.resolve()}


def test_macos_opens_dylibs_and_leaves_the_dll_search_path_alone(tmp_path, monkeypatch):
    native = Native().install(tmp_path, monkeypatch, "darwin")
    monkeypatch.setattr(os, "add_dll_directory", pytest.fail, raising=False)
    Library(native.directory)
    assert [path.name for path, _ in native.loads] == [
        "libggml-base.dylib",
        "libggml.dylib",
        "libllama.dylib",
    ]


def test_windows_registers_the_folder_for_the_dlls_that_depend_on_each_other(tmp_path, monkeypatch):
    native = Native().install(tmp_path, monkeypatch, "win32")
    (native.directory / "libcudart.so.13").touch()  # a Linux name: nothing to preload here
    registered: list[str] = []
    monkeypatch.setattr(os, "add_dll_directory", registered.append, raising=False)
    others = os.pathsep.join(["C:\\Windows", "C:\\Tools"])
    monkeypatch.setenv("PATH", others)
    Library(native.directory)
    folder = str(native.directory.resolve())
    assert registered == [folder]
    assert os.environ["PATH"] == folder + os.pathsep + others  # first, so its DLLs win
    assert [path.name for path, _ in native.loads] == ["ggml-base.dll", "ggml.dll", "llama.dll"]


def test_windows_leaves_a_path_that_already_has_the_folder_as_it_is(tmp_path, monkeypatch):
    native = Native().install(tmp_path, monkeypatch, "win32")
    monkeypatch.setattr(os, "add_dll_directory", lambda folder: None, raising=False)
    present = os.pathsep.join(["C:\\Tools", str(native.directory.resolve()), "C:\\Windows"])
    monkeypatch.setenv("PATH", present)
    Library(native.directory)
    assert os.environ["PATH"] == present


def test_windows_starts_a_path_when_there_is_none(tmp_path, monkeypatch):
    native = Native().install(tmp_path, monkeypatch, "win32")
    monkeypatch.setattr(os, "add_dll_directory", lambda folder: None, raising=False)
    monkeypatch.delenv("PATH", raising=False)
    Library(native.directory)
    assert os.environ["PATH"].strip(os.pathsep) == str(native.directory.resolve())


@pytest.mark.parametrize(
    ("platform", "name"),
    [
        ("win32", "ggml-base.dll"),
        ("darwin", "libggml-base.dylib"),
        ("linux", "libggml-base.so"),
        ("freebsd14", "libggml-base.so"),
    ],
)
def test_shared_libraries_are_named_the_way_the_platform_names_them(
    tmp_path, monkeypatch, platform, name
):
    for candidate in ("ggml-base.dll", "libggml-base.dylib", "libggml-base.so"):
        (tmp_path / candidate).touch()
    monkeypatch.setattr(sys, "platform", platform)
    assert llama_cpp._shared(tmp_path, "ggml-base") == tmp_path / name
    assert llama_cpp._shared(tmp_path, "ggml") is None


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        (0, "cpu"),
        (1, "gpu"),
        (2, "igpu"),
        (3, "accel"),
        (4, "meta"),
        (5, "unknown"),
        (99, "unknown"),
    ],
)
def test_device_kinds_are_named_after_the_ggml_enum(runtime, code, kind):
    runtime.hardware.clear()
    runtime.add_device("X0", "An X", code, "X", GIB)
    assert [d.kind for d in Library(runtime.directory).devices()] == [kind]


def test_devices_are_enumerated_from_the_ggml_registry(runtime):
    runtime.add_device("CUDA1", "NVIDIA GeForce RTX 4090", GPU, "CUDA", 24 * GIB, 20 * GIB)
    runtime.add_device("Vulkan0", "AMD Radeon 780M", IGPU, "Vulkan", 8 * GIB, 7 * GIB)
    library = Library(runtime.directory)
    found = library.devices()
    assert found == [
        Device(runtime.hardware[0].handle, "CPU", "Fake CPU", "cpu", "CPU", 64 * GIB),
        Device(runtime.gpu.handle, "CUDA0", "NVIDIA GeForce RTX 5060 Ti", "gpu", "CUDA", 16 * GIB),
        Device(
            runtime.hardware[2].handle, "CUDA1", "NVIDIA GeForce RTX 4090", "gpu", "CUDA", 24 * GIB
        ),
        Device(runtime.hardware[3].handle, "Vulkan0", "AMD Radeon 780M", "igpu", "Vulkan", 8 * GIB),
    ]
    assert len({d.handle for d in found}) == 4
    assert library.free_bytes(found[1]) == 15 * GIB  # the free memory, not the total
    assert library.free_bytes(found[3]) == 7 * GIB
    runtime.gpu.free = 3 * GIB
    assert library.free_bytes(found[1]) == 3 * GIB  # read when asked


def test_no_devices_before_the_backends_are_loaded(runtime):
    runtime.hardware.clear()
    assert Library(runtime.directory).devices() == []


# --- the session: loading -------------------------------------------------------------------------


def test_load_puts_the_whole_model_on_the_device_auto_picks(runtime):
    runtime.add_device("Vulkan0", "Intel(R) UHD Graphics", IGPU, "Vulkan", 32 * GIB)
    runtime.add_device("Vulkan1", "AMD Radeon RX 6600", GPU, "Vulkan", 8 * GIB, 7 * GIB)
    session = load(runtime)
    (loaded,) = runtime.model_loads
    assert loaded.path == str(runtime.gguf).encode()
    assert loaded.devices == [runtime.gpu.handle]  # one device, never a split
    assert loaded.params.split_mode == 0
    assert loaded.params.n_gpu_layers == 999
    untouched = runtime.llama_model_default_params()
    assert changed(loaded.params, untouched) == {"devices", "split_mode", "n_gpu_layers"}
    expected = Device(
        runtime.gpu.handle, "CUDA0", "NVIDIA GeForce RTX 5060 Ti", "gpu", "CUDA", 16 * GIB
    )
    assert session.device == expected
    assert session.library is Library.open(runtime.directory)


def test_load_on_the_cpu_offloads_nothing(runtime):
    session = load(runtime, device="cpu")
    (loaded,) = runtime.model_loads
    assert loaded.devices == []  # the list holds only its terminator
    assert loaded.params.n_gpu_layers == 0
    assert loaded.params.split_mode == 0
    assert runtime.context_inits[0].offload_kqv is False
    assert session.device is None
    assert session.idle_free is None
    assert session.free_bytes() is None


def test_load_reserves_a_16k_context_for_five_sequences_unless_told_otherwise(runtime):
    load(runtime)
    (params,) = runtime.context_inits
    sizes = (params.n_ctx, params.n_batch, params.n_ubatch, params.n_seq_max)
    assert sizes == (16384, 2048, 512, 5)
    assert params.n_outputs_max == 5


def test_load_sizes_the_context_for_scoring(runtime):
    session = load(runtime, n_ctx=1000, n_batch=2048, n_ubatch=300, n_seq_max=5)
    (params,) = runtime.context_inits
    sizes = (params.n_ctx, params.n_batch, params.n_ubatch, params.n_seq_max)
    assert sizes == (1000, 1000, 300, 5)  # the batch cannot be larger than the context
    assert params.n_outputs_max == 5  # logits at one position per sequence
    assert params.kv_unified is True  # one buffer: branching the prefix shares its cells
    assert params.swa_full is True  # the branches of a long prefix still find free cells
    assert params.no_perf is True
    assert params.offload_kqv is True  # a device was chosen
    assert changed(params, runtime.llama_context_default_params()) == {
        "n_ctx",
        "n_batch",
        "n_ubatch",
        "n_seq_max",
        "n_outputs_max",
        "kv_unified",
        "swa_full",
        "no_perf",
    }
    assert (session.n_ctx, session.n_batch) == (1024, 1000)  # n_ctx: padded


def test_the_session_reports_what_llama_cpp_granted_not_what_was_asked(runtime):
    runtime.granted = {"n_ctx": 4096, "n_batch": 512}
    session = load(runtime, n_ctx=1000, n_batch=700)
    assert (session.n_ctx, session.n_batch) == (4096, 512)


@pytest.mark.parametrize(
    ("n_ctx", "n_batch", "n_ubatch", "expected"),
    [
        (1000, 2048, 300, (1000, 300)),  # the batch is cut to the context
        (1000, 400, 512, (400, 400)),  # the micro-batch is cut to the batch
        (300, 2048, 512, (300, 300)),  # both
        (5000, 2048, 512, (2048, 512)),  # neither
    ],
)
def test_batch_sizes_are_cut_to_what_fits(runtime, n_ctx, n_batch, n_ubatch, expected):
    load(runtime, n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch)
    params = runtime.context_inits[-1]
    assert (params.n_batch, params.n_ubatch) == expected


@PROPERTY
@given(n_ctx=st.integers(1, 5000), n_batch=st.integers(1, 5000), n_ubatch=st.integers(1, 5000))
def test_a_batch_never_exceeds_the_context_nor_a_micro_batch_the_batch(
    runtime, n_ctx, n_batch, n_ubatch
):
    session = load(runtime, n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch)
    params = runtime.context_inits[-1]
    assert params.n_ctx == n_ctx
    assert params.n_batch == min(n_batch, n_ctx)
    assert params.n_ubatch == min(n_ubatch, n_batch, n_ctx)
    session.close()


@pytest.mark.parametrize(("kv_type", "code"), [("f16", 1), ("q8_0", 8), ("q4_0", 2)])
def test_the_kv_cache_type_reaches_both_halves_of_the_cache(runtime, kv_type, code):
    load(runtime, kv_type=kv_type)
    params = runtime.context_inits[-1]
    assert (params.type_k, params.type_v) == (code, code)


@pytest.mark.parametrize("kv_type", [None, ""])
def test_no_kv_cache_type_keeps_the_default_of_llama_cpp(runtime, kv_type):
    load(runtime, kv_type=kv_type)
    params = runtime.context_inits[-1]
    assert (params.type_k, params.type_v) == (1, 1)  # F16, as the fake's defaults say


@pytest.mark.parametrize(
    ("threads", "expected"),
    [(6, (6, 6)), (None, (4, 8)), (0, (4, 8))],  # what is not asked for stays llama.cpp's default
)
def test_threads_set_both_pools_only_when_asked(runtime, threads, expected):
    load(runtime, threads=threads)
    params = runtime.context_inits[-1]
    assert (params.n_threads, params.n_threads_batch) == expected


def test_free_memory_is_read_before_the_weights_take_it(runtime):
    session = load(runtime)
    assert session.idle_free == 15 * GIB
    assert session.free_bytes() == 15 * GIB - MODEL_BYTES
    runtime.gpu.free = 2 * GIB
    assert session.free_bytes() == 2 * GIB  # read when asked


def test_a_model_that_does_not_load_is_reported(runtime):
    runtime.model_ok = False
    with pytest.raises(ValueError, match=exactly(f"llama.cpp cannot load {runtime.gguf}")):
        load(runtime)
    assert runtime.context_inits == []
    assert runtime.calls("llama_model_free") == []


def test_a_context_that_cannot_be_created_frees_the_model_and_names_the_size(runtime):
    runtime.context_ok = False
    message = "llama.cpp cannot create a 5000-token context; lower --ctx or --batch-size"
    with pytest.raises(ValueError, match=exactly(message)):
        load(runtime, n_ctx=5000)
    assert runtime.calls("llama_model_free") == [(runtime.MODEL,)]
    assert runtime.model_alive is False


def test_a_device_that_does_not_exist_is_refused_before_anything_is_loaded(runtime):
    with pytest.raises(ValueError, match="--device rocm: no such GPU"):
        load(runtime, device="rocm")
    assert runtime.model_loads == []


def test_close_frees_the_context_then_the_model_and_only_once(runtime):
    session = load(runtime)
    session.close()
    freed = [name for name, _ in runtime.journal if name in ("llama_free", "llama_model_free")]
    assert freed == ["llama_free", "llama_model_free"]
    assert runtime.calls("llama_free") == [(runtime.CONTEXT,)]
    assert runtime.calls("llama_model_free") == [(runtime.MODEL,)]
    assert (session.context, session.model, session.vocab, session.memory) == (None,) * 4
    session.close()
    assert len(runtime.calls("llama_free")) == len(runtime.calls("llama_model_free")) == 1


# --- the session: text and model information ----------------------------------------------------


def test_tokenize_hands_over_utf8_bytes_and_lets_control_tokens_be_parsed(runtime):
    session = load(runtime)
    text = "héllo <|Bot|>"  # 13 characters, 14 bytes
    assert session.tokenize(text) == list(text.encode())
    vocab, raw, length, _, _, add_special, _ = runtime.calls("llama_tokenize")[0]
    assert (vocab, raw, length, add_special) == (runtime.VOCAB, text.encode(), 14, False)
    assert session.tokenize("ab", add_special=True) == [runtime.BOS, 97, 98]
    assert session.tokenize("") == []
    calls = runtime.calls("llama_tokenize")
    assert [call[5] for call in calls] == [False, True, False]
    assert [call[6] for call in calls] == [True] * 3  # control tokens in the text are parsed


def test_tokenize_has_room_for_the_bytes_and_eight_more_tokens(runtime):
    session = load(runtime)
    runtime.extra_tokens = 8
    assert len(session.tokenize("abc")) == 3 + 8
    runtime.extra_tokens = 9
    with pytest.raises(ValueError, match=exactly("llama_tokenize: buffer too small")):
        session.tokenize("abc")


@PROPERTY
@given(text=st.text(max_size=200), add_special=st.booleans())
def test_tokenize_gives_back_every_token_the_vocabulary_made(runtime, text, add_special):
    session = load(runtime)
    expected = list(text.encode("utf-8"))
    if add_special:
        expected.insert(0, runtime.BOS)
    assert session.tokenize(text, add_special) == expected
    session.close()


def test_meta_reads_a_string_value_or_none(runtime):
    runtime.metadata = {
        b"general.architecture": b"spark2_5",
        b"general.name": "Città".encode(),
        b"empty": b"",
    }
    session = load(runtime)
    assert session.meta("general.architecture") == "spark2_5"
    assert session.meta("general.name") == "Città"
    assert session.meta("empty") == ""  # present, so not None
    assert session.meta("missing") is None
    model, key, _, size = runtime.calls("llama_model_meta_val_str")[0]
    assert (model, key, size) == (runtime.MODEL, b"general.architecture", 1024)


def test_chat_template_is_the_default_one_or_none(runtime):
    session = load(runtime)
    assert session.chat_template() is None
    runtime.template = "{{ messages }} · é".encode()
    assert session.chat_template() == "{{ messages }} · é"
    assert runtime.calls("llama_model_chat_template") == [(runtime.MODEL, None)] * 2


@pytest.mark.parametrize(
    ("pad", "eos", "expected"),
    [(-1, 7, (None, 7)), (3, -1, (3, None)), (0, 0, (0, 0)), (151643, 151645, (151643, 151645))],
)
def test_special_token_ids_are_none_when_the_vocabulary_has_none(runtime, pad, eos, expected):
    runtime.pad, runtime.eos = pad, eos
    session = load(runtime)
    assert (session.pad_token, session.eos_token) == expected


# --- the session: compute -----------------------------------------------------------------------


def test_decode_lays_the_batch_out_as_llama_h_describes_it(runtime):
    session = load(runtime, n_batch=8)
    session.decode([11, 12, 13], [4, 5, 6], [0, 1, 1], outputs=[2, 0])
    (seen,) = runtime.decodes
    assert seen.tokens == [11, 12, 13]
    assert seen.positions == [4, 5, 6]
    assert seen.sequences == [0, 1, 1]
    assert seen.n_seq_ids == [1, 1, 1]  # every token belongs to one sequence
    assert seen.outputs == [0, 2]
    assert seen.embeddings is False
    assert runtime.calls("llama_decode")[0][0] == runtime.CONTEXT


def test_decode_asks_for_no_logits_unless_told_where(runtime):
    session = load(runtime)
    assert session.decode([1, 2], [0, 1], [0, 0]) is None
    assert runtime.decodes[0].outputs == []


@PROPERTY
@given(data=st.data())
def test_decode_hands_the_native_side_exactly_what_it_was_given(runtime, data):
    count = data.draw(st.integers(1, 32))
    int32 = st.integers(0, 2**31 - 1)
    tokens = data.draw(st.lists(int32, min_size=count, max_size=count))
    positions = data.draw(st.lists(int32, min_size=count, max_size=count))
    sequences = data.draw(st.lists(st.integers(0, 255), min_size=count, max_size=count))
    outputs = data.draw(st.lists(st.integers(0, count - 1), unique=True))
    runtime.decodes.clear()
    session = load(runtime, n_batch=32)
    session.decode(tokens, positions, sequences, outputs)
    (seen,) = runtime.decodes
    assert seen.tokens == tokens
    assert seen.positions == positions
    assert seen.sequences == sequences
    assert seen.n_seq_ids == [1] * count
    assert seen.outputs == sorted(outputs)
    session.close()


def test_decode_refuses_an_empty_or_oversized_batch_before_calling_native(runtime):
    session = load(runtime, n_batch=4)
    with pytest.raises(ValueError, match=exactly("Batch of 0 tokens; the limit is 4")):
        session.decode([], [], [])
    with pytest.raises(ValueError, match=exactly("Batch of 5 tokens; the limit is 4")):
        session.decode([1] * 5, list(range(5)), [0] * 5)
    assert runtime.calls("llama_decode") == []
    session.decode([1] * 4, list(range(4)), [0] * 4)  # exactly the limit is fine
    assert len(runtime.calls("llama_decode")) == 1


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (1, "the context is full"),
        (2, "compute error"),
        (-1, "compute error"),
        (-3, "compute error"),
    ],
)
def test_a_failed_decode_says_what_llama_cpp_answered(runtime, status, reason):
    runtime.decode_status = status
    session = load(runtime)
    with pytest.raises(ValueError, match=exactly(f"llama_decode returned {status}: {reason}")):
        session.decode([1], [0], [0])


def test_logits_are_the_requested_rows_of_the_output_at_that_batch_position(runtime):
    session = load(runtime)
    session.decode([5, 6, 7], [0, 1, 2], [0, 0, 0], outputs=[1, 2])
    assert session.logits(1, [0, 3, 299]) == [1000.0, 1003.0, 1299.0]
    assert session.logits(2, [4]) == [2004.0]
    assert session.logits(2, [7, 1, 7]) == [2007.0, 2001.0, 2007.0]  # in the order asked
    assert session.logits(1, []) == []
    assert all(type(value) is float for value in session.logits(1, [0, 1]))
    assert runtime.calls("llama_get_logits_ith")[0] == (runtime.CONTEXT, 1)


def test_logits_where_none_were_produced_are_an_error(runtime):
    session = load(runtime)
    session.decode([5, 6], [0, 1], [0, 0], outputs=[1])
    with pytest.raises(ValueError, match=exactly("No logits were produced at batch position 0")):
        session.logits(0, [1])
    runtime.decode_status = 2
    with pytest.raises(ValueError, match="llama_decode returned 2"):
        session.decode([5], [0], [0], outputs=[0])
    with pytest.raises(ValueError, match=exactly("No logits were produced at batch position 0")):
        session.logits(0, [1])  # a failed decode leaves no row behind


@PROPERTY
@given(
    row=st.lists(st.floats(width=32, allow_nan=False), min_size=8, max_size=8),
    slots=st.lists(st.integers(0, 7), max_size=12),
)
def test_logits_are_those_entries_of_the_row_in_that_order(runtime, row, slots):
    session = load(runtime)
    runtime.rows = {4: row}
    assert session.logits(4, slots) == [row[slot] for slot in slots]
    session.close()


def test_the_memory_operations_address_whole_sequences(runtime):
    session = load(runtime)
    session.clear()
    session.branch(2, 5)
    session.drop(5)
    session.synchronize()
    assert runtime.calls("llama_memory_clear") == [(runtime.MEMORY, True)]  # data, not just cells
    assert runtime.calls("llama_memory_seq_cp") == [(runtime.MEMORY, 2, 5, -1, -1)]
    assert runtime.calls("llama_memory_seq_rm") == [(runtime.MEMORY, 5, -1, -1)]
    assert runtime.calls("llama_synchronize") == [(runtime.CONTEXT,)]


# --- regressions: defects found while writing these tests, now fixed ---------------------------


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
