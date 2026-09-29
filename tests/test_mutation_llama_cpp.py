"""Mutation tests of `rizzo_flow.llama_cpp`."""

import ctypes
import sys
from pathlib import Path

import pytest
from test_llama_cpp import Native
from test_mutation_support import exactly

from rizzo_flow import llama_cpp, llama_release
from rizzo_flow.llama_cpp import Library, Session


class RecordingLibrary:
    """The symbols `Session` calls to build itself and to run a batch; a batch is kept as flags."""

    def __init__(self, count=0):
        self.flags = []
        self.count = count  # what the sizing calls answer: a length, or minus the length needed

    def llama_model_get_vocab(self, model):
        return 0x3000

    def llama_get_memory(self, context):
        return 0x4000

    def llama_decode(self, context, batch):
        self.flags.append([batch.logits[i] for i in range(batch.n_tokens)])
        return 0

    def llama_tokenize(self, vocab, text, length, tokens, capacity, add_special, parse_special):
        return self.count


def session_of(library):
    return Session(library, 0x1000, 0x2000, None, 64, 64, None)


def test_a_batch_asks_for_logits_with_a_one_and_for_nothing_else_with_a_zero():
    """`Session.decode` writing another value than 1 in the flag of an output position.

    llama.cpp reads any non-zero value as "output wanted", so the run itself could not tell.
    """
    library = RecordingLibrary()
    session = session_of(library)
    session.decode([5, 6, 7, 8], [0, 1, 2, 3], [0, 0, 0, 0], outputs=(1, 3))
    session.decode([5, 6], [0, 1], [0, 0])
    assert library.flags == [[0, 1, 0, 1], [0, 0]]


def test_a_tokenizer_that_needs_a_single_token_more_room_is_an_error():
    """`Session.tokenize` refusing only what needs two tokens more (`count < -1`): the count
    llama.cpp returns for a buffer that is too small is minus the number it needs."""
    with pytest.raises(ValueError, match=exactly("llama_tokenize: buffer too small")):
        session_of(RecordingLibrary(count=-1)).tokenize("x")
    assert session_of(RecordingLibrary(count=0)).tokenize("x") == []


def test_the_libraries_are_loaded_by_name_from_the_folder_whatever_order_it_lists_them_in(
    tmp_path, monkeypatch
):
    """`Library._open` loading the first match of each CUDA pattern in listing order instead of
    by name, or giving `ctypes.CDLL` a path where the name was text."""
    monkeypatch.setattr(sys, "platform", "linux")
    listing = {  # the newer version first, as a directory may list it
        "libcudart.so*": ["libcudart.so.13", "libcudart.so.12"],
        "libcublasLt.so*": ["libcublasLt.so.13", "libcublasLt.so.12"],
        "libcublas.so*": ["libcublas.so.13", "libcublas.so.12"],
    }
    monkeypatch.setattr(Path, "glob", lambda self, pattern: [self / n for n in listing[pattern]])
    for stem in ("ggml-base", "ggml", "llama"):
        (tmp_path / f"lib{stem}.so").touch()
    loaded = []

    def cdll(path, mode=None):
        loaded.append((path, mode))
        return f"handle of {path}"

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    library = Library.__new__(Library)
    library.directory = tmp_path
    handles = library._open()
    names = [
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libggml-base.so",
        "libggml.so",
        "libllama.so",
    ]
    assert loaded == [(str(tmp_path / name), ctypes.RTLD_GLOBAL) for name in names]
    assert handles == [f"handle of {tmp_path / name}" for name in reversed(names[3:])]


def test_the_runtime_folder_is_the_resolved_one_even_when_the_library_sits_behind_a_link(tmp_path):
    """`_runtime_folder` naming the folder as it found it, through a link, and not resolved."""
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


def test_a_runtime_that_is_not_there_is_reported_by_its_full_path(tmp_path, monkeypatch):
    """`Library` without `.resolve()`: the message would name the folder as it was typed."""
    monkeypatch.chdir(tmp_path)
    message = f"{llama_release.library_name()} not found in {(tmp_path / 'nowhere').resolve()}"
    with pytest.raises(ValueError, match=exactly(message)):
        Library(Path("nowhere"))


def test_a_session_loaded_without_naming_a_device_falls_back_to_the_cpu(tmp_path, monkeypatch):
    """`Session.load` defaulting to a device called "" instead of "auto": on a machine without a
    GPU that is an error, not the CPU."""
    monkeypatch.setattr(Library, "_loaded", {})
    native = Native().install(tmp_path, monkeypatch)
    native.hardware.remove(native.gpu)
    session = Session.load(native.gguf, directory=native.directory)
    assert session.device is None
    assert native.model_loads[-1].devices == []
