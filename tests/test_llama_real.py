"""Real llama.cpp runtime and real GGUF weights. Opt in: RIZZO_REAL=1 pytest -m integration.

Needs `rizzo download` (any size). Never part of the default suite, which loads no weights.
"""

import json
import os
import string
from pathlib import Path

import pytest

from rizzo_flow import llama_release
from rizzo_flow.config import GGUF, MODELS

pytestmark = pytest.mark.integration


def smallest():
    """The smallest Q8_0 GGUF on disk, or None."""
    files = [spec.path for spec in GGUF.values() if spec.quant == "q8_0" and spec.path.is_file()]
    return min(files, key=lambda path: path.stat().st_size) if files else None


@pytest.fixture(scope="module")
def backend():
    if os.environ.get("RIZZO_REAL") != "1":
        pytest.skip("set RIZZO_REAL=1 to load real weights")
    if smallest() is None or not llama_release.installed():
        pytest.skip("run `rizzo download` first")
    from rizzo_flow.backend_llama import LlamaBackend

    loaded = LlamaBackend.load(smallest(), ctx=4096)
    yield loaded
    loaded.session.close()


def test_identity_is_pinned(backend):
    meta = backend.metadata
    assert meta["runtime"] == "llama.cpp"
    assert meta["llama_cpp_release"] == llama_release.RELEASE
    assert meta["gguf_source"]
    assert meta["precision"] == "q8_0"
    assert meta["source"] in {spec.repo for spec in MODELS.values()}


def test_prompt_matches_the_original_checkpoint(backend):
    """Same template text and same token ids as the Hugging Face files, when they are here."""
    size = next(s for s, spec in MODELS.items() if spec.repo == backend.metadata["source"])
    original = Path(MODELS[size].path)
    if not (original / "chat_template.jinja").is_file():
        pytest.skip("original checkpoint not downloaded")
    from rizzo_flow.backend_llama import LlamaTokenizer

    # The template text may differ by a comment (1.7B); what it renders may not.
    theirs = LlamaTokenizer(
        backend.session, (original / "chat_template.jinja").read_text(encoding="utf-8")
    )
    tokenizers = pytest.importorskip("tokenizers")
    reference = tokenizers.Tokenizer.from_file(str(original / "tokenizer.json"))
    messages = [
        {"role": "system", "content": "You are a precise decision function."},
        {
            "role": "user",
            "content": "<evidence>\nIl cliente scrive: «non riesco ad accedere» 🙁\n</evidence>",
        },
    ]
    text = backend.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert text.endswith("<|Bot|></think>")
    assert text == theirs.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert backend.tokenizer.encode(text) == reference.encode(text, add_special_tokens=False).ids


def test_answer_letters_are_single_tokens(backend):
    ids = [backend.tokenizer.encode(letter) for letter in string.ascii_uppercase]
    assert all(len(found) == 1 for found in ids)
    assert len({found[0] for found in ids}) == 26


def test_shared_prefix_agrees_with_direct(backend):
    from rizzo_flow.engine import Engine

    engine = Engine(backend, ctx=4096)
    request = json.loads(Path("examples/ticket.json").read_text(encoding="utf-8"))
    shared = engine.decide({**request, "mode": "shared"})
    direct = engine.decide({**request, "mode": "direct"})
    assert shared["timing"]["shared_prefix_tokens"] > 0
    assert shared["timing"]["generated_tokens"] == 0
    for key, answer in shared["answers"].items():
        other = direct["answers"][key]["probabilities"]
        assert max(answer["probabilities"], key=answer["probabilities"].get) == max(
            other, key=other.get
        )
        assert max(abs(answer["probabilities"][o] - other[o]) for o in other) < 0.05


@pytest.mark.parametrize("kv_type", ["q8_0", "q4_0"])
def test_quantized_kv_cache_loads_and_is_fingerprinted(backend, kv_type):
    """A quantized V cache needs flash attention in llama.cpp: the context must still build,
    decide the ticket like the F16 cache does, and carry its own fingerprint."""
    from rizzo_flow.backend_llama import LlamaBackend
    from rizzo_flow.engine import Engine

    quantized = LlamaBackend.load(smallest(), ctx=4096, kv_type=kv_type)
    try:
        assert quantized.metadata["kv_cache"] == kv_type
        assert quantized.metadata["fingerprint"] != backend.metadata["fingerprint"]
        request = json.loads(Path("examples/ticket.json").read_text(encoding="utf-8"))
        ours = Engine(quantized, ctx=4096).decide(request)["answers"]
        reference = Engine(backend, ctx=4096).decide(request)["answers"]
        for key, answer in ours.items():
            probabilities, other = answer["probabilities"], reference[key]["probabilities"]
            assert max(probabilities, key=probabilities.get) == max(other, key=other.get)
    finally:
        quantized.session.close()
