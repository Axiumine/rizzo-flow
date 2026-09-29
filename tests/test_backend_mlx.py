"""SparkBackend, the MLX runtime, on tiny random-weight models of the real Spark architecture.

Scoring runs the real model on the CPU. Loading runs against stand-in checkpoint directories and
a fake `spark_mlx_llm.load`, so no weights, tokenizer files, network or GPU are involved."""

import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import re
import weakref
from functools import cache
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from rizzo_flow import backend as backend_module
from rizzo_flow.backend import SparkBackend, branch_cache, quantize_model, selected_logits
from rizzo_flow.compat import model_name
from rizzo_flow.config import MODELS, RUNTIME_REVISION, FlowCheckpoint
from rizzo_flow.engine import Engine
from rizzo_flow.prompts import PROMPT_VERSION, Compiled, compile_request
from rizzo_flow.runtime import import_mlx
from rizzo_flow.schema import Request

pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(
        importlib.util.find_spec("spark_mlx_llm") is None, reason="Install the mlx extra"
    ),
]

SHARD = "model-00001-of-00001.safetensors"
LAYERS = 4
WINDOW = 16  # sliding_window of the tiny model: prefixes and suffixes are chosen around it
CACHE_LIMIT = 256 * 1024**2
STATS = {
    "inference_seconds",
    "prefill_seconds",
    "shared_prefix_tokens",
    "evaluated_tokens_including_padding",
    "logical_input_tokens",
    "batches",
    "generated_tokens",
    "peak_mlx_bytes",
}
PROPERTY = settings(
    max_examples=25,
    deadline=None,  # the first call of a process pays one-off start-up costs
    database=None,  # no .hypothesis directory in the working tree
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


@pytest.fixture(autouse=True)
def on_cpu():
    """Tiny models gain nothing from a GPU: launch overhead, kernel compilation and TF32
    rounding only slow them down or loosen the exact-equivalence tolerances."""
    import mlx.core as mx

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


# -- models, fakes and spies --------------------------------------------------------------------


class FakeTokenizer:
    """All SparkBackend asks of its tokenizer is the token that pads a microbatch."""

    def __init__(self, pad_token_id=0, eos_token_id=1):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class TextTokenizer(FakeTokenizer):
    """Characters as tokens (ASCII fits the tiny vocabulary) and a minimal chat template."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages) + "\nASSISTANT:"


def raw_model(tied=True):
    """A Spark model as a loader returns it: freshly built, lazy, still in training mode."""
    import mlx.core as mx
    from spark_mlx_llm.model import Model, ModelArgs

    mx.random.seed(42)
    return Model(
        ModelArgs(
            model_type="spark2_5",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=LAYERS,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            vocab_size=128,
            sliding_window=WINDOW,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
            rope_parameters={
                "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1},
                "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
            },
            tie_word_embeddings=tied,
        )
    )


def tiny_model(bits=None, *, tied=True, head_bias=False, bf16=False, quant=None):
    """A ready model. `bits` quantizes the way `load` does; `quant` is a (mode, group size, bits)
    triple for any other quantization (the floating-point formats have no separate biases);
    `head_bias` gives the untied output projection a bias; `bf16` casts the weights like the
    real loader."""
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_map

    model = raw_model(tied)
    if head_bias:
        model.lm_head = nn.Linear(64, 128, bias=True)
    if bf16:
        model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    if quant:
        nn.quantize(model, group_size=quant[1], bits=quant[2], mode=quant[0])
    elif bits:
        quantize_model(model, bits)
    model.eval()
    mx.eval(model.parameters())
    return model


@cache
def frozen_model():
    """One shared model for the property tests, which never change it."""
    return tiny_model()


def leaf_modules(model):
    from mlx import nn
    from mlx.utils import tree_flatten

    return dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))


def quantized_layers(model):
    """(bits, group size) of every quantized layer, by name."""
    from mlx import nn

    return {
        name: (module.bits, module.group_size)
        for name, module in leaf_modules(model).items()
        if isinstance(module, nn.QuantizedLinear | nn.QuantizedEmbedding)
    }


def max_error(actual, expected):
    """Largest absolute difference between two arrays of the same shape."""
    import mlx.core as mx

    assert actual.shape == expected.shape
    return mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()


def reference(model, tokens, slots):
    """The logits of some vocabulary rows after `tokens`: one uncached pass through the whole
    model, the definition of what scoring has to reproduce."""
    import mlx.core as mx

    return model(mx.array([tokens]))[0, -1, :][mx.array(slots)].tolist()


def assert_scores(model, jobs, result, tolerance=1e-4):
    assert set(result) == {job.id for job in jobs}
    for job in jobs:
        assert result[job.id] == pytest.approx(
            reference(model, job.tokens, job.slots), abs=tolerance
        )


def make_job(name, prefix, suffix, slots=(20, 21, 22)):
    return Compiled(name, [*prefix, *suffix], list(slots), "hash")


@pytest.fixture
def body_calls(monkeypatch):
    """Every call of the transformer body: (token rows, cache offset of every layer before it)."""
    from spark_mlx_llm.model import Spark2_5Model

    calls = []
    original = Spark2_5Model.__call__

    def spy(self, inputs, cache=None, input_embeddings=None):
        offsets = None if cache is None else [c.offset for c in cache]
        calls.append((inputs.tolist(), offsets))
        return original(self, inputs, cache, input_embeddings)

    monkeypatch.setattr(Spark2_5Model, "__call__", spy)
    return calls


@pytest.fixture
def projections(monkeypatch):
    """Every output projection made while scoring: (shape of the hidden rows, slots)."""
    calls = []
    original = backend_module.selected_logits

    def spy(model, hidden, slots):
        calls.append((tuple(hidden.shape), list(slots)))
        return original(model, hidden, slots)

    monkeypatch.setattr(backend_module, "selected_logits", spy)
    return calls


@pytest.fixture
def evals(monkeypatch):
    """Every mx.eval call, arguments included; the evaluation itself still happens."""
    import mlx.core as mx

    calls = []
    real = mx.eval

    def spy(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(mx, "eval", spy)
    return calls


def use_clock(monkeypatch, read):
    monkeypatch.setattr(backend_module, "time", SimpleNamespace(perf_counter=read))


# -- selected projection and quantization -------------------------------------------------------

HEADS: dict[str, dict[str, Any]] = {
    "dense": {},
    "q4": {"bits": 4},
    "q8": {"bits": 8},
    "untied": {"tied": False},
    "untied-q4": {"tied": False, "bits": 4},
    "untied-q8": {"tied": False, "bits": 8},
    "biased": {"tied": False, "head_bias": True},
    "biased-q8": {"tied": False, "head_bias": True, "bits": 8},
    "q4-g32": {"quant": ("affine", 32, 4)},
    "mxfp4": {"quant": ("mxfp4", 32, 4)},
    "mxfp8": {"quant": ("mxfp8", 32, 8)},
    "nvfp4": {"quant": ("nvfp4", 16, 4)},
}
PROBE = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]


@cache
def probed(head):
    """A model with the given output head, its hidden states at the last position of each probe
    row, and the full-vocabulary logits the model itself computes from them."""
    import mlx.core as mx

    model = tiny_model(**HEADS[head])
    tokens = mx.array(PROBE)
    return model, model.model(tokens)[:, -1, :], model(tokens)[:, -1, :]


@pytest.mark.parametrize("head", list(HEADS))
def test_selected_logits_equal_the_full_vocabulary_projection(head):
    import mlx.core as mx

    model, hidden, full = probed(head)
    slots = [100, 3, 64, 17]  # neither sorted nor contiguous: the columns follow the request
    actual = selected_logits(model, hidden, slots)
    assert actual.dtype == mx.float32
    assert max_error(actual, full[:, mx.array(slots)]) < 1e-4


@PROPERTY
@given(
    head=st.sampled_from(list(HEADS)),
    slots=st.lists(st.integers(0, 127), min_size=1, max_size=24, unique=True),
)
def test_any_selection_of_rows_projects_like_the_full_vocabulary(head, slots):
    import mlx.core as mx

    model, hidden, full = probed(head)
    actual = selected_logits(model, hidden, slots)
    assert max_error(actual, full[:, mx.array(slots)]) < 1e-4


@pytest.mark.parametrize("bits", [None, 8])
def test_selected_logits_are_float32_even_for_a_bfloat16_model(bits):
    import mlx.core as mx

    model = tiny_model(bits, bf16=True)
    tokens = mx.array(PROBE)
    hidden = model.model(tokens)[:, -1, :]
    assert hidden.dtype == mx.bfloat16
    actual = selected_logits(model, hidden, [10, 11, 12])
    assert actual.dtype == mx.float32
    assert max_error(actual, model(tokens)[:, -1, :][:, mx.array([10, 11, 12])]) < 2e-2


@pytest.mark.parametrize("head", ["q8", "q4-g32", "untied-q4", "mxfp4"])
def test_selected_logits_state_the_quantization_of_the_head_explicitly(monkeypatch, head):
    import mlx.core as mx

    model, hidden, _ = probed(head)
    layer = model.model.embedding if model.args.tie_word_embeddings else model.lm_head
    real = mx.quantized_matmul
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(mx, "quantized_matmul", spy)
    selected_logits(model, hidden, [3, 9])
    ((args, kwargs),) = calls  # none of these is left to the defaults of MLX
    assert kwargs == {
        "transpose": True,
        "group_size": layer.group_size,
        "bits": layer.bits,
        "mode": layer.mode,
    }
    assert args[0] is hidden
    assert (args[3] is None) == ("biases" not in layer)  # the offsets of affine quantization


def test_quantize_model_pins_the_group_size_instead_of_leaving_it_to_mlx(monkeypatch):
    from mlx import nn

    real = nn.quantize
    calls = []

    def spy(model, **kwargs):
        calls.append(kwargs)
        return real(model, **kwargs)

    monkeypatch.setattr(nn, "quantize", spy)
    quantize_model(raw_model(), 4)
    (kwargs,) = calls
    assert (kwargs["group_size"], kwargs["bits"]) == (64, 4)


def test_quantize_model_follows_the_predicate_of_the_model_layer_by_layer(monkeypatch):
    from mlx import nn
    from spark_mlx_llm.model import Model

    def only_linear_layers(self):
        return lambda path, module: isinstance(module, nn.Linear)

    monkeypatch.setattr(Model, "quant_predicate", property(only_linear_layers))
    model = raw_model()
    quantize_model(model, 8)
    packed = set(quantized_layers(model))
    assert "model.embedding" not in packed  # not a Linear layer
    assert {f"model.layers.{i}.self_attn.g_proj" for i in range(LAYERS)} <= packed
    assert len(packed) == 6 * LAYERS  # every Linear layer, nothing else


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_model_packs_every_projection_but_the_attention_gates(bits, tied):
    from mlx import nn

    model = raw_model(tied)
    quantize_model(model, bits)
    packed = quantized_layers(model)
    projections = (
        "self_attn.q_k_v_proj",
        "self_attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    expected = {"model.embedding"}
    expected |= {f"model.layers.{i}.{p}" for i in range(LAYERS) for p in projections}
    if not tied:
        expected.add("lm_head")
    assert set(packed) == expected
    assert set(packed.values()) == {(bits, 64)}
    for layer in range(LAYERS):
        gate = leaf_modules(model)[f"model.layers.{layer}.self_attn.g_proj"]
        assert type(gate) is nn.Linear


# -- branch_cache --------------------------------------------------------------------------------


@PROPERTY
@given(length=st.integers(1, 45), copies=st.integers(1, 16))
@example(length=5, copies=3)
@example(length=WINDOW, copies=2)
@example(length=40, copies=1)
def test_branch_cache_repeats_the_prefix_state_for_every_row(length, copies):
    import mlx.core as mx

    backend = SparkBackend(frozen_model(), None, {}, prefill_chunk=8)
    prefix = backend._prefill([2 + i % 100 for i in range(length)])
    branches = branch_cache(prefix, copies)
    assert len(branches) == len(prefix) == LAYERS
    for original, branch in zip(prefix, branches, strict=True):
        assert branch is not original
        assert type(branch) is type(original)
        assert branch.offset == original.offset == length
        for kept, repeated in zip(original.state, branch.state, strict=True):
            assert kept.shape[0] == 1  # the retained prefix stays a single row
            assert repeated.shape == (copies, *kept.shape[1:])
            assert all(mx.array_equal(repeated[row], kept[0]) for row in range(copies))


def test_branching_no_cache_gives_no_cache():
    assert branch_cache([], 3) == []


# -- construction --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("batch_size", "prefill_chunk"),
    [(0, 512), (17, 512), (-1, 512), (4, 0), (4, 2049), (4, -1), (0, 0), (17, 2049)],
)
def test_constructor_rejects_limits_outside_the_supported_range(batch_size, prefill_chunk):
    with pytest.raises(ValueError, match=r"^batch_size must be 1–16 and prefill_chunk 1–2048$"):
        SparkBackend(None, None, {}, batch_size, prefill_chunk)


@pytest.mark.parametrize(("batch_size", "prefill_chunk"), [(1, 1), (16, 2048), (1, 2048), (16, 1)])
def test_constructor_accepts_the_limits_themselves(batch_size, prefill_chunk):
    backend = SparkBackend(None, None, {}, batch_size, prefill_chunk)
    assert (backend.batch_size, backend.prefill_chunk) == (batch_size, prefill_chunk)


def test_constructor_keeps_its_arguments_and_defaults_to_4_by_512():
    model, tokenizer, metadata = object(), object(), {"fingerprint": "f"}
    backend = SparkBackend(model, tokenizer, metadata)
    assert backend.model is model
    assert backend.tokenizer is tokenizer
    assert backend.metadata is metadata
    assert (backend.batch_size, backend.prefill_chunk) == (4, 512)


# -- prefill -------------------------------------------------------------------------------------


def test_prefill_feeds_the_prompt_in_chunks_and_evaluates_each_one(body_calls, evals):
    backend = SparkBackend(tiny_model(), FakeTokenizer(), {}, prefill_chunk=4)
    tokens = list(range(10, 21))
    evals.clear()  # building the model evaluated its weights
    cache = backend._prefill(tokens)
    assert body_calls == [
        ([tokens[0:4]], [0] * LAYERS),
        ([tokens[4:8]], [4] * LAYERS),
        ([tokens[8:11]], [8] * LAYERS),
    ]
    assert [c.offset for c in cache] == [11] * LAYERS
    assert [len(args[0]) for args in evals] == [LAYERS] * 3  # the states of all layers, per chunk


def test_prefill_of_no_tokens_is_an_empty_cache(body_calls, evals):
    backend = SparkBackend(tiny_model(), FakeTokenizer(), {})
    evals.clear()
    cache = backend._prefill([])
    assert [c.offset for c in cache] == [0] * LAYERS
    assert body_calls == []
    assert evals == []


# -- score: input checks -------------------------------------------------------------------------


def test_score_rejects_bad_input_before_computing_anything(body_calls):
    backend = SparkBackend(tiny_model(), FakeTokenizer(), {})
    good = make_job("a", [1, 2], [3])
    with pytest.raises(ValueError, match=r"^Unknown execution mode$"):
        backend.score([1, 2], [good], "batched")
    with pytest.raises(ValueError, match=r"^Unknown execution mode$"):
        backend.score([1, 2], [], "batched")  # the mode is checked before the jobs
    for mode in ("shared", "direct"):
        with pytest.raises(ValueError, match=r"^No decisions supplied$"):
            backend.score([1, 2], [], mode)
    # Jobs that do not start with the prefix, or have no question after it, in either mode.
    for tokens in ([1, 3, 4], [1, 2], [1], [9, 2, 5]):
        for mode in ("shared", "direct"):
            bad = Compiled("bad", tokens, [20, 21], "hash")
            with pytest.raises(ValueError, match=r"^Invalid shared prefix$"):
                backend.score([1, 2], [good, bad], mode)
    assert body_calls == []


@pytest.mark.parametrize("mode", ["shared", "direct"])
@pytest.mark.parametrize("prefix", [[], [1, 2]])
def test_one_token_after_the_prefix_is_the_shortest_valid_question(prefix, mode):
    model = tiny_model()
    jobs = [make_job("a", prefix, [3])]
    result, stats = SparkBackend(model, FakeTokenizer(), {}).score(prefix, jobs, mode)
    assert_scores(model, jobs, result)
    assert stats["logical_input_tokens"] == len(prefix) + 1


# -- score: how the work is laid out -------------------------------------------------------------


def test_shared_mode_prefills_the_prefix_once_and_scores_padded_microbatches(body_calls):
    model = tiny_model()
    backend = SparkBackend(model, FakeTokenizer(pad_token_id=0, eos_token_id=9), {}, 2, 4)
    prefix = [10, 11, 12, 13, 14, 15]
    jobs = [
        make_job("long", prefix, [50, 51, 52, 53, 54, 55, 56]),
        make_job("short", prefix, [60]),
        make_job("mid", prefix, [70, 71, 72]),
    ]
    result, stats = backend.score(prefix, jobs, "shared")
    assert body_calls == [
        ([[10, 11, 12, 13]], [0] * LAYERS),  # the prefix, four tokens at a time
        ([[14, 15]], [4] * LAYERS),
        ([[60, 0, 0], [70, 71, 72]], [6] * LAYERS),  # shortest questions first, padded right
        ([[50, 51, 52, 53, 54, 55, 56]], [6] * LAYERS),  # the prefix is still 6 tokens long
    ]
    assert stats["shared_prefix_tokens"] == 6
    assert stats["evaluated_tokens_including_padding"] == 6 + 2 * 3 + 1 * 7
    assert stats["logical_input_tokens"] == 13 + 7 + 9
    assert stats["batches"] == 2
    assert_scores(model, jobs, result)


def test_padding_falls_back_to_the_eos_token_and_never_replaces_a_pad_token_of_zero(body_calls):
    prefix = [10, 11]
    jobs = [make_job("a", prefix, [50]), make_job("b", prefix, [60, 61])]
    for tokenizer, padding in [
        (FakeTokenizer(pad_token_id=None, eos_token_id=9), 9),
        (FakeTokenizer(pad_token_id=0, eos_token_id=9), 0),
        (FakeTokenizer(pad_token_id=5, eos_token_id=9), 5),
    ]:
        body_calls.clear()
        SparkBackend(tiny_model(), tokenizer, {}).score(prefix, jobs, "shared")
        assert body_calls[-1][0] == [[50, padding], [60, 61]]


def test_equal_length_questions_keep_their_order(body_calls):
    prefix = [10, 11]
    jobs = [make_job(name, prefix, [tokens]) for name, tokens in [("x", 50), ("y", 60), ("z", 70)]]
    SparkBackend(tiny_model(), FakeTokenizer(), {}, batch_size=2).score(prefix, jobs, "shared")
    assert [rows for rows, _ in body_calls[1:]] == [[[50], [60]], [[70]]]


def test_direct_mode_recomputes_every_question_from_an_empty_cache(body_calls):
    model = tiny_model()
    backend = SparkBackend(model, FakeTokenizer(), {}, 2, 4)
    prefix = [10, 11]
    jobs = [make_job("a", prefix, [50, 51, 52, 53, 54, 55, 56]), make_job("b", prefix, [60])]
    result, stats = backend.score(prefix, jobs, "direct")
    assert body_calls == [
        ([[10, 11, 50, 51]], [0] * LAYERS),  # everything but the last token, in chunks
        ([[52, 53, 54, 55]], [4] * LAYERS),
        ([[56]], [8] * LAYERS),  # then the last token alone
        ([[10, 11]], [0] * LAYERS),  # the next question starts again from nothing
        ([[60]], [2] * LAYERS),
    ]
    assert stats["shared_prefix_tokens"] == 0
    assert stats["prefill_seconds"] == 0.0
    assert stats["evaluated_tokens_including_padding"] == 9 + 3
    assert stats["logical_input_tokens"] == 9 + 3
    assert stats["batches"] == 2
    assert_scores(model, jobs, result)


def test_a_single_token_question_is_scored_in_direct_mode_without_any_prefill(body_calls):
    model = tiny_model()
    jobs = [make_job("only", [], [5])]
    result, _ = SparkBackend(model, FakeTokenizer(), {}).score([], jobs, "direct")
    assert body_calls[0] == ([[5]], [0] * LAYERS)
    assert_scores(model, jobs, result)


def test_shared_mode_without_a_prefix_scores_each_question_alone(body_calls):
    model = tiny_model()
    jobs = [make_job("a", [], [50, 51]), make_job("b", [], [60])]
    result, stats = SparkBackend(model, FakeTokenizer(), {}).score([], jobs, "shared")
    assert [rows for rows, _ in body_calls] == [[[50]], [[51]], [[60]]]
    assert stats["shared_prefix_tokens"] == 0
    assert stats["prefill_seconds"] == 0.0
    assert stats["batches"] == 2
    assert_scores(model, jobs, result)


def test_the_default_mode_is_shared():
    prefix = [10, 11, 12]
    jobs = [make_job("a", prefix, [50]), make_job("b", prefix, [60])]
    _, stats = SparkBackend(tiny_model(), FakeTokenizer(), {}).score(prefix, jobs)
    assert stats["shared_prefix_tokens"] == 3


def test_only_the_declared_rows_of_the_last_positions_are_projected(projections, evals):
    import mlx.core as mx

    model = tiny_model()
    backend = SparkBackend(model, FakeTokenizer(), {}, batch_size=2)
    prefix = [10, 11, 12]
    jobs = [
        make_job("a", prefix, [50], slots=[22, 20]),
        make_job("b", prefix, [51, 52], slots=[21, 20, 25]),
        make_job("c", prefix, [53, 54, 55], slots=[30, 29]),
    ]
    evals.clear()
    shared, _ = backend.score(prefix, jobs, "shared")
    # One row per question, the union of the slots of a microbatch, sorted.
    assert projections == [((2, 64), [20, 21, 22, 25]), ((1, 64), [29, 30])]
    assert [a[0].shape for a in evals if isinstance(a[0], mx.array)] == [(2, 4), (1, 2)]
    projections.clear()
    evals.clear()
    direct, _ = backend.score(prefix, jobs, "direct")
    assert projections == [((1, 64), [22, 20]), ((1, 64), [21, 20, 25]), ((1, 64), [30, 29])]
    assert [a[0].shape for a in evals if isinstance(a[0], mx.array)] == [(2,), (3,), (2,)]
    assert_scores(model, jobs, shared)  # every answer comes back in its own slot order
    assert_scores(model, jobs, direct)


def test_a_projection_that_loses_rows_is_an_error_not_a_missing_answer(monkeypatch):
    real = backend_module.selected_logits
    monkeypatch.setattr(
        backend_module,
        "selected_logits",
        lambda model, hidden, slots: real(model, hidden, slots)[:1],
    )
    prefix = [10, 11]
    jobs = [make_job("a", prefix, [50]), make_job("b", prefix, [60])]
    backend = SparkBackend(tiny_model(), FakeTokenizer(), {}, batch_size=2)
    with pytest.raises(ValueError, match=r"zip\(\)"):
        backend.score(prefix, jobs, "shared")


def test_scoring_never_projects_the_whole_vocabulary(monkeypatch):
    from spark_mlx_llm.model import Model

    model = tiny_model()
    prefix = [10, 11, 12]
    jobs = [make_job("a", prefix, [50]), make_job("b", prefix, [60, 61], slots=[9, 8])]
    expected = {job.id: reference(model, job.tokens, job.slots) for job in jobs}

    def forbidden(self, *args, **kwargs):
        raise AssertionError("the full output head must not run")

    monkeypatch.setattr(Model, "__call__", forbidden)
    for mode in ("shared", "direct"):
        result, _ = SparkBackend(model, FakeTokenizer(), {}).score(prefix, jobs, mode)
        for name, logits in expected.items():
            assert result[name] == pytest.approx(logits, abs=1e-4)


# -- score: numbers ------------------------------------------------------------------------------


def padded_cost(prefix, jobs, batch_size):
    """Tokens run through the model by the shared mode: the prefix once, then every microbatch
    as wide as its longest question."""
    ordered = sorted(jobs, key=lambda job: len(job.tokens))
    cost = len(prefix)
    for start in range(0, len(ordered), batch_size):
        group = ordered[start : start + batch_size]
        cost += len(group) * max(len(job.tokens) - len(prefix) for job in group)
    return cost


@st.composite
def workloads(draw):
    tokens = st.integers(2, 127)
    prefix = draw(st.lists(tokens, max_size=40))
    jobs = []
    for index in range(draw(st.integers(1, 5))):
        suffix = draw(st.lists(tokens, min_size=1, max_size=24))
        slots = draw(st.lists(st.integers(0, 127), min_size=2, max_size=6, unique=True))
        jobs.append(Compiled(f"q{index}", prefix + suffix, slots, "hash"))
    return prefix, jobs


@PROPERTY
@given(
    workload=workloads(),
    mode=st.sampled_from(["shared", "direct"]),
    batch_size=st.integers(1, 16),
    prefill_chunk=st.integers(1, 64),
    padding=st.tuples(st.none() | st.integers(0, 127), st.integers(0, 127)),
)
def test_scores_equal_an_uncached_pass_through_the_whole_model(
    workload, mode, batch_size, prefill_chunk, padding
):
    prefix, jobs = workload
    model = frozen_model()
    tokenizer = FakeTokenizer(*padding)
    backend = SparkBackend(model, tokenizer, {}, batch_size, prefill_chunk)
    result, stats = backend.score(prefix, jobs, mode)
    assert_scores(model, jobs, result, tolerance=1e-3)
    assert stats["generated_tokens"] == 0
    assert stats["logical_input_tokens"] == sum(len(job.tokens) for job in jobs)
    if mode == "direct" or not prefix:
        assert stats["evaluated_tokens_including_padding"] == stats["logical_input_tokens"]
        assert stats["batches"] == len(jobs)
        assert stats["shared_prefix_tokens"] == 0
    else:
        cost = padded_cost(prefix, jobs, batch_size)
        assert stats["evaluated_tokens_including_padding"] == cost
        assert stats["batches"] == math.ceil(len(jobs) / batch_size)
        assert stats["shared_prefix_tokens"] == len(prefix)


def test_stats_have_the_documented_keys_and_finite_numbers():
    prefix = [10, 11, 12]
    jobs = [make_job("a", prefix, [50]), make_job("b", prefix, [60, 61])]
    for mode in ("shared", "direct"):
        _, stats = SparkBackend(tiny_model(), FakeTokenizer(), {}).score(prefix, jobs, mode)
        assert set(stats) == STATS
        for value in stats.values():
            assert isinstance(value, int | float)
            assert math.isfinite(value)
        assert 0 <= stats["prefill_seconds"] <= stats["inference_seconds"]
        assert stats["peak_mlx_bytes"] > 0
        assert isinstance(stats["peak_mlx_bytes"], int)


def test_timings_split_the_prefix_prefill_from_the_whole_call(monkeypatch, body_calls):
    # The clock counts transformer calls, so each interval is known: the prefill is the calls
    # made for the prefix, the inference all of them.
    use_clock(monkeypatch, lambda: float(len(body_calls)))
    backend = SparkBackend(tiny_model(), FakeTokenizer(), {}, 2, 4)
    prefix = [10, 11, 12, 13, 14, 15]
    jobs = [
        make_job("a", prefix, [50]),
        make_job("b", prefix, [60, 61]),
        make_job("c", prefix, [7]),
    ]
    _, stats = backend.score(prefix, jobs, "shared")
    assert stats["prefill_seconds"] == 2.0  # two chunks of the prefix
    assert stats["inference_seconds"] == 4.0  # plus two microbatches
    _, stats = backend.score(prefix, jobs, "direct")
    assert stats["prefill_seconds"] == 0.0
    assert stats["inference_seconds"] == 9.0  # three questions: two chunks and the last token


def test_the_device_is_synchronized_before_each_interval_is_read(monkeypatch):
    import mlx.core as mx

    log = []
    real = mx.synchronize

    def synchronize(*args):
        log.append("synchronize")
        return real(*args)

    def clock():
        log.append("clock")
        return 0.0

    backend = SparkBackend(tiny_model(), FakeTokenizer(), {})
    monkeypatch.setattr(mx, "synchronize", synchronize)
    use_clock(monkeypatch, clock)
    prefix = [10, 11, 12]
    jobs = [make_job("a", prefix, [50]), make_job("b", prefix, [60])]
    backend.score(prefix, jobs, "shared")
    assert log == ["clock", "clock", "synchronize", "clock", "synchronize", "clock"]
    log.clear()
    backend.score(prefix, jobs, "direct")
    assert log == ["clock", "synchronize", "clock"]


def working_memory(backend, prefix, jobs, mode):
    """Peak memory of one score() call above what was already alive when it started; the
    process holds on to other models, so the reported peak alone says little."""
    import mlx.core as mx

    gc.collect()
    alive = mx.get_active_memory()
    return backend.score(prefix, jobs, mode)[1]["peak_mlx_bytes"] - alive


def layers_alive_at_each_build(monkeypatch, owner, name):
    """Wrap `owner.name`, which builds a cache (a list of layers) per call. The list returned
    gets, at every call, how many layers of the caches built before it are still alive. Counting
    objects instead of bytes: the process-wide peak from mx.get_peak_memory varies by more than
    10% under CPU contention, so a byte limit fails on a loaded or single-core machine."""
    build = getattr(owner, name)
    layers: list[weakref.ref[Any]] = []
    alive: list[int] = []

    def spy(*args):
        gc.collect()  # what only a cycle kept is gone: a layer that is left is held by a name
        alive.append(sum(ref() is not None for ref in layers))
        cache = build(*args)
        layers.extend(weakref.ref(layer) for layer in cache)
        return cache

    monkeypatch.setattr(owner, name, spy)
    return alive


def test_direct_mode_frees_a_question_before_the_next_one_is_built(monkeypatch):
    backend = SparkBackend(tiny_model(), FakeTokenizer(), {}, prefill_chunk=32)
    prefix = [2, 3, 4]
    alive = layers_alive_at_each_build(monkeypatch, backend, "_prefill")
    backend.score(prefix, [make_job(f"q{i}", prefix, [5 + i] * 100) for i in range(4)], "direct")
    # Without `del cache`, the layers of the previous question live on: [0, 4, 4, 4].
    assert alive == [0, 0, 0, 0]


def test_shared_mode_frees_a_microbatch_before_the_next_one_is_built(monkeypatch):
    backend = SparkBackend(tiny_model(), FakeTokenizer(), {}, batch_size=1)
    prefix = [2, 3, 4]
    alive = layers_alive_at_each_build(monkeypatch, backend_module, "branch_cache")
    backend.score(prefix, [make_job(f"q{i}", prefix, [5 + i] * 10) for i in range(4)], "shared")
    # Without `del branch`, the layers of the previous microbatch live on: [0, 4, 4, 4].
    assert alive == [0, 0, 0, 0]


def test_peak_memory_is_the_peak_of_this_call_only():
    backend = SparkBackend(tiny_model(), FakeTokenizer(), {})

    def working(length):
        prefix = [2 + i % 100 for i in range(length)]
        return working_memory(backend, prefix, [make_job("q", prefix, [9, 10])], "shared")

    long = working(1500)
    short = working(8)
    assert 0 < short < long // 2  # a peak carried over from the long call would show up here


# -- loading -------------------------------------------------------------------------------------


class Env:
    """What SparkBackend.load does to the process, recorded in order. The loader, the device
    switch, the cache limit and (unless `resolved` is None) the device resolution are fakes;
    mx.eval and mx.synchronize are spied on but do run."""

    def __init__(self, monkeypatch, resolved=("TARGET", "cpu")):
        import mlx.core as mx
        import spark_mlx_llm

        self.events: list[tuple[Any, ...]] = []
        self.model = raw_model()
        self.tokenizer = FakeTokenizer()
        self.resolved = resolved
        real_eval, real_synchronize = mx.eval, mx.synchronize
        real_loader = inspect.signature(spark_mlx_llm.load)

        def resolve(device):
            self.events.append(("resolve", device))
            return self.resolved

        def load(*args, **kwargs):
            real_loader.bind(*args, **kwargs)  # only calls the real loader would accept
            self.events.append(("load", args, kwargs))
            return self.model, self.tokenizer

        def spy_eval(*args):
            self.events.append(("eval", *args))
            return real_eval(*args)

        def spy_synchronize(*args):
            self.events.append(("synchronize",))
            return real_synchronize(*args)

        if resolved is not None:
            monkeypatch.setattr(backend_module, "resolve", resolve)
        monkeypatch.setattr(spark_mlx_llm, "load", load)
        monkeypatch.setattr(mx, "eval", spy_eval)
        monkeypatch.setattr(mx, "synchronize", spy_synchronize)
        monkeypatch.setattr(mx, "set_default_device", lambda d: self.events.append(("device", d)))
        monkeypatch.setattr(mx, "set_cache_limit", lambda n: self.events.append(("cache", n)))

    def names(self):
        return [event[0] for event in self.events]

    def calls(self, name):
        return [event[1:] for event in self.events if event[0] == name]


@pytest.fixture
def env(monkeypatch):
    return Env(monkeypatch)


IGNORED = (
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
    "modeling_spark.py",
    "configuration_spark.py",
)


def write_checkpoint(directory, shards=(SHARD,), config=None, size="4b"):
    """Stand-in checkpoint files; returns {name: sha256} of those `load` is meant to hash."""
    config = config or {"model_type": "spark2_5", "hidden_size": MODELS[size].hidden_size}
    hashed = {name: f"weights of {name}".encode() for name in shards}
    hashed["config.json"] = json.dumps(config).encode()
    hashed["tokenizer.json"] = b'{"version": "1.0"}'
    hashed["tokenizer_config.json"] = b'{"tokenizer_class": "Fake"}'
    hashed["chat_template.jinja"] = b"{{ messages }}"
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in hashed.items():
        (directory / name).write_bytes(content)
    for name in IGNORED:
        (directory / name).write_bytes(f"not hashed: {name}".encode())
    return {name: hashlib.sha256(content).hexdigest() for name, content in hashed.items()}


def identity_of(metadata):
    return {k: v for k, v in metadata.items() if k not in ("fingerprint", "load_seconds")}


def fingerprint_of(identity):
    text = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(text.encode()).hexdigest()


def test_load_needs_an_existing_checkpoint_directory(tmp_path, monkeypatch, env):
    missing = tmp_path / "nowhere"
    message = f"Model not found at {missing.resolve()}. Run `rizzo download` first."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        SparkBackend.load(missing)
    (tmp_path / "weights.bin").write_bytes(b"not a directory")
    with pytest.raises(ValueError, match=r"^Model not found at "):
        SparkBackend.load(tmp_path / "weights.bin")
    monkeypatch.chdir(tmp_path)  # relative paths are reported in full
    message = f"Model not found at {Path('nowhere').resolve()}. Run `rizzo download` first."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        SparkBackend.load("nowhere")
    assert env.events == []


@pytest.mark.parametrize("bits", [0, 1, 2, 3, 5, 16, 32, -4, "8", 8.5])
def test_load_supports_bf16_8_bit_and_4_bit_only(tmp_path, env, bits):
    write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match=r"^Supported precisions: BF16, 8-bit, 4-bit$"):
        SparkBackend.load(tmp_path, bits=bits)
    assert env.events == []


BATCH_LIMITS = [
    {"batch_size": 0},
    {"batch_size": 17},
    {"prefill_chunk": 0},
    {"prefill_chunk": 2049},
]


@pytest.mark.parametrize("options", BATCH_LIMITS)
def test_load_rejects_batch_sizes_outside_the_supported_range(tmp_path, env, options):
    write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match=r"^batch_size must be 1–16 and prefill_chunk 1–2048$"):
        SparkBackend.load(tmp_path, **options)


@pytest.mark.parametrize("options", BATCH_LIMITS)
def test_load_rejects_wrong_batch_sizes_before_touching_the_checkpoint(tmp_path, env, options):
    write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match=r"^batch_size must be 1–16 and prefill_chunk 1–2048$"):
        SparkBackend.load(tmp_path, **options)
    assert env.events == []


def test_load_stops_at_a_device_this_install_cannot_use(tmp_path, monkeypatch):
    env = Env(monkeypatch, resolved=None)  # the real device resolution
    write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="Device must be one of"):
        SparkBackend.load(tmp_path, device="tpu")
    assert env.events == []


def test_load_needs_safetensors_weights(tmp_path, env):
    write_checkpoint(tmp_path, shards=())  # the index file is not weights
    (tmp_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^Model directory contains no safetensors weights$"):
        SparkBackend.load(tmp_path)
    assert "load" not in env.names()


@pytest.mark.parametrize(
    "architecture", [{"model_type": "llama"}, {"model_type": "spark2_5x"}, {"model_type": ""}, {}]
)
def test_load_needs_the_spark_2_5_architecture(tmp_path, env, architecture):
    write_checkpoint(tmp_path, config={**architecture, "hidden_size": MODELS["4b"].hidden_size})
    with pytest.raises(ValueError, match=r"^Only the Spark2\.5 architecture is supported$"):
        SparkBackend.load(tmp_path)
    assert "load" not in env.names()


def test_load_needs_a_known_model_size(tmp_path, env):
    write_checkpoint(tmp_path, config={"model_type": "spark2_5", "hidden_size": 123})
    with pytest.raises(ValueError, match=r"Unrecognized Spark2\.5 checkpoint \(hidden_size=123\)"):
        SparkBackend.load(tmp_path)
    assert "load" not in env.names()


@pytest.mark.parametrize("size", list(MODELS))
def test_load_reports_the_checkpoint_it_loaded(tmp_path, env, size):
    class Custom(SparkBackend):
        pass

    hashes = write_checkpoint(tmp_path / "ckpt", size=size)
    backend = Custom.load(str(tmp_path / "ckpt"), batch_size=3, prefill_chunk=7)
    spec = MODELS[size]
    identity = {
        "source": spec.repo,
        "requested_revision": spec.revision,
        "runtime_revision": RUNTIME_REVISION,
        "source_files": hashes,
        "precision": "bf16",
        "quantization_group_size": None,
        "device": "cpu",
        "backend": "cpu",
        "mlx": import_mlx().__version__,  # untyped: the mlx.core stubs lack __version__
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "prompt_version": PROMPT_VERSION,
    }
    assert identity_of(backend.metadata) == identity
    assert backend.metadata["fingerprint"] == fingerprint_of(identity)
    assert backend.metadata["load_seconds"] >= 0
    assert type(backend) is Custom
    assert backend.model is env.model
    assert backend.tokenizer is env.tokenizer
    assert (backend.batch_size, backend.prefill_chunk) == (3, 7)


def test_load_defaults_to_the_automatic_device_and_the_default_batch(tmp_path, env):
    write_checkpoint(tmp_path)
    backend = SparkBackend.load(tmp_path)
    assert env.calls("resolve") == [("auto",)]
    assert (backend.batch_size, backend.prefill_chunk) == (4, 512)


def test_load_hands_the_resolved_path_to_the_loader_with_safe_options(tmp_path, env):
    write_checkpoint(tmp_path / "ckpt")
    SparkBackend.load(tmp_path / "ckpt")
    ((args, kwargs),) = env.calls("load")
    assert args == ((tmp_path / "ckpt").resolve(),)
    assert kwargs == {
        "lazy": True,
        "strict": True,
        "dtype": "bfloat16",
        "tokenizer_config": {"trust_remote_code": False},
    }


@pytest.mark.parametrize(("name", "device"), [("cpu", "cpu"), ("cuda", "gpu"), ("mlx", "gpu")])
def test_load_reports_the_device_and_backend_it_resolved(tmp_path, env, name, device):
    write_checkpoint(tmp_path)
    env.resolved = (f"target-{name}", name)
    backend = SparkBackend.load(tmp_path, device=f"asked-{name}")
    assert env.calls("resolve") == [(f"asked-{name}",)]
    assert env.calls("device") == [(f"target-{name}",)]
    assert backend.metadata["backend"] == name
    assert backend.metadata["device"] == device


def test_load_switches_device_and_limits_the_cache_before_loading(tmp_path, env):
    write_checkpoint(tmp_path)
    SparkBackend.load(tmp_path, bits=8)
    names = env.names()
    assert env.calls("cache") == [(CACHE_LIMIT,)]
    assert names.count("load") == 1
    assert names.index("resolve") < names.index("device") < names.index("load")
    assert names.index("cache") < names.index("load")
    assert names.index("load") < names.index("eval") < names.index("synchronize")


@pytest.mark.parametrize(
    ("bits", "precision", "group_size"), [(None, "bf16", None), (4, "q4", 64), (8, "q8", 64)]
)
def test_load_applies_the_precision_before_it_evaluates_the_weights(
    tmp_path, env, bits, precision, group_size
):
    from mlx.utils import tree_flatten

    write_checkpoint(tmp_path)
    assert env.model.training is True  # as a loader hands it over
    backend = SparkBackend.load(tmp_path, bits=bits)
    assert backend.model.training is False
    assert backend.metadata["precision"] == precision
    assert backend.metadata["quantization_group_size"] == group_size
    assert set(quantized_layers(backend.model).values()) == ({(bits, 64)} if bits else set())
    (args,) = env.calls("eval")
    assert set(dict(tree_flatten(args[0]))) == set(dict(tree_flatten(backend.model.parameters())))


def test_load_reads_the_config_as_utf_8(tmp_path, env, monkeypatch):
    import codecs

    write_checkpoint(tmp_path)
    real = Path.read_text
    reads = []

    def spy(self, *args, **kwargs):
        reads.append((self.name, args, kwargs))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    SparkBackend.load(tmp_path)
    ((_, args, kwargs),) = [read for read in reads if read[0] == "config.json"]
    encoding = kwargs.get("encoding", args[0] if args else None)
    assert encoding is not None
    assert codecs.lookup(encoding).name == "utf-8"


def test_load_seconds_are_the_time_taken_to_load(tmp_path, env, monkeypatch):
    write_checkpoint(tmp_path)
    ticks = count(100.0, 2.5)
    use_clock(monkeypatch, lambda: next(ticks))
    assert SparkBackend.load(tmp_path).metadata["load_seconds"] == 2.5


def test_the_fingerprint_binds_weights_precision_backend_and_files(tmp_path, env):
    write_checkpoint(tmp_path)

    def fingerprint(**options):
        env.model = raw_model()
        return SparkBackend.load(tmp_path, **options).metadata["fingerprint"]

    base = fingerprint()
    assert fingerprint() == base
    assert fingerprint(batch_size=2, prefill_chunk=8) == base  # how the work is cut changes nothing
    assert len({base, fingerprint(bits=8), fingerprint(bits=4)}) == 3
    env.resolved = ("TARGET", "cuda")
    assert fingerprint() != base
    env.resolved = ("TARGET", "cpu")
    (tmp_path / SHARD).write_bytes(b"other weights")
    assert fingerprint() != base


def test_load_hashes_weights_config_tokenizer_and_templates_in_a_stable_order(
    tmp_path, env, monkeypatch
):
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    hashes = write_checkpoint(tmp_path, shards=shards[::-1])
    real_glob = Path.glob
    monkeypatch.setattr(  # a file system that lists files in reverse
        Path, "glob", lambda self, *args, **kw: iter(sorted(real_glob(self, *args, **kw))[::-1])
    )
    files = SparkBackend.load(tmp_path).metadata["source_files"]
    assert files == hashes
    assert list(files) == [
        *shards,
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]


def test_only_the_pinned_fine_tune_is_tagged_as_flow(tmp_path, env, monkeypatch):
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    hashes = write_checkpoint(tmp_path, shards=shards)
    pinned = {name: hashes[name] for name in shards}

    def pins(weights, size="4b"):
        return {size: FlowCheckpoint(size, "org/rizzo-flow", "f" * 40, weights)}

    def loaded(flow_checkpoints, bits=None):
        monkeypatch.setattr(backend_module, "FLOW_CHECKPOINTS", flow_checkpoints)
        env.model = raw_model()
        return SparkBackend.load(tmp_path, bits=bits).metadata

    flow = loaded(pins(pinned), bits=8)
    base = loaded({}, bits=8)
    assert flow["weights"] == "flow"
    assert "weights" not in base
    assert flow["fingerprint"] != base["fingerprint"]
    assert model_name(flow) == "rizzo-flow-4b-q8"
    assert model_name(base) == "rizzo-spark-x2.5-4b-q8"
    assert "weights" not in loaded(pins({**pinned, shards[1]: "0" * 64}))  # one file differs
    missing = {**pinned, "model-00003-of-00003.safetensors": "0" * 64}
    assert "weights" not in loaded(pins(missing))  # a pinned file is not there
    assert "weights" not in loaded(pins({}))  # pinning nothing proves nothing
    assert "weights" not in loaded(pins(pinned, size="1.7b"))  # pins are of another size


def test_load_switches_the_process_to_the_device_it_resolved(tmp_path, monkeypatch):
    import mlx.core as mx
    import spark_mlx_llm

    model = raw_model()
    monkeypatch.setattr(spark_mlx_llm, "load", lambda path, **kwargs: (model, FakeTokenizer()))
    write_checkpoint(tmp_path)
    previous = mx.set_cache_limit(0)
    try:
        backend = SparkBackend.load(tmp_path, device="cpu")
        assert mx.default_device() == mx.cpu
        assert mx.set_cache_limit(previous) == CACHE_LIMIT
    finally:
        mx.set_cache_limit(previous)
    assert backend.metadata["device"] == "cpu"
    assert backend.metadata["backend"] == "cpu"


@pytest.mark.parametrize("bits", [None, 8, 4])
def test_a_loaded_backend_scores_with_the_model_and_tokenizer_of_the_loader(tmp_path, env, bits):
    write_checkpoint(tmp_path)
    env.tokenizer = FakeTokenizer(pad_token_id=None, eos_token_id=7)
    backend = SparkBackend.load(tmp_path, bits=bits, batch_size=2, prefill_chunk=5)
    prefix = [10, 11, 12, 13, 14, 15, 16]
    jobs = [
        make_job("a", prefix, [50]),
        make_job("b", prefix, [60, 61, 62]),
        make_job("c", prefix, [7, 8]),
    ]
    for mode in ("shared", "direct"):
        result, _ = backend.score(prefix, jobs, mode)
        assert_scores(env.model, jobs, result)


# -- through the engine --------------------------------------------------------------------------


def test_the_engine_answers_from_the_logits_of_the_mlx_backend():
    payload = {
        "state": {"ticket": "Cannot log in"},
        "questions": {
            "supported": {"type": "boolean", "instructions": "Does the user need login help?"},
            "route": {
                "type": "choice",
                "instructions": "Choose a queue",
                "options": [
                    {"id": "billing", "description": "Payment problem"},
                    {"id": "access", "description": "Login problem"},
                    {"id": "other", "description": "Anything else"},
                ],
            },
        },
    }
    model, tokenizer = tiny_model(), TextTokenizer()
    metadata = {"fingerprint": "tiny"}
    engine = Engine(SparkBackend(model, tokenizer, metadata, batch_size=2, prefill_chunk=64))
    prefix, jobs = compile_request(tokenizer, Request.model_validate(payload), 8192)
    expected = {job.id: reference(model, job.tokens, job.slots) for job in jobs}
    for mode in ("shared", "direct"):
        response = engine.decide({**payload, "mode": mode})
        assert response["model"] == metadata
        assert response["timing"]["generated_tokens"] == 0
        assert response["timing"]["shared_prefix_tokens"] == (
            len(prefix) if mode == "shared" else 0
        )
        for name, logits in expected.items():
            answer = response["answers"][name]
            assert answer["input_tokens"] == len(next(j.tokens for j in jobs if j.id == name))
            assert list(answer["option_logits"].values()) == pytest.approx(logits, abs=1e-3)
