"""Mutation tests of `rizzo_flow.backend_llama`."""

from types import SimpleNamespace

import pytest
from test_mutation_support import exactly

from rizzo_flow import backend_llama
from rizzo_flow.backend_llama import LlamaBackend
from rizzo_flow.prompts import Compiled


class Session:
    """Keeps the order of the calls `score` makes, and of its readings of the clock."""

    device = None
    idle_free = 1000

    def __init__(self, free=()):
        self.events = []
        self.readings = list(free)  # what free_bytes() answers, one reading per call

    def clear(self):
        self.events.append("clear")

    def branch(self, source, target):
        self.events.append(f"branch {source}>{target}")

    def drop(self, sequence):
        self.events.append(f"drop {sequence}")

    def decode(self, tokens, positions, sequences, outputs=()):
        self.events.append(f"decode {len(tokens)}")

    def logits(self, index, slots):
        return [0.0] * len(slots)

    def synchronize(self):
        self.events.append("synchronize")

    def free_bytes(self):
        self.events.append("free_bytes")
        return self.readings.pop(0) if self.readings else None


def backend(session):
    return LlamaBackend(session, None, {}, batch_size=4)


def job(name, prefix, suffix):
    return Compiled(name, prefix + suffix, [65, 66], "hash")


def clock_in(monkeypatch, session):
    """A clock that ticks once per reading and writes each reading among the session's events."""
    ticks = iter(range(100))

    def perf_counter():
        session.events.append("clock")
        return float(next(ticks))

    monkeypatch.setattr(backend_llama, "time", SimpleNamespace(perf_counter=perf_counter))


def test_the_device_is_synchronized_before_the_prefill_and_the_inference_are_timed(monkeypatch):
    """`score` without `session.synchronize()` after the prefix, or after the last microbatch:
    the times would stop while the device is still computing."""
    session = Session()
    clock_in(monkeypatch, session)
    prefix = [10, 11]
    jobs = [job("a", prefix, [1]), job("b", prefix, [2, 3])]
    _, timing = backend(session).score(prefix, jobs, "shared")
    assert session.events == [
        "clock",  # the request starts
        "clock",  # the prefill starts
        "clear",
        "decode 2",
        "synchronize",
        "clock",  # the prefill ends
        "branch 0>1",
        "branch 0>2",
        "decode 3",
        "drop 1",
        "drop 2",
        "synchronize",
        "free_bytes",
        "clock",  # the request ends
    ]
    assert (timing["prefill_seconds"], timing["inference_seconds"]) == (1.0, 3.0)


def test_the_device_is_synchronized_before_direct_scoring_is_timed(monkeypatch):
    """`score` without `session.synchronize()` at the end of a pass over separate questions."""
    session = Session()
    clock_in(monkeypatch, session)
    _, timing = backend(session).score([10], [job("a", [10], [1]), job("b", [10], [2])], "direct")
    assert session.events == [
        "clock",
        "clear",
        "decode 2",
        "clear",
        "decode 2",
        "synchronize",
        "free_bytes",
        "clock",
    ]
    assert timing["inference_seconds"] == 1.0


@pytest.mark.parametrize("mode", ["shared", "direct"])
def test_one_job_that_does_not_continue_the_prefix_refuses_the_request(mode):
    """`score` refusing only when every job is wrong (`all` for `any`)."""
    prefix = [1, 2]
    fine = job("fine", prefix, [3])
    wrong = job("wrong", [1, 9], [3])  # another prefix
    bare = job("bare", prefix, [])  # nothing after the prefix
    for jobs in ([fine, wrong], [wrong, fine], [fine, bare], [bare, fine]):
        with pytest.raises(ValueError, match=exactly("Invalid shared prefix")):
            backend(Session()).score(prefix, jobs, mode)


def test_without_a_prefix_every_question_is_scored_on_its_own_even_in_shared_mode():
    """`score` sharing an empty prefix (no `bool(prefix)`): the questions would go into one
    microbatch, and the request would report one batch instead of three."""
    session = Session()
    jobs = [job("a", [], [1, 2]), job("b", [], [3]), job("c", [], [4, 5, 6])]
    _, timing = backend(session).score([], jobs, "shared")
    assert session.events == [
        "clear",
        "decode 2",
        "clear",
        "decode 1",
        "clear",
        "decode 3",
        "synchronize",
        "free_bytes",
    ]
    assert timing["batches"] == 3
    assert timing["shared_prefix_tokens"] == 0
    assert timing["prefill_seconds"] == 0.0
    assert timing["evaluated_tokens_including_padding"] == 6


def test_a_reading_without_a_number_leaves_the_lowest_free_memory_alone():
    """`_track_memory` folding a missing reading into the minimum (`if True`)."""
    engine = backend(Session(free=[500, None, 700]))
    peaks = []
    for _ in range(3):
        _, timing = engine.score([1], [job("a", [1], [2])], "direct")
        peaks.append(timing["peak_device_bytes"])
    assert peaks == [500, 500, 500]  # 1000 free before the load, 500 at the lowest


def test_a_missing_weights_file_is_reported_by_its_full_path(tmp_path, monkeypatch):
    """`LlamaBackend.load` without `.resolve()`: the message would name the path as typed."""
    monkeypatch.chdir(tmp_path)
    full = (tmp_path / "absent.gguf").resolve()
    message = f"GGUF file not found at {full}. Run `rizzo download` first."
    with pytest.raises(ValueError, match=exactly(message)):
        LlamaBackend.load("absent.gguf")
