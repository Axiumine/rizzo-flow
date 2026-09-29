"""Thread-safe long-lived model service with deterministic postprocessing."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .decisions import decode
from .prompts import compile_request
from .responses import Response
from .schema import Request


class BackendError(ValueError):
    """The model failed on a valid request (a ValueError, so callers that catch those still do)."""


class Engine:
    def __init__(self, backend, ctx=8192, calibration=None):
        if ctx < 1:
            raise ValueError("ctx must be positive")
        self.backend = backend
        self.ctx = ctx
        self.calibration = calibration
        if calibration and calibration.fingerprint != backend.metadata["fingerprint"]:
            raise ValueError(
                "Calibration was fitted for a different model/runtime/prompt configuration"
            )
        self._lock = Lock()
        self._closed = False
        # All model work runs on one thread that lives as long as the process. Web servers call
        # decide() from short-lived pool threads, and MLX's CUDA backend aborts the process when
        # a thread that ran computations exits.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rizzo-inference")

    def close(self):
        """Wait for the decode in flight and accept no more: the model must not be freed under it.

        The inference thread stays, since it must live as long as the process (see __init__)."""
        with self._lock:  # a decide() under way finishes first, one that follows is refused
            self._closed = True
            # Queued behind the decode that a Ctrl-C left running on the thread.
            self._worker.submit(bool).result()

    def decide(self, request: Request | dict) -> dict:
        # Re-validate a serialized snapshot, also protecting mutable Pydantic objects.
        request = Request.model_validate(
            request.model_dump() if isinstance(request, Request) else request
        )
        started = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("The engine is closed")
            acquired = time.perf_counter()
            prefix, jobs = compile_request(self.backend.tokenizer, request, self.ctx)
            encoded = time.perf_counter()
            try:
                logits, timing = self._worker.submit(
                    self.backend.score, prefix, jobs, request.mode
                ).result()
                answers = {}
                for job in jobs:
                    question = request.questions[job.id]
                    temperature = (
                        self.calibration.temperatures.get(question.type, 1.0)
                        if self.calibration
                        else 1.0
                    )
                    answers[job.id] = decode(question, logits[job.id], temperature)
                    answers[job.id]["prompt_sha256"] = job.prompt_sha256
                    answers[job.id]["input_tokens"] = len(job.tokens)
            except ValueError as error:
                # The request is validated and compiled: from here on, a ValueError is the
                # runtime's (a failed llama_decode, logits that are not numbers or not enough).
                raise BackendError(str(error)) from error
        response = {
            "model": self.backend.metadata,
            "mode": request.mode,
            "answers": answers,
            "calibration": self.calibration.model_dump() if self.calibration else None,
            "timing": {
                **timing,
                "queue_seconds": acquired - started,
                "compile_seconds": encoded - acquired,
                "total_seconds": time.perf_counter() - started,
            },
        }
        return Response.model_validate(response).model_dump()
