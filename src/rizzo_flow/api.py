"""Local HTTP API. One resident model, serialized GPU access, no external calls."""

import hmac
import os
from math import isfinite
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi import Request as HttpRequest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from .compat import (
    SystemOneRequest,
    UnknownModel,
    from_native,
    list_models,
    resolve_model,
    to_native,
)
from .responses import Response
from .schema import Request

API_KEY_ENV = "RIZZO_API_KEY"
MAX_ECHO_DEPTH = 100  # levels of an echoed input that are kept; deeper ones would exhaust the stack
PLAYGROUND = Path(__file__).with_name("playground.html")
SNAKE = Path(__file__).with_name("snake.html")
LOGO = Path(__file__).with_name("logo.png")


def jsonable(value, depth=0):
    """The same structure, with whatever a JSON response cannot carry replaced by its text.

    Three things reach here: non-JSON floats (NaN, Infinity) and lone surrogates (`"\\ud800"`,
    which UTF-8 cannot encode), both accepted by `json.loads` on the way in and echoed back as
    the offending input, and the exception object a validator raised.

    `json.loads` takes far more nesting than Python's recursion limit lets this follow, and the
    encoder that writes the response has a limit of its own: what lies deeper than MAX_ECHO_DEPTH
    is named, not echoed.
    """
    if isinstance(value, float):
        return value if isfinite(value) else f"<{value}>"
    if isinstance(value, (dict, list, tuple)) and depth >= MAX_ECHO_DEPTH:
        return "<nested too deeply>"
    if isinstance(value, dict):
        return {jsonable(str(key)): jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item, depth + 1) for item in value]
    if isinstance(value, (int, bool)) or value is None:
        return value
    # Text, or the text of anything else; a lone surrogate is written as its escape.
    return str(value).encode("utf-8", "backslashreplace").decode("utf-8")


def check_api_key(api_key=None):
    """The key that Bearer auth compares with: `api_key`, else RIZZO_API_KEY (empty: no auth).

    One with characters above 0x7f is a ValueError. The server reads header bytes as Latin-1;
    clients write those characters as Latin-1 (browsers, Python), as UTF-8 (curl) or not at all
    (beyond Latin-1). No single comparison serves them all, so the key has to be ASCII.
    `rizzo serve` asks before it loads the model, `create_app` again for every other caller.
    """
    key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
    if key and not key.isascii():
        raise ValueError(f"{API_KEY_ENV} must be ASCII")
    return key


def create_app(engine, api_key=None):
    app = FastAPI(
        title="Rizzo Flow",
        version="0.2.0",
        description="Typed decisions with a local Spark-X2.5 model; no text generation.",
    )
    api_key = check_api_key(api_key)

    @app.exception_handler(RequestValidationError)
    def invalid_request(request: HttpRequest, error: RequestValidationError):
        # `json.loads` accepts NaN and Infinity (JSON has neither) and lone surrogates (UTF-8
        # cannot encode them). Validation rejects them, but the 422 body echoes the offending
        # input, and serializing that would fail inside the response and turn a client error into
        # a 500. Report them instead of echoing them.
        return JSONResponse(status_code=422, content={"detail": jsonable(error.errors())})

    def authorize(authorization: str | None = Header(default=None)):
        # Bearer auth mirrors the hosted API; it is enforced only when a key is configured.
        if api_key and not hmac.compare_digest(
            (authorization or "").encode(), f"Bearer {api_key}".encode()
        ):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ready", "model": engine.backend.metadata}

    @app.post("/v1/decisions", response_model=Response)
    def decisions(request: Request):
        try:
            return engine.decide(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    def systemone(request: SystemOneRequest):
        try:
            served = resolve_model(request.model, engine.backend.metadata)
            native, options = to_native(request)
            return from_native(request, engine.decide(native), options, served)
        except UnknownModel as error:
            # The hosted API answers an unserved model name with a 400 and a typed detail.
            raise HTTPException(
                status_code=400,
                detail={"error_type": "api_usage_error", "message": str(error)},
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models():
        return list_models(engine.backend.metadata)

    @app.get("/playground", response_class=HTMLResponse, include_in_schema=False)
    def playground():
        return PLAYGROUND.read_text(encoding="utf-8")

    @app.get("/snake", response_class=HTMLResponse, include_in_schema=False)
    def snake():
        return SNAKE.read_text(encoding="utf-8")

    @app.get("/playground/logo.png", include_in_schema=False)
    def logo():
        return FileResponse(LOGO, media_type="image/png")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/playground")

    return app
