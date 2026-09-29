"""Rizzo Flow: typed decisions over local Spark logits."""

from .runtime import prepare

prepare()  # before anything imports the MLX stack (Windows needs a stub and the CUDA DLL path)

from .responses import Response  # noqa: E402  (prepare() above has to run first)
from .schema import Request  # noqa: E402

__all__ = ["Request", "Response"]
