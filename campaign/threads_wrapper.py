"""Run a repo script with a CPU thread count injected into LlamaBackend.load.

scripts/semif_compare.py, validate_checkpoint.py and typed_decisions.py have no --threads flag,
so on the CPU they run with llama.cpp's C API default of 4 threads. This wrapper sets `threads`
(from RIZZO_HW_THREADS) whenever the caller leaves it unset, then runs the script unchanged.

  RIZZO_HW_THREADS=16 python threads_wrapper.py scripts/semif_compare.py --system rizzo ...
"""

import os
import runpy
import sys
from pathlib import Path

from rizzo_flow import backend_llama

THREADS = int(os.environ["RIZZO_HW_THREADS"])
_original = backend_llama.LlamaBackend.load.__func__


def _load(cls, *args, **kwargs):
    if kwargs.get("threads") is None:
        kwargs["threads"] = THREADS
    return _original(cls, *args, **kwargs)


backend_llama.LlamaBackend.load = classmethod(_load)

script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
sys.path.insert(0, str(script.parent))
runpy.run_path(str(script), run_name="__main__")
