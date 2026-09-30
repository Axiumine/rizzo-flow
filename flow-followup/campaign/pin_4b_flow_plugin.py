"""pytest plugin: make tests/test_llama_real.py load the fine-tuned (flow) 4B Q8_0.

  RIZZO_REAL=1 PYTHONPATH=.research/hw-campaign pytest -q -m integration -p pin_4b_flow_plugin

Without it the suite loads the smallest Q8_0 GGUF on disk, which is a 1.7B file (base or flow).
Same idea as pin_4b_plugin.py, which cannot be used any more: since the fine-tuned weights became
the default, `config.GGUF` is keyed by (size, quant, variant) and that plugin's
GGUF[("4b", "q8_0")] raises KeyError.

When RIZZO_HW_EVIDENCE names a file, the plugin also writes there, at the end of the session, what
it pinned and the metadata of every backend that LlamaBackend.load returned, so that the campaign
runner (run_flow.py) can check that the fine-tuned weights were really loaded: the pytest output
alone does not say which file was.
"""

import json
import os
import sys
from pathlib import Path

SIZE, QUANT, VARIANT = "4b", "q8_0", "flow"
LOADED = []  # metadata of every backend loaded during the session


def _spec():
    from rizzo_flow.config import gguf_spec

    return gguf_spec(SIZE, QUANT, VARIANT)


def pytest_configure(config):
    from rizzo_flow import backend_llama

    original = backend_llama.LlamaBackend.load.__func__

    def load(cls, *args, **kwargs):
        backend = original(cls, *args, **kwargs)
        LOADED.append(dict(backend.metadata))
        return backend

    backend_llama.LlamaBackend.load = classmethod(load)


def pytest_collection_modifyitems(session, config, items):
    path = _spec().path
    for item in items:
        module = getattr(item, "module", None)
        if module is not None and module.__name__.endswith("test_llama_real"):
            module.smallest = lambda: path if path.is_file() else None


def pytest_sessionfinish(session, exitstatus):
    target = os.environ.get("RIZZO_HW_EVIDENCE")
    if not target:
        return
    try:
        spec = _spec()
        evidence = {
            "pinned": {"size": SIZE, "quant": QUANT, "variant": VARIANT, "file": spec.file,
                       "path": str(spec.path), "sha256": spec.sha256},
            "loaded": LOADED,
        }
        Path(target).write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    except Exception as error:  # the runner reports a missing file; never break the session
        print(f"pin_4b_flow_plugin: cannot write {target}: {error!r}", file=sys.stderr)
