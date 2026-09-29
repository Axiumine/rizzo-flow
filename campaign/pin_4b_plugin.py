"""pytest plugin: make tests/test_llama_real.py load the 4B Q8_0 instead of the smallest Q8_0.

  RIZZO_REAL=1 PYTHONPATH=.research/hw-campaign pytest -q -m integration -p pin_4b_plugin
"""


def pytest_collection_modifyitems(session, config, items):
    from rizzo_flow.config import GGUF

    path = GGUF[("4b", "q8_0")].path
    for item in items:
        module = getattr(item, "module", None)
        if module is not None and module.__name__.endswith("test_llama_real"):
            module.smallest = lambda: path if path.is_file() else None
