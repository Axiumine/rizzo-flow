"""Helpers of the mutation tests, `test_mutation_<module>.py`.

Not a test module: pytest collects it because of its name and finds nothing to run.

`mutmut run` edits the code one place at a time and reruns the tests that reach it (see
[tool.mutmut] in pyproject.toml). A test in one of these files exists because such an edit got
through the others, and its docstring says which edit.

Edits that no test can tell from the original get no test, and mutmut keeps reporting them as
survivors. At the time of writing (mutmut 3.8.0, 4680 mutants) 86 are left, of these kinds:
  26  `zip(strict=True)` over lists of one length by construction (or `strict=False`, spelled
      `None` or left out)
  16  "UTF-8" for "utf-8", or `bytes.decode()` without the encoding that is its default
   7  an argparse default or `dest` that argparse gives anyway, or that nothing reads
   6  a header name in capitals: urllib capitalizes it
   5  `None` for `False` in `json.dumps(ensure_ascii=..., allow_nan=...)`
   4  in `config.parse_dotenv`, `-2` for `-1` and `>= 0` for `> 0` on an index that is -1 or at
      least 1, and a `maxsplit` left out or raised where only `[0]` is read
   3  a fallback text that changes no outcome ("XXXX" for "")
   3  the type argument of `typing.cast`, a no-op at run time
   2  "SHA256" for "sha256" in hashlib
   2  a zero-filled ctypes array, whose NULL terminator is written anyway
   2  `None` for `False` where only the truth is read
   2  `rsplit(sep, n)[-1]` for another `n`
   2  `+=` for `=` on an accumulator that still holds 0
   1  "MLX-LM" for "mlx-lm" in importlib.metadata
   1  a clamp that cannot bind, `min(1.0, x)` with x at most 1
   1  a default that is only compared with 1
   1  a float literal spelled another way, `1e+100` for `1e100`
   1  a loop bound past the last non-empty bin
   1  `rindex` for `index` on a text found exactly once
One more mutant, in `schema.require_unicode`, loops forever: mutmut's timeout catches it.

The 15 decorated functions (validators, properties) are never mutated by mutmut; trying their
mutants once, on a copy where each was split into a plain function and its decorator, left
5 equivalent ones out of 161.
"""

import io
import re
from pathlib import Path

import pytest


def exactly(message: str) -> str:
    """A `match` pattern for pytest.raises that accepts the message and nothing around it."""
    return f"^{re.escape(message)}$"


def use_ascii_as_default_encoding(monkeypatch: pytest.MonkeyPatch, probe_dir: Path) -> None:
    """Make ASCII the encoding that a text file gets when the caller names none.

    Linux and macOS default to UTF-8, so code that forgets `encoding="utf-8"` still round-trips
    the text of a test there, and only Windows (the ANSI code page) tells. With this in place the
    code has to name its encoding, or the text of the test fails to decode or encode, on every
    platform. `pathlib` asks `io.text_encoding` for the default, which is what this replaces.

    The last two lines prove the replacement reaches `pathlib` (`Path.read_text` and
    `Path.open`), so that a Python that stops asking `io.text_encoding` fails here, loudly,
    instead of letting the tests that use this pass for nothing.
    """

    def default(encoding: str | None, stacklevel: int = 2) -> str:
        return "ascii" if encoding is None else encoding

    monkeypatch.setattr(io, "text_encoding", default)
    probe = probe_dir / "probe.txt"
    probe.write_bytes("\N{LATIN SMALL LETTER E WITH GRAVE}".encode())
    with pytest.raises(UnicodeDecodeError):
        probe.read_text()
    with pytest.raises(UnicodeDecodeError), probe.open() as stream:
        stream.read()
