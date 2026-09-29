"""Model and file registry, Hugging Face token discovery and the download paths.

Nothing touches the network and no weights are needed.
"""

import pytest

from rizzo_flow import config


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("prefill_chunk", [1, 512, 2048])
def test_the_limits_of_both_backends_include_their_bounds(batch_size, prefill_chunk):
    config.check_limits(batch_size, prefill_chunk)  # nothing to report


@pytest.mark.parametrize(
    ("batch_size", "prefill_chunk"),
    [(0, 512), (-1, 512), (17, 512), (4, 0), (4, -1), (4, 2049), (0, 0), (17, 2049)],
)
def test_the_limits_of_both_backends_refuse_what_is_outside_them(batch_size, prefill_chunk):
    with pytest.raises(ValueError, match=r"^batch_size must be 1–16 and prefill_chunk 1–2048$"):
        config.check_limits(batch_size, prefill_chunk)
