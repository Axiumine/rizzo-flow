- `tests/test_llama_real.py::test_shared_prefix_agrees_with_direct` **fails** when the suite
  loads the 1.7B Q8_0, which it does as soon as the 1.7B is downloaded (it picks the smallest
  Q8_0 on disk): same argmax for every question of `examples/ticket.json`, but a probability
  differs by 0.0595 between shared and direct, above the 0.05 tolerance. Identical value on two
  runs. With the same suite pinned to the 4B Q8_0 (a pytest plugin replacing `smallest()`), all
  tests pass. So either the tolerance only holds for the 4B, or the suite should pin the 4B.
- `scripts/semif_compare.py`, `scripts/validate_checkpoint.py` and `scripts/typed_decisions.py`
  have no `--threads` flag, so on the CPU they run on llama.cpp's default 4 threads. For the CPU
  rows here a small wrapper set `threads` in `LlamaBackend.load` to the same value the CLI runs
  used; nothing in the repository was changed.
- Vulkan: the first request of a new shape is slow. On smoke with the 4B Q8_0 the first numeric
  question (`fill-0`) took 3252 ms and the first four-question request 541 ms, against a median
  of 68 ms, so the Vulkan smoke p95 and dec/s mostly measure that one-off cost. The SemIf
  latencies (p50 66 ms, p95 70 ms) are on a warm model and are not affected. That same run
  printed one line on stderr, `NVVM compilation failed: 3` (NVIDIA's shader compiler), and still
  returned correct answers; no other run printed it.
- The GPU hit its 145 W board limit in every long GPU run: power-cap throttling for 234 of 276 s
  in the 4B Q8_0 `semif_compare` run, 87 °C peak with the room at 29 °C. The GPU numbers here are
  therefore power-limited, not clock-limited.
- Between 05:01 and 05:38 something on the desktop used the GPU (18–28 W, 10–19 % utilization)
  during five CPU-only runs (1.7B Q4_K_M and BF16 with a quantized KV cache). Host CPU
  utilization stayed at 50 % (rizzo's 16 threads) as in every other CPU run, so their timings
  are not affected.
