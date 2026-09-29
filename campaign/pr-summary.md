## Summary

First Linux + NVIDIA report, plus Vulkan on the same card and CPU-only on a desktop CPU: **141
runs, all completed**. They cover both models × three quantizations × three KV cache types on
CUDA and on the CPU, plus one Vulkan configuration (4B Q8_0: `rizzo devices`, smoke,
`semif_compare`). About 9.7 hours from the first run to the last, strictly one model process at
a time. Everything installed and ran with `uv sync --locked` and `rizzo download`, with no manual
steps. One integration test fails with the 1.7B (see *Anything odd*).

**Weights:** tested at `c30cc63`, before 12bc7ec made your fine-tuned weights the default. Every
number here is for XHToken's base GGUF files (what `--weights base` gives on current `main`) and
is compared with your base-weight runs.

- **Same answers as your Windows runs.** 4B Q8_0 CUDA: authored144 0.810 / perturbations108
  0.830 (yours 0.812 / 0.848); paired difference on authored144 −0.002 [−0.027, +0.022], 4 of 252
  rows with a different argmax. BF16 −0.009 [−0.028, 0.000], Q4_K_M −0.004 [−0.028, +0.014],
  1.7B Q8_0 +0.022 [0.000, +0.047]. Every interval includes zero: no difference claimed.
- **Faster, even power-capped.** 4B Q8_0: p50 36 ms per decision (yours 49), shape777 shared
  30.03 dec/s (20.99), direct 3.37 dec/s (2.60); 1.7B Q8_0 18 ms and 49.70 dec/s. The board limit
  is 145 W (DCGM enforced power limit): during the 4B Q8_0 `semif_compare` run the GPU was
  power-throttled for 234 of 276 s and reached 87 °C (29 °C room).
- **The shared-vs-direct flips are not CUDA-specific.** With all 37 states in direct mode, the
  4B Q8_0 changes 7 of 777 argmaxes on CUDA (max Δp 0.146) and also 7 of 777 with the Vulkan
  build on the same card (max Δp 0.080). BF16: 1/777, as in your run. Your Vulkan row and the
  0/20 on smoke in #5 (Metal) and #11 (Vulkan) are samples too small to see them (63 and 20
  decisions).
- **Vulkan on NVIDIA under Linux works**, with the same answers as CUDA (authored144 +0.006
  [0.000, +0.020], 4/252 rows). Per decision it is 1.8× slower (p50 66 vs 36 ms), the same ratio
  as your Windows runs (90 vs 49 ms); in shared mode 1.3× slower (23.10 vs 30.03 dec/s; yours
  1.4×). The first request of a new shape can take seconds: 3.25 s for the first numeric question
  on smoke, against a median of 68 ms.
- **CPU-only is usable for low volume, and only with `--threads`.** Ryzen 9 9950X3D, `cpu`
  package, 16 threads: 4B Q8_0 gives the same answers as CUDA (authored144 +0.006 [0.000, +0.020],
  5/252 rows; perturbations108 +0.043 [0.000, +0.071]). p50 727 ms per decision, 20× this GPU's
  36 ms; 1.68 dec/s in shared mode, 18× slower; 0.14 dec/s direct (3 states, the script default).
  1.7B Q8_0: 293 ms. BF16 is as fast as Q8_0 on this CPU (1.73 vs 1.68 dec/s shared).
  **With `--threads` unset llama.cpp uses 4 threads**: smoke median 2542 ms at the default,
  838 ms with `--threads 16` (3.0× faster), 1308 ms with `--threads 32` (SMT hurts).
- **KV cache: `q8_0` is safe, `q4_0` is not.** `--kv-type q8_0` changes nothing significant in any
  of the 12 configurations (every 95 % interval includes zero; 1 to 11 of 252 rows differ).
  `q4_0` is significantly worse in 3 of 12: CPU 4B BF16 −0.070 [−0.109, −0.033], CPU 1.7B Q8_0
  −0.054 [−0.102, −0.010], CUDA 1.7B Q4_K_M −0.042 [−0.076, −0.008], with up to 41 of 252 rows
  changing answer; on CUDA the 1.7B Q8_0 with `q4_0` also changes 21 of 63 shared-vs-direct
  argmaxes (3 direct states). At the default `--ctx 8192` the saving is small (4B Q8_0 peak
  5.61 → 4.97 → 4.62 GiB); it grows with `--ctx`. On CUDA a quantized KV cache costs no speed
  (29.5 vs 30.0 dec/s shared); on the CPU it costs 30–45 % (1.68 → 1.15 → 0.94 dec/s).
- **`--batch-size`**: 4 (the default) is close to the best. Shared dec/s on CUDA 25.1 at 1,
  30.3 at 4, 30.9 at 16; on the CPU 1.51, 1.69, 1.72.
- **typed-decisions** (400 test cases, zero-shot): 4B Q8_0 0.574, KL 2.888, Brier 0.479, 143 ms
  per case (yours 0.574 / 2.899 / 0.480 / 201 ms). Same accuracy on the CPU at 3.0 s per case.
- **1.7B, as documented, abstains far too often**: on `examples/ticket.json` all six 1.7B
  configurations (CUDA and CPU × Q8_0, Q4_K_M, BF16) answer 3 of 4 questions with
  `insufficient_evidence`; smoke accuracy 0.35–0.45 against 0.90–0.95 for the 4B.

Suggestions, if useful: default `--threads` to the number of physical cores; add `--threads`
to `semif_compare.py`, `validate_checkpoint.py` and `typed_decisions.py`; pin the integration
suite to the 4B, or widen its tolerance for the 1.7B. A small docs PR will follow, adding this
report to the hardware table (English and Italian README) and the landing page as a community
report.
