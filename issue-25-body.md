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

## Machine

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D 16-Core Processor, 32 threads |
| RAM | 62 GiB usable — 2 × 32 GB DDR5 Crucial Pro CP32G64C40U5B (DDR5-5600 CL40 modules) configured at 6000 MT/s, dual channel |
| GPU | NVIDIA RTX PRO 4000 Blackwell, 24467 MiB, 615.71.09, 12.0 — also drives the desktop display |
| Ambient temperature | 29 °C at the start of the campaign (room, measured by hand); GPU board power cap 145 W |
| Driver / CUDA | 615.71.09 / CUDA 13.4 |
| OS | Debian GNU/Linux 13 (trixie), kernel 6.12.107+deb13-amd64 |
| Python | Python 3.14.7 (the repo pins 3.12; `.venv` created with 3.14) |
| uv | uv 0.12.17 (x86_64-unknown-linux-gnu) |
| Commit | `c30cc63` |
| llama.cpp | release `b11081`, official prebuilt packages `llama-b11081-linux-x64-cuda`, `-vulkan`, `-cpu` installed side by side |
| Models | Spark-X2.5-4B and 1.7B, XHToken's official (base) GGUF: Q8_0, Q4_K_M, BF16 — `--weights base` on `main` since 12bc7ec |

## Commands

```bash
uv sync --extra test --locked
uv run rizzo download                                   # CUDA runtime + 4B Q8_0
uv run rizzo download --only runtime --runtime vulkan
uv run rizzo download --only runtime --runtime cpu
uv run rizzo download --only weights --quant q4_k_m     # and bf16; and --size 1.7b for all three
uv run rizzo devices                                    # also with RIZZO_LLAMA_DIR=runtimes/<package>
uv run pytest -q && RIZZO_REAL=1 uv run pytest -q -m integration
# per configuration (--device cuda|vulkan|cpu, --size, --quant, --kv-type, --threads):
uv run rizzo decide examples/{ticket,house,numeric}.json --device D --size S --quant Q
uv run rizzo evaluate benchmarks/smoke.jsonl --compare-modes --device D --size S --quant Q --output …
uv run rizzo evaluate benchmarks/perturbations.jsonl --compare-modes …
uv run python scripts/validate_checkpoint.py --device D --size S --quant Q --output …
uv run python scripts/semif_compare.py --system rizzo --semif ../SemIf --device D --size S --quant Q --output …
uv run python scripts/semif_report.py RUN --semif ../SemIf --against maintainers=results/semif-compare/…
uv run python scripts/typed_decisions.py typed-decisions/all/test.jsonl --model GGUF --device D --output …
```

Every command loads the model again. Runs were strictly sequential (never two model processes at once). SemIf at `ca3ba65`, typed-decisions at `c76749e`.

## `rizzo devices`

<details><summary>cuda package: auto_selects <code>CUDA0</code></summary>

```json
{
  "llama.cpp": {
    "release": "b11081",
    "host": "linux/x64",
    "packages": [
      "cuda",
      "rocm",
      "sycl",
      "vulkan",
      "cpu"
    ],
    "recommended": "cuda",
    "installed": [
      "cuda",
      "vulkan",
      "cpu"
    ],
    "directory": "runtimes/llama-b11081-linux-x64-cuda",
    "devices": [
      {
        "name": "CUDA0",
        "description": "NVIDIA RTX PRO 4000 Blackwell",
        "kind": "gpu",
        "backend": "CUDA",
        "total_bytes": 25239027712
      },
      {
        "name": "CPU",
        "description": "AMD Ryzen 9 9950X3D 16-Core Processor",
        "kind": "cpu",
        "backend": "CPU",
        "total_bytes": 66462609408
      }
    ],
    "auto_selects": "CUDA0"
  },
  "mlx": {
    "installed": false
  }
}
```

</details>

<details><summary>cpu package: auto_selects <code>CPU</code></summary>

```json
{
  "llama.cpp": {
    "release": "b11081",
    "host": "linux/x64",
    "packages": [
      "cuda",
      "rocm",
      "sycl",
      "vulkan",
      "cpu"
    ],
    "recommended": "cuda",
    "installed": [
      "cuda",
      "vulkan",
      "cpu"
    ],
    "directory": "runtimes/llama-b11081-linux-x64-cpu",
    "devices": [
      {
        "name": "CPU",
        "description": "AMD Ryzen 9 9950X3D 16-Core Processor",
        "kind": "cpu",
        "backend": "CPU",
        "total_bytes": 66462609408
      }
    ],
    "auto_selects": "CPU"
  },
  "mlx": {
    "installed": false
  }
}
```

</details>

<details><summary>vulkan package: auto_selects <code>Vulkan0</code></summary>

```json
{
  "llama.cpp": {
    "release": "b11081",
    "host": "linux/x64",
    "packages": [
      "cuda",
      "rocm",
      "sycl",
      "vulkan",
      "cpu"
    ],
    "recommended": "cuda",
    "installed": [
      "cuda",
      "vulkan",
      "cpu"
    ],
    "directory": "runtimes/llama-b11081-linux-x64-vulkan",
    "devices": [
      {
        "name": "Vulkan0",
        "description": "NVIDIA RTX PRO 4000 Blackwell",
        "kind": "gpu",
        "backend": "Vulkan",
        "total_bytes": 25655508992
      },
      {
        "name": "CPU",
        "description": "AMD Ryzen 9 9950X3D 16-Core Processor",
        "kind": "cpu",
        "backend": "CPU",
        "total_bytes": 66462609408
      }
    ],
    "auto_selects": "Vulkan0"
  },
  "mlx": {
    "installed": false
  }
}
```

</details>

## Test suites

| Suite | Passed | Failed | Skipped | Summary |
|---|---|---|---|---|
| unit tests | 71 | 0 | 13 | `71 passed, 13 skipped, 2 warnings in 0.73s` |
| integration tests (default: smallest Q8_0 on disk = 1.7B) | 4 | 1 | 1 | `1 failed, 4 passed, 1 skipped, 78 deselected, 2 warnings in 3.58s` |
| integration tests (pinned to the 4B Q8_0) | 5 | 0 | 1 | `5 passed, 1 skipped, 78 deselected, 2 warnings in 8.29s` |

The integration suite picks the smallest Q8_0 GGUF on disk (the 1.7B once it is downloaded) and `--device auto` (CUDA here); the second row pins it to the 4B with a local pytest plugin. The 13 unit-test skips are the 6 integration tests (they need `RIZZO_REAL=1`, run separately above) and 7 MLX tests (the `mlx` extra is not installed).

## SemIf fixtures (`scripts/semif_compare.py`)

Balanced accuracy is the mean over families; held-out = the odd source groups (`semif_report.py`); latency is per short-state decision on a warm model. Direct mode uses 37 states (777 decisions) for the CUDA and Vulkan 4B Q8_0 and CUDA 4B BF16 rows, the script's default 3 states (63) elsewhere. Peak memory is rizzo's own `peak_device_bytes` (drop in free GPU memory) over the whole run; on the CPU, the process's peak RSS.

| Config | authored144 | perturbations108 | held-out | p50 / p95 ms | shape777 shared / direct dec/s | shared vs direct flips | peak memory |
|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | 0.810 | 0.830 | 0.785 / 0.843 | 36 / 39 | 30.03 / 3.37 | 7/777 | 5.61 GiB |
| 4B BF16 · CUDA | 0.819 | 0.848 | 0.806 / 0.843 | 39 / 41 | 29.12 / 3.21 | 1/777 | 9.29 GiB |
| 4B Q4_K_M · CUDA | 0.765 | 0.830 | 0.724 / 0.806 | 37 / 40 | 28.54 / 3.12 | 2/63 | 3.96 GiB |
| 1.7B Q8_0 · CUDA | 0.700 | 0.646 | 0.690 / 0.532 | 18 / 20 | 49.70 / 7.87 | 2/63 | 2.35 GiB |
| 1.7B Q4_K_M · CUDA | 0.607 | 0.472 | 0.603 / 0.481 | 18 / 20 | 48.65 / 7.40 | 6/63 | 1.69 GiB |
| 1.7B BF16 · CUDA | 0.683 | 0.640 | 0.690 / 0.532 | 17 / 18 | 48.40 / 7.03 | 0/63 | 3.93 GiB |
| 4B Q8_0 · CUDA · KV q8_0 | 0.816 | 0.835 | 0.799 / 0.856 | 35 / 37 | 29.54 / 3.32 | 1/63 | 4.97 GiB |
| 4B Q8_0 · CUDA · KV q4_0 | 0.811 | 0.859 | 0.812 / 0.875 | 34 / 37 | 29.56 / 3.33 | 1/63 | 4.62 GiB |
| 4B BF16 · CUDA · KV q8_0 | 0.823 | 0.865 | 0.810 / 0.875 | 38 / 40 | 28.90 / 3.20 | 0/63 | 8.65 GiB |
| 4B BF16 · CUDA · KV q4_0 | 0.807 | 0.826 | 0.795 / 0.833 | 37 / 40 | 28.80 / 3.21 | 3/63 | 8.30 GiB |
| 4B Q4_K_M · CUDA · KV q8_0 | 0.782 | 0.822 | 0.742 / 0.801 | 36 / 39 | 28.28 / 3.11 | 1/63 | 3.32 GiB |
| 4B Q4_K_M · CUDA · KV q4_0 | 0.762 | 0.847 | 0.724 / 0.819 | 36 / 38 | 28.31 / 3.12 | 5/63 | 2.97 GiB |
| 1.7B Q8_0 · CUDA · KV q8_0 | 0.687 | 0.652 | 0.708 / 0.532 | 17 / 19 | 49.29 / 7.69 | 2/63 | 2.10 GiB |
| 1.7B Q8_0 · CUDA · KV q4_0 | 0.687 | 0.576 | 0.677 / 0.532 | 17 / 19 | 49.43 / 7.74 | 21/63 | 1.96 GiB |
| 1.7B Q4_K_M · CUDA · KV q8_0 | 0.594 | 0.464 | 0.577 / 0.412 | 18 / 20 | 48.28 / 7.28 | 9/63 | 1.44 GiB |
| 1.7B Q4_K_M · CUDA · KV q4_0 | 0.566 | 0.476 | 0.533 / 0.449 | 17 / 19 | 48.31 / 7.31 | 15/63 | 1.30 GiB |
| 1.7B BF16 · CUDA · KV q8_0 | 0.683 | 0.640 | 0.690 / 0.532 | 17 / 18 | 47.83 / 6.92 | 0/63 | 3.68 GiB |
| 1.7B BF16 · CUDA · KV q4_0 | 0.668 | 0.602 | 0.685 / 0.532 | 17 / 18 | 47.77 / 6.93 | 14/63 | 3.54 GiB |
| 4B Q8_0 · Vulkan | 0.816 | 0.854 | 0.799 / 0.861 | 66 / 70 | 23.10 / 2.60 | 7/777 | 6.08 GiB |
| 4B Q8_0 · CPU | 0.816 | 0.873 | 0.799 / 0.880 | 727 / 818 | 1.68 / 0.14 | 1/63 | RSS 5.67 GiB |
| 4B BF16 · CPU | 0.829 | 0.842 | 0.824 / 0.843 | 718 / 775 | 1.73 / 0.14 | 0/63 | RSS 9.27 GiB |
| 4B Q4_K_M · CPU | 0.765 | 0.841 | 0.724 / 0.819 | 652 / 709 | 1.86 / 0.15 | 0/63 | RSS 5.59 GiB |
| 1.7B Q8_0 · CPU | 0.713 | 0.664 | 0.708 / 0.551 | 293 / 341 | 4.13 / 0.35 | 1/63 | RSS 2.41 GiB |
| 1.7B Q4_K_M · CPU | 0.585 | 0.471 | 0.582 / 0.472 | 261 / 282 | 4.56 / 0.39 | 9/63 | RSS 2.35 GiB |
| 1.7B BF16 · CPU | 0.683 | 0.640 | 0.690 / 0.532 | 287 / 311 | 4.42 / 0.38 | 0/63 | RSS 3.90 GiB |
| 4B Q8_0 · CPU · KV q8_0 | 0.810 | 0.854 | 0.785 / 0.861 | 738 / 822 | 1.15 / 0.09 | 0/63 | RSS 5.02 GiB |
| 4B Q8_0 · CPU · KV q4_0 | 0.801 | 0.884 | 0.779 / 0.894 | 778 / 886 | 0.94 / 0.08 | 0/63 | RSS 4.66 GiB |
| 4B BF16 · CPU · KV q8_0 | 0.814 | 0.842 | 0.793 / 0.843 | 731 / 800 | 1.23 / 0.10 | 0/63 | RSS 8.61 GiB |
| 4B BF16 · CPU · KV q4_0 | 0.759 | 0.830 | 0.736 / 0.843 | 767 / 831 | 0.99 / 0.08 | 0/63 | RSS 8.26 GiB |
| 4B Q4_K_M · CPU · KV q8_0 | 0.783 | 0.811 | 0.756 / 0.787 | 668 / 728 | 1.29 / 0.11 | 1/63 | RSS 4.93 GiB |
| 4B Q4_K_M · CPU · KV q4_0 | 0.768 | 0.841 | 0.747 / 0.801 | 700 / 766 | 1.03 / 0.09 | 3/63 | RSS 4.58 GiB |
| 1.7B Q8_0 · CPU · KV q8_0 | 0.694 | 0.646 | 0.696 / 0.532 | 301 / 357 | 3.21 / 0.27 | 0/63 | RSS 2.16 GiB |
| 1.7B Q8_0 · CPU · KV q4_0 | 0.659 | 0.537 | 0.660 / 0.514 | 310 / 368 | 2.49 / 0.21 | 0/63 | RSS 2.02 GiB |
| 1.7B Q4_K_M · CPU · KV q8_0 | 0.591 | 0.472 | 0.580 / 0.468 | 264 / 290 | 3.51 / 0.29 | 1/63 | RSS 2.10 GiB |
| 1.7B Q4_K_M · CPU · KV q4_0 | 0.572 | 0.434 | 0.582 / 0.412 | 278 / 309 | 2.68 / 0.22 | 4/63 | RSS 1.96 GiB |
| 1.7B BF16 · CPU · KV q8_0 | 0.683 | 0.640 | 0.690 / 0.514 | 291 / 317 | 3.34 / 0.28 | 0/63 | RSS 3.65 GiB |
| 1.7B BF16 · CPU · KV q4_0 | 0.653 | 0.599 | 0.651 / 0.551 | 302 / 331 | 2.59 / 0.22 | 0/63 | RSS 3.51 GiB |
| *ref* 4B Q8_0 CUDA (maintainers) | 0.812 | 0.848 | 0.793 / 0.861 | 49 / 52 | 20.99 / 2.60 | 13/777 | 5.6 GiB |
| *ref* 4B BF16 CUDA (maintainers) | 0.829 | 0.859 | 0.824 / 0.875 | 60 / 63 | 17.75 / 1.97 | 1/777 | 9.3 GiB |
| *ref* 4B Q4_K_M CUDA (maintainers) | 0.769 | 0.835 | 0.730 / 0.801 | 51 / 54 | 19.98 / 2.39 | 2/63 | 3.9 GiB |
| *ref* 4B Q8_0 Vulkan (maintainers) | 0.807 | 0.854 | 0.781 / 0.861 | 90 / 94 | 14.84 / 1.74 | 0/63 | 6.0 GiB |
| *ref* 1.7B Q8_0 CUDA (maintainers) | 0.678 | 0.640 | 0.690 / 0.514 | 25 / 27 | 31.59 / 5.51 | 31/777 | 2.3 GiB |

<details><summary>Paired differences on authored144 (ours − theirs, 95 % bootstrap over source groups) and rows with a different argmax</summary>

| Config | Against | Difference | 95 % CI | Different argmax |
|---|---|---|---|---|
| 4B Q8_0 · CUDA | maintainers | -0.002 | [-0.027, +0.022] | 4/252 |
| 4B BF16 · CUDA | maintainers | -0.009 | [-0.028, +0.000] | 4/252 |
| 4B Q4_K_M · CUDA | maintainers | -0.004 | [-0.028, +0.014] | 4/252 |
| 1.7B Q8_0 · CUDA | maintainers | 0.022 | [+0.000, +0.047] | 6/252 |
| 4B Q8_0 · CUDA · KV q8_0 | this-cuda-kv-f16 | 0.006 | [+0.000, +0.020] | 4/252 |
| 4B Q8_0 · CUDA · KV q4_0 | this-cuda-kv-f16 | 0.001 | [-0.030, +0.035] | 10/252 |
| 4B BF16 · CUDA · KV q8_0 | this-cuda-kv-f16 | 0.003 | [-0.018, +0.028] | 4/252 |
| 4B BF16 · CUDA · KV q4_0 | this-cuda-kv-f16 | -0.012 | [-0.051, +0.026] | 16/252 |
| 4B Q4_K_M · CUDA · KV q8_0 | this-cuda-kv-f16 | 0.017 | [+0.000, +0.041] | 4/252 |
| 4B Q4_K_M · CUDA · KV q4_0 | this-cuda-kv-f16 | -0.003 | [-0.028, +0.022] | 7/252 |
| 1.7B Q8_0 · CUDA · KV q8_0 | this-cuda-kv-f16 | -0.013 | [-0.042, +0.018] | 5/252 |
| 1.7B Q8_0 · CUDA · KV q4_0 | this-cuda-kv-f16 | -0.013 | [-0.051, +0.029] | 26/252 |
| 1.7B Q4_K_M · CUDA · KV q8_0 | this-cuda-kv-f16 | -0.013 | [-0.033, +0.000] | 9/252 |
| 1.7B Q4_K_M · CUDA · KV q4_0 | this-cuda-kv-f16 | -0.042 | [-0.076, -0.008] | 28/252 |
| 1.7B BF16 · CUDA · KV q8_0 | this-cuda-kv-f16 | 0.000 | [+0.000, +0.000] | 1/252 |
| 1.7B BF16 · CUDA · KV q4_0 | this-cuda-kv-f16 | -0.015 | [-0.062, +0.032] | 41/252 |
| 4B Q8_0 · Vulkan | maintainers | 0.009 | [+0.000, +0.028] | 1/252 |
| 4B Q8_0 · Vulkan | this-cuda | 0.006 | [+0.000, +0.020] | 4/252 |
| 4B Q8_0 · CPU | this-cuda | 0.006 | [+0.000, +0.020] | 5/252 |
| 4B BF16 · CPU | this-cuda | 0.009 | [+0.000, +0.028] | 2/252 |
| 4B Q4_K_M · CPU | this-cuda | 0.000 | [+0.000, +0.000] | 1/252 |
| 1.7B Q8_0 · CPU | this-cuda | 0.013 | [-0.010, +0.041] | 7/252 |
| 1.7B Q4_K_M · CPU | this-cuda | -0.022 | [-0.045, -0.005] | 9/252 |
| 1.7B BF16 · CPU | this-cuda | 0.000 | [+0.000, +0.000] | 0/252 |
| 4B Q8_0 · CPU · KV q8_0 | this-cpu-kv-f16 | -0.006 | [-0.020, +0.000] | 3/252 |
| 4B Q8_0 · CPU · KV q4_0 | this-cpu-kv-f16 | -0.015 | [-0.050, +0.016] | 11/252 |
| 4B BF16 · CPU · KV q8_0 | this-cpu-kv-f16 | -0.015 | [-0.037, +0.000] | 2/252 |
| 4B BF16 · CPU · KV q4_0 | this-cpu-kv-f16 | -0.070 | [-0.109, -0.033] | 17/252 |
| 4B Q4_K_M · CPU · KV q8_0 | this-cpu-kv-f16 | 0.017 | [-0.005, +0.046] | 6/252 |
| 4B Q4_K_M · CPU · KV q4_0 | this-cpu-kv-f16 | 0.003 | [-0.033, +0.037] | 9/252 |
| 1.7B Q8_0 · CPU · KV q8_0 | this-cpu-kv-f16 | -0.019 | [-0.050, +0.007] | 6/252 |
| 1.7B Q8_0 · CPU · KV q4_0 | this-cpu-kv-f16 | -0.054 | [-0.102, -0.010] | 36/252 |
| 1.7B Q4_K_M · CPU · KV q8_0 | this-cpu-kv-f16 | 0.006 | [-0.015, +0.028] | 11/252 |
| 1.7B Q4_K_M · CPU · KV q4_0 | this-cpu-kv-f16 | -0.013 | [-0.041, +0.012] | 16/252 |
| 1.7B BF16 · CPU · KV q8_0 | this-cpu-kv-f16 | 0.000 | [+0.000, +0.000] | 2/252 |
| 1.7B BF16 · CPU · KV q4_0 | this-cpu-kv-f16 | -0.030 | [-0.080, +0.023] | 31/252 |

</details>

<details><summary>Stability and weak spots</summary>

| Config | Flips reversal / wrapper / context | Confident on missing evidence | rule_application perturbed (NLL) |
|---|---|---|---|
| 4B Q8_0 · CUDA | 6 / 5 / 3 | 6/36 | 0.556 (1.69) |
| 4B BF16 · CUDA | 5 / 2 / 4 | 6/36 | 0.611 (1.69) |
| 4B Q4_K_M · CUDA | 6 / 1 / 2 | 6/36 | 0.611 (1.82) |
| 1.7B Q8_0 · CUDA | 18 / 5 / 4 | 2/36 | 0.315 (3.50) |
| 1.7B Q4_K_M · CUDA | 17 / 4 / 2 | 6/36 | 0.056 (4.26) |
| 1.7B BF16 · CUDA | 18 / 4 / 4 | 2/36 | 0.296 (3.55) |
| 4B Q8_0 · CUDA · KV q8_0 | 4 / 3 / 3 | 6/36 | 0.537 (1.69) |
| 4B Q8_0 · CUDA · KV q4_0 | 4 / 5 / 3 | 7/36 | 0.611 (1.58) |
| 4B BF16 · CUDA · KV q8_0 | 5 / 3 / 2 | 6/36 | 0.630 (1.67) |
| 4B BF16 · CUDA · KV q4_0 | 3 / 3 / 3 | 7/36 | 0.611 (1.60) |
| 4B Q4_K_M · CUDA · KV q8_0 | 6 / 2 / 1 | 6/36 | 0.556 (1.81) |
| 4B Q4_K_M · CUDA · KV q4_0 | 7 / 3 / 2 | 6/36 | 0.630 (2.03) |
| 1.7B Q8_0 · CUDA · KV q8_0 | 18 / 5 / 5 | 2/36 | 0.333 (3.45) |
| 1.7B Q8_0 · CUDA · KV q4_0 | 18 / 8 / 8 | 1/36 | 0.296 (3.32) |
| 1.7B Q4_K_M · CUDA · KV q8_0 | 18 / 4 / 2 | 7/36 | 0.056 (4.33) |
| 1.7B Q4_K_M · CUDA · KV q4_0 | 18 / 3 / 3 | 9/36 | 0.167 (5.04) |
| 1.7B BF16 · CUDA · KV q8_0 | 18 / 4 / 5 | 2/36 | 0.296 (3.57) |
| 1.7B BF16 · CUDA · KV q4_0 | 15 / 12 / 8 | 2/36 | 0.407 (3.65) |
| 4B Q8_0 · Vulkan | 4 / 3 / 5 | 6/36 | 0.630 (1.66) |
| 4B Q8_0 · CPU | 3 / 3 / 5 | 6/36 | 0.685 (1.65) |
| 4B BF16 · CPU | 5 / 3 / 4 | 6/36 | 0.593 (1.69) |
| 4B Q4_K_M · CPU | 6 / 2 / 2 | 6/36 | 0.611 (1.82) |
| 1.7B Q8_0 · CPU | 17 / 4 / 6 | 3/36 | 0.370 (3.41) |
| 1.7B Q4_K_M · CPU | 18 / 4 / 2 | 4/36 | 0.074 (4.10) |
| 1.7B BF16 · CPU | 18 / 4 / 4 | 2/36 | 0.296 (3.55) |
| 4B Q8_0 · CPU · KV q8_0 | 6 / 4 / 4 | 6/36 | 0.630 (1.66) |
| 4B Q8_0 · CPU · KV q4_0 | 5 / 3 / 3 | 5/36 | 0.685 (1.60) |
| 4B BF16 · CPU · KV q8_0 | 5 / 3 / 4 | 6/36 | 0.593 (1.69) |
| 4B BF16 · CPU · KV q4_0 | 5 / 3 / 6 | 7/36 | 0.556 (1.79) |
| 4B Q4_K_M · CPU · KV q8_0 | 5 / 2 / 2 | 5/36 | 0.556 (1.82) |
| 4B Q4_K_M · CPU · KV q4_0 | 5 / 2 / 3 | 6/36 | 0.630 (1.78) |
| 1.7B Q8_0 · CPU · KV q8_0 | 18 / 5 / 4 | 2/36 | 0.315 (3.51) |
| 1.7B Q8_0 · CPU · KV q4_0 | 14 / 7 / 4 | 3/36 | 0.241 (3.97) |
| 1.7B Q4_K_M · CPU · KV q8_0 | 16 / 4 / 2 | 8/36 | 0.056 (4.00) |
| 1.7B Q4_K_M · CPU · KV q4_0 | 18 / 4 / 4 | 8/36 | 0.056 (4.72) |
| 1.7B BF16 · CPU · KV q8_0 | 18 / 6 / 4 | 2/36 | 0.315 (3.60) |
| 1.7B BF16 · CPU · KV q4_0 | 17 / 7 / 8 | 2/36 | 0.315 (3.68) |

</details>

## Smoke (`benchmarks/smoke.jsonl --compare-modes`)

| Config | Accuracy | NLL | Brier | ECE | Median / p95 ms | dec/s | Shared vs direct (max Δp) | Peak memory |
|---|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | 0.950 | 0.450 | 0.077 | 0.038 | 43 / 118 | 22.46 | 0/20 (0.000027) | 5.61 GiB |
| 4B BF16 · CUDA | 0.950 | 0.449 | 0.078 | 0.040 | 42 / 134 | 23.14 | 0/20 (0.000002) | 9.29 GiB |
| 4B Q4_K_M · CUDA | 0.900 | 0.735 | 0.191 | 0.075 | 43 / 121 | 22.52 | 0/20 (0.004331) | 3.96 GiB |
| 1.7B Q8_0 · CUDA | 0.400 | 3.865 | 1.097 | 0.528 | 22 / 61 | 46.95 | 0/20 (0.059544) | 2.35 GiB |
| 1.7B Q4_K_M · CUDA | 0.350 | 3.584 | 1.071 | 0.578 | 23 / 62 | 46.64 | 0/20 (0.167532) | 1.68 GiB |
| 1.7B BF16 · CUDA | 0.450 | 3.462 | 1.014 | 0.512 | 21 / 72 | 46.81 | 0/20 (0.032929) | 3.93 GiB |
| 4B Q8_0 · CUDA · KV q8_0 | 0.950 | 0.445 | 0.077 | 0.038 | 42 / 122 | 23.54 | 0/20 (0.000028) | 4.97 GiB |
| 4B Q8_0 · CUDA · KV q4_0 | 0.950 | 0.549 | 0.099 | 0.051 | 40 / 119 | 24.45 | 0/20 (0.000029) | 4.62 GiB |
| 4B BF16 · CUDA · KV q8_0 | 0.950 | 0.448 | 0.077 | 0.039 | 41 / 135 | 23.44 | 0/20 (0.000003) | 8.65 GiB |
| 4B BF16 · CUDA · KV q4_0 | 0.950 | 0.339 | 0.076 | 0.036 | 41 / 133 | 23.68 | 0/20 (0.000040) | 8.30 GiB |
| 4B Q4_K_M · CUDA · KV q8_0 | 0.900 | 0.663 | 0.174 | 0.091 | 41 / 123 | 22.67 | 0/20 (0.040437) | 3.32 GiB |
| 4B Q4_K_M · CUDA · KV q4_0 | 0.900 | 0.621 | 0.221 | 0.127 | 41 / 123 | 23.37 | 0/20 (0.009315) | 2.96 GiB |
| 1.7B Q8_0 · CUDA · KV q8_0 | 0.450 | 3.601 | 1.043 | 0.552 | 23 / 62 | 47.29 | 0/20 (0.027087) | 2.10 GiB |
| 1.7B Q8_0 · CUDA · KV q4_0 | 0.400 | 3.770 | 1.129 | 0.597 | 22 / 61 | 48.42 | 0/20 (0.028102) | 1.96 GiB |
| 1.7B Q4_K_M · CUDA · KV q8_0 | 0.400 | 3.341 | 1.043 | 0.577 | 25 / 71 | 44.40 | 0/20 (0.155390) | 1.43 GiB |
| 1.7B Q4_K_M · CUDA · KV q4_0 | 0.300 | 4.207 | 1.224 | 0.660 | 22 / 61 | 47.67 | 0/20 (0.330020) | 1.30 GiB |
| 1.7B BF16 · CUDA · KV q8_0 | 0.450 | 3.470 | 1.007 | 0.512 | 21 / 72 | 46.52 | 0/20 (0.012723) | 3.68 GiB |
| 1.7B BF16 · CUDA · KV q4_0 | 0.350 | 4.232 | 1.195 | 0.589 | 22 / 72 | 46.75 | 0/20 (0.253036) | 3.54 GiB |
| 4B Q8_0 · Vulkan | 0.950 | 0.458 | 0.079 | 0.040 | 68 / 3252 | 4.14 | 0/20 (0.000006) | 5.65 GiB |
| 4B Q8_0 · CPU · threads default (4) | 0.950 | 0.429 | 0.077 | 0.037 | 2542 / 7807 | 0.36 | 0/20 (0.000012) | RSS 5.66 GiB |
| 4B Q8_0 · CPU · threads 16 | 0.950 | 0.429 | 0.077 | 0.037 | 838 / 2454 | 1.10 | 0/20 (0.000012) | RSS 5.66 GiB |
| 4B Q8_0 · CPU · threads 32 | 0.950 | 0.429 | 0.077 | 0.037 | 1308 / 3752 | 0.86 | 0/20 (0.000012) | RSS 5.66 GiB |
| 4B Q8_0 · CPU | 0.950 | 0.429 | 0.077 | 0.037 | 922 / 2580 | 1.07 | 0/20 (0.000012) | RSS 5.66 GiB |
| 4B BF16 · CPU | 0.950 | 0.445 | 0.078 | 0.041 | 825 / 2432 | 1.12 | 0/20 (0.000004) | RSS 9.25 GiB |
| 4B Q4_K_M · CPU | 0.900 | 0.730 | 0.188 | 0.057 | 746 / 2227 | 1.23 | 0/20 (0.012698) | RSS 5.57 GiB |
| 1.7B Q8_0 · CPU | 0.400 | 3.699 | 1.076 | 0.526 | 358 / 1000 | 2.64 | 0/20 (0.059870) | RSS 2.39 GiB |
| 1.7B Q4_K_M · CPU | 0.350 | 3.368 | 1.085 | 0.549 | 324 / 901 | 3.06 | 0/20 (0.102410) | RSS 2.33 GiB |
| 1.7B BF16 · CPU | 0.450 | 3.470 | 1.014 | 0.515 | 387 / 976 | 2.80 | 0/20 (0.006990) | RSS 3.89 GiB |
| 4B Q8_0 · CPU · KV q8_0 | 0.950 | 0.433 | 0.077 | 0.038 | 887 / 2746 | 1.04 | 0/20 (0.000000) | RSS 5.00 GiB |
| 4B Q8_0 · CPU · KV q4_0 | 0.950 | 0.472 | 0.088 | 0.048 | 1011 / 3064 | 0.95 | 0/20 (0.000000) | RSS 4.65 GiB |
| 4B BF16 · CPU · KV q8_0 | 0.950 | 0.436 | 0.077 | 0.038 | 854 / 2683 | 1.05 | 0/20 (0.000000) | RSS 8.59 GiB |
| 4B BF16 · CPU · KV q4_0 | 0.950 | 0.578 | 0.093 | 0.041 | 896 / 2999 | 0.97 | 0/20 (0.000000) | RSS 8.24 GiB |
| 4B Q4_K_M · CPU · KV q8_0 | 0.900 | 0.697 | 0.178 | 0.087 | 770 / 2461 | 1.15 | 0/20 (0.039762) | RSS 4.92 GiB |
| 4B Q4_K_M · CPU · KV q4_0 | 0.900 | 0.657 | 0.169 | 0.063 | 849 / 2815 | 1.05 | 0/20 (0.136352) | RSS 4.57 GiB |
| 1.7B Q8_0 · CPU · KV q8_0 | 0.450 | 3.628 | 1.057 | 0.562 | 371 / 1048 | 2.60 | 0/20 (0.000000) | RSS 2.14 GiB |
| 1.7B Q8_0 · CPU · KV q4_0 | 0.350 | 3.742 | 1.122 | 0.598 | 379 / 1177 | 2.41 | 0/20 (0.000000) | RSS 2.00 GiB |
| 1.7B Q4_K_M · CPU · KV q8_0 | 0.350 | 3.384 | 1.045 | 0.544 | 324 / 990 | 2.88 | 0/20 (0.130118) | RSS 2.08 GiB |
| 1.7B Q4_K_M · CPU · KV q4_0 | 0.250 | 4.159 | 1.324 | 0.717 | 339 / 1091 | 2.63 | 1/20 (0.213939) | RSS 1.94 GiB |
| 1.7B BF16 · CPU · KV q8_0 | 0.450 | 3.503 | 1.018 | 0.518 | 336 / 1046 | 2.69 | 0/20 (0.000005) | RSS 3.63 GiB |
| 1.7B BF16 · CPU · KV q4_0 | 0.400 | 3.868 | 1.066 | 0.526 | 393 / 1164 | 2.47 | 0/20 (0.000003) | RSS 3.49 GiB |

*ref* maintainers, 4B Q8_0 CUDA (RTX 5060 Ti): accuracy 0.95, NLL 0.459, median 66 ms. Community: M3 Pro Metal 0.95 / 0.450 / 518 ms, 0/20 changed (#5); Radeon 780M Vulkan 0.95 / 0.448 / 620 ms, 0/20 (#11); i5-1334U CPU-only Q4_K_M 0.90 / 0.685 / 11.9 s (#7).

## Perturbations (`benchmarks/perturbations.jsonl --compare-modes`)

| Config | Accuracy | NLL | Brier | ECE | Median / p95 ms | dec/s | Shared vs direct (max Δp) | Peak memory |
|---|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | 1 | 0.020 | 0.003 | 0.019 | 40 / 43 | 25.10 | 0/9 (0.000000) | 5.57 GiB |
| 4B BF16 · CUDA | 1 | 0.018 | 0.002 | 0.018 | 40 / 45 | 24.06 | 0/9 (0.000000) | 9.29 GiB |
| 4B Q4_K_M · CUDA | 1 | 0.012 | 0.001 | 0.012 | 40 / 43 | 24.65 | 0/9 (0.000000) | 3.96 GiB |
| 1.7B Q8_0 · CUDA | 0.444 | 3.126 | 0.960 | 0.551 | 19 / 22 | 52.47 | 0/9 (0.000000) | 2.35 GiB |
| 1.7B Q4_K_M · CUDA | 0.556 | 2.412 | 0.895 | 0.520 | 20 / 23 | 50.54 | 0/9 (0.000000) | 1.68 GiB |
| 1.7B BF16 · CUDA | 0.556 | 2.759 | 0.871 | 0.499 | 19 / 22 | 52.39 | 0/9 (0.000000) | 3.93 GiB |
| 4B Q8_0 · CPU | 1 | 0.013 | 0.001 | 0.013 | 829 / 944 | 1.20 | 0/9 (0.000000) | RSS 5.63 GiB |
| 4B BF16 · CPU | 1 | 0.017 | 0.002 | 0.017 | 822 / 885 | 1.21 | 0/9 (0.000000) | RSS 9.22 GiB |
| 4B Q4_K_M · CPU | 1 | 0.010 | 0.000 | 0.009 | 772 / 807 | 1.32 | 0/9 (0.000000) | RSS 5.55 GiB |
| 1.7B Q8_0 · CPU | 0.556 | 3.113 | 0.935 | 0.535 | 351 / 395 | 2.85 | 0/9 (0.000000) | RSS 2.38 GiB |
| 1.7B Q4_K_M · CPU | 0.556 | 2.294 | 0.840 | 0.466 | 306 / 321 | 3.31 | 0/9 (0.000000) | RSS 2.32 GiB |
| 1.7B BF16 · CPU | 0.556 | 2.782 | 0.865 | 0.455 | 336 / 376 | 2.98 | 0/9 (0.000000) | RSS 3.87 GiB |

## Batch-size sweep (`semif_compare.py --batch-size N --direct-states 0`, 4B Q8_0)

| Config | shape777 shared dec/s | Peak memory | Wall s |
|---|---|---|---|
| 4B Q8_0 · CUDA · batch 1 | 25.11 | 5.61 GiB | 44 |
| 4B Q8_0 · CUDA · batch 2 | 27.70 | 5.61 GiB | 41 |
| 4B Q8_0 · CUDA · batch 4 | 30.26 | 5.61 GiB | 38 |
| 4B Q8_0 · CUDA · batch 8 | 30.62 | 5.61 GiB | 38 |
| 4B Q8_0 · CUDA · batch 16 | 30.90 | 5.61 GiB | 38 |
| 4B Q8_0 · CPU · batch 1 | 1.51 | RSS 5.67 GiB | 722 |
| 4B Q8_0 · CPU · batch 2 | 1.64 | RSS 5.67 GiB | 681 |
| 4B Q8_0 · CPU · batch 4 | 1.69 | RSS 5.67 GiB | 667 |
| 4B Q8_0 · CPU · batch 8 | 1.70 | RSS 5.67 GiB | 664 |
| 4B Q8_0 · CPU · batch 16 | 1.72 | RSS 5.68 GiB | 659 |

## typed-decisions (`scripts/typed_decisions.py`, 400 test cases, zero-shot)

| Config | Accuracy | KL | Brier | ECE | p50 / p95 ms per case | dec/s |
|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | 0.574 | 2.888 | 0.479 | 0.349 | 143 / 180 | 34.44 |
| 4B BF16 · CUDA | 0.575 | 2.935 | 0.479 | 0.348 | 148 / 186 | 33.43 |
| 4B Q4_K_M · CUDA | 0.574 | 2.321 | 0.445 | 0.324 | 151 / 189 | 32.92 |
| 1.7B Q8_0 · CUDA | 0.530 | 3.023 | 0.496 | 0.346 | 72 / 90 | 69.36 |
| 1.7B Q4_K_M · CUDA | 0.481 | 2.799 | 0.579 | 0.427 | 74 / 92 | 67.43 |
| 1.7B BF16 · CUDA | 0.528 | 3.019 | 0.494 | 0.348 | 73 / 92 | 68.20 |
| 4B Q8_0 · CPU | 0.576 | 2.902 | 0.480 | 0.348 | 3008 / 3711 | 1.68 |
| 4B BF16 · CPU | 0.575 | 2.935 | 0.479 | 0.348 | 2904 / 3616 | 1.74 |
| 4B Q4_K_M · CPU | 0.576 | 2.331 | 0.444 | 0.323 | 2683 / 3291 | 1.88 |
| 1.7B Q8_0 · CPU | 0.528 | 2.997 | 0.495 | 0.347 | 1199 / 1488 | 4.21 |
| 1.7B Q4_K_M · CPU | 0.481 | 2.839 | 0.583 | 0.431 | 1070 / 1336 | 4.70 |
| 1.7B BF16 · CPU | 0.526 | 3.028 | 0.495 | 0.351 | 1155 / 1432 | 4.35 |

*ref* maintainers (docs/training.md, RTX 5060 Ti): 4B Q8_0 0.574 / KL 2.899 / Brier 0.480 / 201 ms; 4B BF16 0.574 / 2.935 / 0.479 / 250 ms.

## `scripts/validate_checkpoint.py`

| Config | Smoke acc (NLL) | Long state shared / direct ms | Long state changed | Load s | Peak memory |
|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | 0.950 (0.450) | 345 / 1014 | 0 | 2.5 | 5.61 GiB |
| 4B BF16 · CUDA | 0.950 (0.449) | 354 / 1005 | 0 | 4.1 | 9.29 GiB |
| 4B Q4_K_M · CUDA | 0.900 (0.735) | 355 / 1049 | 0 | 1.6 | 3.96 GiB |
| 1.7B Q8_0 · CUDA | 0.400 (3.865) | 169 / 443 | 0 | 1.2 | 2.35 GiB |
| 1.7B Q4_K_M · CUDA | 0.350 (3.584) | 171 / 466 | 0 | 0.9 | 1.69 GiB |
| 1.7B BF16 · CUDA | 0.450 (3.462) | 183 / 482 | 0 | 1.9 | 3.93 GiB |
| 4B Q8_0 · CPU | 0.950 (0.429) | 7765 / 24916 | 0 | 2.1 | RSS 5.67 GiB |
| 4B BF16 · CPU | 0.950 (0.445) | 7418 / 23880 | 0 | 3.5 | RSS 9.27 GiB |
| 4B Q4_K_M · CPU | 0.900 (0.730) | 6901 / 22261 | 0 | 1.8 | RSS 5.59 GiB |
| 1.7B Q8_0 · CPU | 0.400 (3.699) | 3095 / 9829 | 0 | 0.9 | RSS 2.41 GiB |
| 1.7B Q4_K_M · CPU | 0.350 (3.368) | 2728 / 8734 | 0 | 0.8 | RSS 2.35 GiB |
| 1.7B BF16 · CPU | 0.450 (3.470) | 2929 / 9196 | 0 | 1.5 | RSS 3.90 GiB |

## `rizzo decide examples/*.json`

| Config | ticket.json answers | ticket / house / numeric ms |
|---|---|---|
| 4B Q8_0 · CUDA | needs_access_support=True, queue=access, urgency=1.9999213168228909, deadline_hours: insufficient_evidence | 140 / 103 / 78 |
| 4B BF16 · CUDA | needs_access_support=True, queue=access, urgency=1.999934620732511, deadline_hours: insufficient_evidence | 219 / 134 / 137 |
| 4B Q4_K_M · CUDA | needs_access_support=True, queue=access, urgency=1.9998314898287899, deadline_hours: insufficient_evidence | 145 / 102 / 84 |
| 1.7B Q8_0 · CUDA | needs_access_support=False, queue: insufficient_evidence, urgency: insufficient_evidence, deadline_hours: insufficient_evidence | 82 / 58 / 48 |
| 1.7B Q4_K_M · CUDA | needs_access_support=False, queue: insufficient_evidence, urgency: insufficient_evidence, deadline_hours: insufficient_evidence | 82 / 60 / 52 |
| 1.7B BF16 · CUDA | needs_access_support=False, queue: insufficient_evidence, urgency: insufficient_evidence, deadline_hours: insufficient_evidence | 117 / 91 / 79 |
| 4B Q8_0 · CPU | needs_access_support=True, queue=access, urgency=1.9999318522082647, deadline_hours: insufficient_evidence | 2441 / 1860 / 1537 |
| 4B BF16 · CPU | needs_access_support=True, queue=access, urgency=1.9999307522271041, deadline_hours: insufficient_evidence | 2453 / 1950 / 1332 |
| 4B Q4_K_M · CPU | needs_access_support=True, queue=access, urgency=1.9998857226744353, deadline_hours: insufficient_evidence | 2176 / 1750 / 1220 |
| 1.7B Q8_0 · CPU | needs_access_support=False, queue: insufficient_evidence, urgency: insufficient_evidence, deadline_hours: insufficient_evidence | 943 / 760 / 521 |
| 1.7B Q4_K_M · CPU | needs_access_support=False, queue: insufficient_evidence, urgency: insufficient_evidence, deadline_hours: insufficient_evidence | 880 / 675 / 484 |
| 1.7B BF16 · CPU | needs_access_support=False, queue: insufficient_evidence, urgency: insufficient_evidence, deadline_hours: insufficient_evidence | 953 / 721 / 517 |

## Telemetry per run

GPU: `nvidia-smi` at 100 ms (power, energy, memory, utilization, temperature) and DCGM at 1 s (SM activity; runs of 5 s or more); CPU: the child's rusage. Net energy = integral of GPU power over the run − 10 s idle baseline power × duration; each window covers the whole command, model loading included. The GPU also drives the desktop, so the baseline is not zero.

<details><summary>Table</summary>

| Run | Wall s | GPU avg / peak W | Net GPU kJ | GPU util % | VRAM max MiB | SM active | Temp max °C | CPU cores avg | Peak RSS GiB |
|---|---|---|---|---|---|---|---|---|---|
| T0-unit · unit tests | 1.1 | 32 / 33 | 0.002 | 24 | 1295 | — | 47 | 0.55 | 0.07 |
| T2-devices-cuda · rizzo devices | 0.5 | 33 / 39 | 0.001 | 24 | 1474 | — | 48 | 0.79 | 0.47 |
| T2-devices-cpu · rizzo devices | 0.3 | 34 / 34 | 0.002 | 24 | 1253 | — | 47 | 0.74 | 0.06 |
| T2-devices-vulkan · rizzo devices | 0.4 | 34 / 34 | 0.002 | 30 | 1256 | — | 47 | 0.77 | 0.17 |
| T1-integration · integration tests | 3.9 | 64 / 144 | 0.124 | 34 | 5681 | — | 55 | 0.99 | 2.59 |
| T1-integration-4b · integration tests | 8.6 | 43 / 94 | 0.122 | 25 | 11411 | 0.19 | 53 | 0.95 | 5.04 |
| 4B Q8_0 · CUDA · rizzo decide examples | 8.2 | 40 / 62 | 0.071 | 15 | 7265 | 0.04 | 50 | 0.98 | 4.54 |
| 4B Q8_0 · CUDA · smoke | 4.6 | 80 / 155 | 0.232 | 47 | 7269 | — | 59 | 0.99 | 4.54 |
| 4B Q8_0 · CUDA · perturbations | 3.6 | 49 / 144 | 0.078 | 36 | 7225 | — | 56 | 0.94 | 4.53 |
| 4B Q8_0 · CUDA · validate_checkpoint | 7.6 | 98 / 154 | 0.531 | 59 | 7273 | 0.35 | 63 | 0.99 | 4.55 |
| 4B Q8_0 · CUDA · semif_compare | 275.7 | 141 / 180 | 31.258 | 91 | 7271 | 0.77 | 87 | 1.00 | 4.53 |
| 4B Q8_0 · CUDA · typed-decisions | 60.8 | 140 / 164 | 6.557 | 82 | 7271 | 0.66 | 87 | 1.00 | 4.54 |
| 4B BF16 · CUDA · rizzo decide examples | 13.3 | 42 / 79 | 0.129 | 12 | 11037 | 0.02 | 68 | 0.98 | 8.13 |
| 4B BF16 · CUDA · smoke | 6.2 | 62 / 145 | 0.195 | 34 | 11037 | 0.11 | 67 | 0.99 | 8.12 |
| 4B BF16 · CUDA · perturbations | 5.1 | 52 / 159 | 0.114 | 32 | 11037 | 0.75 | 64 | 0.99 | 8.12 |
| 4B BF16 · CUDA · validate_checkpoint | 9.3 | 90 / 157 | 0.558 | 56 | 11039 | 0.25 | 68 | 1.00 | 8.14 |
| 4B BF16 · CUDA · semif_compare | 290.0 | 141 / 177 | 32.652 | 91 | 11041 | 0.79 | 87 | 1.00 | 8.13 |
| 4B BF16 · CUDA · typed-decisions | 64.4 | 138 / 171 | 6.892 | 81 | 11039 | 0.65 | 87 | 1.00 | 8.13 |
| 4B Q4_K_M · CUDA · rizzo decide examples | 5.6 | 50 / 102 | 0.087 | 8 | 5575 | 0.24 | 70 | 0.94 | 2.88 |
| 4B Q4_K_M · CUDA · smoke | 3.8 | 95 / 165 | 0.238 | 48 | 5579 | — | 71 | 0.98 | 2.88 |
| 4B Q4_K_M · CUDA · perturbations | 2.7 | 67 / 147 | 0.105 | 39 | 5577 | — | 68 | 0.97 | 2.88 |
| 4B Q4_K_M · CUDA · validate_checkpoint | 6.9 | 106 / 158 | 0.538 | 64 | 5583 | 0.67 | 71 | 0.99 | 2.89 |
| 4B Q4_K_M · CUDA · semif_compare | 66.4 | 133 / 180 | 6.968 | 81 | 5581 | 0.67 | 85 | 1.00 | 2.88 |
| 4B Q4_K_M · CUDA · typed-decisions | 62.8 | 140 / 164 | 6.828 | 82 | 5581 | 0.68 | 87 | 1.00 | 2.88 |
| 1.7B Q8_0 · CUDA · rizzo decide examples | 4.2 | 48 / 99 | 0.073 | 16 | 3927 | — | 54 | 0.95 | 2.16 |
| 1.7B Q8_0 · CUDA · smoke | 2.4 | 66 / 139 | 0.089 | 34 | 3927 | — | 57 | 0.96 | 2.15 |
| 1.7B Q8_0 · CUDA · perturbations | 1.8 | 39 / 146 | 0.025 | 17 | 3927 | — | 57 | 0.97 | 2.15 |
| 1.7B Q8_0 · CUDA · validate_checkpoint | 3.9 | 91 / 159 | 0.252 | 49 | 3929 | — | 61 | 0.96 | 2.17 |
| 1.7B Q8_0 · CUDA · semif_compare | 32.9 | 120 / 178 | 3.062 | 64 | 3929 | 0.48 | 76 | 1.00 | 2.16 |
| 1.7B Q8_0 · CUDA · typed-decisions | 30.4 | 138 / 161 | 3.306 | 67 | 3929 | 0.51 | 81 | 1.00 | 2.16 |
| 1.7B Q4_K_M · CUDA · rizzo decide examples | 3.5 | 40 / 79 | 0.030 | 10 | 3249 | — | 72 | 0.91 | 1.49 |
| 1.7B Q4_K_M · CUDA · smoke | 2.0 | 71 / 145 | 0.079 | 33 | 3249 | — | 70 | 0.98 | 1.49 |
| 1.7B Q4_K_M · CUDA · perturbations | 1.5 | 50 / 145 | 0.032 | 22 | 3249 | — | 65 | 0.96 | 1.49 |
| 1.7B Q4_K_M · CUDA · validate_checkpoint | 3.7 | 102 / 154 | 0.266 | 56 | 3251 | — | 67 | 0.97 | 1.50 |
| 1.7B Q4_K_M · CUDA · semif_compare | 33.6 | 123 / 180 | 3.146 | 65 | 3251 | 0.49 | 80 | 1.00 | 1.49 |
| 1.7B Q4_K_M · CUDA · typed-decisions | 30.9 | 139 / 156 | 3.367 | 67 | 3251 | 0.52 | 83 | 1.00 | 1.49 |
| 1.7B BF16 · CUDA · rizzo decide examples | 6.6 | 41 / 51 | 0.067 | 12 | 5551 | 0.05 | 64 | 0.97 | 3.65 |
| 1.7B BF16 · CUDA · smoke | 3.1 | 54 / 146 | 0.072 | 29 | 5551 | — | 64 | 0.98 | 3.65 |
| 1.7B BF16 · CUDA · perturbations | 2.5 | 38 / 144 | 0.031 | 24 | 5551 | — | 61 | 0.98 | 3.65 |
| 1.7B BF16 · CUDA · validate_checkpoint | 4.7 | 87 / 152 | 0.282 | 46 | 5553 | — | 64 | 0.98 | 3.66 |
| 1.7B BF16 · CUDA · semif_compare | 35.0 | 120 / 180 | 3.273 | 65 | 5553 | 0.49 | 78 | 1.00 | 3.65 |
| 1.7B BF16 · CUDA · typed-decisions | 31.6 | 136 / 156 | 3.387 | 68 | 5553 | 0.52 | 81 | 1.00 | 3.65 |
| 4B Q8_0 · CUDA · KV q8_0 · smoke | 4.6 | 81 / 149 | 0.231 | 42 | 6611 | — | 72 | 0.98 | 4.53 |
| 4B Q8_0 · CUDA · KV q8_0 · semif_compare | 64.2 | 129 / 177 | 6.812 | 78 | 6645 | 0.63 | 84 | 1.00 | 4.53 |
| 4B Q8_0 · CUDA · KV q4_0 · smoke | 4.4 | 62 / 148 | 0.188 | 35 | 6283 | — | 73 | 0.99 | 4.53 |
| 4B Q8_0 · CUDA · KV q4_0 · semif_compare | 64.0 | 130 / 180 | 7.181 | 78 | 6285 | 0.64 | 84 | 1.00 | 4.53 |
| 4B BF16 · CUDA · KV q8_0 · smoke | 6.2 | 51 / 153 | 0.195 | 27 | 10415 | 0.00 | 72 | 0.99 | 8.12 |
| 4B BF16 · CUDA · KV q8_0 · semif_compare | 67.9 | 127 / 180 | 7.527 | 77 | 10417 | 0.64 | 84 | 1.00 | 8.12 |
| 4B BF16 · CUDA · KV q4_0 · smoke | 6.1 | 49 / 146 | 0.177 | 23 | 10055 | 0.06 | 72 | 0.99 | 8.12 |
| 4B BF16 · CUDA · KV q4_0 · semif_compare | 67.8 | 127 / 181 | 7.435 | 77 | 10057 | 0.65 | 84 | 1.00 | 8.13 |
| 4B Q4_K_M · CUDA · KV q8_0 · smoke | 3.7 | 81 / 152 | 0.227 | 41 | 4953 | — | 75 | 0.97 | 2.88 |
| 4B Q4_K_M · CUDA · KV q8_0 · semif_compare | 66.7 | 133 / 182 | 7.703 | 81 | 4955 | 0.67 | 85 | 1.00 | 2.88 |
| 4B Q4_K_M · CUDA · KV q4_0 · smoke | 3.6 | 79 / 157 | 0.210 | 41 | 4593 | — | 74 | 0.99 | 2.88 |
| 4B Q4_K_M · CUDA · KV q4_0 · semif_compare | 66.3 | 133 / 182 | 7.663 | 80 | 4595 | 0.66 | 85 | 1.00 | 2.88 |
| 1.7B Q8_0 · CUDA · KV q8_0 · smoke | 2.4 | 64 / 146 | 0.106 | 25 | 3703 | — | 72 | 0.96 | 2.15 |
| 1.7B Q8_0 · CUDA · KV q8_0 · semif_compare | 33.2 | 121 / 179 | 3.432 | 63 | 3705 | 0.46 | 80 | 1.00 | 2.16 |
| 1.7B Q8_0 · CUDA · KV q4_0 · smoke | 2.4 | 62 / 144 | 0.099 | 25 | 3563 | — | 68 | 0.97 | 2.15 |
| 1.7B Q8_0 · CUDA · KV q4_0 · semif_compare | 32.9 | 118 / 173 | 3.313 | 61 | 3565 | 0.48 | 79 | 1.00 | 2.16 |
| 1.7B Q4_K_M · CUDA · KV q8_0 · smoke | 2.1 | 64 / 143 | 0.098 | 26 | 3025 | — | 70 | 0.96 | 1.49 |
| 1.7B Q4_K_M · CUDA · KV q8_0 · semif_compare | 34.0 | 122 / 181 | 3.596 | 64 | 3027 | 0.49 | 80 | 1.00 | 1.49 |
| 1.7B Q4_K_M · CUDA · KV q4_0 · smoke | 2.1 | 76 / 149 | 0.118 | 30 | 2885 | — | 70 | 0.96 | 1.49 |
| 1.7B Q4_K_M · CUDA · KV q4_0 · semif_compare | 33.8 | 119 / 180 | 3.495 | 63 | 2887 | 0.48 | 80 | 1.00 | 1.49 |
| 1.7B BF16 · CUDA · KV q8_0 · smoke | 3.1 | 46 / 154 | 0.094 | 27 | 5323 | — | 68 | 0.97 | 3.65 |
| 1.7B BF16 · CUDA · KV q8_0 · semif_compare | 35.5 | 124 / 181 | 3.803 | 67 | 5325 | 0.48 | 79 | 1.00 | 3.65 |
| 1.7B BF16 · CUDA · KV q4_0 · smoke | 3.1 | 41 / 146 | 0.067 | 16 | 5183 | — | 67 | 0.98 | 3.65 |
| 1.7B BF16 · CUDA · KV q4_0 · semif_compare | 35.6 | 121 / 181 | 3.720 | 65 | 5185 | 0.49 | 79 | 0.99 | 3.65 |
| 4B Q8_0 · Vulkan · smoke | 12.9 | 50 / 159 | 0.419 | 17 | 7107 | 0.13 | 68 | 0.84 | 4.28 |
| 4B Q8_0 · Vulkan · semif_compare | 361.4 | 141 / 180 | 45.314 | 91 | 7552 | 0.68 | 87 | 0.70 | 4.28 |
| 4B Q8_0 · CUDA · batch 1 · semif_compare | 43.7 | 122 / 179 | 4.529 | 71 | 7303 | 0.55 | 83 | 1.00 | 4.54 |
| 4B Q8_0 · CUDA · batch 2 · semif_compare | 40.8 | 123 / 175 | 4.231 | 70 | 7303 | 0.54 | 82 | 1.00 | 4.54 |
| 4B Q8_0 · CUDA · batch 4 · semif_compare | 38.2 | 120 / 180 | 3.852 | 68 | 7303 | 0.54 | 81 | 1.00 | 4.53 |
| 4B Q8_0 · CUDA · batch 8 · semif_compare | 38.0 | 120 / 181 | 3.831 | 68 | 7303 | 0.55 | 81 | 1.00 | 4.54 |
| 4B Q8_0 · CUDA · batch 16 · semif_compare | 37.7 | 121 / 181 | 3.859 | 67 | 7305 | 0.54 | 81 | 1.00 | 4.54 |
| 4B Q8_0 · CPU · threads default (4) · smoke | 123.0 | 8 / 12 | — (GPU idle) | 0 | 1327 | 0.00 | 62 | 3.90 | 5.66 |
| 4B Q8_0 · CPU · threads 16 · smoke | 41.4 | 7 / 8 | — (GPU idle) | 0 | 1327 | 0.00 | 43 | 14.47 | 5.66 |
| 4B Q8_0 · CPU · threads 32 · smoke | 51.1 | 6 / 7 | — (GPU idle) | 0 | 1327 | 0.00 | 41 | 28.56 | 5.66 |
| 4B Q8_0 · CPU · rizzo decide examples | 12.5 | 6 / 7 | — (GPU idle) | 0 | 1327 | 0.00 | 39 | 7.78 | 5.66 |
| 4B Q8_0 · CPU · smoke | 41.6 | 6 / 7 | — (GPU idle) | 0 | 1327 | 0.00 | 39 | 14.47 | 5.66 |
| 4B Q8_0 · CPU · perturbations | 18.2 | 6 / 8 | — (GPU idle) | 0 | 1327 | 0.00 | 38 | 13.33 | 5.63 |
| 4B Q8_0 · CPU · validate_checkpoint | 108.3 | 6 / 17 | — (GPU idle) | 1 | 1327 | 0.00 | 38 | 15.17 | 5.67 |
| 4B Q8_0 · CPU · semif_compare | 1292.3 | 6 / 18 | — (GPU idle) | 1 | 1327 | 0.00 | 38 | 15.63 | 5.67 |
| 4B Q8_0 · CPU · typed-decisions | 1194.8 | 6 / 17 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 15.60 | 5.66 |
| 4B BF16 · CPU · rizzo decide examples | 16.6 | 6 / 7 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 6.03 | 9.25 |
| 4B BF16 · CPU · smoke | 41.8 | 6 / 14 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 13.99 | 9.25 |
| 4B BF16 · CPU · perturbations | 19.4 | 7 / 15 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 12.45 | 9.22 |
| 4B BF16 · CPU · validate_checkpoint | 105.9 | 6 / 14 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 14.97 | 9.27 |
| 4B BF16 · CPU · semif_compare | 1248.0 | 6 / 20 | — (GPU idle) | 1 | 1327 | 0.00 | 38 | 15.62 | 9.27 |
| 4B BF16 · CPU · typed-decisions | 1158.2 | 6 / 18 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 15.60 | 9.26 |
| 4B Q4_K_M · CPU · rizzo decide examples | 10.8 | 6 / 7 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 7.93 | 5.57 |
| 4B Q4_K_M · CPU · smoke | 36.9 | 8 / 17 | — (GPU idle) | 2 | 1327 | 0.00 | 37 | 14.42 | 5.57 |
| 4B Q4_K_M · CPU · perturbations | 16.2 | 7 / 16 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 13.34 | 5.55 |
| 4B Q4_K_M · CPU · validate_checkpoint | 96.7 | 6 / 16 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 15.14 | 5.59 |
| 4B Q4_K_M · CPU · semif_compare | 1155.7 | 6 / 16 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 15.61 | 5.59 |
| 4B Q4_K_M · CPU · typed-decisions | 1065.1 | 6 / 17 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 15.59 | 5.57 |
| 1.7B Q8_0 · CPU · rizzo decide examples | 5.2 | 6 / 7 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 7.18 | 2.39 |
| 1.7B Q8_0 · CPU · smoke | 17.1 | 6 / 7 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 14.41 | 2.39 |
| 1.7B Q8_0 · CPU · perturbations | 7.5 | 6 / 7 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 13.22 | 2.38 |
| 1.7B Q8_0 · CPU · validate_checkpoint | 43.8 | 7 / 9 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 15.08 | 2.41 |
| 1.7B Q8_0 · CPU · semif_compare | 516.0 | 7 / 12 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 15.48 | 2.41 |
| 1.7B Q8_0 · CPU · typed-decisions | 476.9 | 7 / 17 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 15.49 | 2.40 |
| 1.7B Q4_K_M · CPU · rizzo decide examples | 4.7 | 6 / 6 | — (GPU idle) | 0 | 1327 | — | 37 | 7.18 | 2.33 |
| 1.7B Q4_K_M · CPU · smoke | 14.9 | 7 / 15 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 14.24 | 2.33 |
| 1.7B Q4_K_M · CPU · perturbations | 6.7 | 7 / 16 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 13.04 | 2.32 |
| 1.7B Q4_K_M · CPU · validate_checkpoint | 38.7 | 6 / 14 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 14.99 | 2.35 |
| 1.7B Q4_K_M · CPU · semif_compare | 462.3 | 6 / 18 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 15.43 | 2.35 |
| 1.7B Q4_K_M · CPU · typed-decisions | 427.3 | 6 / 9 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 15.44 | 2.34 |
| 1.7B BF16 · CPU · rizzo decide examples | 7.1 | 6 / 6 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 5.47 | 3.89 |
| 1.7B BF16 · CPU · smoke | 16.9 | 6 / 7 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 13.80 | 3.89 |
| 1.7B BF16 · CPU · perturbations | 8.0 | 6 / 6 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 12.13 | 3.87 |
| 1.7B BF16 · CPU · validate_checkpoint | 42.0 | 6 / 7 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 14.82 | 3.90 |
| 1.7B BF16 · CPU · semif_compare | 482.1 | 6 / 16 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 15.43 | 3.90 |
| 1.7B BF16 · CPU · typed-decisions | 462.3 | 6 / 17 | — (GPU idle) | 0 | 1327 | 0.00 | 38 | 15.46 | 3.89 |
| 4B Q8_0 · CPU · KV q8_0 · smoke | 43.8 | 6 / 16 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 14.85 | 5.00 |
| 4B Q8_0 · CPU · KV q8_0 · semif_compare | 1796.9 | 6 / 17 | — (GPU idle) | 0 | 1327 | 0.00 | 38 | 15.51 | 5.02 |
| 4B Q8_0 · CPU · KV q4_0 · smoke | 47.0 | 6 / 7 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 15.09 | 4.65 |
| 4B Q8_0 · CPU · KV q4_0 · semif_compare | 2152.5 | 6 / 16 | — (GPU idle) | 0 | 1327 | 0.00 | 38 | 15.41 | 4.66 |
| 4B BF16 · CPU · KV q8_0 · smoke | 44.2 | 6 / 16 | — (GPU idle) | 1 | 1327 | 0.00 | 38 | 14.42 | 8.59 |
| 4B BF16 · CPU · KV q8_0 · semif_compare | 1681.4 | 6 / 17 | — (GPU idle) | 0 | 1327 | 0.00 | 38 | 15.69 | 8.61 |
| 4B BF16 · CPU · KV q4_0 · smoke | 47.6 | 6 / 16 | — (GPU idle) | 1 | 1327 | 0.00 | 38 | 14.67 | 8.24 |
| 4B BF16 · CPU · KV q4_0 · semif_compare | 2032.4 | 6 / 17 | — (GPU idle) | 1 | 1327 | 0.00 | 38 | 15.60 | 8.26 |
| 4B Q4_K_M · CPU · KV q8_0 · smoke | 39.3 | 7 / 11 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 14.87 | 4.92 |
| 4B Q4_K_M · CPU · KV q8_0 · semif_compare | 1593.9 | 6 / 17 | — (GPU idle) | 1 | 1327 | 0.00 | 38 | 15.69 | 4.93 |
| 4B Q4_K_M · CPU · KV q4_0 · smoke | 42.5 | 6 / 16 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 15.14 | 4.57 |
| 4B Q4_K_M · CPU · KV q4_0 · semif_compare | 1948.8 | 6 / 17 | — (GPU idle) | 0 | 1327 | 0.00 | 38 | 15.60 | 4.58 |
| 1.7B Q8_0 · CPU · KV q8_0 · smoke | 17.6 | 6 / 8 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 14.74 | 2.14 |
| 1.7B Q8_0 · CPU · KV q8_0 · semif_compare | 646.4 | 6 / 14 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 15.65 | 2.16 |
| 1.7B Q8_0 · CPU · KV q4_0 · smoke | 18.8 | 6 / 14 | — (GPU idle) | 1 | 1327 | 0.00 | 37 | 14.96 | 2.00 |
| 1.7B Q8_0 · CPU · KV q4_0 · semif_compare | 813.3 | 6 / 16 | — (GPU idle) | 0 | 1327 | 0.00 | 38 | 15.66 | 2.02 |
| 1.7B Q4_K_M · CPU · KV q8_0 · smoke | 15.7 | 6 / 6 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 14.65 | 2.08 |
| 1.7B Q4_K_M · CPU · KV q8_0 · semif_compare | 586.3 | 6 / 7 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 15.63 | 2.10 |
| 1.7B Q4_K_M · CPU · KV q4_0 · smoke | 17.1 | 7 / 8 | — (GPU idle) | 0 | 1327 | 0.00 | 37 | 14.94 | 1.94 |
| 1.7B Q4_K_M · CPU · KV q4_0 · semif_compare | 747.4 | 18 / 48 | — (GPU idle) | 10 | 1508 | 0.03 | 46 | 15.66 | 1.96 |
| 1.7B BF16 · CPU · KV q8_0 · smoke | 17.6 | 28 / 30 | — (GPU idle) | 19 | 1402 | 0.09 | 46 | 14.19 | 3.63 |
| 1.7B BF16 · CPU · KV q8_0 · semif_compare | 619.0 | 28 / 35 | — (GPU idle) | 18 | 1402 | 0.08 | 47 | 15.63 | 3.65 |
| 1.7B BF16 · CPU · KV q4_0 · smoke | 18.8 | 28 / 28 | — (GPU idle) | 18 | 1402 | 0.04 | 47 | 14.47 | 3.49 |
| 1.7B BF16 · CPU · KV q4_0 · semif_compare | 774.7 | 26 / 35 | — (GPU idle) | 17 | 1402 | 0.05 | 47 | 15.66 | 3.51 |
| 4B Q8_0 · CPU · batch 1 · semif_compare | 722.4 | 6 / 17 | — (GPU idle) | 1 | 1327 | 0.00 | 41 | 15.47 | 5.67 |
| 4B Q8_0 · CPU · batch 2 · semif_compare | 680.6 | 6 / 17 | — (GPU idle) | 1 | 1327 | 0.00 | 38 | 15.44 | 5.67 |
| 4B Q8_0 · CPU · batch 4 · semif_compare | 667.1 | 6 / 16 | — (GPU idle) | 1 | 1327 | 0.00 | 38 | 15.43 | 5.67 |
| 4B Q8_0 · CPU · batch 8 · semif_compare | 663.9 | 6 / 17 | — (GPU idle) | 1 | 1328 | 0.00 | 37 | 15.42 | 5.67 |
| 4B Q8_0 · CPU · batch 16 · semif_compare | 658.7 | 6 / 17 | — (GPU idle) | 1 | 1328 | 0.00 | 38 | 15.41 | 5.68 |

</details>

## Anything odd

- With `--threads` unset, llama.cpp's C API default is 4 threads (`llama_context_default_params().n_threads == 4`), so CPU runs use 4 of the 32 threads unless `--threads` is given; see the threads rows in the smoke table. The CPU runs after that sweep use the fastest explicit count.
- integration tests (`T1-integration`): `tests/test_llama_real.py::test_shared_prefix_agrees_with_direct` failed.
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
