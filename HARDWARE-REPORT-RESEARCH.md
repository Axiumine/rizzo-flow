# Hardware report research — Linux + RTX PRO 4000 Blackwell

Working notes, 28 September 2026, repo at `c30cc63`. Goal: contribute a hardware report to
Rizzo Flow (`rizzo devices` output + timings) covering every combination this machine can run,
with GPU/CPU telemetry from Prometheus/DCGM. **Nothing has been run or downloaded yet**: this is
the plan and the state of the setup.

## 1. What the project asks for

README, end of *Contributors*: "The most useful next reports are from Linux, other AMD or Intel
GPUs, different Apple Silicon models, ROCm/SYCL builds, or a dedicated CPU-only machine. Please
open an issue with the output of `rizzo devices` and your timings." It does not say which
timings; the past reports ([#5](https://github.com/Rizzo-AI-Academy/rizzo-flow/issues/5) M3 Pro,
[#7](https://github.com/Rizzo-AI-Academy/rizzo-flow/issues/7) Iris Xe + CPU,
[#11](https://github.com/Rizzo-AI-Academy/rizzo-flow/issues/11) Radeon 780M) all share one format:

1. **Machine**: GPU, driver, CPU, RAM, OS/kernel, Python, uv, commit, runtime folder, model/quant.
2. **Exact commands** run.
3. **`rizzo devices` output** (devices, `auto_selects`).
4. **Table vs the maintainers' CUDA numbers**: accuracy, NLL, Brier, ECE, median and p95 latency,
   decisions/s, peak GPU memory, `mode_comparison` (changed argmaxes, max Δp).
5. **Anything odd**: crashes, exit codes, wrong device picked.

The hardware table says CUDA was tested only on "Windows 10 + RTX 5060 Ti. **Linux not tried**",
so this machine fills an open gap. Both community reports (#5 Metal, #11 Vulkan) saw 0 argmax
flips shared vs direct on smoke and guessed the maintainers' 13/777 flips are CUDA-specific: a
second CUDA data point matters (smoke has only 20 decisions, so it is a hint, not proof).

## 2. This machine

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D, 16C/32T |
| RAM | 64 GB: 2 × 32 GB DDR5 Crucial Pro CP32G64C40U5B (DDR5-5600 CL40), configured at 6000 MT/s, dual channel, 1.1 V (61 GiB usable) |
| GPU | NVIDIA RTX PRO 4000 Blackwell, 24 GB, compute capability 12.0 |
| Driver / CUDA | 615.71.09 / 13.4 |
| Board power limit | 145 W |
| Ambient temperature | 29 °C at the start of the campaign (reported by the user) |
| Vulkan | `libvulkan.so.1` 1.4.309 + `nvidia_icd.json` present (`vulkaninfo` not installed) |
| Disk | 180 GB free on `/media/ai` |
| Python / uv | `.venv` on 3.14 (`.python-version` changed 3.12 → 3.14, uncommitted) / uv 0.12.17 |
| Runtime installed | `runtimes/llama-b11081-linux-x64-cuda` only (ships 14 `libggml-cpu-*` too) |
| `rizzo devices` | `CUDA0` (23.5 GiB) + `CPU` (61.9 GiB), `auto_selects: CUDA0` |
| GGUF on disk | 4B `q8_0`, 4B `bf16` |
| GGUF missing | 4B `q4_k_m`; 1.7B `q8_0`, `q4_k_m`, `bf16` |
| Missing extras | `test` extra not synced (no pytest), no MLX, no `.research/SemIf`, no `.research/typed-decisions` |

**The GPU also drives the host display** and cannot be freed: at the time of the check the
desktop held ~1.65 GiB of VRAM and ~13 % utilization. Consequences:

- VRAM is not a problem: ~22 GB left, the largest run (4B BF16) needs ~9.3 GiB.
- Timing noise from the desktop: keep desktop use light during the GPU block (no video,
  games or WebGL). Repeat a run if Prometheus shows a utilization spike in its window.
- Energy: record 30 s of baseline before each run and report net energy =
  `increase(TOTAL_ENERGY)` − baseline power × duration.
- Disclose it in the report ("GPU also drives the display, ~1.6 GiB / ~13 % baseline").

## 3. Monitoring: Prometheus + DCGM

| Source | Address | Interval | Notes |
|---|---|---|---|
| Prometheus | `http://prometheus.gio.lan:9090` | — | v3.14, 15 d retention, API and range queries verified |
| DCGM exporter | job `dcgm` → `pcgio:9400` | **1 s** | custom counters in `/etc/dcgm-exporter/custom-counters.csv` |
| node_exporter | job `node_exporter` → `pcgio:9100` | 2 s | 32 CPU threads, memory, hwmon temps, PSI; **no RAPL** (no CPU energy) |

Changes made on 28 September 2026 to get 1 s resolution (both are needed; the second alone
only repeats the same value):

- exporter: `/etc/systemd/system/nvidia-dcgm-exporter.service.d/override.conf` → `-c 1000`
  (was `-c 5000`), then `daemon-reload` + restart;
- Prometheus job `dcgm`: `scrape_interval: 1s`, `scrape_timeout: 900ms` (was 5s / 4s).

Verified after the change: power, energy, `PROF_SM_ACTIVE`, `PROF_DRAM_ACTIVE` change 15/15
samples; `GPU_TEMP` moves slowly (integer), `FB_USED` flat while idle; scrape ~6 ms. Overhead
negligible on one GPU; the `PROF_*` fields hold the performance counters (pause the exporter for
Nsight/ncu).

Limits:

- Per-decision latency (~50 ms) and GPU smoke runs (~1–2 s) are below Prometheus resolution:
  rizzo's own `latency_seconds` stays the reference; short runs also get a local
  `nvidia-smi --query-gpu=... -lms 100` sampler.
- Energy edge error is now ~1 s of idle power per edge (was ~5 s).
- `FB_USED` includes other processes; rizzo's `peak_device_bytes` (free-memory delta) is cleaner.

Per-run queries (window `[D]` = run duration, evaluated `@ END`):

```promql
max_over_time(DCGM_FI_DEV_FB_USED{hostname="pcgio"}[D])                       # peak VRAM, MiB
avg_over_time(DCGM_FI_DEV_POWER_USAGE{hostname="pcgio"}[D])                   # avg power, W
increase(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{hostname="pcgio"}[D]) / 1e3     # energy, J (counter in mJ)
avg_over_time(DCGM_FI_PROF_SM_ACTIVE{hostname="pcgio"}[D])                    # SM activity 0–1
avg_over_time(DCGM_FI_PROF_DRAM_ACTIVE{hostname="pcgio"}[D])                  # memory-bandwidth activity 0–1
max_over_time(DCGM_FI_DEV_GPU_TEMP{hostname="pcgio"}[D])
increase(DCGM_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP_NS{hostname="pcgio"}[D]) / 1e9   # s throttled
1 - avg(rate(node_cpu_seconds_total{instance="pcgio:9100",mode="idle"}[D]))  # CPU utilization
```

On CPU runs, `DCGM_FI_DEV_GPU_UTIL` staying at 0 confirms the run was really CPU-only.

## 4. Configuration axes

| Axis | Values | Status here |
|---|---|---|
| Device | GPU via CUDA (`--device cuda`) | installed |
| | GPU via Vulkan (`--device vulkan`) | **one check only** (4B q8_0, see section 6); needs `rizzo download --only runtime --runtime vulkan` (~30 MB) |
| | CPU (`--device cpu`) | runnable today: empty device list → `n_gpu_layers = 0`, `offload_kqv = False` (`llama_cpp.py:357-387`), CPU backend bundled in the CUDA package |
| Size × quant | 4B `q8_0` 4.4 GB, `bf16` 8.2 GB, `q4_k_m` 2.6 GB | q8_0, bf16 present |
| | 1.7B `q8_0` 1.8 GB, `q4_k_m` 1.1 GB, `bf16` 3.4 GB | none present |
| `--kv-type` | `f16` (default), `q8_0`, `q4_0` | llama only |
| `--batch-size` | 1–16, default 4 | throughput in shared mode only |
| `--threads` | default / 16 / 32 | CPU only |
| Excluded | ROCm, SYCL, Metal | no AMD/Intel GPU, no Mac |
| | MLX-CPU | documented unusable (~185 s for 8 tokens) |
| | MLX-CUDA | dropped (user decision) |

Invalid combinations (raise `ValueError`, `loader.py`): `--backend mlx` with `--quant`,
`--kv-type` or a llama device family; `--backend llama` with `--bits` or `--device mlx`;
`--batch-size` outside 1–16.

Base grid: 2 devices (CUDA, CPU) × 6 GGUF = **12 configurations**; × 3 KV types = 36. Vulkan is
not a grid axis: one targeted check (section 6).

## 5. Test suites

| ID | Command | Measures | GPU | CPU (estimate) |
|---|---|---|---|---|
| T0 | `uv run pytest -q` | 78 unit tests, no weights | ~4 s, once | — |
| T1 | `RIZZO_REAL=1 uv run pytest -q -m integration` | 6 real-weight tests incl. kv-type; smallest Q8_0 on disk, device auto | once | — |
| T2 | `uv run rizzo devices` | runtime + devices | per runtime | — |
| T3 | `uv run rizzo decide examples/{ticket,house,numeric}.json` | sanity of answers | seconds | ~1 min |
| T4 | `uv run rizzo evaluate benchmarks/smoke.jsonl --compare-modes --output …` | the community test: acc, NLL, Brier, ECE, p50/p95, dec/s, shared vs direct | ~2 s | 1–3 min |
| T5 | `uv run rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --output …` | stability | ~1 s | ~1 min |
| T6 | `uv run python scripts/validate_checkpoint.py --output …` | real-weight sanity + long-state reuse | ~1 min | 5–10 min |
| T7 | `uv run python scripts/semif_compare.py --system rizzo --semif .research/SemIf --output …` | the maintainers' main table: authored144, perturbations108, stability, shape777 shared/direct, p50/p95, peak | 3–6 min | 15–40 min |
| T8 | `uv run python scripts/semif_report.py RUN --semif .research/SemIf --against NAME=DIR` | held-out halves + paired differences vs published runs; no model | seconds | — |
| T9 | `uv run python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --quant … --device … --output …` | HF typed-decisions benchmark (optional; 1.7B via `--model`) | minutes | tens of min |

CPU estimates are scaled from the i5-1334U CPU-only run in #7 (11.9 s median on smoke, Q4_K_M)
and are unmeasured: the first CPU run will replace them.

Excluded on purpose: `smoke-v1`/`perturbations-v1` (old prompt), `prompt_lab.py` (maintainer
tool; the held-out split is looked at once), `record_demo.py` (rewrites `docs/index.html`),
`training/` (not a benchmark).

To match the published "13/777 changed argmaxes", run T7 on GPU `q8_0` and `bf16` with
`--direct-states 37`; the default (3 states, 63 decisions) matches the Q4_K_M and Vulkan rows.

## 6. Matrix

| Block | Configurations | Suites | Runs |
|---|---|---|---|
| One-off | — | T0, T1, T2 (CUDA package, CPU package) | 4 |
| **Core** | 12 (CUDA, CPU × 6 GGUF, KV f16) | T3, T4, T5, T6, T7 | **60** |
| KV cache | 24 (12 × `q8_0`, `q4_0`) | T4, T7 | 48 |
| Vulkan check | 1 (4B q8_0, KV f16) | T2, T4, T7 `--direct-states 37` | 3 |
| Sweeps | `--batch-size` {1,2,4,8,16} on CUDA and CPU, 4B q8_0, shape777 shared (10) + `--threads` {default,16,32} on CPU smoke (3) | — | 13 |
| Optional: typed-decisions | 12 | T9 | 12 |
| **Total** | | | **140** |

Dropped on 28 September 2026 (user decisions): MLX-CUDA (24 runs), HTTP round-trip T10 (3 runs),
the full Vulkan grid (61 runs → replaced by the 3-run check). Without typed-decisions: 128 runs.

Why Vulkan stays as one check: `--device auto` picks CUDA here and Vulkan on Linux is already
covered by #11 (Radeon 780M), so a full grid would mostly repeat CUDA, slower. But the same card
with the same weights, CUDA vs Vulkan, both with `--direct-states 37`, tests whether the 13/777
shared-vs-direct flips are CUDA-specific (#5 Metal and #11 Vulkan saw 0/20), and repeats the
maintainers' Windows CUDA-vs-Vulkan comparison on Linux.

### Time estimate (revised 28 September 2026)

Runs are strictly sequential: a CPU run in parallel would steal cores from the GPU run's host
side, and vice versa, and both timings would be wrong.

| Block | GPU | CPU |
|---|---|---|
| One-off (T0, T1, T2) | ~5 min | — |
| Core (6 + 6 configs) | ~30 min | ~2.2 h |
| KV cache (12 + 12 configs) | ~20 min | ~3.5 h |
| Vulkan check | ~10 min | — |
| Sweeps (13) | ~10 min | ~1 h |
| typed-decisions (6 + 6, 400 test cases) | ~10 min | ~1.8 h |
| **Total** | **~1.4 h** (range 1–2 h) | **~8.5 h** (range 4–17 h) |

About 10 h in total (range 5–19 h), or about 8 h without typed-decisions. Downloads come on
top: 8.9 GB of GGUF, about 12 min at 100 Mbit/s.

How the numbers were derived:

- **GPU**: the maintainers' RTX 5060 Ti timings (T7 = model load ~15 s + 252 × ~50 ms +
  shape777 shared 777 / 21 dec/s + direct 63 / 2.6 dec/s ≈ 1.5 min; with `--direct-states 37`
  ≈ 6 min). Multipliers: Vulkan ×1.6, BF16 ×1.2, 1.7B ×0.6. Every command reloads the model
  (~11–20 s including the GGUF sha256), which dominates the short runs. The RTX PRO 4000 has more
  cores and bandwidth than the 5060 Ti but a 145 W cap, so these are treated as an upper bound.
- **CPU**: assumes ~350 prompt tokens/s for the 4B Q8_0 on the 9950X3D, about 13× the
  i5-1334U CPU-only run in #7. Per 4B Q8_0 config: T3 ~1 min, T4 ~1 min, T5 ~0.5 min,
  T6 ~2 min, T7 ~17–20 min (252 short decisions ≈ 88k tokens, shape777 shared ≈ 136k, direct
  3 states ≈ 132k). Time multipliers: Q4_K_M ×1.15, BF16 ×1.4, 1.7B ×0.5–0.7. The CPU range
  is ×0.5 to ×2 because the throughput is unmeasured.
- The CPU batch-size sweep runs T7 with `--direct-states 0` (supported: `semif_compare.py`
  skips a mode with no states), about 11 min each instead of 20.
- typed-decisions: 400 test cases; maintainers' p50 201 ms/case (Q8_0) and 250 ms (BF16) on
  the 5060 Ti; on CPU ~3 s/case assumed.

Order: GPU block first (predictable), then a single T4 smoke run on CPU with the 4B Q8_0
(~1–2 min) to measure the real CPU throughput and recompute the CPU estimate, then the CPU
block (overnight).

**New data** (never published by the maintainers): Linux + CUDA (4B q8_0/bf16 rows compare
directly with their RTX 5060 Ti table), CUDA vs Vulkan on the same card under Linux (the shared-vs-direct flip question), a dedicated desktop CPU,
any `--kv-type` numbers, 1.7B at bf16 and q4_k_m.

## 7. Reference numbers (maintainers, llama.cpp, prompt v3, Windows 10 + RTX 5060 Ti 16 GB)

| Config | authored144 | perturbations108 | held-out | shape777 shared / direct dec/s | p50 / p95 | peak GPU |
|---|---|---|---|---|---|---|
| 4B Q8_0 CUDA | 0.812 | 0.848 | 0.793 / 0.861 | 20.99 / 2.60 | 49 / 52 ms | 5.6 GiB |
| 4B BF16 CUDA | 0.829 | 0.859 | 0.824 / 0.875 | 17.75 / 1.97 | 60 / 63 ms | 9.3 GiB |
| 4B Q4_K_M CUDA | 0.769 | 0.835 | 0.730 / 0.801 | 19.98 / 2.39 | 51 / 54 ms | 3.9 GiB |
| 4B Q8_0 Vulkan | 0.807 | 0.854 | 0.781 / 0.861 | 14.84 / 1.74 | 90 / 94 ms | 6.0 GiB |
| 1.7B Q8_0 CUDA | 0.678 | 0.640 | 0.690 / 0.514 | 31.59 / 5.51 | 25 / 27 ms | 2.3 GiB |

Smoke, 4B Q8_0 CUDA: 0.95 (19/20), NLL 0.459, median 66 ms, perturbations 9/9. Shared vs
direct: 13/777 changed argmaxes (all near-ties, margin < 0.24).

Community smoke results for context: M3 Pro/Metal Q8_0 0.95, NLL 0.450, median 518 ms, 0/20
flips (#5); Iris Xe/Vulkan Q4_K_M 0.90, 2.22 s, CPU-only 11.9 s (#7); Radeon 780M/Vulkan Q8_0
0.95, NLL 0.448, 620 ms, 0/20 flips (#11).

Reports and their folders: `results/README.md`, `results/semif-compare/*-llama-*`.

## 8. Prerequisites

- `uv sync --extra test --locked`
- `uv run rizzo download --quant q4_k_m` and `uv run rizzo download --size 1.7b --quant {q8_0,q4_k_m,bf16}` (8.9 GB)
- `uv run rizzo download --only runtime --runtime vulkan` (~30 MB); optionally `--runtime cpu` to also test a GPU-free install
- SemIf clone at `ca3ba65` in `.research/SemIf` (fixtures + `evaluate.py` only, no model) — needed for T7/T8
- Optional: `LocalLLaMA/typed-decisions` (config `all`: 1200 train, 400 test) in `.research/typed-decisions/`

## 9. Run rules

- One new output path per run: `results/local-<device>-<size>-<quant>-<kv>-<suite>` (create-only;
  `results/local-*` is git-ignored).
- One process with weights at a time; no `rizzo serve` running while measuring; light desktop use only (the GPU drives the display).
- Each run wrapped with start/end timestamps → Prometheus queries above + 100 ms `nvidia-smi`
  sampler → `*.metrics.json` next to the report.
- Record in the report: Python 3.14 (not the pinned 3.12), commit, runtime folder, driver.

## 10. Decisions and outcome

- Scope: 141 runs (140 planned + the integration suite pinned to the 4B), downloads approved,
  MLX-CUDA and HTTP dropped, Vulkan reduced to one configuration, Python 3.14 kept.
- Campaign: 28 September 21:08 → 29 September 06:36, 141/141 completed, 0 failed runs.
- CPU runs used `--threads 16` (fastest in the sweep; llama.cpp defaults to 4).
- Paste-ready report: `HARDWARE-REPORT-PR.md` (summary in `.research/hw-campaign/pr-summary.md`,
  notes in `pr-notes.md`; every claim checked against the raw outputs by a verification pass:
  123 claims correct, 14 corrected).
- Rebuild the reports from the ledger: `.venv/bin/python .research/hw-campaign/run_campaign.py --render`.

<!-- RESULTS:BEGIN (auto-generated by .research/hw-campaign/run_campaign.py) -->

## 11. Results log

Progress: **141/141** runs, 0 failed; elapsed 9h30m, estimated remaining 0h00m (observed/estimated time: GPU ×0.61, CPU ×1.00). Updated 2026-09-29 07:24.

### 1. `T0-unit` — unit tests — ok

- T0-unit · started 2026-09-28 20:52:58 · wall 1.1 s (estimate 10 s) · exit [0]
- `pytest -q`
- passed 71, failed 0, skipped 13 — `71 passed, 13 skipped, 2 warnings in 0.73s`
- Telemetry: GPU avg 32.4 W (baseline 30.7 W), peak 33.0 W, net energy 0.002 kJ (DCGM —), util avg 24 %, SM active —, DRAM active —, VRAM max 1295 MiB (baseline 1295), temp max 47 °C, power-cap throttle — s; CPU 0.55 cores avg (host util — %), peak RSS 0.07 GiB

### 2. `T2-devices-cuda` — rizzo devices — ok

- T2-devices-cuda · started 2026-09-28 20:53:13 · wall 0.5 s (estimate 5 s) · exit [0]
- `rizzo devices`
- runtime `llama-b11081-linux-x64-cuda`; devices: CUDA0 (gpu, NVIDIA RTX PRO 4000 Blackwell), CPU (cpu, AMD Ryzen 9 9950X3D 16-Core Processor); auto_selects `CUDA0`
- Telemetry: GPU avg 33.5 W (baseline 31.7 W), peak 38.7 W, net energy 0.001 kJ (DCGM —), util avg 24 %, SM active —, DRAM active —, VRAM max 1474 MiB (baseline 1295), temp max 48 °C, power-cap throttle — s; CPU 0.79 cores avg (host util — %), peak RSS 0.47 GiB

### 3. `T2-devices-cpu` — rizzo devices — ok

- T2-devices-cpu · started 2026-09-28 20:53:26 · wall 0.3 s (estimate 5 s) · exit [0]
- `rizzo devices`
- runtime `llama-b11081-linux-x64-cpu`; devices: CPU (cpu, AMD Ryzen 9 9950X3D 16-Core Processor); auto_selects `CPU`
- Telemetry: GPU avg 33.7 W (baseline 28.2 W), peak 33.7 W, net energy 0.002 kJ (DCGM —), util avg 24 %, SM active —, DRAM active —, VRAM max 1253 MiB (baseline 1256), temp max 47 °C, power-cap throttle — s; CPU 0.74 cores avg (host util — %), peak RSS 0.06 GiB

### 4. `T2-devices-vulkan` — rizzo devices — ok

- T2-devices-vulkan · started 2026-09-28 20:53:40 · wall 0.4 s (estimate 5 s) · exit [0]
- `rizzo devices`
- runtime `llama-b11081-linux-x64-vulkan`; devices: Vulkan0 (gpu, NVIDIA RTX PRO 4000 Blackwell), CPU (cpu, AMD Ryzen 9 9950X3D 16-Core Processor); auto_selects `Vulkan0`
- Telemetry: GPU avg 34.0 W (baseline 29.6 W), peak 34.0 W, net energy 0.002 kJ (DCGM —), util avg 30 %, SM active —, DRAM active —, VRAM max 1256 MiB (baseline 1253), temp max 47 °C, power-cap throttle — s; CPU 0.77 cores avg (host util — %), peak RSS 0.17 GiB

### 5. `T1-integration` — integration tests — ok

- T1-integration · started 2026-09-28 20:54:38 · wall 3.9 s (estimate 90 s) · exit [1]
- `pytest -q -m integration`
- passed 4, failed 1, skipped 1 — `1 failed, 4 passed, 1 skipped, 78 deselected, 2 warnings in 3.58s`
- Telemetry: GPU avg 64.3 W (baseline 31.5 W), peak 143.9 W, net energy 0.124 kJ (DCGM —), util avg 34 %, SM active —, DRAM active —, VRAM max 5681 MiB (baseline 1295), temp max 55 °C, power-cap throttle — s; CPU 0.99 cores avg (host util — %), peak RSS 2.59 GiB

### 6. `T1-integration-4b` — integration tests — ok

- T1-integration-4b · started 2026-09-28 20:54:55 · wall 8.6 s (estimate 90 s) · exit [0]
- `pytest -q -m integration -p pin_4b_plugin`
- passed 5, failed 0, skipped 1 — `5 passed, 1 skipped, 78 deselected, 2 warnings in 8.29s`
- Telemetry: GPU avg 43.4 W (baseline 29.2 W), peak 94.2 W, net energy 0.122 kJ (DCGM 0.186), util avg 25 %, SM active 0.19, DRAM active 0.03, VRAM max 11411 MiB (baseline 1295), temp max 53 °C, power-cap throttle 0.3 s; CPU 0.95 cores avg (host util 6 %), peak RSS 5.04 GiB

### 7. `cuda-4b-q8_0-f16-T3` — rizzo decide examples — ok

- 4B Q8_0 · CUDA · started 2026-09-28 21:09:05 · wall 8.2 s (estimate 58 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T3/decide-numeric.json`
- ticket: needs_access_support=True (ok, p=1.000); queue=access (ok, p=1.000); urgency=1.9999213168228909 (ok, p=1.000); deadline_hours=None (insufficient_evidence, p=1.000) — total 140 ms, inference 125 ms, 1287 tokens, load 2.5 s
- house: estimated_price=None (insufficient_evidence, p=0.998) — total 103 ms, inference 97 ms, 562 tokens, load 2.4 s
- numeric: fill=74.99991115394128 (ok, p=1.000) — total 78 ms, inference 71 ms, 398 tokens, load 2.4 s
- Telemetry: GPU avg 40.5 W (baseline 31.8 W), peak 61.8 W, net energy 0.071 kJ (DCGM 0.076), util avg 15 %, SM active 0.04, DRAM active 0.02, VRAM max 7265 MiB (baseline 1295), temp max 50 °C, power-cap throttle 0.1 s; CPU 0.98 cores avg (host util 5 %), peak RSS 4.54 GiB

### 8. `cuda-4b-q8_0-f16-T4` — smoke — ok

- 4B Q8_0 · CUDA · started 2026-09-28 21:09:26 · wall 4.6 s (estimate 38 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.450, Brier 0.077, ECE 0.038
- latency median 43 ms, p95 118 ms, 22.46 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000027, median 43 / 45 ms
- peak device 5.61 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 2.4 s
- Telemetry: GPU avg 79.9 W (baseline 29.3 W), peak 154.9 W, net energy 0.232 kJ (DCGM —), util avg 47 %, SM active —, DRAM active —, VRAM max 7269 MiB (baseline 1295), temp max 59 °C, power-cap throttle — s; CPU 0.99 cores avg (host util — %), peak RSS 4.54 GiB

### 9. `cuda-4b-q8_0-f16-T5` — perturbations — ok

- 4B Q8_0 · CUDA · started 2026-09-28 21:09:44 · wall 3.6 s (estimate 33 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T5/perturbations.json`
- accuracy 1 (9 rows), accepted 1, coverage 0.889, NLL 0.020, Brier 0.003, ECE 0.019
- latency median 40 ms, p95 43 ms, 25.10 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 40 / 41 ms
- peak device 5.57 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 2.4 s
- Telemetry: GPU avg 49.3 W (baseline 27.6 W), peak 144.3 W, net energy 0.078 kJ (DCGM —), util avg 36 %, SM active —, DRAM active —, VRAM max 7225 MiB (baseline 1295), temp max 56 °C, power-cap throttle — s; CPU 0.94 cores avg (host util — %), peak RSS 4.53 GiB

### 10. `cuda-4b-q8_0-f16-T6` — validate_checkpoint — ok

- 4B Q8_0 · CUDA · started 2026-09-28 21:10:00 · wall 7.6 s (estimate 73 s) · exit [0]
- `python scripts/validate_checkpoint.py --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T6/validate`
- smoke accuracy 0.950 (NLL 0.450, 0 changed), perturbations 1
- long state: shared 345 ms vs direct 1014 ms, 0 changed (max Δp 0.011618)
- validation 4.7 s, load 2.5 s, peak 5.61 GiB
- Telemetry: GPU avg 97.8 W (baseline 27.5 W), peak 154.2 W, net energy 0.531 kJ (DCGM 0.401), util avg 59 %, SM active 0.35, DRAM active 0.15, VRAM max 7273 MiB (baseline 1295), temp max 63 °C, power-cap throttle 1.8 s; CPU 0.99 cores avg (host util 5 %), peak RSS 4.55 GiB

### 11. `cuda-4b-q8_0-f16-T7` — semif_compare — ok

- 4B Q8_0 · CUDA · started 2026-09-28 21:10:21 · wall 275.7 s (estimate 373 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T7/semif --direct-states 37`
- authored144 0.810, perturbations108 0.830, held-out 0.785 / 0.843, macro-F1 0.804
- latency p50 36 ms, p95 39 ms; shape777 shared 30.03 dec/s (37 states, 2111 tok/dec), direct 3.37 dec/s (37 states)
- shared vs direct 7/777 flips (max Δp 0.146); stability flips reversal/wrapper/context 6/5/3; confident on missing evidence 6/36; rule_application perturbed 0.556 (NLL 1.69)
- peak device 5.61 GiB
- vs maintainers: authored144 difference -0.002 [-0.027, +0.022], different argmax 4/252
- Telemetry: GPU avg 141.2 W (baseline 27.9 W), peak 180.1 W, net energy 31.258 kJ (DCGM 31.135), util avg 91 %, SM active 0.77, DRAM active 0.27, VRAM max 7271 MiB (baseline 1295), temp max 87 °C, power-cap throttle 234.4 s; CPU 1.00 cores avg (host util 4 %), peak RSS 4.53 GiB

### 12. `cuda-4b-q8_0-f16-T9` — typed-decisions — ok

- 4B Q8_0 · CUDA · started 2026-09-28 21:15:10 · wall 60.8 s (estimate 113 s) · exit [0]
- `python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-4B-GGUF/Spark-X2.5-4B-Q8_0.gguf --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T9/typed.json`
- accuracy 0.574, KL 2.888, Brier 0.479, ECE 0.349, macro-F1 0.414
- p50 143 ms/case, p95 180 ms, 34.44 dec/s
- by workflow: agent_trace_observability 0.370, customer_service 0.636, invoice_processing 0.658, security_incidents 0.630
- Telemetry: GPU avg 139.6 W (baseline 31.8 W), peak 163.6 W, net energy 6.557 kJ (DCGM 6.427), util avg 82 %, SM active 0.66, DRAM active 0.28, VRAM max 7271 MiB (baseline 1295), temp max 87 °C, power-cap throttle 42.9 s; CPU 1.00 cores avg (host util 4 %), peak RSS 4.54 GiB

### 13. `cuda-4b-bf16-f16-T3` — rizzo decide examples — ok

- 4B BF16 · CUDA · started 2026-09-28 21:16:24 · wall 13.3 s (estimate 67 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 4b --quant bf16 --device cuda --output results/local-hw/cuda-4b-bf16-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 4b --quant bf16 --device cuda --output results/local-hw/cuda-4b-bf16-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 4b --quant bf16 --device cuda --output results/local-hw/cuda-4b-bf16-f16-T3/decide-numeric.json`
- ticket: needs_access_support=True (ok, p=1.000); queue=access (ok, p=1.000); urgency=1.999934620732511 (ok, p=1.000); deadline_hours=None (insufficient_evidence, p=1.000) — total 219 ms, inference 203 ms, 1287 tokens, load 4.1 s
- house: estimated_price=None (insufficient_evidence, p=0.998) — total 134 ms, inference 128 ms, 562 tokens, load 4.0 s
- numeric: fill=74.99989584062547 (ok, p=1.000) — total 137 ms, inference 130 ms, 398 tokens, load 4.0 s
- Telemetry: GPU avg 42.1 W (baseline 32.4 W), peak 79.5 W, net energy 0.129 kJ (DCGM 0.126), util avg 12 %, SM active 0.02, DRAM active 0.02, VRAM max 11037 MiB (baseline 1295), temp max 68 °C, power-cap throttle 0.1 s; CPU 0.98 cores avg (host util 4 %), peak RSS 8.13 GiB

### 14. `cuda-4b-bf16-f16-T4` — smoke — ok

- 4B BF16 · CUDA · started 2026-09-28 21:16:51 · wall 6.2 s (estimate 43 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant bf16 --device cuda --output results/local-hw/cuda-4b-bf16-f16-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.449, Brier 0.078, ECE 0.040
- latency median 42 ms, p95 134 ms, 23.14 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000002, median 42 / 43 ms
- peak device 9.29 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 4.1 s
- Telemetry: GPU avg 61.6 W (baseline 30.0 W), peak 145.4 W, net energy 0.195 kJ (DCGM 0.152), util avg 34 %, SM active 0.11, DRAM active 0.06, VRAM max 11037 MiB (baseline 1295), temp max 67 °C, power-cap throttle 0.5 s; CPU 0.99 cores avg (host util 4 %), peak RSS 8.12 GiB

### 15. `cuda-4b-bf16-f16-T5` — perturbations — ok

- 4B BF16 · CUDA · started 2026-09-28 21:17:10 · wall 5.1 s (estimate 37 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 4b --quant bf16 --device cuda --output results/local-hw/cuda-4b-bf16-f16-T5/perturbations.json`
- accuracy 1 (9 rows), accepted 1, coverage 0.889, NLL 0.018, Brier 0.002, ECE 0.018
- latency median 40 ms, p95 45 ms, 24.06 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 40 / 42 ms
- peak device 9.29 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 4.1 s
- Telemetry: GPU avg 51.9 W (baseline 28.8 W), peak 159.1 W, net energy 0.114 kJ (DCGM 0.021), util avg 32 %, SM active 0.75, DRAM active 0.01, VRAM max 11037 MiB (baseline 1295), temp max 64 °C, power-cap throttle 0.0 s; CPU 0.99 cores avg (host util 4 %), peak RSS 8.12 GiB

### 16. `cuda-4b-bf16-f16-T6` — validate_checkpoint — ok

- 4B BF16 · CUDA · started 2026-09-28 21:17:28 · wall 9.3 s (estimate 85 s) · exit [0]
- `python scripts/validate_checkpoint.py --size 4b --quant bf16 --device cuda --output results/local-hw/cuda-4b-bf16-f16-T6/validate`
- smoke accuracy 0.950 (NLL 0.449, 0 changed), perturbations 1
- long state: shared 354 ms vs direct 1005 ms, 0 changed (max Δp 0.008014)
- validation 4.8 s, load 4.1 s, peak 9.29 GiB
- Telemetry: GPU avg 90.0 W (baseline 30.0 W), peak 157.3 W, net energy 0.558 kJ (DCGM 0.424), util avg 56 %, SM active 0.25, DRAM active 0.15, VRAM max 11039 MiB (baseline 1295), temp max 68 °C, power-cap throttle 2.5 s; CPU 1.00 cores avg (host util 5 %), peak RSS 8.14 GiB

### 17. `cuda-4b-bf16-f16-T7` — semif_compare — ok

- 4B BF16 · CUDA · started 2026-09-28 21:17:51 · wall 290.0 s (estimate 445 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant bf16 --device cuda --output results/local-hw/cuda-4b-bf16-f16-T7/semif --direct-states 37`
- authored144 0.819, perturbations108 0.848, held-out 0.806 / 0.843, macro-F1 0.815
- latency p50 39 ms, p95 41 ms; shape777 shared 29.12 dec/s (37 states, 2111 tok/dec), direct 3.21 dec/s (37 states)
- shared vs direct 1/777 flips (max Δp 0.040); stability flips reversal/wrapper/context 5/2/4; confident on missing evidence 6/36; rule_application perturbed 0.611 (NLL 1.69)
- peak device 9.29 GiB
- vs maintainers: authored144 difference -0.009 [-0.028, +0.000], different argmax 4/252
- Telemetry: GPU avg 141.3 W (baseline 28.7 W), peak 177.4 W, net energy 32.652 kJ (DCGM 32.215), util avg 91 %, SM active 0.79, DRAM active 0.35, VRAM max 11041 MiB (baseline 1295), temp max 87 °C, power-cap throttle 251.4 s; CPU 1.00 cores avg (host util 4 %), peak RSS 8.13 GiB

### 18. `cuda-4b-bf16-f16-T9` — typed-decisions — ok

- 4B BF16 · CUDA · started 2026-09-28 21:22:54 · wall 64.4 s (estimate 133 s) · exit [0]
- `python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-4B-GGUF/Spark-X2.5-4B.gguf --device cuda --output results/local-hw/cuda-4b-bf16-f16-T9/typed.json`
- accuracy 0.575, KL 2.935, Brier 0.479, ECE 0.348, macro-F1 0.414
- p50 148 ms/case, p95 186 ms, 33.43 dec/s
- by workflow: agent_trace_observability 0.372, customer_service 0.630, invoice_processing 0.662, security_incidents 0.636
- Telemetry: GPU avg 138.5 W (baseline 31.4 W), peak 171.3 W, net energy 6.892 kJ (DCGM 6.558), util avg 81 %, SM active 0.65, DRAM active 0.38, VRAM max 11039 MiB (baseline 1295), temp max 87 °C, power-cap throttle 45.1 s; CPU 1.00 cores avg (host util 4 %), peak RSS 8.13 GiB

### 19. `cuda-4b-q4_k_m-f16-T3` — rizzo decide examples — ok

- 4B Q4_K_M · CUDA · started 2026-09-28 21:24:11 · wall 5.6 s (estimate 58 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 4b --quant q4_k_m --device cuda --output results/local-hw/cuda-4b-q4_k_m-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 4b --quant q4_k_m --device cuda --output results/local-hw/cuda-4b-q4_k_m-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 4b --quant q4_k_m --device cuda --output results/local-hw/cuda-4b-q4_k_m-f16-T3/decide-numeric.json`
- ticket: needs_access_support=True (ok, p=0.934); queue=access (ok, p=1.000); urgency=1.9998314898287899 (ok, p=1.000); deadline_hours=None (insufficient_evidence, p=1.000) — total 145 ms, inference 130 ms, 1287 tokens, load 1.5 s
- house: estimated_price=None (insufficient_evidence, p=0.998) — total 102 ms, inference 96 ms, 562 tokens, load 1.4 s
- numeric: fill=74.99983841021684 (ok, p=1.000) — total 84 ms, inference 76 ms, 398 tokens, load 1.4 s
- Telemetry: GPU avg 50.2 W (baseline 34.3 W), peak 102.4 W, net energy 0.087 kJ (DCGM 0.088), util avg 8 %, SM active 0.24, DRAM active 0.03, VRAM max 5575 MiB (baseline 1295), temp max 70 °C, power-cap throttle 0.1 s; CPU 0.94 cores avg (host util 4 %), peak RSS 2.88 GiB

### 20. `cuda-4b-q4_k_m-f16-T4` — smoke — ok

- 4B Q4_K_M · CUDA · started 2026-09-28 21:24:30 · wall 3.8 s (estimate 38 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q4_k_m --device cuda --output results/local-hw/cuda-4b-q4_k_m-f16-T4/smoke.json`
- accuracy 0.900 (20 rows), accepted 1, coverage 0.750, NLL 0.735, Brier 0.191, ECE 0.075
- latency median 43 ms, p95 121 ms, 22.52 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.004331, median 43 / 43 ms
- peak device 3.96 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.5 s
- Telemetry: GPU avg 95.2 W (baseline 30.6 W), peak 165.3 W, net energy 0.238 kJ (DCGM —), util avg 48 %, SM active —, DRAM active —, VRAM max 5579 MiB (baseline 1295), temp max 71 °C, power-cap throttle — s; CPU 0.98 cores avg (host util — %), peak RSS 2.88 GiB

### 21. `cuda-4b-q4_k_m-f16-T5` — perturbations — ok

- 4B Q4_K_M · CUDA · started 2026-09-28 21:24:47 · wall 2.7 s (estimate 33 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 4b --quant q4_k_m --device cuda --output results/local-hw/cuda-4b-q4_k_m-f16-T5/perturbations.json`
- accuracy 1 (9 rows), accepted 1, coverage 0.889, NLL 0.012, Brier 0.001, ECE 0.012
- latency median 40 ms, p95 43 ms, 24.65 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 40 / 42 ms
- peak device 3.96 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.6 s
- Telemetry: GPU avg 66.8 W (baseline 28.5 W), peak 147.5 W, net energy 0.105 kJ (DCGM —), util avg 39 %, SM active —, DRAM active —, VRAM max 5577 MiB (baseline 1295), temp max 68 °C, power-cap throttle — s; CPU 0.97 cores avg (host util — %), peak RSS 2.88 GiB

### 22. `cuda-4b-q4_k_m-f16-T6` — validate_checkpoint — ok

- 4B Q4_K_M · CUDA · started 2026-09-28 21:25:03 · wall 6.9 s (estimate 73 s) · exit [0]
- `python scripts/validate_checkpoint.py --size 4b --quant q4_k_m --device cuda --output results/local-hw/cuda-4b-q4_k_m-f16-T6/validate`
- smoke accuracy 0.900 (NLL 0.735, 0 changed), perturbations 1
- long state: shared 355 ms vs direct 1049 ms, 0 changed (max Δp 0.021567)
- validation 4.8 s, load 1.6 s, peak 3.96 GiB
- Telemetry: GPU avg 106.5 W (baseline 28.1 W), peak 157.7 W, net energy 0.538 kJ (DCGM 0.547), util avg 64 %, SM active 0.67, DRAM active 0.19, VRAM max 5583 MiB (baseline 1295), temp max 71 °C, power-cap throttle 3.2 s; CPU 0.99 cores avg (host util 4 %), peak RSS 2.89 GiB

### 23. `cuda-4b-q4_k_m-f16-T7` — semif_compare — ok

- 4B Q4_K_M · CUDA · started 2026-09-28 21:25:23 · wall 66.4 s (estimate 103 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q4_k_m --device cuda --output results/local-hw/cuda-4b-q4_k_m-f16-T7/semif`
- authored144 0.765, perturbations108 0.830, held-out 0.724 / 0.806, macro-F1 0.756
- latency p50 37 ms, p95 40 ms; shape777 shared 28.54 dec/s (37 states, 2111 tok/dec), direct 3.12 dec/s (3 states)
- shared vs direct 2/63 flips (max Δp 0.215); stability flips reversal/wrapper/context 6/1/2; confident on missing evidence 6/36; rule_application perturbed 0.611 (NLL 1.82)
- peak device 3.96 GiB
- vs maintainers: authored144 difference -0.004 [-0.028, +0.014], different argmax 4/252
- Telemetry: GPU avg 133.1 W (baseline 28.1 W), peak 180.4 W, net energy 6.968 kJ (DCGM 6.860), util avg 81 %, SM active 0.67, DRAM active 0.19, VRAM max 5581 MiB (baseline 1295), temp max 85 °C, power-cap throttle 45.3 s; CPU 1.00 cores avg (host util 4 %), peak RSS 2.88 GiB

### 24. `cuda-4b-q4_k_m-f16-T9` — typed-decisions — ok

- 4B Q4_K_M · CUDA · started 2026-09-28 21:26:42 · wall 62.8 s (estimate 113 s) · exit [0]
- `python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-4B-GGUF/Spark-X2.5-4B-Q4_K_M.gguf --device cuda --output results/local-hw/cuda-4b-q4_k_m-f16-T9/typed.json`
- accuracy 0.574, KL 2.321, Brier 0.445, ECE 0.324, macro-F1 0.412
- p50 151 ms/case, p95 189 ms, 32.92 dec/s
- by workflow: agent_trace_observability 0.418, customer_service 0.638, invoice_processing 0.650, security_incidents 0.588
- Telemetry: GPU avg 140.1 W (baseline 31.5 W), peak 164.4 W, net energy 6.828 kJ (DCGM 6.600), util avg 82 %, SM active 0.68, DRAM active 0.21, VRAM max 5581 MiB (baseline 1295), temp max 87 °C, power-cap throttle 43.3 s; CPU 1.00 cores avg (host util 4 %), peak RSS 2.88 GiB

### 25. `cuda-1.7b-q8_0-f16-T3` — rizzo decide examples — ok

- 1.7B Q8_0 · CUDA · started 2026-09-28 20:55:50 · wall 4.2 s (estimate 40 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 1.7b --quant q8_0 --device cuda --output results/local-hw/cuda-1.7b-q8_0-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 1.7b --quant q8_0 --device cuda --output results/local-hw/cuda-1.7b-q8_0-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 1.7b --quant q8_0 --device cuda --output results/local-hw/cuda-1.7b-q8_0-f16-T3/decide-numeric.json`
- ticket: needs_access_support=False (ok, p=0.973); queue=None (insufficient_evidence, p=0.897); urgency=None (insufficient_evidence, p=0.929); deadline_hours=None (insufficient_evidence, p=1.000) — total 82 ms, inference 67 ms, 1287 tokens, load 1.2 s
- house: estimated_price=None (insufficient_evidence, p=1.000) — total 58 ms, inference 50 ms, 562 tokens, load 1.1 s
- numeric: fill=74.94570939589498 (ok, p=0.730) — total 48 ms, inference 41 ms, 398 tokens, load 1.1 s
- Telemetry: GPU avg 48.2 W (baseline 30.5 W), peak 98.6 W, net energy 0.073 kJ (DCGM —), util avg 16 %, SM active —, DRAM active —, VRAM max 3927 MiB (baseline 1295), temp max 54 °C, power-cap throttle — s; CPU 0.95 cores avg (host util — %), peak RSS 2.16 GiB

### 26. `cuda-1.7b-q8_0-f16-T4` — smoke — ok

- 1.7B Q8_0 · CUDA · started 2026-09-28 20:56:07 · wall 2.4 s (estimate 28 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q8_0 --device cuda --output results/local-hw/cuda-1.7b-q8_0-f16-T4/smoke.json`
- accuracy 0.400 (20 rows), accepted 0.833, coverage 0.300, NLL 3.865, Brier 1.097, ECE 0.528
- latency median 22 ms, p95 61 ms, 46.95 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.059544, median 22 / 23 ms
- peak device 2.35 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.2 s
- Telemetry: GPU avg 65.6 W (baseline 28.9 W), peak 138.7 W, net energy 0.089 kJ (DCGM —), util avg 34 %, SM active —, DRAM active —, VRAM max 3927 MiB (baseline 1295), temp max 57 °C, power-cap throttle — s; CPU 0.96 cores avg (host util — %), peak RSS 2.15 GiB

### 27. `cuda-1.7b-q8_0-f16-T5` — perturbations — ok

- 1.7B Q8_0 · CUDA · started 2026-09-28 20:56:23 · wall 1.8 s (estimate 25 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 1.7b --quant q8_0 --device cuda --output results/local-hw/cuda-1.7b-q8_0-f16-T5/perturbations.json`
- accuracy 0.444 (9 rows), accepted 1, coverage 0.333, NLL 3.126, Brier 0.960, ECE 0.551
- latency median 19 ms, p95 22 ms, 52.47 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 19 / 20 ms
- peak device 2.35 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.2 s
- Telemetry: GPU avg 38.7 W (baseline 26.8 W), peak 146.0 W, net energy 0.025 kJ (DCGM —), util avg 17 %, SM active —, DRAM active —, VRAM max 3927 MiB (baseline 1295), temp max 57 °C, power-cap throttle — s; CPU 0.97 cores avg (host util — %), peak RSS 2.15 GiB

### 28. `cuda-1.7b-q8_0-f16-T6` — validate_checkpoint — ok

- 1.7B Q8_0 · CUDA · started 2026-09-28 20:56:37 · wall 3.9 s (estimate 49 s) · exit [0]
- `python scripts/validate_checkpoint.py --size 1.7b --quant q8_0 --device cuda --output results/local-hw/cuda-1.7b-q8_0-f16-T6/validate`
- smoke accuracy 0.400 (NLL 3.865, 0 changed), perturbations 0.444
- long state: shared 169 ms vs direct 443 ms, 0 changed (max Δp 0.005816)
- validation 2.3 s, load 1.2 s, peak 2.35 GiB
- Telemetry: GPU avg 91.2 W (baseline 26.8 W), peak 159.1 W, net energy 0.252 kJ (DCGM —), util avg 49 %, SM active —, DRAM active —, VRAM max 3929 MiB (baseline 1295), temp max 61 °C, power-cap throttle — s; CPU 0.96 cores avg (host util — %), peak RSS 2.17 GiB

### 29. `cuda-1.7b-q8_0-f16-T7` — semif_compare — ok

- 1.7B Q8_0 · CUDA · started 2026-09-28 20:56:54 · wall 32.9 s (estimate 67 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q8_0 --device cuda --output results/local-hw/cuda-1.7b-q8_0-f16-T7/semif`
- authored144 0.700, perturbations108 0.646, held-out 0.690 / 0.532, macro-F1 0.675
- latency p50 18 ms, p95 20 ms; shape777 shared 49.70 dec/s (37 states, 2111 tok/dec), direct 7.87 dec/s (3 states)
- shared vs direct 2/63 flips (max Δp 0.159); stability flips reversal/wrapper/context 18/5/4; confident on missing evidence 2/36; rule_application perturbed 0.315 (NLL 3.50)
- peak device 2.35 GiB
- vs maintainers: authored144 difference 0.022 [+0.000, +0.047], different argmax 6/252
- Telemetry: GPU avg 119.8 W (baseline 26.7 W), peak 177.5 W, net energy 3.062 kJ (DCGM 2.885), util avg 64 %, SM active 0.48, DRAM active 0.18, VRAM max 3929 MiB (baseline 1295), temp max 76 °C, power-cap throttle 9.8 s; CPU 1.00 cores avg (host util 5 %), peak RSS 2.16 GiB

### 30. `cuda-1.7b-q8_0-f16-T9` — typed-decisions — ok

- 1.7B Q8_0 · CUDA · started 2026-09-28 20:57:41 · wall 30.4 s (estimate 73 s) · exit [0]
- `python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-1.7B-GGUF/Spark-X2.5-1.7B-Q8_0.gguf --device cuda --output results/local-hw/cuda-1.7b-q8_0-f16-T9/typed.json`
- accuracy 0.530, KL 3.023, Brier 0.496, ECE 0.346, macro-F1 0.334
- p50 72 ms/case, p95 90 ms, 69.36 dec/s
- by workflow: agent_trace_observability 0.376, customer_service 0.642, invoice_processing 0.486, security_incidents 0.618
- Telemetry: GPU avg 137.6 W (baseline 28.7 W), peak 161.1 W, net energy 3.306 kJ (DCGM 3.186), util avg 67 %, SM active 0.51, DRAM active 0.21, VRAM max 3929 MiB (baseline 1295), temp max 81 °C, power-cap throttle 10.5 s; CPU 1.00 cores avg (host util 4 %), peak RSS 2.16 GiB

### 31. `cuda-1.7b-q4_k_m-f16-T3` — rizzo decide examples — ok

- 1.7B Q4_K_M · CUDA · started 2026-09-28 21:27:58 · wall 3.5 s (estimate 40 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 1.7b --quant q4_k_m --device cuda --output results/local-hw/cuda-1.7b-q4_k_m-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 1.7b --quant q4_k_m --device cuda --output results/local-hw/cuda-1.7b-q4_k_m-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 1.7b --quant q4_k_m --device cuda --output results/local-hw/cuda-1.7b-q4_k_m-f16-T3/decide-numeric.json`
- ticket: needs_access_support=False (ok, p=0.735); queue=None (insufficient_evidence, p=0.774); urgency=None (insufficient_evidence, p=0.974); deadline_hours=None (insufficient_evidence, p=1.000) — total 82 ms, inference 66 ms, 1287 tokens, load 0.9 s
- house: estimated_price=None (insufficient_evidence, p=1.000) — total 60 ms, inference 54 ms, 562 tokens, load 0.8 s
- numeric: fill=None (insufficient_evidence, p=0.998) — total 52 ms, inference 45 ms, 398 tokens, load 0.8 s
- Telemetry: GPU avg 40.3 W (baseline 31.9 W), peak 79.1 W, net energy 0.030 kJ (DCGM —), util avg 10 %, SM active —, DRAM active —, VRAM max 3249 MiB (baseline 1295), temp max 72 °C, power-cap throttle — s; CPU 0.91 cores avg (host util — %), peak RSS 1.49 GiB

### 32. `cuda-1.7b-q4_k_m-f16-T4` — smoke — ok

- 1.7B Q4_K_M · CUDA · started 2026-09-28 21:28:15 · wall 2.0 s (estimate 28 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q4_k_m --device cuda --output results/local-hw/cuda-1.7b-q4_k_m-f16-T4/smoke.json`
- accuracy 0.350 (20 rows), accepted 0.800, coverage 0.250, NLL 3.584, Brier 1.071, ECE 0.578
- latency median 23 ms, p95 62 ms, 46.64 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.167532, median 23 / 23 ms
- peak device 1.68 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 0.9 s
- Telemetry: GPU avg 70.5 W (baseline 31.7 W), peak 144.8 W, net energy 0.079 kJ (DCGM —), util avg 33 %, SM active —, DRAM active —, VRAM max 3249 MiB (baseline 1295), temp max 70 °C, power-cap throttle — s; CPU 0.98 cores avg (host util — %), peak RSS 1.49 GiB

### 33. `cuda-1.7b-q4_k_m-f16-T5` — perturbations — ok

- 1.7B Q4_K_M · CUDA · started 2026-09-28 21:28:30 · wall 1.5 s (estimate 25 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 1.7b --quant q4_k_m --device cuda --output results/local-hw/cuda-1.7b-q4_k_m-f16-T5/perturbations.json`
- accuracy 0.556 (9 rows), accepted 1, coverage 0.444, NLL 2.412, Brier 0.895, ECE 0.520
- latency median 20 ms, p95 23 ms, 50.54 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 20 / 20 ms
- peak device 1.68 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 0.9 s
- Telemetry: GPU avg 50.0 W (baseline 29.0 W), peak 144.6 W, net energy 0.032 kJ (DCGM —), util avg 22 %, SM active —, DRAM active —, VRAM max 3249 MiB (baseline 1295), temp max 65 °C, power-cap throttle — s; CPU 0.96 cores avg (host util — %), peak RSS 1.49 GiB

### 34. `cuda-1.7b-q4_k_m-f16-T6` — validate_checkpoint — ok

- 1.7B Q4_K_M · CUDA · started 2026-09-28 21:28:44 · wall 3.7 s (estimate 49 s) · exit [0]
- `python scripts/validate_checkpoint.py --size 1.7b --quant q4_k_m --device cuda --output results/local-hw/cuda-1.7b-q4_k_m-f16-T6/validate`
- smoke accuracy 0.350 (NLL 3.584, 0 changed), perturbations 0.556
- long state: shared 171 ms vs direct 466 ms, 0 changed (max Δp 0.060829)
- validation 2.3 s, load 0.9 s, peak 1.69 GiB
- Telemetry: GPU avg 102.1 W (baseline 29.1 W), peak 153.5 W, net energy 0.266 kJ (DCGM —), util avg 56 %, SM active —, DRAM active —, VRAM max 3251 MiB (baseline 1295), temp max 67 °C, power-cap throttle — s; CPU 0.97 cores avg (host util — %), peak RSS 1.50 GiB

### 35. `cuda-1.7b-q4_k_m-f16-T7` — semif_compare — ok

- 1.7B Q4_K_M · CUDA · started 2026-09-28 21:29:01 · wall 33.6 s (estimate 67 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q4_k_m --device cuda --output results/local-hw/cuda-1.7b-q4_k_m-f16-T7/semif`
- authored144 0.607, perturbations108 0.472, held-out 0.603 / 0.481, macro-F1 0.576
- latency p50 18 ms, p95 20 ms; shape777 shared 48.65 dec/s (37 states, 2111 tok/dec), direct 7.40 dec/s (3 states)
- shared vs direct 6/63 flips (max Δp 0.330); stability flips reversal/wrapper/context 17/4/2; confident on missing evidence 6/36; rule_application perturbed 0.056 (NLL 4.26)
- peak device 1.69 GiB
- Telemetry: GPU avg 122.5 W (baseline 29.0 W), peak 179.9 W, net energy 3.146 kJ (DCGM 3.068), util avg 65 %, SM active 0.49, DRAM active 0.15, VRAM max 3251 MiB (baseline 1295), temp max 80 °C, power-cap throttle 11.1 s; CPU 1.00 cores avg (host util 5 %), peak RSS 1.49 GiB

### 36. `cuda-1.7b-q4_k_m-f16-T9` — typed-decisions — ok

- 1.7B Q4_K_M · CUDA · started 2026-09-28 21:29:48 · wall 30.9 s (estimate 73 s) · exit [0]
- `python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-1.7B-GGUF/Spark-X2.5-1.7B-Q4_K_M.gguf --device cuda --output results/local-hw/cuda-1.7b-q4_k_m-f16-T9/typed.json`
- accuracy 0.481, KL 2.799, Brier 0.579, ECE 0.427, macro-F1 0.289
- p50 74 ms/case, p95 92 ms, 67.43 dec/s
- by workflow: agent_trace_observability 0.432, customer_service 0.552, invoice_processing 0.348, security_incidents 0.592
- Telemetry: GPU avg 138.7 W (baseline 29.7 W), peak 156.2 W, net energy 3.367 kJ (DCGM 3.282), util avg 67 %, SM active 0.52, DRAM active 0.17, VRAM max 3251 MiB (baseline 1295), temp max 83 °C, power-cap throttle 11.1 s; CPU 1.00 cores avg (host util 4 %), peak RSS 1.49 GiB

### 37. `cuda-1.7b-bf16-f16-T3` — rizzo decide examples — ok

- 1.7B BF16 · CUDA · started 2026-09-28 21:30:32 · wall 6.6 s (estimate 44 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 1.7b --quant bf16 --device cuda --output results/local-hw/cuda-1.7b-bf16-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 1.7b --quant bf16 --device cuda --output results/local-hw/cuda-1.7b-bf16-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 1.7b --quant bf16 --device cuda --output results/local-hw/cuda-1.7b-bf16-f16-T3/decide-numeric.json`
- ticket: needs_access_support=False (ok, p=0.933); queue=None (insufficient_evidence, p=0.799); urgency=None (insufficient_evidence, p=0.841); deadline_hours=None (insufficient_evidence, p=1.000) — total 117 ms, inference 102 ms, 1287 tokens, load 1.9 s
- house: estimated_price=None (insufficient_evidence, p=1.000) — total 91 ms, inference 85 ms, 562 tokens, load 1.8 s
- numeric: fill=74.96895565092947 (ok, p=0.872) — total 79 ms, inference 72 ms, 398 tokens, load 1.8 s
- Telemetry: GPU avg 40.9 W (baseline 30.7 W), peak 50.7 W, net energy 0.067 kJ (DCGM 0.088), util avg 12 %, SM active 0.05, DRAM active 0.02, VRAM max 5551 MiB (baseline 1295), temp max 64 °C, power-cap throttle 0.0 s; CPU 0.97 cores avg (host util 4 %), peak RSS 3.65 GiB

### 38. `cuda-1.7b-bf16-f16-T4` — smoke — ok

- 1.7B BF16 · CUDA · started 2026-09-28 21:30:52 · wall 3.1 s (estimate 30 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant bf16 --device cuda --output results/local-hw/cuda-1.7b-bf16-f16-T4/smoke.json`
- accuracy 0.450 (20 rows), accepted 0.857, coverage 0.350, NLL 3.462, Brier 1.014, ECE 0.512
- latency median 21 ms, p95 72 ms, 46.81 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.032929, median 21 / 19 ms
- peak device 3.93 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.9 s
- Telemetry: GPU avg 54.0 W (baseline 29.9 W), peak 145.7 W, net energy 0.072 kJ (DCGM —), util avg 29 %, SM active —, DRAM active —, VRAM max 5551 MiB (baseline 1295), temp max 64 °C, power-cap throttle — s; CPU 0.98 cores avg (host util — %), peak RSS 3.65 GiB

### 39. `cuda-1.7b-bf16-f16-T5` — perturbations — ok

- 1.7B BF16 · CUDA · started 2026-09-28 21:31:08 · wall 2.5 s (estimate 27 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 1.7b --quant bf16 --device cuda --output results/local-hw/cuda-1.7b-bf16-f16-T5/perturbations.json`
- accuracy 0.556 (9 rows), accepted 1, coverage 0.444, NLL 2.759, Brier 0.871, ECE 0.499
- latency median 19 ms, p95 22 ms, 52.39 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 19 / 19 ms
- peak device 3.93 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.9 s
- Telemetry: GPU avg 37.6 W (baseline 27.4 W), peak 143.7 W, net energy 0.031 kJ (DCGM —), util avg 24 %, SM active —, DRAM active —, VRAM max 5551 MiB (baseline 1295), temp max 61 °C, power-cap throttle — s; CPU 0.98 cores avg (host util — %), peak RSS 3.65 GiB

### 40. `cuda-1.7b-bf16-f16-T6` — validate_checkpoint — ok

- 1.7B BF16 · CUDA · started 2026-09-28 21:31:23 · wall 4.7 s (estimate 55 s) · exit [0]
- `python scripts/validate_checkpoint.py --size 1.7b --quant bf16 --device cuda --output results/local-hw/cuda-1.7b-bf16-f16-T6/validate`
- smoke accuracy 0.450 (NLL 3.462, 0 changed), perturbations 0.556
- long state: shared 183 ms vs direct 482 ms, 0 changed (max Δp 0.012355)
- validation 2.3 s, load 1.9 s, peak 3.93 GiB
- Telemetry: GPU avg 87.5 W (baseline 27.6 W), peak 151.8 W, net energy 0.282 kJ (DCGM —), util avg 46 %, SM active —, DRAM active —, VRAM max 5553 MiB (baseline 1295), temp max 64 °C, power-cap throttle — s; CPU 0.98 cores avg (host util — %), peak RSS 3.66 GiB

### 41. `cuda-1.7b-bf16-f16-T7` — semif_compare — ok

- 1.7B BF16 · CUDA · started 2026-09-28 21:31:41 · wall 35.0 s (estimate 76 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant bf16 --device cuda --output results/local-hw/cuda-1.7b-bf16-f16-T7/semif`
- authored144 0.683, perturbations108 0.640, held-out 0.690 / 0.532, macro-F1 0.658
- latency p50 17 ms, p95 18 ms; shape777 shared 48.40 dec/s (37 states, 2111 tok/dec), direct 7.03 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.047); stability flips reversal/wrapper/context 18/4/4; confident on missing evidence 2/36; rule_application perturbed 0.296 (NLL 3.55)
- peak device 3.93 GiB
- Telemetry: GPU avg 120.4 W (baseline 27.0 W), peak 179.8 W, net energy 3.273 kJ (DCGM 3.054), util avg 65 %, SM active 0.49, DRAM active 0.25, VRAM max 5553 MiB (baseline 1295), temp max 78 °C, power-cap throttle 14.0 s; CPU 1.00 cores avg (host util 4 %), peak RSS 3.65 GiB

### 42. `cuda-1.7b-bf16-f16-T9` — typed-decisions — ok

- 1.7B BF16 · CUDA · started 2026-09-28 21:32:29 · wall 31.6 s (estimate 83 s) · exit [0]
- `python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-1.7B-GGUF/Spark-X2.5-1.7B.gguf --device cuda --output results/local-hw/cuda-1.7b-bf16-f16-T9/typed.json`
- accuracy 0.528, KL 3.019, Brier 0.494, ECE 0.348, macro-F1 0.329
- p50 73 ms/case, p95 92 ms, 68.20 dec/s
- by workflow: agent_trace_observability 0.374, customer_service 0.648, invoice_processing 0.476, security_incidents 0.616
- Telemetry: GPU avg 136.0 W (baseline 28.9 W), peak 155.9 W, net energy 3.387 kJ (DCGM 3.189), util avg 68 %, SM active 0.52, DRAM active 0.31, VRAM max 5553 MiB (baseline 1295), temp max 81 °C, power-cap throttle 14.0 s; CPU 1.00 cores avg (host util 4 %), peak RSS 3.65 GiB

### 43. `cuda-4b-q8_0-q8_0-T4` — smoke — ok

- 4B Q8_0 · CUDA · KV q8_0 · started 2026-09-28 21:33:14 · wall 4.6 s (estimate 38 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --device cuda --kv-type q8_0 --output results/local-hw/cuda-4b-q8_0-q8_0-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.445, Brier 0.077, ECE 0.038
- latency median 42 ms, p95 122 ms, 23.54 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000028, median 42 / 42 ms
- peak device 4.97 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 2.4 s
- Telemetry: GPU avg 80.8 W (baseline 30.1 W), peak 149.2 W, net energy 0.231 kJ (DCGM —), util avg 42 %, SM active —, DRAM active —, VRAM max 6611 MiB (baseline 1295), temp max 72 °C, power-cap throttle — s; CPU 0.98 cores avg (host util — %), peak RSS 4.53 GiB

### 44. `cuda-4b-q8_0-q8_0-T7` — semif_compare — ok

- 4B Q8_0 · CUDA · KV q8_0 · started 2026-09-28 21:33:32 · wall 64.2 s (estimate 103 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cuda --kv-type q8_0 --output results/local-hw/cuda-4b-q8_0-q8_0-T7/semif`
- authored144 0.816, perturbations108 0.835, held-out 0.799 / 0.856, macro-F1 0.811
- latency p50 35 ms, p95 37 ms; shape777 shared 29.54 dec/s (37 states, 2111 tok/dec), direct 3.32 dec/s (3 states)
- shared vs direct 1/63 flips (max Δp 0.106); stability flips reversal/wrapper/context 4/3/3; confident on missing evidence 6/36; rule_application perturbed 0.537 (NLL 1.69)
- peak device 4.97 GiB
- vs this-cuda-kv-f16: authored144 difference 0.006 [+0.000, +0.020], different argmax 4/252
- Telemetry: GPU avg 129.4 W (baseline 23.1 W), peak 177.0 W, net energy 6.812 kJ (DCGM 6.631), util avg 78 %, SM active 0.63, DRAM active 0.25, VRAM max 6645 MiB (baseline 1309), temp max 84 °C, power-cap throttle 44.4 s; CPU 1.00 cores avg (host util 4 %), peak RSS 4.53 GiB

### 45. `cuda-4b-q8_0-q4_0-T4` — smoke — ok

- 4B Q8_0 · CUDA · KV q4_0 · started 2026-09-28 21:34:49 · wall 4.4 s (estimate 38 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --device cuda --kv-type q4_0 --output results/local-hw/cuda-4b-q8_0-q4_0-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.549, Brier 0.099, ECE 0.051
- latency median 40 ms, p95 119 ms, 24.45 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000029, median 40 / 39 ms
- peak device 4.62 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 2.4 s
- Telemetry: GPU avg 61.9 W (baseline 19.7 W), peak 148.2 W, net energy 0.188 kJ (DCGM —), util avg 35 %, SM active —, DRAM active —, VRAM max 6283 MiB (baseline 1327), temp max 73 °C, power-cap throttle — s; CPU 0.99 cores avg (host util — %), peak RSS 4.53 GiB

### 46. `cuda-4b-q8_0-q4_0-T7` — semif_compare — ok

- 4B Q8_0 · CUDA · KV q4_0 · started 2026-09-28 21:35:07 · wall 64.0 s (estimate 103 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cuda --kv-type q4_0 --output results/local-hw/cuda-4b-q8_0-q4_0-T7/semif`
- authored144 0.811, perturbations108 0.859, held-out 0.812 / 0.875, macro-F1 0.802
- latency p50 34 ms, p95 37 ms; shape777 shared 29.56 dec/s (37 states, 2111 tok/dec), direct 3.33 dec/s (3 states)
- shared vs direct 1/63 flips (max Δp 0.417); stability flips reversal/wrapper/context 4/5/3; confident on missing evidence 7/36; rule_application perturbed 0.611 (NLL 1.58)
- peak device 4.62 GiB
- vs this-cuda-kv-f16: authored144 difference 0.001 [-0.030, +0.035], different argmax 10/252
- Telemetry: GPU avg 130.2 W (baseline 18.1 W), peak 179.6 W, net energy 7.181 kJ (DCGM 7.043), util avg 78 %, SM active 0.64, DRAM active 0.24, VRAM max 6285 MiB (baseline 1327), temp max 84 °C, power-cap throttle 44.6 s; CPU 1.00 cores avg (host util 4 %), peak RSS 4.53 GiB

### 47. `cuda-4b-bf16-q8_0-T4` — smoke — ok

- 4B BF16 · CUDA · KV q8_0 · started 2026-09-28 21:36:24 · wall 6.2 s (estimate 43 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant bf16 --device cuda --kv-type q8_0 --output results/local-hw/cuda-4b-bf16-q8_0-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.448, Brier 0.077, ECE 0.039
- latency median 41 ms, p95 135 ms, 23.44 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000003, median 41 / 40 ms
- peak device 8.65 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 4.1 s
- Telemetry: GPU avg 51.3 W (baseline 19.2 W), peak 153.4 W, net energy 0.195 kJ (DCGM 0.105), util avg 27 %, SM active 0.00, DRAM active 0.00, VRAM max 10415 MiB (baseline 1327), temp max 72 °C, power-cap throttle 0.0 s; CPU 0.99 cores avg (host util 4 %), peak RSS 8.12 GiB

### 48. `cuda-4b-bf16-q8_0-T7` — semif_compare — ok

- 4B BF16 · CUDA · KV q8_0 · started 2026-09-28 21:36:44 · wall 67.9 s (estimate 121 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant bf16 --device cuda --kv-type q8_0 --output results/local-hw/cuda-4b-bf16-q8_0-T7/semif`
- authored144 0.823, perturbations108 0.865, held-out 0.810 / 0.875, macro-F1 0.818
- latency p50 38 ms, p95 40 ms; shape777 shared 28.90 dec/s (37 states, 2111 tok/dec), direct 3.20 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.063); stability flips reversal/wrapper/context 5/3/2; confident on missing evidence 6/36; rule_application perturbed 0.630 (NLL 1.67)
- peak device 8.65 GiB
- vs this-cuda-kv-f16: authored144 difference 0.003 [-0.018, +0.028], different argmax 4/252
- Telemetry: GPU avg 127.4 W (baseline 16.5 W), peak 180.3 W, net energy 7.527 kJ (DCGM 6.923), util avg 77 %, SM active 0.64, DRAM active 0.32, VRAM max 10417 MiB (baseline 1327), temp max 84 °C, power-cap throttle 49.3 s; CPU 1.00 cores avg (host util 4 %), peak RSS 8.12 GiB

### 49. `cuda-4b-bf16-q4_0-T4` — smoke — ok

- 4B BF16 · CUDA · KV q4_0 · started 2026-09-28 21:38:05 · wall 6.1 s (estimate 43 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant bf16 --device cuda --kv-type q4_0 --output results/local-hw/cuda-4b-bf16-q4_0-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 1, coverage 0.750, NLL 0.339, Brier 0.076, ECE 0.036
- latency median 41 ms, p95 133 ms, 23.68 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000040, median 41 / 40 ms
- peak device 8.30 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 4.0 s
- Telemetry: GPU avg 49.5 W (baseline 19.7 W), peak 145.9 W, net energy 0.177 kJ (DCGM 0.089), util avg 23 %, SM active 0.06, DRAM active 0.05, VRAM max 10055 MiB (baseline 1327), temp max 72 °C, power-cap throttle 0.6 s; CPU 0.99 cores avg (host util 4 %), peak RSS 8.12 GiB

### 50. `cuda-4b-bf16-q4_0-T7` — semif_compare — ok

- 4B BF16 · CUDA · KV q4_0 · started 2026-09-28 21:38:24 · wall 67.8 s (estimate 121 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant bf16 --device cuda --kv-type q4_0 --output results/local-hw/cuda-4b-bf16-q4_0-T7/semif`
- authored144 0.807, perturbations108 0.826, held-out 0.795 / 0.833, macro-F1 0.799
- latency p50 37 ms, p95 40 ms; shape777 shared 28.80 dec/s (37 states, 2111 tok/dec), direct 3.21 dec/s (3 states)
- shared vs direct 3/63 flips (max Δp 0.405); stability flips reversal/wrapper/context 3/3/3; confident on missing evidence 7/36; rule_application perturbed 0.611 (NLL 1.60)
- peak device 8.30 GiB
- vs this-cuda-kv-f16: authored144 difference -0.012 [-0.051, +0.026], different argmax 16/252
- Telemetry: GPU avg 127.5 W (baseline 17.7 W), peak 181.1 W, net energy 7.435 kJ (DCGM 7.173), util avg 77 %, SM active 0.65, DRAM active 0.32, VRAM max 10057 MiB (baseline 1327), temp max 84 °C, power-cap throttle 50.7 s; CPU 1.00 cores avg (host util 4 %), peak RSS 8.13 GiB

### 51. `cuda-4b-q4_k_m-q8_0-T4` — smoke — ok

- 4B Q4_K_M · CUDA · KV q8_0 · started 2026-09-28 21:39:45 · wall 3.7 s (estimate 38 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q4_k_m --device cuda --kv-type q8_0 --output results/local-hw/cuda-4b-q4_k_m-q8_0-T4/smoke.json`
- accuracy 0.900 (20 rows), accepted 1, coverage 0.750, NLL 0.663, Brier 0.174, ECE 0.091
- latency median 41 ms, p95 123 ms, 22.67 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.040437, median 41 / 42 ms
- peak device 3.32 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.5 s
- Telemetry: GPU avg 81.2 W (baseline 19.3 W), peak 151.6 W, net energy 0.227 kJ (DCGM —), util avg 41 %, SM active —, DRAM active —, VRAM max 4953 MiB (baseline 1327), temp max 75 °C, power-cap throttle — s; CPU 0.97 cores avg (host util — %), peak RSS 2.88 GiB

### 52. `cuda-4b-q4_k_m-q8_0-T7` — semif_compare — ok

- 4B Q4_K_M · CUDA · KV q8_0 · started 2026-09-28 21:40:02 · wall 66.7 s (estimate 103 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q4_k_m --device cuda --kv-type q8_0 --output results/local-hw/cuda-4b-q4_k_m-q8_0-T7/semif`
- authored144 0.782, perturbations108 0.822, held-out 0.742 / 0.801, macro-F1 0.773
- latency p50 36 ms, p95 39 ms; shape777 shared 28.28 dec/s (37 states, 2111 tok/dec), direct 3.11 dec/s (3 states)
- shared vs direct 1/63 flips (max Δp 0.146); stability flips reversal/wrapper/context 6/2/1; confident on missing evidence 6/36; rule_application perturbed 0.556 (NLL 1.81)
- peak device 3.32 GiB
- vs this-cuda-kv-f16: authored144 difference 0.017 [+0.000, +0.041], different argmax 4/252
- Telemetry: GPU avg 133.3 W (baseline 17.8 W), peak 182.2 W, net energy 7.703 kJ (DCGM 7.281), util avg 81 %, SM active 0.67, DRAM active 0.19, VRAM max 4955 MiB (baseline 1327), temp max 85 °C, power-cap throttle 46.9 s; CPU 1.00 cores avg (host util 4 %), peak RSS 2.88 GiB

### 53. `cuda-4b-q4_k_m-q4_0-T4` — smoke — ok

- 4B Q4_K_M · CUDA · KV q4_0 · started 2026-09-28 21:41:22 · wall 3.6 s (estimate 38 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q4_k_m --device cuda --kv-type q4_0 --output results/local-hw/cuda-4b-q4_k_m-q4_0-T4/smoke.json`
- accuracy 0.900 (20 rows), accepted 1, coverage 0.750, NLL 0.621, Brier 0.221, ECE 0.127
- latency median 41 ms, p95 123 ms, 23.37 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.009315, median 41 / 41 ms
- peak device 2.96 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.5 s
- Telemetry: GPU avg 78.9 W (baseline 19.0 W), peak 157.0 W, net energy 0.210 kJ (DCGM —), util avg 41 %, SM active —, DRAM active —, VRAM max 4593 MiB (baseline 1327), temp max 74 °C, power-cap throttle — s; CPU 0.99 cores avg (host util — %), peak RSS 2.88 GiB

### 54. `cuda-4b-q4_k_m-q4_0-T7` — semif_compare — ok

- 4B Q4_K_M · CUDA · KV q4_0 · started 2026-09-28 21:41:39 · wall 66.3 s (estimate 103 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q4_k_m --device cuda --kv-type q4_0 --output results/local-hw/cuda-4b-q4_k_m-q4_0-T7/semif`
- authored144 0.762, perturbations108 0.847, held-out 0.724 / 0.819, macro-F1 0.751
- latency p50 36 ms, p95 38 ms; shape777 shared 28.31 dec/s (37 states, 2111 tok/dec), direct 3.12 dec/s (3 states)
- shared vs direct 5/63 flips (max Δp 0.696); stability flips reversal/wrapper/context 7/3/2; confident on missing evidence 6/36; rule_application perturbed 0.630 (NLL 2.03)
- peak device 2.97 GiB
- vs this-cuda-kv-f16: authored144 difference -0.003 [-0.028, +0.022], different argmax 7/252
- Telemetry: GPU avg 133.3 W (baseline 17.6 W), peak 182.1 W, net energy 7.663 kJ (DCGM 7.316), util avg 80 %, SM active 0.66, DRAM active 0.19, VRAM max 4595 MiB (baseline 1327), temp max 85 °C, power-cap throttle 46.1 s; CPU 1.00 cores avg (host util 4 %), peak RSS 2.88 GiB

### 55. `cuda-1.7b-q8_0-q8_0-T4` — smoke — ok

- 1.7B Q8_0 · CUDA · KV q8_0 · started 2026-09-28 21:42:58 · wall 2.4 s (estimate 28 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q8_0 --device cuda --kv-type q8_0 --output results/local-hw/cuda-1.7b-q8_0-q8_0-T4/smoke.json`
- accuracy 0.450 (20 rows), accepted 0.857, coverage 0.350, NLL 3.601, Brier 1.043, ECE 0.552
- latency median 23 ms, p95 62 ms, 47.29 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.027087, median 23 / 21 ms
- peak device 2.10 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.2 s
- Telemetry: GPU avg 64.4 W (baseline 18.1 W), peak 146.5 W, net energy 0.106 kJ (DCGM —), util avg 25 %, SM active —, DRAM active —, VRAM max 3703 MiB (baseline 1327), temp max 72 °C, power-cap throttle — s; CPU 0.96 cores avg (host util — %), peak RSS 2.15 GiB

### 56. `cuda-1.7b-q8_0-q8_0-T7` — semif_compare — ok

- 1.7B Q8_0 · CUDA · KV q8_0 · started 2026-09-28 21:43:14 · wall 33.2 s (estimate 67 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q8_0 --device cuda --kv-type q8_0 --output results/local-hw/cuda-1.7b-q8_0-q8_0-T7/semif`
- authored144 0.687, perturbations108 0.652, held-out 0.708 / 0.532, macro-F1 0.660
- latency p50 17 ms, p95 19 ms; shape777 shared 49.29 dec/s (37 states, 2111 tok/dec), direct 7.69 dec/s (3 states)
- shared vs direct 2/63 flips (max Δp 0.188); stability flips reversal/wrapper/context 18/5/5; confident on missing evidence 2/36; rule_application perturbed 0.333 (NLL 3.45)
- peak device 2.10 GiB
- vs this-cuda-kv-f16: authored144 difference -0.013 [-0.042, +0.018], different argmax 5/252
- Telemetry: GPU avg 120.6 W (baseline 17.0 W), peak 179.0 W, net energy 3.432 kJ (DCGM 3.224), util avg 63 %, SM active 0.46, DRAM active 0.18, VRAM max 3705 MiB (baseline 1327), temp max 80 °C, power-cap throttle 11.5 s; CPU 1.00 cores avg (host util 4 %), peak RSS 2.16 GiB

### 57. `cuda-1.7b-q8_0-q4_0-T4` — smoke — ok

- 1.7B Q8_0 · CUDA · KV q4_0 · started 2026-09-28 21:44:00 · wall 2.4 s (estimate 28 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q8_0 --device cuda --kv-type q4_0 --output results/local-hw/cuda-1.7b-q8_0-q4_0-T4/smoke.json`
- accuracy 0.400 (20 rows), accepted 0.833, coverage 0.300, NLL 3.770, Brier 1.129, ECE 0.597
- latency median 22 ms, p95 61 ms, 48.42 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.028102, median 22 / 21 ms
- peak device 1.96 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.2 s
- Telemetry: GPU avg 62.4 W (baseline 17.6 W), peak 143.7 W, net energy 0.099 kJ (DCGM —), util avg 25 %, SM active —, DRAM active —, VRAM max 3563 MiB (baseline 1327), temp max 68 °C, power-cap throttle — s; CPU 0.97 cores avg (host util — %), peak RSS 2.15 GiB

### 58. `cuda-1.7b-q8_0-q4_0-T7` — semif_compare — ok

- 1.7B Q8_0 · CUDA · KV q4_0 · started 2026-09-28 21:44:16 · wall 32.9 s (estimate 67 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q8_0 --device cuda --kv-type q4_0 --output results/local-hw/cuda-1.7b-q8_0-q4_0-T7/semif`
- authored144 0.687, perturbations108 0.576, held-out 0.677 / 0.532, macro-F1 0.658
- latency p50 17 ms, p95 19 ms; shape777 shared 49.43 dec/s (37 states, 2111 tok/dec), direct 7.74 dec/s (3 states)
- shared vs direct 21/63 flips (max Δp 0.717); stability flips reversal/wrapper/context 18/8/8; confident on missing evidence 1/36; rule_application perturbed 0.296 (NLL 3.32)
- peak device 1.96 GiB
- vs this-cuda-kv-f16: authored144 difference -0.013 [-0.051, +0.029], different argmax 26/252
- Telemetry: GPU avg 117.8 W (baseline 17.2 W), peak 173.4 W, net energy 3.313 kJ (DCGM 3.246), util avg 61 %, SM active 0.48, DRAM active 0.19, VRAM max 3565 MiB (baseline 1327), temp max 79 °C, power-cap throttle 11.6 s; CPU 1.00 cores avg (host util 4 %), peak RSS 2.16 GiB

### 59. `cuda-1.7b-q4_k_m-q8_0-T4` — smoke — ok

- 1.7B Q4_K_M · CUDA · KV q8_0 · started 2026-09-28 21:45:02 · wall 2.1 s (estimate 28 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q4_k_m --device cuda --kv-type q8_0 --output results/local-hw/cuda-1.7b-q4_k_m-q8_0-T4/smoke.json`
- accuracy 0.400 (20 rows), accepted 0.833, coverage 0.300, NLL 3.341, Brier 1.043, ECE 0.577
- latency median 25 ms, p95 71 ms, 44.40 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.155390, median 25 / 27 ms
- peak device 1.43 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 0.9 s
- Telemetry: GPU avg 63.6 W (baseline 17.9 W), peak 143.1 W, net energy 0.098 kJ (DCGM —), util avg 26 %, SM active —, DRAM active —, VRAM max 3025 MiB (baseline 1327), temp max 70 °C, power-cap throttle — s; CPU 0.96 cores avg (host util — %), peak RSS 1.49 GiB

### 60. `cuda-1.7b-q4_k_m-q8_0-T7` — semif_compare — ok

- 1.7B Q4_K_M · CUDA · KV q8_0 · started 2026-09-28 21:45:17 · wall 34.0 s (estimate 67 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q4_k_m --device cuda --kv-type q8_0 --output results/local-hw/cuda-1.7b-q4_k_m-q8_0-T7/semif`
- authored144 0.594, perturbations108 0.464, held-out 0.577 / 0.412, macro-F1 0.558
- latency p50 18 ms, p95 20 ms; shape777 shared 48.28 dec/s (37 states, 2111 tok/dec), direct 7.28 dec/s (3 states)
- shared vs direct 9/63 flips (max Δp 0.265); stability flips reversal/wrapper/context 18/4/2; confident on missing evidence 7/36; rule_application perturbed 0.056 (NLL 4.33)
- peak device 1.44 GiB
- vs this-cuda-kv-f16: authored144 difference -0.013 [-0.033, +0.000], different argmax 9/252
- Telemetry: GPU avg 121.9 W (baseline 15.8 W), peak 180.6 W, net energy 3.596 kJ (DCGM 3.397), util avg 64 %, SM active 0.49, DRAM active 0.15, VRAM max 3027 MiB (baseline 1327), temp max 80 °C, power-cap throttle 11.6 s; CPU 1.00 cores avg (host util 4 %), peak RSS 1.49 GiB

### 61. `cuda-1.7b-q4_k_m-q4_0-T4` — smoke — ok

- 1.7B Q4_K_M · CUDA · KV q4_0 · started 2026-09-28 21:46:04 · wall 2.1 s (estimate 28 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q4_k_m --device cuda --kv-type q4_0 --output results/local-hw/cuda-1.7b-q4_k_m-q4_0-T4/smoke.json`
- accuracy 0.300 (20 rows), accepted 0.750, coverage 0.200, NLL 4.207, Brier 1.224, ECE 0.660
- latency median 22 ms, p95 61 ms, 47.67 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.330020, median 22 / 20 ms
- peak device 1.30 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 0.9 s
- Telemetry: GPU avg 76.0 W (baseline 17.1 W), peak 148.6 W, net energy 0.118 kJ (DCGM —), util avg 30 %, SM active —, DRAM active —, VRAM max 2885 MiB (baseline 1327), temp max 70 °C, power-cap throttle — s; CPU 0.96 cores avg (host util — %), peak RSS 1.49 GiB

### 62. `cuda-1.7b-q4_k_m-q4_0-T7` — semif_compare — ok

- 1.7B Q4_K_M · CUDA · KV q4_0 · started 2026-09-28 21:46:20 · wall 33.8 s (estimate 67 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q4_k_m --device cuda --kv-type q4_0 --output results/local-hw/cuda-1.7b-q4_k_m-q4_0-T7/semif`
- authored144 0.566, perturbations108 0.476, held-out 0.533 / 0.449, macro-F1 0.524
- latency p50 17 ms, p95 19 ms; shape777 shared 48.31 dec/s (37 states, 2111 tok/dec), direct 7.31 dec/s (3 states)
- shared vs direct 15/63 flips (max Δp 0.833); stability flips reversal/wrapper/context 18/3/3; confident on missing evidence 9/36; rule_application perturbed 0.167 (NLL 5.04)
- peak device 1.30 GiB
- vs this-cuda-kv-f16: authored144 difference -0.042 [-0.076, -0.008], different argmax 28/252
- Telemetry: GPU avg 118.8 W (baseline 15.2 W), peak 179.9 W, net energy 3.495 kJ (DCGM 3.282), util avg 63 %, SM active 0.48, DRAM active 0.14, VRAM max 2887 MiB (baseline 1327), temp max 80 °C, power-cap throttle 11.0 s; CPU 1.00 cores avg (host util 4 %), peak RSS 1.49 GiB

### 63. `cuda-1.7b-bf16-q8_0-T4` — smoke — ok

- 1.7B BF16 · CUDA · KV q8_0 · started 2026-09-28 21:47:07 · wall 3.1 s (estimate 30 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant bf16 --device cuda --kv-type q8_0 --output results/local-hw/cuda-1.7b-bf16-q8_0-T4/smoke.json`
- accuracy 0.450 (20 rows), accepted 0.857, coverage 0.350, NLL 3.470, Brier 1.007, ECE 0.512
- latency median 21 ms, p95 72 ms, 46.52 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.012723, median 21 / 20 ms
- peak device 3.68 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.9 s
- Telemetry: GPU avg 46.0 W (baseline 17.2 W), peak 153.7 W, net energy 0.094 kJ (DCGM —), util avg 27 %, SM active —, DRAM active —, VRAM max 5323 MiB (baseline 1327), temp max 68 °C, power-cap throttle — s; CPU 0.97 cores avg (host util — %), peak RSS 3.65 GiB

### 64. `cuda-1.7b-bf16-q8_0-T7` — semif_compare — ok

- 1.7B BF16 · CUDA · KV q8_0 · started 2026-09-28 21:47:23 · wall 35.5 s (estimate 76 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant bf16 --device cuda --kv-type q8_0 --output results/local-hw/cuda-1.7b-bf16-q8_0-T7/semif`
- authored144 0.683, perturbations108 0.640, held-out 0.690 / 0.532, macro-F1 0.658
- latency p50 17 ms, p95 18 ms; shape777 shared 47.83 dec/s (37 states, 2111 tok/dec), direct 6.92 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.067); stability flips reversal/wrapper/context 18/4/5; confident on missing evidence 2/36; rule_application perturbed 0.296 (NLL 3.57)
- peak device 3.68 GiB
- vs this-cuda-kv-f16: authored144 difference 0.000 [+0.000, +0.000], different argmax 1/252
- Telemetry: GPU avg 123.8 W (baseline 16.8 W), peak 180.7 W, net energy 3.803 kJ (DCGM 3.335), util avg 67 %, SM active 0.48, DRAM active 0.24, VRAM max 5325 MiB (baseline 1327), temp max 79 °C, power-cap throttle 14.7 s; CPU 1.00 cores avg (host util 4 %), peak RSS 3.65 GiB

### 65. `cuda-1.7b-bf16-q4_0-T4` — smoke — ok

- 1.7B BF16 · CUDA · KV q4_0 · started 2026-09-28 21:48:12 · wall 3.1 s (estimate 30 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant bf16 --device cuda --kv-type q4_0 --output results/local-hw/cuda-1.7b-bf16-q4_0-T4/smoke.json`
- accuracy 0.350 (20 rows), accepted 0.800, coverage 0.250, NLL 4.232, Brier 1.195, ECE 0.589
- latency median 22 ms, p95 72 ms, 46.75 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.253036, median 22 / 22 ms
- peak device 3.54 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (cuda), load 1.9 s
- Telemetry: GPU avg 40.6 W (baseline 18.4 W), peak 146.3 W, net energy 0.067 kJ (DCGM —), util avg 16 %, SM active —, DRAM active —, VRAM max 5183 MiB (baseline 1327), temp max 67 °C, power-cap throttle — s; CPU 0.98 cores avg (host util — %), peak RSS 3.65 GiB

### 66. `cuda-1.7b-bf16-q4_0-T7` — semif_compare — ok

- 1.7B BF16 · CUDA · KV q4_0 · started 2026-09-28 21:48:28 · wall 35.6 s (estimate 76 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant bf16 --device cuda --kv-type q4_0 --output results/local-hw/cuda-1.7b-bf16-q4_0-T7/semif`
- authored144 0.668, perturbations108 0.602, held-out 0.685 / 0.532, macro-F1 0.634
- latency p50 17 ms, p95 18 ms; shape777 shared 47.77 dec/s (37 states, 2111 tok/dec), direct 6.93 dec/s (3 states)
- shared vs direct 14/63 flips (max Δp 0.500); stability flips reversal/wrapper/context 15/12/8; confident on missing evidence 2/36; rule_application perturbed 0.407 (NLL 3.65)
- peak device 3.54 GiB
- vs this-cuda-kv-f16: authored144 difference -0.015 [-0.062, +0.032], different argmax 41/252
- Telemetry: GPU avg 121.2 W (baseline 16.8 W), peak 180.8 W, net energy 3.720 kJ (DCGM 3.373), util avg 65 %, SM active 0.49, DRAM active 0.24, VRAM max 5185 MiB (baseline 1327), temp max 79 °C, power-cap throttle 15.5 s; CPU 0.99 cores avg (host util 4 %), peak RSS 3.65 GiB

### 67. `vulkan-4b-q8_0-f16-T4` — smoke — ok

- 4B Q8_0 · Vulkan · started 2026-09-28 21:49:17 · wall 12.9 s (estimate 53 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --device vulkan --output results/local-hw/vulkan-4b-q8_0-f16-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.458, Brier 0.079, ECE 0.040
- latency median 68 ms, p95 3252 ms, 4.14 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000006, median 68 / 66 ms
- peak device 5.65 GiB, device `NVIDIA RTX PRO 4000 Blackwell` (vulkan), load 2.5 s
- Telemetry: GPU avg 49.8 W (baseline 17.4 W), peak 159.4 W, net energy 0.419 kJ (DCGM 0.398), util avg 17 %, SM active 0.13, DRAM active 0.04, VRAM max 7107 MiB (baseline 1327), temp max 68 °C, power-cap throttle 0.4 s; CPU 0.84 cores avg (host util 4 %), peak RSS 4.28 GiB

### 68. `vulkan-4b-q8_0-f16-T7` — semif_compare — ok

- 4B Q8_0 · Vulkan · started 2026-09-28 21:49:43 · wall 361.4 s (estimate 589 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device vulkan --output results/local-hw/vulkan-4b-q8_0-f16-T7/semif --direct-states 37`
- authored144 0.816, perturbations108 0.854, held-out 0.799 / 0.861, macro-F1 0.811
- latency p50 66 ms, p95 70 ms; shape777 shared 23.10 dec/s (37 states, 2111 tok/dec), direct 2.60 dec/s (37 states)
- shared vs direct 7/777 flips (max Δp 0.080); stability flips reversal/wrapper/context 4/3/5; confident on missing evidence 6/36; rule_application perturbed 0.630 (NLL 1.66)
- peak device 6.08 GiB
- vs maintainers: authored144 difference 0.009 [+0.000, +0.028], different argmax 1/252
- vs this-cuda: authored144 difference 0.006 [+0.000, +0.020], different argmax 4/252
- Telemetry: GPU avg 141.1 W (baseline 15.7 W), peak 180.2 W, net energy 45.314 kJ (DCGM 44.357), util avg 91 %, SM active 0.68, DRAM active 0.23, VRAM max 7552 MiB (baseline 1327), temp max 87 °C, power-cap throttle 230.7 s; CPU 0.70 cores avg (host util 3 %), peak RSS 4.28 GiB

### 69. `cuda-4b-q8_0-f16-T7-batch1` — semif_compare — ok

- 4B Q8_0 · CUDA · batch 1 · started 2026-09-28 21:55:58 · wall 43.7 s (estimate 73 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T7-batch1/semif --batch-size 1 --direct-states 0`
- authored144 0.810, perturbations108 0.830, held-out 0.785 / 0.843, macro-F1 0.804
- latency p50 35 ms, p95 38 ms; shape777 shared 25.11 dec/s (37 states, 2111 tok/dec), direct — dec/s (None states)
- shared vs direct None/None flips (max Δp —); stability flips reversal/wrapper/context 6/5/3; confident on missing evidence 6/36; rule_application perturbed 0.556 (NLL 1.69)
- peak device 5.61 GiB
- Telemetry: GPU avg 122.5 W (baseline 18.8 W), peak 178.7 W, net energy 4.529 kJ (DCGM 4.399), util avg 71 %, SM active 0.55, DRAM active 0.28, VRAM max 7303 MiB (baseline 1327), temp max 83 °C, power-cap throttle 19.5 s; CPU 1.00 cores avg (host util 4 %), peak RSS 4.54 GiB

### 70. `cuda-4b-q8_0-f16-T7-batch2` — semif_compare — ok

- 4B Q8_0 · CUDA · batch 2 · started 2026-09-28 21:56:55 · wall 40.8 s (estimate 73 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T7-batch2/semif --batch-size 2 --direct-states 0`
- authored144 0.810, perturbations108 0.830, held-out 0.785 / 0.843, macro-F1 0.804
- latency p50 35 ms, p95 38 ms; shape777 shared 27.70 dec/s (37 states, 2111 tok/dec), direct — dec/s (None states)
- shared vs direct None/None flips (max Δp —); stability flips reversal/wrapper/context 6/5/3; confident on missing evidence 6/36; rule_application perturbed 0.556 (NLL 1.69)
- peak device 5.61 GiB
- Telemetry: GPU avg 122.7 W (baseline 18.9 W), peak 175.2 W, net energy 4.231 kJ (DCGM 3.951), util avg 70 %, SM active 0.54, DRAM active 0.24, VRAM max 7303 MiB (baseline 1327), temp max 82 °C, power-cap throttle 22.7 s; CPU 1.00 cores avg (host util 4 %), peak RSS 4.54 GiB

### 71. `cuda-4b-q8_0-f16-T7-batch4` — semif_compare — ok

- 4B Q8_0 · CUDA · batch 4 · started 2026-09-28 21:57:49 · wall 38.2 s (estimate 73 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T7-batch4/semif --batch-size 4 --direct-states 0`
- authored144 0.810, perturbations108 0.830, held-out 0.785 / 0.843, macro-F1 0.804
- latency p50 35 ms, p95 37 ms; shape777 shared 30.26 dec/s (37 states, 2111 tok/dec), direct — dec/s (None states)
- shared vs direct None/None flips (max Δp —); stability flips reversal/wrapper/context 6/5/3; confident on missing evidence 6/36; rule_application perturbed 0.556 (NLL 1.69)
- peak device 5.61 GiB
- Telemetry: GPU avg 120.3 W (baseline 19.4 W), peak 180.0 W, net energy 3.852 kJ (DCGM 3.770), util avg 68 %, SM active 0.54, DRAM active 0.22, VRAM max 7303 MiB (baseline 1327), temp max 81 °C, power-cap throttle 21.6 s; CPU 1.00 cores avg (host util 4 %), peak RSS 4.53 GiB

### 72. `cuda-4b-q8_0-f16-T7-batch8` — semif_compare — ok

- 4B Q8_0 · CUDA · batch 8 · started 2026-09-28 21:58:40 · wall 38.0 s (estimate 73 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T7-batch8/semif --batch-size 8 --direct-states 0`
- authored144 0.810, perturbations108 0.830, held-out 0.785 / 0.843, macro-F1 0.804
- latency p50 35 ms, p95 38 ms; shape777 shared 30.62 dec/s (37 states, 2111 tok/dec), direct — dec/s (None states)
- shared vs direct None/None flips (max Δp —); stability flips reversal/wrapper/context 6/5/3; confident on missing evidence 6/36; rule_application perturbed 0.556 (NLL 1.69)
- peak device 5.61 GiB
- Telemetry: GPU avg 120.3 W (baseline 19.4 W), peak 180.7 W, net energy 3.831 kJ (DCGM 3.697), util avg 68 %, SM active 0.55, DRAM active 0.23, VRAM max 7303 MiB (baseline 1327), temp max 81 °C, power-cap throttle 20.2 s; CPU 1.00 cores avg (host util 4 %), peak RSS 4.54 GiB

### 73. `cuda-4b-q8_0-f16-T7-batch16` — semif_compare — ok

- 4B Q8_0 · CUDA · batch 16 · started 2026-09-28 21:59:31 · wall 37.7 s (estimate 73 s) · exit [0]
- `python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cuda --output results/local-hw/cuda-4b-q8_0-f16-T7-batch16/semif --batch-size 16 --direct-states 0`
- authored144 0.810, perturbations108 0.830, held-out 0.785 / 0.843, macro-F1 0.804
- latency p50 35 ms, p95 37 ms; shape777 shared 30.90 dec/s (37 states, 2111 tok/dec), direct — dec/s (None states)
- shared vs direct None/None flips (max Δp —); stability flips reversal/wrapper/context 6/5/3; confident on missing evidence 6/36; rule_application perturbed 0.556 (NLL 1.69)
- peak device 5.61 GiB
- Telemetry: GPU avg 121.1 W (baseline 18.7 W), peak 181.2 W, net energy 3.859 kJ (DCGM 3.665), util avg 67 %, SM active 0.54, DRAM active 0.22, VRAM max 7305 MiB (baseline 1327), temp max 81 °C, power-cap throttle 20.0 s; CPU 1.00 cores avg (host util 4 %), peak RSS 4.54 GiB

### 74. `cpu-4b-q8_0-f16-T4-threadsdefault` — smoke — ok

- 4B Q8_0 · CPU · threads default (4) · started 2026-09-28 22:00:22 · wall 123.0 s (estimate 73 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --device cpu --output results/local-hw/cpu-4b-q8_0-f16-T4-threadsdefault/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.429, Brier 0.077, ECE 0.037
- latency median 2542 ms, p95 7807 ms, 0.36 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000012, median 2542 / 2543 ms
- peak device —, device `None` (cpu), load 2.0 s
- Telemetry: GPU avg 7.8 W (baseline 18.1 W), peak 12.3 W, net energy -1.266 kJ (DCGM -1.393), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 62 °C, power-cap throttle 0.0 s; CPU 3.90 cores avg (host util 13 %), peak RSS 5.66 GiB

### 75. `cpu-4b-q8_0-f16-T4-threads16` — smoke — ok

- 4B Q8_0 · CPU · threads 16 · 16 threads · started 2026-09-28 22:02:38 · wall 41.4 s (estimate 73 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-4b-q8_0-f16-T4-threads16/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.429, Brier 0.077, ECE 0.037
- latency median 838 ms, p95 2454 ms, 1.10 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000012, median 838 / 930 ms
- peak device —, device `None` (cpu), load 2.1 s
- Telemetry: GPU avg 6.6 W (baseline 6.4 W), peak 7.5 W, net energy 0.008 kJ (DCGM 0.020), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 43 °C, power-cap throttle 0.0 s; CPU 14.47 cores avg (host util 48 %), peak RSS 5.66 GiB

### 76. `cpu-4b-q8_0-f16-T4-threads32` — smoke — ok

- 4B Q8_0 · CPU · threads 32 · 32 threads · started 2026-09-28 22:03:32 · wall 51.1 s (estimate 73 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --device cpu --threads 32 --output results/local-hw/cpu-4b-q8_0-f16-T4-threads32/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.429, Brier 0.077, ECE 0.037
- latency median 1308 ms, p95 3752 ms, 0.86 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000012, median 1308 / 1242 ms
- peak device —, device `None` (cpu), load 2.1 s
- Telemetry: GPU avg 6.1 W (baseline 5.9 W), peak 6.7 W, net energy 0.015 kJ (DCGM 0.032), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 41 °C, power-cap throttle 0.0 s; CPU 28.56 cores avg (host util 92 %), peak RSS 5.66 GiB

### 77. `cpu-4b-q8_0-f16-T3` — rizzo decide examples — ok

- 4B Q8_0 · CPU · 16 threads · started 2026-09-28 22:04:37 · wall 12.5 s (estimate 73 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 4b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-4b-q8_0-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 4b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-4b-q8_0-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 4b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-4b-q8_0-f16-T3/decide-numeric.json`
- ticket: needs_access_support=True (ok, p=1.000); queue=access (ok, p=1.000); urgency=1.9999318522082647 (ok, p=1.000); deadline_hours=None (insufficient_evidence, p=1.000) — total 2441 ms, inference 2425 ms, 1287 tokens, load 2.1 s
- house: estimated_price=None (insufficient_evidence, p=0.998) — total 1860 ms, inference 1854 ms, 562 tokens, load 2.0 s
- numeric: fill=74.99990119850396 (ok, p=1.000) — total 1537 ms, inference 1529 ms, 398 tokens, load 2.1 s
- Telemetry: GPU avg 6.0 W (baseline 6.0 W), peak 6.7 W, net energy 0.000 kJ (DCGM -0.001), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 39 °C, power-cap throttle 0.0 s; CPU 7.78 cores avg (host util 30 %), peak RSS 5.66 GiB

### 78. `cpu-4b-q8_0-f16-T4` — smoke — ok

- 4B Q8_0 · CPU · 16 threads · started 2026-09-28 22:05:02 · wall 41.6 s (estimate 73 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-4b-q8_0-f16-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.429, Brier 0.077, ECE 0.037
- latency median 922 ms, p95 2580 ms, 1.07 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000012, median 922 / 879 ms
- peak device —, device `None` (cpu), load 2.1 s
- Telemetry: GPU avg 6.0 W (baseline 5.8 W), peak 7.1 W, net energy 0.008 kJ (DCGM 0.022), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 39 °C, power-cap throttle 0.0 s; CPU 14.47 cores avg (host util 48 %), peak RSS 5.66 GiB

### 79. `cpu-4b-q8_0-f16-T5` — perturbations — ok

- 4B Q8_0 · CPU · 16 threads · started 2026-09-28 22:05:57 · wall 18.2 s (estimate 43 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 4b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-4b-q8_0-f16-T5/perturbations.json`
- accuracy 1 (9 rows), accepted 1, coverage 0.889, NLL 0.013, Brier 0.001, ECE 0.013
- latency median 829 ms, p95 944 ms, 1.20 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 829 / 842 ms
- peak device —, device `None` (cpu), load 2.1 s
- Telemetry: GPU avg 6.1 W (baseline 5.8 W), peak 7.8 W, net energy 0.007 kJ (DCGM 0.005), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 13.33 cores avg (host util 49 %), peak RSS 5.63 GiB

### 80. `cpu-4b-q8_0-f16-T6` — validate_checkpoint — ok

- 4B Q8_0 · CPU · 16 threads · started 2026-09-28 22:06:28 · wall 108.3 s (estimate 133 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/validate_checkpoint.py --size 4b --quant q8_0 --device cpu --output results/local-hw/cpu-4b-q8_0-f16-T6/validate`
- smoke accuracy 0.950 (NLL 0.429, 0 changed), perturbations 1
- long state: shared 7765 ms vs direct 24916 ms, 0 changed (max Δp 0.008573)
- validation 103.5 s, load 2.1 s, peak —
- Telemetry: GPU avg 6.2 W (baseline 5.7 W), peak 16.6 W, net energy 0.061 kJ (DCGM 0.038), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.17 cores avg (host util 49 %), peak RSS 5.67 GiB

### 81. `cpu-4b-q8_0-f16-T7` — semif_compare — ok

- 4B Q8_0 · CPU · 16 threads · started 2026-09-28 22:08:30 · wall 1292.3 s (estimate 1153 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cpu --output results/local-hw/cpu-4b-q8_0-f16-T7/semif`
- authored144 0.816, perturbations108 0.873, held-out 0.799 / 0.880, macro-F1 0.811
- latency p50 727 ms, p95 818 ms; shape777 shared 1.68 dec/s (37 states, 2111 tok/dec), direct 0.14 dec/s (3 states)
- shared vs direct 1/63 flips (max Δp 0.166); stability flips reversal/wrapper/context 3/3/5; confident on missing evidence 6/36; rule_application perturbed 0.685 (NLL 1.65)
- peak device —
- vs this-cuda: authored144 difference 0.006 [+0.000, +0.020], different argmax 5/252
- Telemetry: GPU avg 6.2 W (baseline 6.6 W), peak 17.9 W, net energy -0.454 kJ (DCGM 0.656), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.63 cores avg (host util 50 %), peak RSS 5.67 GiB

### 82. `cpu-4b-q8_0-f16-T9` — typed-decisions — ok

- 4B Q8_0 · CPU · 16 threads · started 2026-09-28 22:30:15 · wall 1194.8 s (estimate 1213 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-4B-GGUF/Spark-X2.5-4B-Q8_0.gguf --device cpu --output results/local-hw/cpu-4b-q8_0-f16-T9/typed.json`
- accuracy 0.576, KL 2.902, Brier 0.480, ECE 0.348, macro-F1 0.414
- p50 3008 ms/case, p95 3711 ms, 1.68 dec/s
- by workflow: agent_trace_observability 0.370, customer_service 0.642, invoice_processing 0.660, security_incidents 0.632
- Telemetry: GPU avg 6.2 W (baseline 5.6 W), peak 17.2 W, net energy 0.743 kJ (DCGM 0.815), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.60 cores avg (host util 50 %), peak RSS 5.66 GiB

### 83. `cpu-4b-bf16-f16-T3` — rizzo decide examples — ok

- 4B BF16 · CPU · 16 threads · started 2026-09-28 22:50:23 · wall 16.6 s (estimate 97 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 4b --quant bf16 --device cpu --threads 16 --output results/local-hw/cpu-4b-bf16-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 4b --quant bf16 --device cpu --threads 16 --output results/local-hw/cpu-4b-bf16-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 4b --quant bf16 --device cpu --threads 16 --output results/local-hw/cpu-4b-bf16-f16-T3/decide-numeric.json`
- ticket: needs_access_support=True (ok, p=1.000); queue=access (ok, p=1.000); urgency=1.9999307522271041 (ok, p=1.000); deadline_hours=None (insufficient_evidence, p=1.000) — total 2453 ms, inference 2438 ms, 1287 tokens, load 3.5 s
- house: estimated_price=None (insufficient_evidence, p=0.998) — total 1950 ms, inference 1944 ms, 562 tokens, load 3.5 s
- numeric: fill=74.99989759156095 (ok, p=1.000) — total 1332 ms, inference 1324 ms, 398 tokens, load 3.5 s
- Telemetry: GPU avg 6.0 W (baseline 5.8 W), peak 6.8 W, net energy 0.002 kJ (DCGM -0.003), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 6.03 cores avg (host util 21 %), peak RSS 9.25 GiB

### 84. `cpu-4b-bf16-f16-T4` — smoke — ok

- 4B BF16 · CPU · 16 threads · started 2026-09-28 22:50:53 · wall 41.8 s (estimate 97 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant bf16 --device cpu --threads 16 --output results/local-hw/cpu-4b-bf16-f16-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.445, Brier 0.078, ECE 0.041
- latency median 825 ms, p95 2432 ms, 1.12 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000004, median 825 / 812 ms
- peak device —, device `None` (cpu), load 3.5 s
- Telemetry: GPU avg 6.3 W (baseline 5.9 W), peak 14.2 W, net energy 0.018 kJ (DCGM 0.003), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 13.99 cores avg (host util 48 %), peak RSS 9.25 GiB

### 85. `cpu-4b-bf16-f16-T5` — perturbations — ok

- 4B BF16 · CPU · 16 threads · started 2026-09-28 22:51:48 · wall 19.4 s (estimate 55 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 4b --quant bf16 --device cpu --threads 16 --output results/local-hw/cpu-4b-bf16-f16-T5/perturbations.json`
- accuracy 1 (9 rows), accepted 1, coverage 0.889, NLL 0.017, Brier 0.002, ECE 0.017
- latency median 822 ms, p95 885 ms, 1.21 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 822 / 833 ms
- peak device —, device `None` (cpu), load 3.5 s
- Telemetry: GPU avg 7.0 W (baseline 6.2 W), peak 14.9 W, net energy 0.014 kJ (DCGM 0.014), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 12.45 cores avg (host util 43 %), peak RSS 9.22 GiB

### 86. `cpu-4b-bf16-f16-T6` — validate_checkpoint — ok

- 4B BF16 · CPU · 16 threads · started 2026-09-28 22:52:20 · wall 105.9 s (estimate 181 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/validate_checkpoint.py --size 4b --quant bf16 --device cpu --output results/local-hw/cpu-4b-bf16-f16-T6/validate`
- smoke accuracy 0.950 (NLL 0.445, 0 changed), perturbations 1
- long state: shared 7418 ms vs direct 23880 ms, 0 changed (max Δp 0.003072)
- validation 99.8 s, load 3.5 s, peak —
- Telemetry: GPU avg 6.2 W (baseline 6.4 W), peak 14.0 W, net energy -0.016 kJ (DCGM 0.047), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.97 cores avg (host util 50 %), peak RSS 9.27 GiB

### 87. `cpu-4b-bf16-f16-T7` — semif_compare — ok

- 4B BF16 · CPU · 16 threads · started 2026-09-28 22:54:19 · wall 1248.0 s (estimate 1609 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant bf16 --device cpu --output results/local-hw/cpu-4b-bf16-f16-T7/semif`
- authored144 0.829, perturbations108 0.842, held-out 0.824 / 0.843, macro-F1 0.825
- latency p50 718 ms, p95 775 ms; shape777 shared 1.73 dec/s (37 states, 2111 tok/dec), direct 0.14 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.039); stability flips reversal/wrapper/context 5/3/4; confident on missing evidence 6/36; rule_application perturbed 0.593 (NLL 1.69)
- peak device —
- vs this-cuda: authored144 difference 0.009 [+0.000, +0.028], different argmax 2/252
- Telemetry: GPU avg 6.4 W (baseline 6.4 W), peak 20.2 W, net energy 0.077 kJ (DCGM 0.649), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.62 cores avg (host util 51 %), peak RSS 9.27 GiB

### 88. `cpu-4b-bf16-f16-T9` — typed-decisions — ok

- 4B BF16 · CPU · 16 threads · started 2026-09-28 23:15:21 · wall 1158.2 s (estimate 1693 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-4B-GGUF/Spark-X2.5-4B.gguf --device cpu --output results/local-hw/cpu-4b-bf16-f16-T9/typed.json`
- accuracy 0.575, KL 2.935, Brier 0.479, ECE 0.348, macro-F1 0.413
- p50 2904 ms/case, p95 3616 ms, 1.74 dec/s
- by workflow: agent_trace_observability 0.370, customer_service 0.630, invoice_processing 0.664, security_incidents 0.636
- Telemetry: GPU avg 6.4 W (baseline 6.2 W), peak 18.2 W, net energy 0.213 kJ (DCGM 0.241), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.60 cores avg (host util 51 %), peak RSS 9.26 GiB

### 89. `cpu-4b-q4_k_m-f16-T3` — rizzo decide examples — ok

- 4B Q4_K_M · CPU · 16 threads · started 2026-09-28 23:34:52 · wall 10.8 s (estimate 82 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 4b --quant q4_k_m --device cpu --threads 16 --output results/local-hw/cpu-4b-q4_k_m-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 4b --quant q4_k_m --device cpu --threads 16 --output results/local-hw/cpu-4b-q4_k_m-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 4b --quant q4_k_m --device cpu --threads 16 --output results/local-hw/cpu-4b-q4_k_m-f16-T3/decide-numeric.json`
- ticket: needs_access_support=True (ok, p=0.942); queue=access (ok, p=1.000); urgency=1.9998857226744353 (ok, p=1.000); deadline_hours=None (insufficient_evidence, p=1.000) — total 2176 ms, inference 2161 ms, 1287 tokens, load 1.8 s
- house: estimated_price=None (insufficient_evidence, p=0.996) — total 1750 ms, inference 1744 ms, 562 tokens, load 1.8 s
- numeric: fill=74.99981008560538 (ok, p=1.000) — total 1220 ms, inference 1213 ms, 398 tokens, load 1.7 s
- Telemetry: GPU avg 6.5 W (baseline 7.4 W), peak 6.9 W, net energy -0.010 kJ (DCGM 0.003), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 7.93 cores avg (host util 27 %), peak RSS 5.57 GiB

### 90. `cpu-4b-q4_k_m-f16-T4` — smoke — ok

- 4B Q4_K_M · CPU · 16 threads · started 2026-09-28 23:35:16 · wall 36.9 s (estimate 82 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q4_k_m --device cpu --threads 16 --output results/local-hw/cpu-4b-q4_k_m-f16-T4/smoke.json`
- accuracy 0.900 (20 rows), accepted 1, coverage 0.750, NLL 0.730, Brier 0.188, ECE 0.057
- latency median 746 ms, p95 2227 ms, 1.23 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.012698, median 746 / 773 ms
- peak device —, device `None` (cpu), load 1.8 s
- Telemetry: GPU avg 7.5 W (baseline 6.5 W), peak 17.4 W, net energy 0.040 kJ (DCGM 0.022), util avg 2 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.42 cores avg (host util 48 %), peak RSS 5.57 GiB

### 91. `cpu-4b-q4_k_m-f16-T5` — perturbations — ok

- 4B Q4_K_M · CPU · 16 threads · started 2026-09-28 23:36:06 · wall 16.2 s (estimate 48 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 4b --quant q4_k_m --device cpu --threads 16 --output results/local-hw/cpu-4b-q4_k_m-f16-T5/perturbations.json`
- accuracy 1 (9 rows), accepted 1, coverage 0.889, NLL 0.010, Brier 0.000, ECE 0.009
- latency median 772 ms, p95 807 ms, 1.32 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 772 / 755 ms
- peak device —, device `None` (cpu), load 1.7 s
- Telemetry: GPU avg 6.6 W (baseline 6.0 W), peak 15.7 W, net energy 0.010 kJ (DCGM 0.004), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 13.34 cores avg (host util 46 %), peak RSS 5.55 GiB

### 92. `cpu-4b-q4_k_m-f16-T6` — validate_checkpoint — ok

- 4B Q4_K_M · CPU · 16 threads · started 2026-09-28 23:36:35 · wall 96.7 s (estimate 151 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/validate_checkpoint.py --size 4b --quant q4_k_m --device cpu --output results/local-hw/cpu-4b-q4_k_m-f16-T6/validate`
- smoke accuracy 0.900 (NLL 0.730, 0 changed), perturbations 1
- long state: shared 6901 ms vs direct 22261 ms, 0 changed (max Δp 0.104480)
- validation 92.5 s, load 1.8 s, peak —
- Telemetry: GPU avg 6.3 W (baseline 5.8 W), peak 16.0 W, net energy 0.052 kJ (DCGM 0.063), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.14 cores avg (host util 50 %), peak RSS 5.59 GiB

### 93. `cpu-4b-q4_k_m-f16-T7` — semif_compare — ok

- 4B Q4_K_M · CPU · 16 threads · started 2026-09-28 23:38:25 · wall 1155.7 s (estimate 1324 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q4_k_m --device cpu --output results/local-hw/cpu-4b-q4_k_m-f16-T7/semif`
- authored144 0.765, perturbations108 0.841, held-out 0.724 / 0.819, macro-F1 0.756
- latency p50 652 ms, p95 709 ms; shape777 shared 1.86 dec/s (37 states, 2111 tok/dec), direct 0.15 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.153); stability flips reversal/wrapper/context 6/2/2; confident on missing evidence 6/36; rule_application perturbed 0.611 (NLL 1.82)
- peak device —
- vs this-cuda: authored144 difference 0.000 [+0.000, +0.000], different argmax 1/252
- Telemetry: GPU avg 6.0 W (baseline 5.8 W), peak 15.7 W, net energy 0.320 kJ (DCGM 0.170), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.61 cores avg (host util 50 %), peak RSS 5.59 GiB

### 94. `cpu-4b-q4_k_m-f16-T9` — typed-decisions — ok

- 4B Q4_K_M · CPU · 16 threads · started 2026-09-28 23:57:54 · wall 1065.1 s (estimate 1393 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-4B-GGUF/Spark-X2.5-4B-Q4_K_M.gguf --device cpu --output results/local-hw/cpu-4b-q4_k_m-f16-T9/typed.json`
- accuracy 0.576, KL 2.331, Brier 0.444, ECE 0.323, macro-F1 0.415
- p50 2683 ms/case, p95 3291 ms, 1.88 dec/s
- by workflow: agent_trace_observability 0.420, customer_service 0.630, invoice_processing 0.658, security_incidents 0.594
- Telemetry: GPU avg 6.3 W (baseline 5.7 W), peak 16.8 W, net energy 0.624 kJ (DCGM 0.174), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.59 cores avg (host util 50 %), peak RSS 5.57 GiB

### 95. `cpu-1.7b-q8_0-f16-T3` — rizzo decide examples — ok

- 1.7B Q8_0 · CPU · 16 threads · started 2026-09-29 00:15:53 · wall 5.2 s (estimate 43 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 1.7b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-1.7b-q8_0-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 1.7b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-1.7b-q8_0-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 1.7b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-1.7b-q8_0-f16-T3/decide-numeric.json`
- ticket: needs_access_support=False (ok, p=0.966); queue=None (insufficient_evidence, p=0.914); urgency=None (insufficient_evidence, p=0.899); deadline_hours=None (insufficient_evidence, p=1.000) — total 943 ms, inference 928 ms, 1287 tokens, load 0.9 s
- house: estimated_price=None (insufficient_evidence, p=1.000) — total 760 ms, inference 754 ms, 562 tokens, load 0.9 s
- numeric: fill=74.94451500387777 (ok, p=0.685) — total 521 ms, inference 513 ms, 398 tokens, load 0.9 s
- Telemetry: GPU avg 6.5 W (baseline 6.3 W), peak 6.9 W, net energy 0.001 kJ (DCGM -0.002), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 7.18 cores avg (host util 23 %), peak RSS 2.39 GiB

### 96. `cpu-1.7b-q8_0-f16-T4` — smoke — ok

- 1.7B Q8_0 · CPU · 16 threads · started 2026-09-29 00:16:11 · wall 17.1 s (estimate 43 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-1.7b-q8_0-f16-T4/smoke.json`
- accuracy 0.400 (20 rows), accepted 0.833, coverage 0.300, NLL 3.699, Brier 1.076, ECE 0.526
- latency median 358 ms, p95 1000 ms, 2.64 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.059870, median 358 / 413 ms
- peak device —, device `None` (cpu), load 0.9 s
- Telemetry: GPU avg 6.0 W (baseline 5.8 W), peak 6.6 W, net energy 0.003 kJ (DCGM 0.010), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.41 cores avg (host util 47 %), peak RSS 2.39 GiB

### 97. `cpu-1.7b-q8_0-f16-T5` — perturbations — ok

- 1.7B Q8_0 · CPU · 16 threads · started 2026-09-29 00:16:41 · wall 7.5 s (estimate 28 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 1.7b --quant q8_0 --device cpu --threads 16 --output results/local-hw/cpu-1.7b-q8_0-f16-T5/perturbations.json`
- accuracy 0.556 (9 rows), accepted 1, coverage 0.444, NLL 3.113, Brier 0.935, ECE 0.535
- latency median 351 ms, p95 395 ms, 2.85 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 351 / 333 ms
- peak device —, device `None` (cpu), load 0.9 s
- Telemetry: GPU avg 6.3 W (baseline 5.8 W), peak 6.7 W, net energy 0.003 kJ (DCGM 0.008), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 13.22 cores avg (host util 48 %), peak RSS 2.38 GiB

### 98. `cpu-1.7b-q8_0-f16-T6` — validate_checkpoint — ok

- 1.7B Q8_0 · CPU · 16 threads · started 2026-09-29 00:17:02 · wall 43.8 s (estimate 73 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/validate_checkpoint.py --size 1.7b --quant q8_0 --device cpu --output results/local-hw/cpu-1.7b-q8_0-f16-T6/validate`
- smoke accuracy 0.400 (NLL 3.699, 0 changed), perturbations 0.556
- long state: shared 3095 ms vs direct 9829 ms, 0 changed (max Δp 0.000076)
- validation 41.8 s, load 0.9 s, peak —
- Telemetry: GPU avg 6.9 W (baseline 5.8 W), peak 8.8 W, net energy 0.048 kJ (DCGM 0.024), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.08 cores avg (host util 50 %), peak RSS 2.41 GiB

### 99. `cpu-1.7b-q8_0-f16-T7` — semif_compare — ok

- 1.7B Q8_0 · CPU · 16 threads · started 2026-09-29 00:17:59 · wall 516.0 s (estimate 583 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q8_0 --device cpu --output results/local-hw/cpu-1.7b-q8_0-f16-T7/semif`
- authored144 0.713, perturbations108 0.664, held-out 0.708 / 0.551, macro-F1 0.684
- latency p50 293 ms, p95 341 ms; shape777 shared 4.13 dec/s (37 states, 2111 tok/dec), direct 0.35 dec/s (3 states)
- shared vs direct 1/63 flips (max Δp 0.154); stability flips reversal/wrapper/context 17/4/6; confident on missing evidence 3/36; rule_application perturbed 0.370 (NLL 3.41)
- peak device —
- vs this-cuda: authored144 difference 0.013 [-0.010, +0.041], different argmax 7/252
- Telemetry: GPU avg 6.5 W (baseline 6.4 W), peak 11.8 W, net energy 0.062 kJ (DCGM 0.352), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.48 cores avg (host util 50 %), peak RSS 2.41 GiB

### 100. `cpu-1.7b-q8_0-f16-T9` — typed-decisions — ok

- 1.7B Q8_0 · CPU · 16 threads · started 2026-09-29 00:26:48 · wall 476.9 s (estimate 613 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-1.7B-GGUF/Spark-X2.5-1.7B-Q8_0.gguf --device cpu --output results/local-hw/cpu-1.7b-q8_0-f16-T9/typed.json`
- accuracy 0.528, KL 2.997, Brier 0.495, ECE 0.347, macro-F1 0.331
- p50 1199 ms/case, p95 1488 ms, 4.21 dec/s
- by workflow: agent_trace_observability 0.368, customer_service 0.648, invoice_processing 0.484, security_incidents 0.614
- Telemetry: GPU avg 6.6 W (baseline 5.9 W), peak 16.7 W, net energy 0.316 kJ (DCGM 0.343), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.49 cores avg (host util 50 %), peak RSS 2.40 GiB

### 101. `cpu-1.7b-q4_k_m-f16-T3` — rizzo decide examples — ok

- 1.7B Q4_K_M · CPU · 16 threads · started 2026-09-29 00:34:58 · wall 4.7 s (estimate 46 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 1.7b --quant q4_k_m --device cpu --threads 16 --output results/local-hw/cpu-1.7b-q4_k_m-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 1.7b --quant q4_k_m --device cpu --threads 16 --output results/local-hw/cpu-1.7b-q4_k_m-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 1.7b --quant q4_k_m --device cpu --threads 16 --output results/local-hw/cpu-1.7b-q4_k_m-f16-T3/decide-numeric.json`
- ticket: needs_access_support=False (ok, p=0.600); queue=None (insufficient_evidence, p=0.768); urgency=None (insufficient_evidence, p=0.948); deadline_hours=None (insufficient_evidence, p=1.000) — total 880 ms, inference 861 ms, 1287 tokens, load 0.8 s
- house: estimated_price=None (insufficient_evidence, p=1.000) — total 675 ms, inference 668 ms, 562 tokens, load 0.8 s
- numeric: fill=None (insufficient_evidence, p=0.999) — total 484 ms, inference 477 ms, 398 tokens, load 0.8 s
- Telemetry: GPU avg 5.8 W (baseline 6.1 W), peak 6.3 W, net energy -0.002 kJ (DCGM —), util avg 0 %, SM active —, DRAM active —, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle — s; CPU 7.18 cores avg (host util — %), peak RSS 2.33 GiB

### 102. `cpu-1.7b-q4_k_m-f16-T4` — smoke — ok

- 1.7B Q4_K_M · CPU · 16 threads · started 2026-09-29 00:35:16 · wall 14.9 s (estimate 46 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q4_k_m --device cpu --threads 16 --output results/local-hw/cpu-1.7b-q4_k_m-f16-T4/smoke.json`
- accuracy 0.350 (20 rows), accepted 0.800, coverage 0.250, NLL 3.368, Brier 1.085, ECE 0.549
- latency median 324 ms, p95 901 ms, 3.06 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.102410, median 324 / 318 ms
- peak device —, device `None` (cpu), load 0.8 s
- Telemetry: GPU avg 6.5 W (baseline 5.8 W), peak 15.1 W, net energy 0.011 kJ (DCGM 0.010), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.24 cores avg (host util 49 %), peak RSS 2.33 GiB

### 103. `cpu-1.7b-q4_k_m-f16-T5` — perturbations — ok

- 1.7B Q4_K_M · CPU · 16 threads · started 2026-09-29 00:35:44 · wall 6.7 s (estimate 30 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 1.7b --quant q4_k_m --device cpu --threads 16 --output results/local-hw/cpu-1.7b-q4_k_m-f16-T5/perturbations.json`
- accuracy 0.556 (9 rows), accepted 1, coverage 0.444, NLL 2.294, Brier 0.840, ECE 0.466
- latency median 306 ms, p95 321 ms, 3.31 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 306 / 305 ms
- peak device —, device `None` (cpu), load 0.8 s
- Telemetry: GPU avg 7.0 W (baseline 5.7 W), peak 16.1 W, net energy 0.009 kJ (DCGM 0.007), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 13.04 cores avg (host util 48 %), peak RSS 2.32 GiB

### 104. `cpu-1.7b-q4_k_m-f16-T6` — validate_checkpoint — ok

- 1.7B Q4_K_M · CPU · 16 threads · started 2026-09-29 00:36:04 · wall 38.7 s (estimate 79 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/validate_checkpoint.py --size 1.7b --quant q4_k_m --device cpu --output results/local-hw/cpu-1.7b-q4_k_m-f16-T6/validate`
- smoke accuracy 0.350 (NLL 3.368, 0 changed), perturbations 0.556
- long state: shared 2728 ms vs direct 8734 ms, 0 changed (max Δp 0.001874)
- validation 36.8 s, load 0.8 s, peak —
- Telemetry: GPU avg 6.3 W (baseline 6.1 W), peak 14.2 W, net energy 0.007 kJ (DCGM 0.017), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.99 cores avg (host util 50 %), peak RSS 2.35 GiB

### 105. `cpu-1.7b-q4_k_m-f16-T7` — semif_compare — ok

- 1.7B Q4_K_M · CPU · 16 threads · started 2026-09-29 00:36:56 · wall 462.3 s (estimate 640 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q4_k_m --device cpu --output results/local-hw/cpu-1.7b-q4_k_m-f16-T7/semif`
- authored144 0.585, perturbations108 0.471, held-out 0.582 / 0.472, macro-F1 0.550
- latency p50 261 ms, p95 282 ms; shape777 shared 4.56 dec/s (37 states, 2111 tok/dec), direct 0.39 dec/s (3 states)
- shared vs direct 9/63 flips (max Δp 0.207); stability flips reversal/wrapper/context 18/4/2; confident on missing evidence 4/36; rule_application perturbed 0.074 (NLL 4.10)
- peak device —
- vs this-cuda: authored144 difference -0.022 [-0.045, -0.005], different argmax 9/252
- Telemetry: GPU avg 6.3 W (baseline 5.7 W), peak 18.0 W, net energy 0.286 kJ (DCGM 0.215), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.43 cores avg (host util 50 %), peak RSS 2.35 GiB

### 106. `cpu-1.7b-q4_k_m-f16-T9` — typed-decisions — ok

- 1.7B Q4_K_M · CPU · 16 threads · started 2026-09-29 00:44:52 · wall 427.3 s (estimate 673 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-1.7B-GGUF/Spark-X2.5-1.7B-Q4_K_M.gguf --device cpu --output results/local-hw/cpu-1.7b-q4_k_m-f16-T9/typed.json`
- accuracy 0.481, KL 2.839, Brier 0.583, ECE 0.431, macro-F1 0.290
- p50 1070 ms/case, p95 1336 ms, 4.70 dec/s
- by workflow: agent_trace_observability 0.436, customer_service 0.554, invoice_processing 0.348, security_incidents 0.588
- Telemetry: GPU avg 6.4 W (baseline 6.2 W), peak 8.6 W, net energy 0.067 kJ (DCGM 0.252), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.44 cores avg (host util 50 %), peak RSS 2.34 GiB

### 107. `cpu-1.7b-bf16-f16-T3` — rizzo decide examples — ok

- 1.7B BF16 · CPU · 16 threads · started 2026-09-29 00:52:12 · wall 7.1 s (estimate 55 s) · exit [0, 0, 0]
- `rizzo decide examples/ticket.json --size 1.7b --quant bf16 --device cpu --threads 16 --output results/local-hw/cpu-1.7b-bf16-f16-T3/decide-ticket.json`
- `rizzo decide examples/house.json --size 1.7b --quant bf16 --device cpu --threads 16 --output results/local-hw/cpu-1.7b-bf16-f16-T3/decide-house.json`
- `rizzo decide examples/numeric.json --size 1.7b --quant bf16 --device cpu --threads 16 --output results/local-hw/cpu-1.7b-bf16-f16-T3/decide-numeric.json`
- ticket: needs_access_support=False (ok, p=0.931); queue=None (insufficient_evidence, p=0.787); urgency=None (insufficient_evidence, p=0.861); deadline_hours=None (insufficient_evidence, p=1.000) — total 953 ms, inference 937 ms, 1287 tokens, load 1.5 s
- house: estimated_price=None (insufficient_evidence, p=1.000) — total 721 ms, inference 715 ms, 562 tokens, load 1.5 s
- numeric: fill=74.9688557542975 (ok, p=0.859) — total 517 ms, inference 510 ms, 398 tokens, load 1.5 s
- Telemetry: GPU avg 6.0 W (baseline 5.9 W), peak 6.5 W, net energy 0.000 kJ (DCGM 0.001), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 5.47 cores avg (host util 23 %), peak RSS 3.89 GiB

### 108. `cpu-1.7b-bf16-f16-T4` — smoke — ok

- 1.7B BF16 · CPU · 16 threads · started 2026-09-29 00:52:32 · wall 16.9 s (estimate 55 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant bf16 --device cpu --threads 16 --output results/local-hw/cpu-1.7b-bf16-f16-T4/smoke.json`
- accuracy 0.450 (20 rows), accepted 0.857, coverage 0.350, NLL 3.470, Brier 1.014, ECE 0.515
- latency median 387 ms, p95 976 ms, 2.80 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.006990, median 387 / 343 ms
- peak device —, device `None` (cpu), load 1.5 s
- Telemetry: GPU avg 6.2 W (baseline 5.8 W), peak 7.1 W, net energy 0.006 kJ (DCGM 0.009), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 13.80 cores avg (host util 47 %), peak RSS 3.89 GiB

### 109. `cpu-1.7b-bf16-f16-T5` — perturbations — ok

- 1.7B BF16 · CPU · 16 threads · started 2026-09-29 00:53:02 · wall 8.0 s (estimate 34 s) · exit [0]
- `rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 1.7b --quant bf16 --device cpu --threads 16 --output results/local-hw/cpu-1.7b-bf16-f16-T5/perturbations.json`
- accuracy 0.556 (9 rows), accepted 1, coverage 0.444, NLL 2.782, Brier 0.865, ECE 0.455
- latency median 336 ms, p95 376 ms, 2.98 dec/s
- shared vs direct: 0/9 changed argmaxes, max Δp 0.000000, median 336 / 337 ms
- peak device —, device `None` (cpu), load 1.5 s
- Telemetry: GPU avg 6.1 W (baseline 5.9 W), peak 6.5 W, net energy 0.002 kJ (DCGM 0.004), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 12.13 cores avg (host util 46 %), peak RSS 3.87 GiB

### 110. `cpu-1.7b-bf16-f16-T6` — validate_checkpoint — ok

- 1.7B BF16 · CPU · 16 threads · started 2026-09-29 00:53:23 · wall 42.0 s (estimate 97 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/validate_checkpoint.py --size 1.7b --quant bf16 --device cpu --output results/local-hw/cpu-1.7b-bf16-f16-T6/validate`
- smoke accuracy 0.450 (NLL 3.470, 0 changed), perturbations 0.556
- long state: shared 2929 ms vs direct 9196 ms, 0 changed (max Δp 0.000074)
- validation 39.4 s, load 1.5 s, peak —
- Telemetry: GPU avg 6.1 W (baseline 5.8 W), peak 6.5 W, net energy 0.012 kJ (DCGM 0.005), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.82 cores avg (host util 50 %), peak RSS 3.90 GiB

### 111. `cpu-1.7b-bf16-f16-T7` — semif_compare — ok

- 1.7B BF16 · CPU · 16 threads · started 2026-09-29 00:54:18 · wall 482.1 s (estimate 811 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant bf16 --device cpu --output results/local-hw/cpu-1.7b-bf16-f16-T7/semif`
- authored144 0.683, perturbations108 0.640, held-out 0.690 / 0.532, macro-F1 0.658
- latency p50 287 ms, p95 311 ms; shape777 shared 4.42 dec/s (37 states, 2111 tok/dec), direct 0.38 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.038); stability flips reversal/wrapper/context 18/4/4; confident on missing evidence 2/36; rule_application perturbed 0.296 (NLL 3.55)
- peak device —
- vs this-cuda: authored144 difference 0.000 [+0.000, +0.000], different argmax 0/252
- Telemetry: GPU avg 6.3 W (baseline 6.0 W), peak 15.9 W, net energy 0.126 kJ (DCGM 0.285), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.43 cores avg (host util 50 %), peak RSS 3.90 GiB

### 112. `cpu-1.7b-bf16-f16-T9` — typed-decisions — ok

- 1.7B BF16 · CPU · 16 threads · started 2026-09-29 01:02:34 · wall 462.3 s (estimate 853 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --model models/Spark-X2.5-1.7B-GGUF/Spark-X2.5-1.7B.gguf --device cpu --output results/local-hw/cpu-1.7b-bf16-f16-T9/typed.json`
- accuracy 0.526, KL 3.028, Brier 0.495, ECE 0.351, macro-F1 0.330
- p50 1155 ms/case, p95 1432 ms, 4.35 dec/s
- by workflow: agent_trace_observability 0.374, customer_service 0.650, invoice_processing 0.460, security_incidents 0.622
- Telemetry: GPU avg 6.3 W (baseline 6.1 W), peak 16.8 W, net energy 0.128 kJ (DCGM 0.271), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.46 cores avg (host util 50 %), peak RSS 3.89 GiB

### 113. `cpu-4b-q8_0-q8_0-T4` — smoke — ok

- 4B Q8_0 · CPU · KV q8_0 · 16 threads · started 2026-09-29 01:10:29 · wall 43.8 s (estimate 73 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --device cpu --kv-type q8_0 --threads 16 --output results/local-hw/cpu-4b-q8_0-q8_0-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.433, Brier 0.077, ECE 0.038
- latency median 887 ms, p95 2746 ms, 1.04 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000000, median 887 / 890 ms
- peak device —, device `None` (cpu), load 2.0 s
- Telemetry: GPU avg 5.8 W (baseline 5.5 W), peak 15.9 W, net energy 0.014 kJ (DCGM 0.013), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.85 cores avg (host util 50 %), peak RSS 5.00 GiB

### 114. `cpu-4b-q8_0-q8_0-T7` — semif_compare — ok

- 4B Q8_0 · CPU · KV q8_0 · 16 threads · started 2026-09-29 01:11:26 · wall 1796.9 s (estimate 1153 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cpu --kv-type q8_0 --output results/local-hw/cpu-4b-q8_0-q8_0-T7/semif`
- authored144 0.810, perturbations108 0.854, held-out 0.785 / 0.861, macro-F1 0.804
- latency p50 738 ms, p95 822 ms; shape777 shared 1.15 dec/s (37 states, 2111 tok/dec), direct 0.09 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.000); stability flips reversal/wrapper/context 6/4/4; confident on missing evidence 6/36; rule_application perturbed 0.630 (NLL 1.66)
- peak device —
- vs this-cpu-kv-f16: authored144 difference -0.006 [-0.020, +0.000], different argmax 3/252
- Telemetry: GPU avg 5.9 W (baseline 5.5 W), peak 17.1 W, net energy 0.637 kJ (DCGM 0.398), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.51 cores avg (host util 50 %), peak RSS 5.02 GiB

### 115. `cpu-4b-q8_0-q4_0-T4` — smoke — ok

- 4B Q8_0 · CPU · KV q4_0 · 16 threads · started 2026-09-29 01:41:37 · wall 47.0 s (estimate 73 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --device cpu --kv-type q4_0 --threads 16 --output results/local-hw/cpu-4b-q8_0-q4_0-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.472, Brier 0.088, ECE 0.048
- latency median 1011 ms, p95 3064 ms, 0.95 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000000, median 1011 / 926 ms
- peak device —, device `None` (cpu), load 2.0 s
- Telemetry: GPU avg 6.0 W (baseline 5.6 W), peak 7.1 W, net energy 0.019 kJ (DCGM 0.037), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.09 cores avg (host util 49 %), peak RSS 4.65 GiB

### 116. `cpu-4b-q8_0-q4_0-T7` — semif_compare — ok

- 4B Q8_0 · CPU · KV q4_0 · 16 threads · started 2026-09-29 01:42:37 · wall 2152.5 s (estimate 1153 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cpu --kv-type q4_0 --output results/local-hw/cpu-4b-q8_0-q4_0-T7/semif`
- authored144 0.801, perturbations108 0.884, held-out 0.779 / 0.894, macro-F1 0.796
- latency p50 778 ms, p95 886 ms; shape777 shared 0.94 dec/s (37 states, 2111 tok/dec), direct 0.08 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.000); stability flips reversal/wrapper/context 5/3/3; confident on missing evidence 5/36; rule_application perturbed 0.685 (NLL 1.60)
- peak device —
- vs this-cpu-kv-f16: authored144 difference -0.015 [-0.050, +0.016], different argmax 11/252
- Telemetry: GPU avg 5.9 W (baseline 5.7 W), peak 16.4 W, net energy 0.406 kJ (DCGM 0.664), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.41 cores avg (host util 49 %), peak RSS 4.66 GiB

### 117. `cpu-4b-bf16-q8_0-T4` — smoke — ok

- 4B BF16 · CPU · KV q8_0 · 16 threads · started 2026-09-29 02:18:43 · wall 44.2 s (estimate 97 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant bf16 --device cpu --kv-type q8_0 --threads 16 --output results/local-hw/cpu-4b-bf16-q8_0-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.436, Brier 0.077, ECE 0.038
- latency median 854 ms, p95 2683 ms, 1.05 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000000, median 854 / 843 ms
- peak device —, device `None` (cpu), load 3.4 s
- Telemetry: GPU avg 6.4 W (baseline 5.6 W), peak 16.0 W, net energy 0.035 kJ (DCGM 0.025), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 14.42 cores avg (host util 47 %), peak RSS 8.59 GiB

### 118. `cpu-4b-bf16-q8_0-T7` — semif_compare — ok

- 4B BF16 · CPU · KV q8_0 · 16 threads · started 2026-09-29 02:19:40 · wall 1681.4 s (estimate 1609 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant bf16 --device cpu --kv-type q8_0 --output results/local-hw/cpu-4b-bf16-q8_0-T7/semif`
- authored144 0.814, perturbations108 0.842, held-out 0.793 / 0.843, macro-F1 0.810
- latency p50 731 ms, p95 800 ms; shape777 shared 1.23 dec/s (37 states, 2111 tok/dec), direct 0.10 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.000); stability flips reversal/wrapper/context 5/3/4; confident on missing evidence 6/36; rule_application perturbed 0.593 (NLL 1.69)
- peak device —
- vs this-cpu-kv-f16: authored144 difference -0.015 [-0.037, +0.000], different argmax 2/252
- Telemetry: GPU avg 6.1 W (baseline 5.5 W), peak 17.0 W, net energy 0.872 kJ (DCGM 0.578), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.69 cores avg (host util 51 %), peak RSS 8.61 GiB

### 119. `cpu-4b-bf16-q4_0-T4` — smoke — ok

- 4B BF16 · CPU · KV q4_0 · 16 threads · started 2026-09-29 02:47:55 · wall 47.6 s (estimate 97 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant bf16 --device cpu --kv-type q4_0 --threads 16 --output results/local-hw/cpu-4b-bf16-q4_0-T4/smoke.json`
- accuracy 0.950 (20 rows), accepted 0.938, coverage 0.800, NLL 0.578, Brier 0.093, ECE 0.041
- latency median 896 ms, p95 2999 ms, 0.97 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000000, median 896 / 881 ms
- peak device —, device `None` (cpu), load 3.4 s
- Telemetry: GPU avg 6.4 W (baseline 7.3 W), peak 15.9 W, net energy -0.040 kJ (DCGM 0.000), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 14.67 cores avg (host util 50 %), peak RSS 8.24 GiB

### 120. `cpu-4b-bf16-q4_0-T7` — semif_compare — ok

- 4B BF16 · CPU · KV q4_0 · 16 threads · started 2026-09-29 02:48:56 · wall 2032.4 s (estimate 1609 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant bf16 --device cpu --kv-type q4_0 --output results/local-hw/cpu-4b-bf16-q4_0-T7/semif`
- authored144 0.759, perturbations108 0.830, held-out 0.736 / 0.843, macro-F1 0.748
- latency p50 767 ms, p95 831 ms; shape777 shared 0.99 dec/s (37 states, 2111 tok/dec), direct 0.08 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.000); stability flips reversal/wrapper/context 5/3/6; confident on missing evidence 7/36; rule_application perturbed 0.556 (NLL 1.79)
- peak device —
- vs this-cpu-kv-f16: authored144 difference -0.070 [-0.109, -0.033], different argmax 17/252
- Telemetry: GPU avg 6.3 W (baseline 7.2 W), peak 17.1 W, net energy -1.796 kJ (DCGM 0.208), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.60 cores avg (host util 50 %), peak RSS 8.26 GiB

### 121. `cpu-4b-q4_k_m-q8_0-T4` — smoke — ok

- 4B Q4_K_M · CPU · KV q8_0 · 16 threads · started 2026-09-29 03:23:01 · wall 39.3 s (estimate 82 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q4_k_m --device cpu --kv-type q8_0 --threads 16 --output results/local-hw/cpu-4b-q4_k_m-q8_0-T4/smoke.json`
- accuracy 0.900 (20 rows), accepted 1, coverage 0.750, NLL 0.697, Brier 0.178, ECE 0.087
- latency median 770 ms, p95 2461 ms, 1.15 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.039762, median 770 / 841 ms
- peak device —, device `None` (cpu), load 1.7 s
- Telemetry: GPU avg 6.7 W (baseline 6.4 W), peak 10.7 W, net energy 0.012 kJ (DCGM 0.018), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.87 cores avg (host util 50 %), peak RSS 4.92 GiB

### 122. `cpu-4b-q4_k_m-q8_0-T7` — semif_compare — ok

- 4B Q4_K_M · CPU · KV q8_0 · 16 threads · started 2026-09-29 03:23:54 · wall 1593.9 s (estimate 1324 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q4_k_m --device cpu --kv-type q8_0 --output results/local-hw/cpu-4b-q4_k_m-q8_0-T7/semif`
- authored144 0.783, perturbations108 0.811, held-out 0.756 / 0.787, macro-F1 0.774
- latency p50 668 ms, p95 728 ms; shape777 shared 1.29 dec/s (37 states, 2111 tok/dec), direct 0.11 dec/s (3 states)
- shared vs direct 1/63 flips (max Δp 0.232); stability flips reversal/wrapper/context 5/2/2; confident on missing evidence 5/36; rule_application perturbed 0.556 (NLL 1.82)
- peak device —
- vs this-cpu-kv-f16: authored144 difference 0.017 [-0.005, +0.046], different argmax 6/252
- Telemetry: GPU avg 6.3 W (baseline 5.5 W), peak 17.0 W, net energy 1.259 kJ (DCGM 0.588), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.69 cores avg (host util 50 %), peak RSS 4.93 GiB

### 123. `cpu-4b-q4_k_m-q4_0-T4` — smoke — ok

- 4B Q4_K_M · CPU · KV q4_0 · 16 threads · started 2026-09-29 03:50:41 · wall 42.5 s (estimate 82 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q4_k_m --device cpu --kv-type q4_0 --threads 16 --output results/local-hw/cpu-4b-q4_k_m-q4_0-T4/smoke.json`
- accuracy 0.900 (20 rows), accepted 0.938, coverage 0.800, NLL 0.657, Brier 0.169, ECE 0.063
- latency median 849 ms, p95 2815 ms, 1.05 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.136352, median 849 / 820 ms
- peak device —, device `None` (cpu), load 1.7 s
- Telemetry: GPU avg 6.0 W (baseline 5.7 W), peak 15.6 W, net energy 0.011 kJ (DCGM 0.019), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.14 cores avg (host util 51 %), peak RSS 4.57 GiB

### 124. `cpu-4b-q4_k_m-q4_0-T7` — semif_compare — ok

- 4B Q4_K_M · CPU · KV q4_0 · 16 threads · started 2026-09-29 03:51:37 · wall 1948.8 s (estimate 1324 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q4_k_m --device cpu --kv-type q4_0 --output results/local-hw/cpu-4b-q4_k_m-q4_0-T7/semif`
- authored144 0.768, perturbations108 0.841, held-out 0.747 / 0.801, macro-F1 0.754
- latency p50 700 ms, p95 766 ms; shape777 shared 1.03 dec/s (37 states, 2111 tok/dec), direct 0.09 dec/s (3 states)
- shared vs direct 3/63 flips (max Δp 0.385); stability flips reversal/wrapper/context 5/2/3; confident on missing evidence 6/36; rule_application perturbed 0.630 (NLL 1.78)
- peak device —
- vs this-cpu-kv-f16: authored144 difference 0.003 [-0.033, +0.037], different argmax 9/252
- Telemetry: GPU avg 6.1 W (baseline 5.9 W), peak 17.3 W, net energy 0.373 kJ (DCGM 1.125), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.60 cores avg (host util 50 %), peak RSS 4.58 GiB

### 125. `cpu-1.7b-q8_0-q8_0-T4` — smoke — ok

- 1.7B Q8_0 · CPU · KV q8_0 · 16 threads · started 2026-09-29 04:24:19 · wall 17.6 s (estimate 43 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q8_0 --device cpu --kv-type q8_0 --threads 16 --output results/local-hw/cpu-1.7b-q8_0-q8_0-T4/smoke.json`
- accuracy 0.450 (20 rows), accepted 0.857, coverage 0.350, NLL 3.628, Brier 1.057, ECE 0.562
- latency median 371 ms, p95 1048 ms, 2.60 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000000, median 371 / 442 ms
- peak device —, device `None` (cpu), load 0.9 s
- Telemetry: GPU avg 6.2 W (baseline 5.8 W), peak 7.5 W, net energy 0.007 kJ (DCGM 0.012), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.74 cores avg (host util 50 %), peak RSS 2.14 GiB

### 126. `cpu-1.7b-q8_0-q8_0-T7` — semif_compare — ok

- 1.7B Q8_0 · CPU · KV q8_0 · 16 threads · started 2026-09-29 04:24:50 · wall 646.4 s (estimate 583 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q8_0 --device cpu --kv-type q8_0 --output results/local-hw/cpu-1.7b-q8_0-q8_0-T7/semif`
- authored144 0.694, perturbations108 0.646, held-out 0.696 / 0.532, macro-F1 0.658
- latency p50 301 ms, p95 357 ms; shape777 shared 3.21 dec/s (37 states, 2111 tok/dec), direct 0.27 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.000); stability flips reversal/wrapper/context 18/5/4; confident on missing evidence 2/36; rule_application perturbed 0.315 (NLL 3.51)
- peak device —
- vs this-cpu-kv-f16: authored144 difference -0.019 [-0.050, +0.007], different argmax 6/252
- Telemetry: GPU avg 6.1 W (baseline 5.8 W), peak 14.5 W, net energy 0.187 kJ (DCGM 0.222), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.65 cores avg (host util 50 %), peak RSS 2.16 GiB

### 127. `cpu-1.7b-q8_0-q4_0-T4` — smoke — ok

- 1.7B Q8_0 · CPU · KV q4_0 · 16 threads · started 2026-09-29 04:35:50 · wall 18.8 s (estimate 43 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q8_0 --device cpu --kv-type q4_0 --threads 16 --output results/local-hw/cpu-1.7b-q8_0-q4_0-T4/smoke.json`
- accuracy 0.350 (20 rows), accepted 0.800, coverage 0.250, NLL 3.742, Brier 1.122, ECE 0.598
- latency median 379 ms, p95 1177 ms, 2.41 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000000, median 379 / 452 ms
- peak device —, device `None` (cpu), load 0.8 s
- Telemetry: GPU avg 6.3 W (baseline 5.8 W), peak 14.1 W, net energy 0.008 kJ (DCGM 0.007), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.96 cores avg (host util 50 %), peak RSS 2.00 GiB

### 128. `cpu-1.7b-q8_0-q4_0-T7` — semif_compare — ok

- 1.7B Q8_0 · CPU · KV q4_0 · 16 threads · started 2026-09-29 04:36:22 · wall 813.3 s (estimate 583 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q8_0 --device cpu --kv-type q4_0 --output results/local-hw/cpu-1.7b-q8_0-q4_0-T7/semif`
- authored144 0.659, perturbations108 0.537, held-out 0.660 / 0.514, macro-F1 0.633
- latency p50 310 ms, p95 368 ms; shape777 shared 2.49 dec/s (37 states, 2111 tok/dec), direct 0.21 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.000); stability flips reversal/wrapper/context 14/7/4; confident on missing evidence 3/36; rule_application perturbed 0.241 (NLL 3.97)
- peak device —
- vs this-cpu-kv-f16: authored144 difference -0.054 [-0.102, -0.010], different argmax 36/252
- Telemetry: GPU avg 6.0 W (baseline 6.6 W), peak 16.0 W, net energy -0.509 kJ (DCGM 0.212), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.66 cores avg (host util 50 %), peak RSS 2.02 GiB

### 129. `cpu-1.7b-q4_k_m-q8_0-T4` — smoke — ok

- 1.7B Q4_K_M · CPU · KV q8_0 · 16 threads · started 2026-09-29 04:50:08 · wall 15.7 s (estimate 46 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q4_k_m --device cpu --kv-type q8_0 --threads 16 --output results/local-hw/cpu-1.7b-q4_k_m-q8_0-T4/smoke.json`
- accuracy 0.350 (20 rows), accepted 0.800, coverage 0.250, NLL 3.384, Brier 1.045, ECE 0.544
- latency median 324 ms, p95 990 ms, 2.88 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.130118, median 324 / 300 ms
- peak device —, device `None` (cpu), load 0.8 s
- Telemetry: GPU avg 5.9 W (baseline 6.4 W), peak 6.5 W, net energy -0.009 kJ (DCGM 0.003), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.65 cores avg (host util 50 %), peak RSS 2.08 GiB

### 130. `cpu-1.7b-q4_k_m-q8_0-T7` — semif_compare — ok

- 1.7B Q4_K_M · CPU · KV q8_0 · 16 threads · started 2026-09-29 04:50:37 · wall 586.3 s (estimate 640 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q4_k_m --device cpu --kv-type q8_0 --output results/local-hw/cpu-1.7b-q4_k_m-q8_0-T7/semif`
- authored144 0.591, perturbations108 0.472, held-out 0.580 / 0.468, macro-F1 0.558
- latency p50 264 ms, p95 290 ms; shape777 shared 3.51 dec/s (37 states, 2111 tok/dec), direct 0.29 dec/s (3 states)
- shared vs direct 1/63 flips (max Δp 0.224); stability flips reversal/wrapper/context 16/4/2; confident on missing evidence 8/36; rule_application perturbed 0.056 (NLL 4.00)
- peak device —
- vs this-cpu-kv-f16: authored144 difference 0.006 [-0.015, +0.028], different argmax 11/252
- Telemetry: GPU avg 5.9 W (baseline 5.4 W), peak 7.4 W, net energy 0.283 kJ (DCGM 0.394), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.63 cores avg (host util 50 %), peak RSS 2.10 GiB

### 131. `cpu-1.7b-q4_k_m-q4_0-T4` — smoke — ok

- 1.7B Q4_K_M · CPU · KV q4_0 · 16 threads · started 2026-09-29 05:00:37 · wall 17.1 s (estimate 46 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant q4_k_m --device cpu --kv-type q4_0 --threads 16 --output results/local-hw/cpu-1.7b-q4_k_m-q4_0-T4/smoke.json`
- accuracy 0.250 (20 rows), accepted 0.667, coverage 0.150, NLL 4.159, Brier 1.324, ECE 0.717
- latency median 339 ms, p95 1091 ms, 2.63 dec/s
- shared vs direct: 1/20 changed argmaxes, max Δp 0.213939, median 339 / 373 ms
- peak device —, device `None` (cpu), load 0.8 s
- Telemetry: GPU avg 6.8 W (baseline 5.7 W), peak 8.2 W, net energy 0.019 kJ (DCGM 0.011), util avg 0 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 14.94 cores avg (host util 50 %), peak RSS 1.94 GiB

### 132. `cpu-1.7b-q4_k_m-q4_0-T7` — semif_compare — ok

- 1.7B Q4_K_M · CPU · KV q4_0 · 16 threads · started 2026-09-29 05:01:07 · wall 747.4 s (estimate 640 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant q4_k_m --device cpu --kv-type q4_0 --output results/local-hw/cpu-1.7b-q4_k_m-q4_0-T7/semif`
- authored144 0.572, perturbations108 0.434, held-out 0.582 / 0.412, macro-F1 0.533
- latency p50 278 ms, p95 309 ms; shape777 shared 2.68 dec/s (37 states, 2111 tok/dec), direct 0.22 dec/s (3 states)
- shared vs direct 4/63 flips (max Δp 0.738); stability flips reversal/wrapper/context 18/4/4; confident on missing evidence 8/36; rule_application perturbed 0.056 (NLL 4.72)
- peak device —
- vs this-cpu-kv-f16: authored144 difference -0.013 [-0.041, +0.012], different argmax 16/252
- Telemetry: GPU avg 18.1 W (baseline 6.3 W), peak 47.9 W, net energy 8.875 kJ (DCGM 9.027), util avg 10 %, SM active 0.03, DRAM active 0.00, VRAM max 1508 MiB (baseline 1327), temp max 46 °C, power-cap throttle 0.0 s; CPU 15.66 cores avg (host util 50 %), peak RSS 1.96 GiB

### 133. `cpu-1.7b-bf16-q8_0-T4` — smoke — ok

- 1.7B BF16 · CPU · KV q8_0 · 16 threads · started 2026-09-29 05:13:48 · wall 17.6 s (estimate 55 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant bf16 --device cpu --kv-type q8_0 --threads 16 --output results/local-hw/cpu-1.7b-bf16-q8_0-T4/smoke.json`
- accuracy 0.450 (20 rows), accepted 0.857, coverage 0.350, NLL 3.503, Brier 1.018, ECE 0.518
- latency median 336 ms, p95 1046 ms, 2.69 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000005, median 336 / 366 ms
- peak device —, device `None` (cpu), load 1.5 s
- Telemetry: GPU avg 28.2 W (baseline 29.0 W), peak 30.1 W, net energy -0.013 kJ (DCGM 0.012), util avg 19 %, SM active 0.09, DRAM active 0.00, VRAM max 1402 MiB (baseline 1402), temp max 46 °C, power-cap throttle 0.0 s; CPU 14.19 cores avg (host util 50 %), peak RSS 3.63 GiB

### 134. `cpu-1.7b-bf16-q8_0-T7` — semif_compare — ok

- 1.7B BF16 · CPU · KV q8_0 · 16 threads · started 2026-09-29 05:14:18 · wall 619.0 s (estimate 811 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant bf16 --device cpu --kv-type q8_0 --output results/local-hw/cpu-1.7b-bf16-q8_0-T7/semif`
- authored144 0.683, perturbations108 0.640, held-out 0.690 / 0.514, macro-F1 0.658
- latency p50 291 ms, p95 317 ms; shape777 shared 3.34 dec/s (37 states, 2111 tok/dec), direct 0.28 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.001); stability flips reversal/wrapper/context 18/6/4; confident on missing evidence 2/36; rule_application perturbed 0.315 (NLL 3.60)
- peak device —
- vs this-cpu-kv-f16: authored144 difference 0.000 [+0.000, +0.000], different argmax 2/252
- Telemetry: GPU avg 28.1 W (baseline 27.7 W), peak 35.3 W, net energy 0.245 kJ (DCGM -0.017), util avg 18 %, SM active 0.08, DRAM active 0.00, VRAM max 1402 MiB (baseline 1402), temp max 47 °C, power-cap throttle 0.0 s; CPU 15.63 cores avg (host util 51 %), peak RSS 3.65 GiB

### 135. `cpu-1.7b-bf16-q4_0-T4` — smoke — ok

- 1.7B BF16 · CPU · KV q4_0 · 16 threads · started 2026-09-29 05:24:51 · wall 18.8 s (estimate 55 s) · exit [0]
- `rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 1.7b --quant bf16 --device cpu --kv-type q4_0 --threads 16 --output results/local-hw/cpu-1.7b-bf16-q4_0-T4/smoke.json`
- accuracy 0.400 (20 rows), accepted 0.833, coverage 0.300, NLL 3.868, Brier 1.066, ECE 0.526
- latency median 393 ms, p95 1164 ms, 2.47 dec/s
- shared vs direct: 0/20 changed argmaxes, max Δp 0.000003, median 393 / 364 ms
- peak device —, device `None` (cpu), load 1.5 s
- Telemetry: GPU avg 27.6 W (baseline 27.6 W), peak 28.2 W, net energy 0.001 kJ (DCGM -0.048), util avg 18 %, SM active 0.04, DRAM active 0.00, VRAM max 1402 MiB (baseline 1402), temp max 47 °C, power-cap throttle 0.0 s; CPU 14.47 cores avg (host util 51 %), peak RSS 3.49 GiB

### 136. `cpu-1.7b-bf16-q4_0-T7` — semif_compare — ok

- 1.7B BF16 · CPU · KV q4_0 · 16 threads · started 2026-09-29 05:25:23 · wall 774.7 s (estimate 811 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 1.7b --quant bf16 --device cpu --kv-type q4_0 --output results/local-hw/cpu-1.7b-bf16-q4_0-T7/semif`
- authored144 0.653, perturbations108 0.599, held-out 0.651 / 0.551, macro-F1 0.615
- latency p50 302 ms, p95 331 ms; shape777 shared 2.59 dec/s (37 states, 2111 tok/dec), direct 0.22 dec/s (3 states)
- shared vs direct 0/63 flips (max Δp 0.000); stability flips reversal/wrapper/context 17/7/8; confident on missing evidence 2/36; rule_application perturbed 0.315 (NLL 3.68)
- peak device —
- vs this-cpu-kv-f16: authored144 difference -0.030 [-0.080, +0.023], different argmax 31/252
- Telemetry: GPU avg 26.2 W (baseline 27.6 W), peak 35.1 W, net energy -1.066 kJ (DCGM -1.561), util avg 17 %, SM active 0.05, DRAM active 0.00, VRAM max 1402 MiB (baseline 1402), temp max 47 °C, power-cap throttle 0.0 s; CPU 15.66 cores avg (host util 51 %), peak RSS 3.51 GiB

### 137. `cpu-4b-q8_0-f16-T7-batch1` — semif_compare — ok

- 4B Q8_0 · CPU · batch 1 · 16 threads · started 2026-09-29 05:38:31 · wall 722.4 s (estimate 673 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cpu --output results/local-hw/cpu-4b-q8_0-f16-T7-batch1/semif --batch-size 1 --direct-states 0`
- authored144 0.816, perturbations108 0.873, held-out 0.799 / 0.880, macro-F1 0.811
- latency p50 733 ms, p95 825 ms; shape777 shared 1.51 dec/s (37 states, 2111 tok/dec), direct — dec/s (None states)
- shared vs direct None/None flips (max Δp —); stability flips reversal/wrapper/context 3/3/5; confident on missing evidence 6/36; rule_application perturbed 0.685 (NLL 1.65)
- peak device —
- Telemetry: GPU avg 6.3 W (baseline 5.8 W), peak 17.4 W, net energy 0.361 kJ (DCGM 0.215), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 41 °C, power-cap throttle 0.0 s; CPU 15.47 cores avg (host util 50 %), peak RSS 5.67 GiB

### 138. `cpu-4b-q8_0-f16-T7-batch2` — semif_compare — ok

- 4B Q8_0 · CPU · batch 2 · 16 threads · started 2026-09-29 05:50:46 · wall 680.6 s (estimate 673 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cpu --output results/local-hw/cpu-4b-q8_0-f16-T7-batch2/semif --batch-size 2 --direct-states 0`
- authored144 0.816, perturbations108 0.873, held-out 0.799 / 0.880, macro-F1 0.811
- latency p50 725 ms, p95 802 ms; shape777 shared 1.64 dec/s (37 states, 2111 tok/dec), direct — dec/s (None states)
- shared vs direct None/None flips (max Δp —); stability flips reversal/wrapper/context 3/3/5; confident on missing evidence 6/36; rule_application perturbed 0.685 (NLL 1.65)
- peak device —
- Telemetry: GPU avg 6.3 W (baseline 5.5 W), peak 17.0 W, net energy 0.557 kJ (DCGM 0.329), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.44 cores avg (host util 50 %), peak RSS 5.67 GiB

### 139. `cpu-4b-q8_0-f16-T7-batch4` — semif_compare — ok

- 4B Q8_0 · CPU · batch 4 · 16 threads · started 2026-09-29 06:02:20 · wall 667.1 s (estimate 673 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cpu --output results/local-hw/cpu-4b-q8_0-f16-T7-batch4/semif --batch-size 4 --direct-states 0`
- authored144 0.816, perturbations108 0.873, held-out 0.799 / 0.880, macro-F1 0.811
- latency p50 725 ms, p95 821 ms; shape777 shared 1.69 dec/s (37 states, 2111 tok/dec), direct — dec/s (None states)
- shared vs direct None/None flips (max Δp —); stability flips reversal/wrapper/context 3/3/5; confident on missing evidence 6/36; rule_application perturbed 0.685 (NLL 1.65)
- peak device —
- Telemetry: GPU avg 6.3 W (baseline 6.6 W), peak 16.4 W, net energy -0.195 kJ (DCGM 0.206), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1327 MiB (baseline 1327), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.43 cores avg (host util 50 %), peak RSS 5.67 GiB

### 140. `cpu-4b-q8_0-f16-T7-batch8` — semif_compare — ok

- 4B Q8_0 · CPU · batch 8 · 16 threads · started 2026-09-29 06:13:40 · wall 663.9 s (estimate 673 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cpu --output results/local-hw/cpu-4b-q8_0-f16-T7-batch8/semif --batch-size 8 --direct-states 0`
- authored144 0.816, perturbations108 0.873, held-out 0.799 / 0.880, macro-F1 0.811
- latency p50 730 ms, p95 813 ms; shape777 shared 1.70 dec/s (37 states, 2111 tok/dec), direct — dec/s (None states)
- shared vs direct None/None flips (max Δp —); stability flips reversal/wrapper/context 3/3/5; confident on missing evidence 6/36; rule_application perturbed 0.685 (NLL 1.65)
- peak device —
- Telemetry: GPU avg 6.2 W (baseline 5.6 W), peak 16.7 W, net energy 0.421 kJ (DCGM 0.346), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1328 MiB (baseline 1327), temp max 37 °C, power-cap throttle 0.0 s; CPU 15.42 cores avg (host util 50 %), peak RSS 5.67 GiB

### 141. `cpu-4b-q8_0-f16-T7-batch16` — semif_compare — ok

- 4B Q8_0 · CPU · batch 16 · 16 threads · started 2026-09-29 06:24:57 · wall 658.7 s (estimate 673 s) · exit [0]
- `python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --device cpu --output results/local-hw/cpu-4b-q8_0-f16-T7-batch16/semif --batch-size 16 --direct-states 0`
- authored144 0.816, perturbations108 0.873, held-out 0.799 / 0.880, macro-F1 0.811
- latency p50 729 ms, p95 805 ms; shape777 shared 1.72 dec/s (37 states, 2111 tok/dec), direct — dec/s (None states)
- shared vs direct None/None flips (max Δp —); stability flips reversal/wrapper/context 3/3/5; confident on missing evidence 6/36; rule_application perturbed 0.685 (NLL 1.65)
- peak device —
- Telemetry: GPU avg 6.4 W (baseline 6.2 W), peak 17.0 W, net energy 0.086 kJ (DCGM 0.203), util avg 1 %, SM active 0.00, DRAM active 0.00, VRAM max 1328 MiB (baseline 1328), temp max 38 °C, power-cap throttle 0.0 s; CPU 15.41 cores avg (host util 50 %), peak RSS 5.68 GiB

<!-- RESULTS:END -->
