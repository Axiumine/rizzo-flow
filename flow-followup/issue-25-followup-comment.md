## Follow-up: the fine-tuned weights (`--weights flow`) on this machine

Same machine, same commands, same run ids as the base campaign, now with `--weights flow`: **39 runs, all completed** (1 h 16 min from the first run to the last, one model process at a time, idle gate on). Tested at `b9ba007`; the base numbers are the ones of this report, at `c30cc63`. Every flow run is paired with the base run of the same id. Differences are flow − base; an interval that contains 0 means no difference shown. Probabilities are not calibrated.

- **Your flow 4B Q8_0 run reproduces exactly.** SemIf authored144 0.845 / perturbations108 0.946, flips 1 / 1 / 1, missing evidence 21/36 with 5 confident wrong, rule_application perturbed 0.870: same as yours. Paired difference against your run +0.000 [+0.000, +0.000] on both fixtures, 0 of 252 rows with a different argmax. CPU-only gives the same 0 of 252 rows against CUDA.
- **Flow vs base, SemIf, 4B Q8_0:** authored144 0.810 → 0.845, +0.035 [−0.027, +0.100], no difference shown; perturbations108 0.830 → 0.946, +0.116 [+0.044, +0.205]. BF16: +0.026 [−0.039, +0.093] and +0.098 [+0.000, +0.206], no difference shown on either. Q4_K_M: +0.059 [−0.004, +0.128] (no difference shown) and +0.104 [+0.027, +0.191]. 43 to 54 of 252 rows change argmax.
- **1.7B on SemIf, first numbers on flow weights:** Q8_0 0.794 / 0.783, Q4_K_M 0.812 / 0.770, BF16 0.801 / 0.783. All six differences from base have intervals that exclude 0, for example Q8_0 +0.094 [+0.021, +0.163] and +0.137 [+0.052, +0.231]; Q4_K_M from 0.607 / 0.472 at base, +0.204 [+0.129, +0.280] and +0.298 [+0.176, +0.389].
- **Smoke, first numbers on flow weights:** 19/20 for all six CUDA configs and the 4B CPU run (base: 18 to 19/20 for the 4B, 7 to 9/20 for the 1.7B). 4B Q8_0 NLL 0.450 → 0.228; 1.7B Q8_0 3.865 → 0.208. Perturbations 9/9 everywhere (base 1.7B: 4 to 5/9). On the 4B, Brier and ECE are slightly higher with flow (Q8_0 Brier 0.077 → 0.088, ECE 0.038 → 0.060) while NLL is lower. N is 20 and 9: one decision is 5.0 and 11.1 points.
- **Missing evidence is lower on the 4B, as on your machine.** 4B Q8_0: 28/36 → 21/36 correct, confident wrong (p ≥ 0.8) 6/36 → 5/36; BF16 27 → 21, Q4_K_M 25 → 21. No interval computed on 36 rows, so these are counts, not a tested difference. The 1.7B goes 29/36 → 26/36 (Q8_0), 2/36 → 1/36 confident wrong.
- **typed-decisions, 400 cases:** 4B Q8_0 0.574 → 0.651, +0.077 [+0.051, +0.102] (yours +0.074 [+0.050, +0.101]); KL 2.888 → 0.455, Brier 0.479 → 0.205, ECE 0.349 → 0.109 (yours 0.648 / 0.452 / 0.205 / 0.112). Q4_K_M 0.646 (yours 0.650). 1.7B Q8_0 0.530 → 0.543, +0.013 [−0.017, +0.043], no difference shown; KL 3.023 → 0.693. 4B BF16 +0.074 [+0.047, +0.099].
- **KV cache (flow 4B Q8_0, CUDA):** `q8_0` changes nothing: 0 of 252 rows, +0.000 on both fixtures. `q4_0`: authored144 −0.019 [−0.046, +0.000], perturbations108 −0.019 [−0.069, +0.022], 5 of 252 rows; no difference shown. Peak GPU memory (authored144 phase) 5.61 → 4.96 → 4.61 GiB, identical to the base runs; the tables read 5.88 → 5.01 → 4.62 at the end of the shape777 run (the figure is the drop in free GPU memory, which also sees desktop use; the f16 flow run is 0.27 GiB above its own earlier phases and above the base run). Shared dec/s 30.23 → 28.98 → 29.07. Only this weight and only 4B Q8_0 were run for flow.
- **SemIf and typed-decisions timing is the same as base on this machine.** 4B Q8_0 p50 36 ms (base 36), shared 30.23 dec/s (base 30.03), 1.7B Q8_0 18 ms and 50.20 dec/s. typed-decisions 142 ms per case (143). The two campaigns ran on different days, so this is two measurements, not a controlled pair; across all nine configurations the SemIf and typed-decisions p50 and shared dec/s differ by 3 % or less. The short smoke runs are noisier: medians differ by up to 5 %, and by 13 % for KV q8_0 (42 → 47 ms).
- **Against your flow run on the Windows machine:** 36 vs 66 ms p50 per decision, shared 30.23 vs 16.25 dec/s, typed-decisions 142 vs 195 ms per case. Peak GPU memory 5.61 vs 5.60 GiB (authored144 phase; 5.88 at the end of the shape777 run, see the KV note). That run and this one are different machines.
- **CPU-only (Ryzen 9 9950X3D, `--threads 16`):** 4B Q8_0 p50 721 ms per decision (base 727), 19.8× the CUDA run; 1.69 dec/s shared, 0.14 direct; typed-decisions 2976 ms per case (base 3008); smoke 877 ms median.
- **Shared vs direct:** 4B Q8_0 changes 3 of 777 argmaxes on CUDA (base 7 of 777 in the report), BF16 1 of 777, 4B CPU 0 of 63, 1.7B Q8_0 2 of 63. Counts have different denominators (37 vs 3 direct states).
- **`validate_checkpoint`:** all six configs ran and finished; long state (crossing the 512-token window) 0 changed argmaxes everywhere, max Δp 0.0041 to 0.0308.

**Reading notes.** Follow-up to [this report](https://github.com/Rizzo-AI-Academy/rizzo-flow/issues/25), run with the fine-tuned GGUF files (*flow*) instead of XHToken's original ones (*base*, `--weights base`, the numbers of that report). Scope: 39 of the 141 runs of that report, with the same run ids: the unit and integration suites (the integration one pinned to the 4B Q8_0); on CUDA, smoke, perturbations, `validate_checkpoint`, SemIf and typed-decisions for the 4B and the 1.7B at Q8_0, Q4_K_M and BF16; the `--kv-type q8_0` and `q4_0` runs (smoke, SemIf) of the 4B Q8_0; and its CPU-only smoke, SemIf and typed-decisions runs. Not repeated: Vulkan, `rizzo decide`, the device listing, the unpinned integration run, the batch-size and thread-count sweeps, the KV-cache runs of other weights and on the CPU, and the other CPU-only runs. A flow run and the base run of the same id are paired: same configuration and commands, other weights. Differences are flow − base; where an interval is shown, one that contains 0 is marked *includes 0*. The probabilities are not calibrated.

## Machine and weights

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D 16-Core Processor, 32 threads |
| RAM | 62 GiB usable — 2 × 32 GB DDR5 Crucial Pro CP32G64C40U5B (DDR5-5600 CL40 modules) configured at 6000 MT/s, dual channel |
| GPU | NVIDIA RTX PRO 4000 Blackwell, 24467 MiB, 615.71.09, 12.0 — also drives the desktop display |
| Ambient temperature | not measured on 30 September (29 °C at the start of the base campaign) |
| Driver / CUDA | 615.71.09 / CUDA 13.4 |
| OS | Debian GNU/Linux 13 (trixie), kernel 6.12.111+deb13-amd64 |
| Python | Python 3.14.7 (the repository pins 3.12; `.venv` was created with 3.14) |
| uv | uv 0.12.17 (x86_64-unknown-linux-gnu) |
| Commit | `b9ba007` (issue #25 base campaign: `c30cc63`) |
| llama.cpp | release `b11081`, official prebuilt packages `llama-b11081-linux-x64-cuda` and `-cpu`, as in issue #25 |

Differences from the base campaign's machine: os: `Debian GNU/Linux 13 (trixie), kernel 6.12.107+deb13-amd64` → `Debian GNU/Linux 13 (trixie), kernel 6.12.111+deb13-amd64`.

Code between the two commits (`c30cc63` → `b9ba007`), `src/`, `scripts/`, `benchmarks/`, `examples/`: 11 files changed (`scripts/record_demo.py`, `scripts/semif_compare.py`, `scripts/typed_decisions.py`, `scripts/validate_checkpoint.py`, `src/rizzo_flow/backend.py`, `src/rizzo_flow/backend_llama.py`, `src/rizzo_flow/cli.py`, `src/rizzo_flow/compat.py`, `src/rizzo_flow/config.py`, `src/rizzo_flow/llama_release.py`, `src/rizzo_flow/loader.py`); `prompts.py` unchanged; benchmark and example fixtures unchanged.

Weights: the fine-tuned GGUF files, `--weights flow` (the default at this commit), pinned in `rizzo_flow.config`. The base numbers are XHToken's original GGUF files (`--weights base`), from the base campaign.

| Size | Quant | Repository | Revision | File | sha256 |
|---|---|---|---|---|---|
| 4B | q8_0 | rizzoaiacademy/rizzo-flow | 55633c8cbd2b826bd3eefdeb05310450996649df | spark-x2.5-4b-rizzo-flow-lora-q8_0.gguf | dbec3c89d33984772324e65a8ed56b48e958e301b856cb870a7c1b385d01691a |
| 4B | bf16 | rizzoaiacademy/rizzo-flow | 55633c8cbd2b826bd3eefdeb05310450996649df | spark-x2.5-4b-rizzo-flow-lora-bf16.gguf | 8e0f9a73644d50fc319ef4205f64878f17a88ba8d4b875ffd3bb5e61597da865 |
| 4B | q4_k_m | rizzoaiacademy/rizzo-flow | 55633c8cbd2b826bd3eefdeb05310450996649df | spark-x2.5-4b-rizzo-flow-lora-q4_k_m.gguf | 79de5cb8dbfd1a1f5cb3037252251594352841fe5e3dc1ae8cead053010fcd54 |
| 1.7B | q8_0 | rizzoaiacademy/rizzo-flow-1.7b | 532e1586cc60787375ec7d86a5a302deb9a1adec | spark-x2.5-1.7b-rizzo-flow-lora-q8_0.gguf | 685403da9c62e0745cb77817ee2d66418e3ce85002ef403cf2680481e22bf16d |
| 1.7B | q4_k_m | rizzoaiacademy/rizzo-flow-1.7b | 532e1586cc60787375ec7d86a5a302deb9a1adec | spark-x2.5-1.7b-rizzo-flow-lora-q4_k_m.gguf | c2dab675c94226d548afb2d5027507a380c879adad2ef7a4e65289ff9e01123b |
| 1.7B | bf16 | rizzoaiacademy/rizzo-flow-1.7b | 532e1586cc60787375ec7d86a5a302deb9a1adec | spark-x2.5-1.7b-rizzo-flow-lora-bf16.gguf | 21bf261a4c15460d03172b522e1507152d21c31e167ca44ffe083f2f1893c756 |

## Machine state

| Item | Value |
|---|---|
| Idle gate | external CPU ≤ 1.0 threads, GPU utilization ≤ 25 %, no foreign CUDA compute process; a 5 s window before every run, 3 idle windows in a row after a wait |
| External CPU per run, whole run (threads) | median 0.23, max 0.70 (39 runs) |
| External CPU per run, timed window (threads) | median 0.23, max 0.96 (39 runs) |
| Runs that waited at the gate | 1 of 39 runs waited, longest 1.3 min; because of external CPU (1), GPU utilization (1) |
| Contaminated attempts | 1 (listed below) |
| Foreign GPU compute processes | recorded runs: none (1464 polls); thrown-away attempts: none; at the gate: none |

External CPU is the machine's busy CPU time (`/proc/stat`) minus the CPU time of the runner and its children (`getrusage`), in threads, from just before a run to after its telemetry queries; the timed window is the commands alone. A run over the limit, with a foreign CUDA compute process, or whose GPU could not be watched (`nvidia-smi` gave no answer for too long), is not recorded: its output is kept aside and the run is repeated once the machine is idle. Desktop graphics processes are not compute processes.

Contaminated attempts (not results; output kept in `results/local-hw-flow/_contaminated/`):
- `cuda-4b-q8_0-f16-T5` attempt 1: external CPU 1.68 threads over the whole run (27.5 CPU-s in 16 s; limit 1.0); external CPU 1.36 threads in the timed window (limit 1.0)

## Test suites

The unit tests load no weights. The integration run loads the 4B Q8_0 through a pytest plugin of the campaign: the fine-tuned file for the flow row (the plugin records the file and hash it loaded), the original file for the base row, which ran at the earlier commit.

| Suite | Campaign | Status | Passed | Failed | Skipped | Summary |
|---|---|---|---|---|---|---|
| unit tests | base | ok | 71 | 0 | 13 | `71 passed, 13 skipped, 2 warnings in 0.73s` |
| unit tests | flow | ok | 75 | 0 | 13 | `75 passed, 13 skipped, 2 warnings in 0.71s` |
| integration tests (pinned to the 4B Q8_0) | base | ok | 5 | 0 | 1 | `5 passed, 1 skipped, 78 deselected, 2 warnings in 8.29s` |
| integration tests (pinned to the 4B Q8_0) | flow | ok | 5 | 0 | 1 | `5 passed, 1 skipped, 82 deselected, 2 warnings in 8.23s` |

## SemIf fixtures, fine-tuned weights

SemIf's fixtures (`scripts/semif_compare.py`, SemIf's own `evaluate.py`), flow weights. Balanced accuracy is the mean over families; held-out = the odd source groups (`semif_report.py`); flips = argmax changes under option reversal, criterion wrapper and irrelevant context; missing evidence = the 36 rows where `insufficient` is the right answer, confident = a different answer with p ≥ 0.8; rule_application perturbed = balanced accuracy (NLL). Probabilities are not calibrated.

**Quality**

| Config | authored144 | perturbations108 | Held-out (authored144 / perturbations108) | Flips (reversal / wrapper / context) | Missing evidence: accuracy | Confident wrong (p ≥ 0.8) | rule_application perturbed |
|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | 0.845 | 0.946 | 0.809 / 0.949 | 1 / 1 / 1 | 21/36 (0.583) | 5/36 | 0.870 (0.35) |
| 4B BF16 · CUDA | 0.845 | 0.946 | 0.809 / 0.949 | 1 / 1 / 1 | 21/36 (0.583) | 5/36 | 0.870 (0.34) |
| 4B Q4_K_M · CUDA | 0.825 | 0.933 | 0.782 / 0.949 | 3 / 0 / 2 | 21/36 (0.583) | 6/36 | 0.833 (0.37) |
| 1.7B Q8_0 · CUDA | 0.794 | 0.783 | 0.716 / 0.778 | 11 / 5 / 1 | 26/36 (0.722) | 1/36 | 0.759 (0.80) |
| 1.7B Q4_K_M · CUDA | 0.812 | 0.770 | 0.799 / 0.736 | 10 / 5 / 1 | 27/36 (0.750) | 1/36 | 0.778 (0.69) |
| 1.7B BF16 · CUDA | 0.801 | 0.783 | 0.728 / 0.778 | 10 / 4 / 1 | 26/36 (0.722) | 2/36 | 0.759 (0.79) |
| 4B Q8_0 · CUDA · KV q8_0 | 0.845 | 0.946 | 0.809 / 0.949 | 1 / 1 / 1 | 21/36 (0.583) | 5/36 | 0.870 (0.33) |
| 4B Q8_0 · CUDA · KV q4_0 | 0.827 | 0.927 | 0.772 / 0.931 | 1 / 1 / 0 | 19/36 (0.528) | 4/36 | 0.815 (0.37) |
| 4B Q8_0 · CPU | 0.845 | 0.946 | 0.809 / 0.949 | 1 / 1 / 1 | 21/36 (0.583) | 5/36 | 0.870 (0.34) |
| *ref* maintainers, flow 4B Q8_0 CUDA (Windows 10 + RTX 5060 Ti) | 0.845 | 0.946 | 0.809 / 0.949 | 1 / 1 / 1 | 21/36 (0.583) | 5/36 | 0.870 (0.33) |

**Speed and memory**

Direct mode ran on the number of states in the *Direct states* column (37 states = 777 decisions, 3 states = 63 decisions): changed-argmax counts with different denominators are not comparable. Peak memory is rizzo's `peak_device_bytes` on the GPU (the drop in free GPU memory since before the load, so desktop use shows: the column is the shape777 phase, and the same run read lower in earlier phases, e.g. the 4B Q8_0 f16 flow run 5.61 GiB in authored144) and the process's peak RSS on the CPU.

| Config | p50 / p95 ms | shape777 shared / direct dec/s | Shared vs direct changed argmaxes | Direct states | Peak memory |
|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | 36 / 39 | 30.23 / 3.38 | 3/777 | 37 | 5.88 GiB |
| 4B BF16 · CUDA | 39 / 40 | 29.20 / 3.22 | 1/777 | 37 | 9.33 GiB |
| 4B Q4_K_M · CUDA | 37 / 39 | 28.84 / 3.14 | 3/63 | 3 | 4.00 GiB |
| 1.7B Q8_0 · CUDA | 18 / 19 | 50.20 / 7.89 | 2/63 | 3 | 2.39 GiB |
| 1.7B Q4_K_M · CUDA | 18 / 20 | 49.10 / 7.45 | 2/63 | 3 | 1.73 GiB |
| 1.7B BF16 · CUDA | 17 / 19 | 48.36 / 7.03 | 0/63 | 3 | 3.93 GiB |
| 4B Q8_0 · CUDA · KV q8_0 | 36 / 38 | 28.98 / 3.25 | 2/63 | 3 | 5.01 GiB |
| 4B Q8_0 · CUDA · KV q4_0 | 35 / 37 | 29.07 / 3.25 | 1/63 | 3 | 4.62 GiB |
| 4B Q8_0 · CPU | 721 / 804 | 1.69 / 0.14 | 0/63 | 3 | RSS 5.67 GiB |
| *ref* maintainers, flow 4B Q8_0 CUDA (Windows 10 + RTX 5060 Ti) | 66 / 79 | 16.25 / 1.92 | 1/63 | 3 | 5.60 GiB |

**Same weights, other machine** (this run minus the maintainers' flow run, `--against maintainers-flow`; their machine: Windows 10 + RTX 5060 Ti; direct states: 37 here, 3 there, not part of the difference, which uses the 252 rows of the quality fixtures)

| Config | authored144 difference [95 % CI] | perturbations108 difference [95 % CI] | Rows with a different argmax |
|---|---|---|---|
| 4B Q8_0 · CUDA | +0.000 [+0.000, +0.000] · includes 0 | +0.000 [+0.000, +0.000] · includes 0 | 0/252 |

## Fine-tuning effect on this machine

Each pair is the same run id in the base campaign (2026-09-28 to 2026-09-29, XHToken's GGUF) and in this one (fine-tuned GGUF): same configuration and direct-state count, other weights. Difference = flow − base, paired on SemIf's 252 rows (`--against this-base`). 95 % interval: paired source-group bootstrap stratified by family; represented gold classes per draw (1000 draws, seed 217, 36 source groups). An interval that contains 0 at the printed precision is marked *includes 0*; no difference is claimed for it. The timing columns of the second table come from different days (ambient temperature and desktop activity were not controlled), so they are two measurements, not a controlled pair. Commits: base `c30cc63`, flow `b9ba007` (see *Machine and weights* for the files that differ).

**Accuracy on SemIf's fixtures: base → flow**

| Config | authored144 | Difference [95 % CI] | perturbations108 | Difference [95 % CI] | Rows with a different argmax |
|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | 0.810 → 0.845 | +0.035 [-0.027, +0.100] · includes 0 | 0.830 → 0.946 | +0.116 [+0.044, +0.205] | 44/252 |
| 4B BF16 · CUDA | 0.819 → 0.845 | +0.026 [-0.039, +0.093] · includes 0 | 0.848 → 0.946 | +0.098 [+0.000, +0.206] · includes 0 | 43/252 |
| 4B Q4_K_M · CUDA | 0.765 → 0.825 | +0.059 [-0.004, +0.128] · includes 0 | 0.830 → 0.933 | +0.104 [+0.027, +0.191] | 54/252 |
| 1.7B Q8_0 · CUDA | 0.700 → 0.794 | +0.094 [+0.021, +0.163] | 0.646 → 0.783 | +0.137 [+0.052, +0.231] | 85/252 |
| 1.7B Q4_K_M · CUDA | 0.607 → 0.812 | +0.204 [+0.129, +0.280] | 0.472 → 0.770 | +0.298 [+0.176, +0.389] | 104/252 |
| 1.7B BF16 · CUDA | 0.683 → 0.801 | +0.118 [+0.039, +0.190] | 0.640 → 0.783 | +0.143 [+0.058, +0.241] | 87/252 |
| 4B Q8_0 · CUDA · KV q8_0 | 0.816 → 0.845 | +0.029 [-0.033, +0.094] · includes 0 | 0.835 → 0.946 | +0.111 [+0.045, +0.200] | 43/252 |
| 4B Q8_0 · CUDA · KV q4_0 | 0.811 → 0.827 | +0.016 [-0.051, +0.088] · includes 0 | 0.859 → 0.927 | +0.068 [-0.010, +0.155] · includes 0 | 50/252 |
| 4B Q8_0 · CPU | 0.816 → 0.845 | +0.029 [-0.033, +0.094] · includes 0 | 0.873 → 0.946 | +0.073 [-0.004, +0.158] · includes 0 | 41/252 |
| *ref* maintainers, 4B Q8_0 CUDA (Windows 10 + RTX 5060 Ti) | 0.812 → 0.845 | +0.033 [-0.033, +0.105] · includes 0 | 0.848 → 0.946 | +0.098 [+0.001, +0.204] | 44/252 |

**Missing evidence, stability, latency and throughput: base → flow**

| Config | Missing evidence: accuracy | Confident wrong (p ≥ 0.8) | Flips (reversal / wrapper / context) | rule_application perturbed | p50 ms | Shared dec/s |
|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | 28/36 (0.778) → 21/36 (0.583) | 6/36 → 5/36 | 6 / 5 / 3 → 1 / 1 / 1 | 0.556 (1.69) → 0.870 (0.35) | 36 → 36 | 30.03 → 30.23 |
| 4B BF16 · CUDA | 27/36 (0.750) → 21/36 (0.583) | 6/36 → 5/36 | 5 / 2 / 4 → 1 / 1 / 1 | 0.611 (1.69) → 0.870 (0.34) | 39 → 39 | 29.12 → 29.20 |
| 4B Q4_K_M · CUDA | 25/36 (0.694) → 21/36 (0.583) | 6/36 → 6/36 | 6 / 1 / 2 → 3 / 0 / 2 | 0.611 (1.82) → 0.833 (0.37) | 37 → 37 | 28.54 → 28.84 |
| 1.7B Q8_0 · CUDA | 29/36 (0.806) → 26/36 (0.722) | 2/36 → 1/36 | 18 / 5 / 4 → 11 / 5 / 1 | 0.315 (3.50) → 0.759 (0.80) | 18 → 18 | 49.70 → 50.20 |
| 1.7B Q4_K_M · CUDA | 26/36 (0.722) → 27/36 (0.750) | 6/36 → 1/36 | 17 / 4 / 2 → 10 / 5 / 1 | 0.056 (4.26) → 0.778 (0.69) | 18 → 18 | 48.65 → 49.10 |
| 1.7B BF16 · CUDA | 28/36 (0.778) → 26/36 (0.722) | 2/36 → 2/36 | 18 / 4 / 4 → 10 / 4 / 1 | 0.296 (3.55) → 0.759 (0.79) | 17 → 17 | 48.40 → 48.36 |
| 4B Q8_0 · CUDA · KV q8_0 | 28/36 (0.778) → 21/36 (0.583) | 6/36 → 5/36 | 4 / 3 / 3 → 1 / 1 / 1 | 0.537 (1.69) → 0.870 (0.33) | 35 → 36 | 29.54 → 28.98 |
| 4B Q8_0 · CUDA · KV q4_0 | 28/36 (0.778) → 19/36 (0.528) | 7/36 → 4/36 | 4 / 5 / 3 → 1 / 1 / 0 | 0.611 (1.58) → 0.815 (0.37) | 34 → 35 | 29.56 → 29.07 |
| 4B Q8_0 · CPU | 28/36 (0.778) → 21/36 (0.583) | 6/36 → 5/36 | 3 / 3 / 5 → 1 / 1 / 1 | 0.685 (1.65) → 0.870 (0.34) | 727 → 721 | 1.68 → 1.69 |
| *ref* maintainers, 4B Q8_0 CUDA (Windows 10 + RTX 5060 Ti) | 27/36 (0.750) → 21/36 (0.583) | 6/36 → 5/36 | 5 / 4 / 5 → 1 / 1 / 1 | 0.611 (1.69) → 0.870 (0.33) | 66 → 66 | 15.64 → 16.25 |

The *ref* row is the maintainers' own base and flow runs on their machine (`analysis.json` of the flow run). Its timing cells pair their flow run with the base weights run they repeated right after it on the same machine (README.md, Results so far, footnote 1: 66 / 73 ms, 15.64 dec/s shared), not with their published base run (49 ms, 20.99 dec/s shared): the footnote notes that the machine's speed was not the same on the two days, so those two runs are not a like-for-like timing pair.

## Native benchmarks: smoke and perturbations

`rizzo evaluate benchmarks/{smoke,perturbations}.jsonl --compare-modes`, same command for both weights. Accuracy is n of N labelled decisions; NLL, Brier and ECE are computed on the model's own probabilities (not calibrated); *shared vs direct* = argmax changes between the prefix-sharing and the fresh-prefill mode.

**Smoke (`benchmarks/smoke.jsonl`)** — N = 20 labelled decisions, one decision = 0.050 of accuracy

| Config | Weights | Accuracy | NLL | Brier | ECE | Median / p95 ms | dec/s | Shared vs direct changed argmaxes | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | base | 19/20 (0.950) | 0.450 | 0.077 | 0.038 | 43 / 118 | 22.46 | 0/20 | 5.61 GiB |
| 4B Q8_0 · CUDA | flow | 19/20 (0.950) | 0.228 | 0.088 | 0.060 | 41 / 120 | 23.60 | 0/20 | 5.61 GiB |
| 4B BF16 · CUDA | base | 19/20 (0.950) | 0.449 | 0.078 | 0.040 | 42 / 134 | 23.14 | 0/20 | 9.29 GiB |
| 4B BF16 · CUDA | flow | 19/20 (0.950) | 0.238 | 0.087 | 0.060 | 43 / 134 | 22.92 | 0/20 | 9.29 GiB |
| 4B Q4_K_M · CUDA | base | 18/20 (0.900) | 0.735 | 0.191 | 0.075 | 43 / 121 | 22.52 | 0/20 | 3.96 GiB |
| 4B Q4_K_M · CUDA | flow | 19/20 (0.950) | 0.225 | 0.088 | 0.058 | 42 / 123 | 22.51 | 0/20 | 3.96 GiB |
| 1.7B Q8_0 · CUDA | base | 8/20 (0.400) | 3.865 | 1.097 | 0.528 | 22 / 61 | 46.95 | 0/20 | 2.35 GiB |
| 1.7B Q8_0 · CUDA | flow | 19/20 (0.950) | 0.208 | 0.094 | 0.052 | 22 / 61 | 47.74 | 0/20 | 2.35 GiB |
| 1.7B Q4_K_M · CUDA | base | 7/20 (0.350) | 3.584 | 1.071 | 0.578 | 23 / 62 | 46.64 | 0/20 | 1.68 GiB |
| 1.7B Q4_K_M · CUDA | flow | 19/20 (0.950) | 0.337 | 0.144 | 0.094 | 23 / 62 | 47.14 | 1/20 | 1.68 GiB |
| 1.7B BF16 · CUDA | base | 9/20 (0.450) | 3.462 | 1.014 | 0.512 | 21 / 72 | 46.81 | 0/20 | 3.93 GiB |
| 1.7B BF16 · CUDA | flow | 19/20 (0.950) | 0.203 | 0.091 | 0.047 | 21 / 76 | 46.11 | 0/20 | 3.93 GiB |
| 4B Q8_0 · CUDA · KV q8_0 | base | 19/20 (0.950) | 0.445 | 0.077 | 0.038 | 42 / 122 | 23.54 | 0/20 | 4.97 GiB |
| 4B Q8_0 · CUDA · KV q8_0 | flow | 19/20 (0.950) | 0.230 | 0.087 | 0.060 | 47 / 122 | 23.36 | 0/20 | 4.97 GiB |
| 4B Q8_0 · CUDA · KV q4_0 | base | 19/20 (0.950) | 0.549 | 0.099 | 0.051 | 40 / 119 | 24.45 | 0/20 | 4.62 GiB |
| 4B Q8_0 · CUDA · KV q4_0 | flow | 19/20 (0.950) | 0.166 | 0.083 | 0.051 | 39 / 121 | 23.93 | 0/20 | 4.62 GiB |
| 4B Q8_0 · CPU | base | 19/20 (0.950) | 0.429 | 0.077 | 0.037 | 922 / 2580 | 1.07 | 0/20 | RSS 5.66 GiB |
| 4B Q8_0 · CPU | flow | 19/20 (0.950) | 0.224 | 0.088 | 0.060 | 877 / 2439 | 1.09 | 0/20 | RSS 5.66 GiB |
| *ref* maintainers, 4B Q8_0 CUDA (Windows 10 + RTX 5060 Ti) | base | 19/20 (0.950) | 0.459 | 0.078 | 0.039 | 66 / 169 | 16.81 | 0/20 | 5.60 GiB |

**Perturbations (`benchmarks/perturbations.jsonl`)** — N = 9 labelled decisions, one decision = 0.111 of accuracy

| Config | Weights | Accuracy | NLL | Brier | ECE | Median / p95 ms | dec/s | Shared vs direct changed argmaxes | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | base | 9/9 (1.000) | 0.020 | 0.003 | 0.019 | 40 / 43 | 25.10 | 0/9 | 5.57 GiB |
| 4B Q8_0 · CUDA | flow | 9/9 (1.000) | 0.011 | 0.000 | 0.011 | 41 / 45 | 24.66 | 0/9 | 5.61 GiB |
| 4B BF16 · CUDA | base | 9/9 (1.000) | 0.018 | 0.002 | 0.018 | 40 / 45 | 24.06 | 0/9 | 9.29 GiB |
| 4B BF16 · CUDA | flow | 9/9 (1.000) | 0.010 | 0.000 | 0.010 | 41 / 46 | 24.00 | 0/9 | 9.33 GiB |
| 4B Q4_K_M · CUDA | base | 9/9 (1.000) | 0.012 | 0.001 | 0.012 | 40 / 43 | 24.65 | 0/9 | 3.96 GiB |
| 4B Q4_K_M · CUDA | flow | 9/9 (1.000) | 0.012 | 0.000 | 0.012 | 40 / 43 | 24.66 | 0/9 | 3.96 GiB |
| 1.7B Q8_0 · CUDA | base | 4/9 (0.444) | 3.126 | 0.960 | 0.551 | 19 / 22 | 52.47 | 0/9 | 2.35 GiB |
| 1.7B Q8_0 · CUDA | flow | 9/9 (1.000) | 0.004 | 0.000 | 0.004 | 19 / 22 | 51.99 | 0/9 | 2.35 GiB |
| 1.7B Q4_K_M · CUDA | base | 5/9 (0.556) | 2.412 | 0.895 | 0.520 | 20 / 23 | 50.54 | 0/9 | 1.68 GiB |
| 1.7B Q4_K_M · CUDA | flow | 9/9 (1.000) | 0.015 | 0.001 | 0.014 | 20 / 23 | 49.30 | 0/9 | 1.68 GiB |
| 1.7B BF16 · CUDA | base | 5/9 (0.556) | 2.759 | 0.871 | 0.499 | 19 / 22 | 52.39 | 0/9 | 3.93 GiB |
| 1.7B BF16 · CUDA | flow | 9/9 (1.000) | 0.005 | 0.000 | 0.005 | 19 / 22 | 52.09 | 0/9 | 3.93 GiB |
| *ref* maintainers, 4B Q8_0 CUDA (Windows 10 + RTX 5060 Ti) | base | 9/9 (1.000) | 0.018 | 0.002 | 0.017 | 52 / 57 | 18.94 | 0/9 | 5.60 GiB |

The *ref* rows are the maintainers' published base-weight run (`results/llama-q8_0-cuda-validation`), not a flow measurement.

## typed-decisions

Test split of `LocalLLaMA/typed-decisions`, config `all` (400 cases, 2,000 decisions), replayed as `/v1/systemone` request bodies, in process (no HTTP layer), by `scripts/typed_decisions.py`, the same command for both weights. Base = zero-shot original GGUF; flow = the fine-tune. The *ref* rows are the maintainers' published numbers (Windows 10 + RTX 5060 Ti), parsed from the file named in the row (README.md, Results on typed-decisions; docs/training.md, section 10) and not re-measured here; where no row is shown, nothing is published for that configuration. KL, Brier and ECE are computed on uncalibrated probabilities.

| Config | Weights / source | Accuracy | KL | Brier | ECE | p50 / p95 ms per case | dec/s |
|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | base, this machine | 0.574 | 2.888 | 0.479 | 0.349 | 143 / 180 | 34.44 |
| 4B Q8_0 · CUDA | flow, this machine | 0.651 | 0.455 | 0.205 | 0.109 | 142 / 178 | 34.73 |
| 4B Q8_0 · CUDA | *ref* base, maintainers (README.md) | 0.574 | 2.899 | 0.480 | 0.349 | 201 / — | — |
| 4B Q8_0 · CUDA | *ref* flow, maintainers (README.md) | 0.648 | 0.452 | 0.205 | 0.112 | 195 / — | — |
| 4B BF16 · CUDA | base, this machine | 0.575 | 2.935 | 0.479 | 0.348 | 148 / 186 | 33.43 |
| 4B BF16 · CUDA | flow, this machine | 0.649 | 0.456 | 0.206 | 0.110 | 147 / 186 | 33.66 |
| 4B BF16 · CUDA | *ref* base, maintainers (docs/training.md §10) | 0.574 | 2.935 | 0.479 | — | 250 / — | — |
| 4B Q4_K_M · CUDA | base, this machine | 0.574 | 2.321 | 0.445 | 0.324 | 151 / 189 | 32.92 |
| 4B Q4_K_M · CUDA | flow, this machine | 0.646 | 0.437 | 0.202 | 0.095 | 150 / 187 | 33.22 |
| 4B Q4_K_M · CUDA | *ref* flow, maintainers (README.md) | 0.650 | 0.436 | 0.201 | 0.093 | 198 / — | — |
| 1.7B Q8_0 · CUDA | base, this machine | 0.530 | 3.023 | 0.496 | 0.346 | 72 / 90 | 69.36 |
| 1.7B Q8_0 · CUDA | flow, this machine | 0.543 | 0.693 | 0.275 | 0.169 | 71 / 88 | 70.37 |
| 1.7B Q8_0 · CUDA | *ref* base, maintainers (README.md) | 0.530 | 3.031 | 0.496 | 0.348 | 117 / — | — |
| 1.7B Q8_0 · CUDA | *ref* flow, maintainers (README.md) | 0.544 | 0.694 | 0.275 | 0.169 | 109 / — | — |
| 1.7B Q4_K_M · CUDA | base, this machine | 0.481 | 2.799 | 0.579 | 0.427 | 74 / 92 | 67.43 |
| 1.7B Q4_K_M · CUDA | flow, this machine | 0.487 | 0.639 | 0.278 | 0.193 | 73 / 91 | 68.09 |
| 1.7B Q4_K_M · CUDA | *ref* flow, maintainers (README.md) | 0.490 | 0.640 | 0.279 | 0.192 | 107 / — | — |
| 1.7B BF16 · CUDA | base, this machine | 0.528 | 3.019 | 0.494 | 0.348 | 73 / 92 | 68.20 |
| 1.7B BF16 · CUDA | flow, this machine | 0.545 | 0.690 | 0.273 | 0.168 | 72 / 92 | 68.82 |
| 4B Q8_0 · CPU | base, this machine | 0.576 | 2.902 | 0.480 | 0.348 | 3008 / 3711 | 1.68 |
| 4B Q8_0 · CPU | flow, this machine | 0.650 | 0.453 | 0.205 | 0.110 | 2976 / 3674 | 1.70 |

**Difference flow − base on this machine**

Accuracy: paired bootstrap over cases, 1000 draws, seed 217, computed from the two raw reports and the test split (`—` when either run is missing). KL, Brier and ECE: point differences, no interval.

| Config | Accuracy difference [95 % CI] | KL | Brier | ECE |
|---|---|---|---|---|
| 4B Q8_0 · CUDA | +0.077 [+0.051, +0.102] | -2.433 | -0.274 | -0.240 |
| 4B BF16 · CUDA | +0.074 [+0.047, +0.099] | -2.480 | -0.273 | -0.238 |
| 4B Q4_K_M · CUDA | +0.073 [+0.045, +0.097] | -1.884 | -0.243 | -0.230 |
| 1.7B Q8_0 · CUDA | +0.013 [-0.017, +0.043] · includes 0 | -2.330 | -0.221 | -0.178 |
| 1.7B Q4_K_M · CUDA | +0.006 [-0.023, +0.035] · includes 0 | -2.160 | -0.300 | -0.234 |
| 1.7B BF16 · CUDA | +0.017 [-0.012, +0.046] · includes 0 | -2.329 | -0.221 | -0.181 |
| 4B Q8_0 · CPU | +0.074 [+0.048, +0.098] | -2.449 | -0.275 | -0.238 |

Maintainers' paired difference, 4B Q8_0, base → flow (Windows 10 + RTX 5060 Ti): accuracy +0.074 [+0.050, +0.101] (README.md, Results on typed-decisions).

## `scripts/validate_checkpoint.py`

`scripts/validate_checkpoint.py`: smoke, perturbations and a long shared state (crossing Spark's 512-token attention window) evaluated in process on the engine, in shared and direct mode, plus one example request through the HTTP app; the timings have no HTTP layer. Status is the runner's: `ok` = the tool produced its summary.

| Config | Weights | Status | Smoke accuracy, NLL, changed argmaxes shared vs direct | Perturbations accuracy | Long state shared / direct ms | Long state changed argmaxes | Validation s | Load s | Peak memory |
|---|---|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | base | ok | 0.950, NLL 0.450, 0 changed | 1.000 | 345 / 1014 | 0 (max Δp 0.011618) | 4.7 | 2.5 | 5.61 GiB |
| 4B Q8_0 · CUDA | flow | ok | 0.950, NLL 0.228, 0 changed | 1.000 | 343 / 1009 | 0 (max Δp 0.009767) | 4.7 | 2.3 | 5.61 GiB |
| 4B BF16 · CUDA | base | ok | 0.950, NLL 0.449, 0 changed | 1.000 | 354 / 1005 | 0 (max Δp 0.008014) | 4.8 | 4.1 | 9.29 GiB |
| 4B BF16 · CUDA | flow | ok | 0.950, NLL 0.238, 0 changed | 1.000 | 354 / 1006 | 0 (max Δp 0.004087) | 4.7 | 4.0 | 9.29 GiB |
| 4B Q4_K_M · CUDA | base | ok | 0.900, NLL 0.735, 0 changed | 1.000 | 355 / 1049 | 0 (max Δp 0.021567) | 4.8 | 1.6 | 3.96 GiB |
| 4B Q4_K_M · CUDA | flow | ok | 0.950, NLL 0.225, 0 changed | 1.000 | 353 / 1041 | 0 (max Δp 0.004020) | 4.8 | 1.6 | 3.96 GiB |
| 1.7B Q8_0 · CUDA | base | ok | 0.400, NLL 3.865, 0 changed | 0.444 | 169 / 443 | 0 (max Δp 0.005816) | 2.3 | 1.2 | 2.35 GiB |
| 1.7B Q8_0 · CUDA | flow | ok | 0.950, NLL 0.208, 0 changed | 1.000 | 164 / 424 | 0 (max Δp 0.017120) | 2.2 | 1.2 | 2.35 GiB |
| 1.7B Q4_K_M · CUDA | base | ok | 0.350, NLL 3.584, 0 changed | 0.556 | 171 / 466 | 0 (max Δp 0.060829) | 2.3 | 0.9 | 1.69 GiB |
| 1.7B Q4_K_M · CUDA | flow | ok | 0.950, NLL 0.337, 1 changed | 1.000 | 169 / 449 | 0 (max Δp 0.030816) | 2.3 | 0.9 | 1.69 GiB |
| 1.7B BF16 · CUDA | base | ok | 0.450, NLL 3.462, 0 changed | 0.556 | 183 / 482 | 0 (max Δp 0.012355) | 2.3 | 1.9 | 3.93 GiB |
| 1.7B BF16 · CUDA | flow | ok | 0.950, NLL 0.203, 0 changed | 1.000 | 180 / 473 | 0 (max Δp 0.018632) | 2.3 | 1.9 | 3.93 GiB |
| *ref* maintainers, 4B Q8_0 CUDA (Windows 10 + RTX 5060 Ti) | base | published | 0.950, NLL 0.459, 0 changed | 1.000 | 471 / 1297 | 0 (max Δp 0.011680) | 6.3 | 12.0 | 5.60 GiB |

## KV cache on the fine-tuned 4B Q8_0

4B Q8_0 on CUDA, `--kv-type f16` (the default) / `q8_0` / `q4_0`, same weights within a block. Differences are against the f16 run of the same weights (`--against this-cuda-kv-f16`, SemIf's 252 rows). The f16 run used 37 direct states and the quantized-KV run 3; the count does not enter these differences, which use only the 252 rows of the quality fixtures. The base block is the base campaign's own KV runs.

| Weights | KV type | Smoke accuracy, NLL | authored144 | Difference vs f16 [95 % CI] | perturbations108 | Difference vs f16 [95 % CI] | Rows with a different argmax vs f16 | Peak memory | Shared dec/s |
|---|---|---|---|---|---|---|---|---|---|
| flow | f16 | 19/20 (0.950), NLL 0.228 | 0.845 | reference | 0.946 | reference | reference | 5.88 GiB | 30.23 |
| flow | q8_0 | 19/20 (0.950), NLL 0.230 | 0.845 | +0.000 [+0.000, +0.000] · includes 0 | 0.946 | +0.000 [+0.000, +0.000] · includes 0 | 0/252 | 5.01 GiB | 28.98 |
| flow | q4_0 | 19/20 (0.950), NLL 0.166 | 0.827 | -0.019 [-0.046, +0.000] · includes 0 | 0.927 | -0.019 [-0.069, +0.022] · includes 0 | 5/252 | 4.62 GiB | 29.07 |
| base | f16 | 19/20 (0.950), NLL 0.450 | 0.810 | reference | 0.830 | reference | reference | 5.61 GiB | 30.03 |
| base | q8_0 | 19/20 (0.950), NLL 0.445 | 0.816 | +0.006 [+0.000, +0.020] · includes 0 | 0.835 | +0.005 [-0.017, +0.030] · includes 0 | 4/252 | 4.97 GiB | 29.54 |
| base | q4_0 | 19/20 (0.950), NLL 0.549 | 0.811 | +0.001 [-0.030, +0.035] · includes 0 | 0.859 | +0.030 [+0.000, +0.071] · includes 0 | 10/252 | 4.62 GiB | 29.56 |

## CPU-only

AMD Ryzen 9 9950X3D 16-Core Processor, 32 threads, `cpu` package, `--threads 16` (the count of the base campaign's CPU runs; T7 and T9 get it through the campaign's `threads_wrapper.py`, since those scripts have no `--threads` flag). Differences in the T7 table are against the CUDA run of the same weights and precision on the same side of the pair (flow rows: the flow CUDA run; base rows: the base campaign's CUDA run; `--against this-cuda`, SemIf's 252 rows); the *p50 vs CUDA* columns are the ratio of the CPU run's median to that CUDA run's, also per side. The CUDA run used 37 direct states and the CPU run 3; the count does not enter these differences, which use only the 252 rows of the quality fixtures.

**T4 smoke**

| Config | Weights | Accuracy, NLL | Median / p95 ms | dec/s | Shared vs direct changed argmaxes | Peak memory |
|---|---|---|---|---|---|---|
| 4B Q8_0 · CPU | base | 19/20 (0.950), NLL 0.429 | 922 / 2580 | 1.07 | 0/20 | RSS 5.66 GiB |
| 4B Q8_0 · CPU | flow | 19/20 (0.950), NLL 0.224 | 877 / 2439 | 1.09 | 0/20 | RSS 5.66 GiB |

**T7 SemIf fixtures**

| Config | Weights | authored144 | perturbations108 | authored144 difference vs CUDA [95 % CI] | perturbations108 difference vs CUDA [95 % CI] | Rows with a different argmax vs CUDA | p50 / p95 ms | p50 vs CUDA | shape777 shared / direct dec/s | Shared vs direct changed argmaxes | Peak memory |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CPU | base | 0.816 | 0.873 | +0.006 [+0.000, +0.020] · includes 0 | +0.043 [+0.000, +0.071] · includes 0 | 5/252 | 727 / 818 | ×20.1 | 1.68 / 0.14 | 1/63 | RSS 5.67 GiB |
| 4B Q8_0 · CPU | flow | 0.845 | 0.946 | +0.000 [+0.000, +0.000] · includes 0 | +0.000 [+0.000, +0.000] · includes 0 | 0/252 | 721 / 804 | ×19.8 | 1.69 / 0.14 | 0/63 | RSS 5.67 GiB |

**T9 typed-decisions**

| Config | Weights | Accuracy | KL | Brier | ECE | p50 / p95 ms per case | p50 vs CUDA | dec/s |
|---|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CPU | base | 0.576 | 2.902 | 0.480 | 0.348 | 3008 / 3711 | ×21.0 | 1.68 |
| 4B Q8_0 · CPU | flow | 0.650 | 0.453 | 0.205 | 0.110 | 2976 / 3674 | ×20.9 | 1.70 |

<details><summary>Telemetry of the SemIf runs</summary>

GPU: `nvidia-smi` at 100 ms (power, energy, temperature) and DCGM at 1 s (SM activity, power-cap throttling); CPU: the child's rusage. Net energy = integral of GPU power over the run − 10 s idle baseline power × duration (nvidia-smi, then DCGM); each window covers the whole command, model loading included. The GPU also drives the desktop, so the baseline is not zero. Base and flow runs are from different days.

| Config | Weights | Wall s | GPU avg / peak W | Idle baseline W | Net GPU energy kJ (nvidia-smi / DCGM) | SM active (DCGM) | Power-cap throttle s | Temp max °C | CPU cores avg / peak RSS GiB |
|---|---|---|---|---|---|---|---|---|---|
| 4B Q8_0 · CUDA | base | 275.7 | 141.2 / 180.1 | 27.9 | 31.258 / 31.135 | 0.77 | 234.4 | 87 | 1.00 / 4.53 |
| 4B Q8_0 · CUDA | flow | 275.0 | 141.6 / 177.3 | 28.9 | 30.997 / 30.677 | 0.77 | 231.1 | 87 | 1.00 / 4.54 |
| 4B BF16 · CUDA | base | 290.0 | 141.3 / 177.4 | 28.7 | 32.652 / 32.215 | 0.79 | 251.4 | 87 | 1.00 / 8.13 |
| 4B BF16 · CUDA | flow | 289.4 | 141.2 / 175.1 | 30.5 | 32.025 / 31.588 | 0.80 | 251.0 | 87 | 1.00 / 8.13 |
| 4B Q4_K_M · CUDA | base | 66.4 | 133.1 / 180.4 | 28.1 | 6.968 / 6.860 | 0.67 | 45.3 | 85 | 1.00 / 2.88 |
| 4B Q4_K_M · CUDA | flow | 65.9 | 133.2 / 177.9 | 31.6 | 6.687 / 6.642 | 0.68 | 46.1 | 85 | 1.00 / 2.88 |
| 1.7B Q8_0 · CUDA | base | 32.9 | 119.8 / 177.5 | 26.7 | 3.062 / 2.885 | 0.48 | 9.8 | 76 | 1.00 / 2.16 |
| 1.7B Q8_0 · CUDA | flow | 32.6 | 121.8 / 178.2 | 30.5 | 2.973 / 2.842 | 0.48 | 10.9 | 79 | 1.00 / 2.16 |
| 1.7B Q4_K_M · CUDA | base | 33.6 | 122.5 / 179.9 | 29.0 | 3.146 / 3.068 | 0.49 | 11.1 | 80 | 1.00 / 1.49 |
| 1.7B Q4_K_M · CUDA | flow | 33.5 | 119.1 / 178.5 | 31.0 | 2.944 / 2.957 | 0.50 | 11.3 | 79 | 1.00 / 1.49 |
| 1.7B BF16 · CUDA | base | 35.0 | 120.4 / 179.8 | 27.0 | 3.273 / 3.054 | 0.49 | 14.0 | 78 | 1.00 / 3.65 |
| 1.7B BF16 · CUDA | flow | 35.1 | 119.9 / 174.8 | 30.1 | 3.155 / 2.994 | 0.50 | 15.1 | 78 | 1.00 / 3.65 |
| 4B Q8_0 · CUDA · KV q8_0 | base | 64.2 | 129.4 / 177.0 | 23.1 | 6.812 / 6.631 | 0.63 | 44.4 | 84 | 1.00 / 4.53 |
| 4B Q8_0 · CUDA · KV q8_0 | flow | 65.6 | 131.1 / 175.6 | 31.0 | 6.550 / 6.433 | 0.64 | 44.5 | 84 | 1.00 / 4.54 |
| 4B Q8_0 · CUDA · KV q4_0 | base | 64.0 | 130.2 / 179.6 | 18.1 | 7.181 / 7.043 | 0.64 | 44.6 | 84 | 1.00 / 4.53 |
| 4B Q8_0 · CUDA · KV q4_0 | flow | 65.1 | 131.2 / 179.1 | 31.6 | 6.477 / 6.338 | 0.64 | 44.4 | 84 | 1.00 / 4.53 |
| 4B Q8_0 · CPU | base | 1292.3 | 6.2 / 17.9 | 6.6 | — (GPU idle) | 0.00 | 0.0 | 38 | 15.63 / 5.67 |
| 4B Q8_0 · CPU | flow | 1281.7 | 19.8 / 50.2 | 31.0 | — (GPU idle) | 0.03 | 0.0 | 53 | 15.64 / 5.67 |

</details>

## Commands and data

```bash
uv sync --extra test --locked
uv run rizzo download --weights flow           # CUDA runtime + 4B Q8_0 flow GGUF
uv run rizzo download --weights flow --only weights --quant q4_k_m   # and bf16
uv run rizzo download --weights flow --only weights --size 1.7b --quant q8_0   # and q4_k_m, bf16
uv run rizzo download --only runtime --runtime cpu   # CPU-only package, for --device cpu
uv run rizzo devices
# unit tests — as run for T0-unit
pytest -q
# integration tests — as run for T1-integration-4b
RIZZO_REAL=1 PYTHONPATH=.research/hw-campaign RIZZO_HW_EVIDENCE=results/local-hw-flow/T1-integration-4b/weights-evidence.json pytest -q -m integration -p pin_4b_flow_plugin
# smoke, CUDA — as run for cuda-4b-q8_0-f16-T4
rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --weights flow --device cuda --output results/local-hw-flow/cuda-4b-q8_0-f16-T4/smoke.json
# perturbations, CUDA — as run for cuda-4b-q8_0-f16-T5
rizzo evaluate benchmarks/perturbations.jsonl --compare-modes --size 4b --quant q8_0 --weights flow --device cuda --output results/local-hw-flow/cuda-4b-q8_0-f16-T5/perturbations.json
# validate_checkpoint, CUDA — as run for cuda-4b-q8_0-f16-T6
python scripts/validate_checkpoint.py --size 4b --quant q8_0 --weights flow --device cuda --output results/local-hw-flow/cuda-4b-q8_0-f16-T6/validate
# semif_compare, CUDA — as run for cuda-4b-q8_0-f16-T7
python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --weights flow --device cuda --output results/local-hw-flow/cuda-4b-q8_0-f16-T7/semif --direct-states 37
# typed-decisions, CUDA — as run for cuda-4b-q8_0-f16-T9
python scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --size 4b --quant q8_0 --weights flow --device cuda --output results/local-hw-flow/cuda-4b-q8_0-f16-T9/typed.json
# smoke, KV cache runs (--kv-type) — as run for cuda-4b-q8_0-q8_0-T4
rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --weights flow --device cuda --kv-type q8_0 --output results/local-hw-flow/cuda-4b-q8_0-q8_0-T4/smoke.json
# semif_compare, KV cache runs (--kv-type) — as run for cuda-4b-q8_0-q8_0-T7
python scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --weights flow --device cuda --kv-type q8_0 --output results/local-hw-flow/cuda-4b-q8_0-q8_0-T7/semif
# smoke, CPU-only — as run for cpu-4b-q8_0-f16-T4
rizzo evaluate benchmarks/smoke.jsonl --compare-modes --size 4b --quant q8_0 --weights flow --device cpu --threads 16 --output results/local-hw-flow/cpu-4b-q8_0-f16-T4/smoke.json
# semif_compare, CPU-only — as run for cpu-4b-q8_0-f16-T7
RIZZO_HW_THREADS=16 python .research/hw-campaign/threads_wrapper.py scripts/semif_compare.py --system rizzo --semif .research/SemIf --size 4b --quant q8_0 --weights flow --device cpu --output results/local-hw-flow/cpu-4b-q8_0-f16-T7/semif
# typed-decisions, CPU-only — as run for cpu-4b-q8_0-f16-T9
RIZZO_HW_THREADS=16 python .research/hw-campaign/threads_wrapper.py scripts/typed_decisions.py .research/typed-decisions/all/test.jsonl --size 4b --quant q8_0 --weights flow --device cpu --output results/local-hw-flow/cpu-4b-q8_0-f16-T9/typed.json
# After every T7 run the runner calls semif_report.py; --against blocks per run:
#   maintainers-flow  cuda-4b-q8_0-f16-T7 only: results/semif-compare/rizzo-flow-q8_0-v3-llama-cuda
#   this-base         every flow T7 run: the base run of the same id (results/local-hw/ID/semif)
#   this-cuda-kv-f16  the KV runs: results/local-hw-flow/cuda-4b-q8_0-f16-T7/semif
#   this-cuda         the CPU run: results/local-hw-flow/cuda-4b-q8_0-f16-T7/semif
uv run python scripts/semif_report.py results/local-hw-flow/ID/semif --semif .research/SemIf \
  --against this-base=results/local-hw/ID/semif
```

Every command loads the model again; runs were strictly sequential (one model process at a time). Other configurations differ only in `--size`, `--quant`, `--device`, `--kv-type` and `--threads`. Some commands use campaign scripts that are not part of the repository (`.research/` is git-ignored): `.research/hw-campaign/pin_4b_flow_plugin.py` (pytest plugin: pins the integration suite to the 4B Q8_0 and records what it loaded) and `.research/hw-campaign/threads_wrapper.py` (sets `--threads` for the CPU-only SemIf and typed-decisions runs, whose scripts have no such flag). T7 runs with an explicit `--direct-states`: `cuda-4b-bf16-f16-T7` (37), `cuda-4b-q8_0-f16-T7` (37); the others use the script default, as in the base campaign.

Data and raw outputs of this follow-up: https://github.com/Axiumine/rizzo-flow/tree/hardware-report-issue-25/flow-followup. Base campaign data: https://github.com/Axiumine/rizzo-flow/tree/hardware-report-issue-25.

## Anything odd

- **One contaminated attempt, thrown away.** `cuda-4b-q8_0-f16-T5` (perturbations) was first run while other workload on the desktop (an IDE) was using CPU: 1.68 external threads over the run (27.5 CPU-s in 16 s) and 1.36 in the timed window, against a limit of 1.0. The idle gate caught it, the output went to `results/local-hw-flow/_contaminated/`, and the run was repeated once the machine was idle. It is not in any table. The repeated run is clean, and the gate allows up to 3 attempts.
- **Idle gate:** external CPU (machine busy time minus the runner's own) ≤ 1.0 threads, GPU utilization ≤ 25 %, no foreign CUDA compute process; a 5 s window before each run, 3 idle windows in a row after a wait, retry every 60 s. One run waited: the first (`T0-unit`), 1.3 min, at 3.21 external threads and 27 % GPU utilization. Over the 39 recorded runs external CPU was median 0.23 and max 0.70 threads; in the timed window median 0.23, max 0.96 (limit 1.0). No foreign GPU compute process in any of 1464 polls.
- **Exit codes:** all 39 runs exited 0. No weights-guard failures.
- **stderr:** empty in every run except the six `validate_checkpoint` runs, which print one `StarletteDeprecationWarning` (`httpx` with `starlette.testclient`, suggests installing `httpx2`). Python 3.14 is used here though the repository pins 3.12. Unit tests: 75 passed, 13 skipped, vs 71 + 13 at the base commit (4 more tests in `b9ba007`). The 13 skips are 7 `test_mlx.py` tests (the `mlx` extra is not installed here) and 6 real-model tests (`RIZZO_REAL=1`); with MLX installed the suite is 82 passed + 6 skipped, the number in your CLAUDE.md.
- **Power limit:** same as the base report. The board limit is 145 W and the GPU was power-throttled for 231 of 275 s in the 4B Q8_0 SemIf run (87 °C peak); the GPU numbers are power-limited.
- **Base and flow ran on different days**, and ambient temperature was not measured for the follow-up. Timings are two measurements, not a controlled pair.
- **T7 direct states:** 37 states on the 4B Q8_0 and 4B BF16 CUDA runs with the f16 KV type, 3 on all the others (4B Q4_K_M, the 1.7B, the quantized-KV and the CPU runs), as in the base campaign, so changed-argmax counts are not comparable across rows.
- **Not repeated:** Vulkan, `rizzo decide`, batch-size and thread-count sweeps, KV cache on the 1.7B and the CPU, other CPU configs, the unpinned integration run. The integration suite was pinned to the 4B Q8_0 as in the base report.
- The 1.7B Q4_K_M smoke has 1 of 20 shared-vs-direct argmax changes (base: 0).
- **CPU-only flow run, GPU state.** The GPU is not compute-idle in the telemetry of the flow CPU-only SemIf run: 19.8 W average, 12 % utilization, 53 °C, idle baseline 31 W (base run: 6.2 W, 0.6 %, 38 °C, baseline 6.6 W), with no foreign CUDA compute process in any poll. It is desktop activity and another power state of the card, so the `— (GPU idle)` cell means no CUDA work, not a quiet board.
