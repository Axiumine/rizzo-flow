# rizzo-flow hardware report — data for issue #25

Supporting material for
[Rizzo-AI-Academy/rizzo-flow#25](https://github.com/Rizzo-AI-Academy/rizzo-flow/issues/25):
141 runs of rizzo-flow at commit `c30cc63` (XHToken's base GGUF weights) on Debian 13 with an
NVIDIA RTX PRO 4000 Blackwell (24 GB, CUDA and Vulkan builds) and an AMD Ryzen 9 9950X3D (CPU-only
`cpu` package), 28–29 September 2026.

| Path | Content |
|---|---|
| [`HARDWARE-REPORT-RESEARCH.md`](HARDWARE-REPORT-RESEARCH.md) | Planning notes and the full results log: every run with its command, statistics and GPU/CPU telemetry |
| [`issue-25-body.md`](issue-25-body.md) | The body of issue #25 as posted |
| [`data/ledger.json`](data/ledger.json) | Every run in one JSON file: run spec, extracted statistics, rusage, `nvidia-smi` and Prometheus/DCGM telemetry, timestamps |
| [`data/local-hw.tar.gz`](data/local-hw.tar.gz) | Raw outputs of all 141 runs (`results/local-hw/<run-id>/`): the tools' own JSON reports, stdout/stderr, `nvidia-smi` samples every 100 ms, `metrics.json` |
| [`campaign/`](campaign/) | The runner (`run_campaign.py`), the `--threads` wrapper for the scripts that lack the flag, the pytest plugin that pins the integration suite to the 4B, the summary and notes used for the issue, and the run log |

The runner was executed from the repository root:
`.venv/bin/python .research/hw-campaign/run_campaign.py` (resumable; `--dry-run` prints the plan,
`--render` rebuilds the reports from `ledger.json`). Paths and hostnames in these files are those
of the test machine; the Prometheus/DCGM telemetry is optional and the runner works without it.

## Follow-up

[`flow-followup/`](flow-followup/): the same machine with the fine-tuned weights (`--weights flow`), 39 runs, 30 September 2026.
