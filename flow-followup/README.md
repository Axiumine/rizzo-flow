# Follow-up: fine-tuned weights (`--weights flow`)

39 runs of rizzo-flow at commit `b9ba007` with the fine-tuned GGUF (`rizzoaiacademy/rizzo-flow`
rev `55633c8c…`, `rizzoaiacademy/rizzo-flow-1.7b`), same machine and same run ids as the base
campaign one level up, 30 September 2026. Posted as a comment on
[Rizzo-AI-Academy/rizzo-flow#25](https://github.com/Rizzo-AI-Academy/rizzo-flow/issues/25).

| Path | Content |
|---|---|
| [`HARDWARE-REPORT-FLOW.md`](HARDWARE-REPORT-FLOW.md) | All tables and the per-run log (command, statistics, telemetry, idle-gate and external-load measurements) |
| [`issue-25-followup-comment.md`](issue-25-followup-comment.md) | The comment as posted |
| [`data/ledger-flow.json`](data/ledger-flow.json) | Every run: spec, statistics, weights check, gate and external CPU load, rusage, telemetry; plus the one discarded (contaminated) attempt |
| [`data/local-hw-flow.tar.gz`](data/local-hw-flow.tar.gz) | Raw outputs of the 39 runs (`results/local-hw-flow/<run-id>/`) and the discarded attempt in `_contaminated/` |
| [`campaign/`](campaign/) | `run_flow.py` (imports `../campaign/run_campaign.py` read-only; idle gate), `render_flow.py`, the pytest plugin pinning the integration suite to the 4B flow Q8_0, summary, notes and run log |

Run from the repository root with the files in `.research/hw-campaign/`:
`.venv/bin/python .research/hw-campaign/run_flow.py` (`--check-idle`, `--dry-run`, `--render`).
