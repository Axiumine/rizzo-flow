#!/usr/bin/env python3
"""Follow-up hardware campaign on the fine-tuned (flow) weights: 39 runs, same machine.

The first campaign (run_campaign.py: 141 runs, ledger.json, results/local-hw/) ran on the original
XHToken GGUF, now `--weights base`. This one repeats the runs that matter on the LoRA fine-tune
that became the default (`--weights flow`). Run ids are identical to the base scheme, so a flow
run and its base run pair by id.

It is run_campaign.py with other inputs, not a fork: run_campaign is imported and its module
globals and functions are replaced (install()). The rest is run_campaign's own code: main() with
--dry-run/--only/--limit/--render/--reextract, execute() and extract() (inside thin wrappers that
add the weights guard), the telemetry, the create-only outputs, the resume logic.

  python .research/hw-campaign/run_flow.py --dry-run        # print the plan
  python .research/hw-campaign/run_flow.py --only T4,T5     # subset (substring of id)
  python .research/hw-campaign/run_flow.py                  # everything still to do
  python .research/hw-campaign/run_flow.py --render         # rewrite the reports only
  python .research/hw-campaign/run_flow.py --reextract      # recompute stats from files, then render
  python .research/hw-campaign/run_flow.py --check-idle     # one 5 s measurement, no lock, exit 0/1

Never written: ledger.json, results/local-hw/, HARDWARE-REPORT-RESEARCH.md, HARDWARE-REPORT-ISSUE.md.
Written: ledger-flow.json, ledger-flow.lock, results/local-hw-flow/ (thrown-away attempts under
_contaminated/), campaign-flow.log, errors-flow.log (only when a report fails) and, through
render_flow.render, HARDWARE-REPORT-FLOW.md and HARDWARE-REPORT-FLOW-COMMENT.md.

Options of the idle gate (--check-idle takes the two limits too; --dry-run prints the settings):
  --check-idle              one 5 s measurement, no lock, exit status 0 if idle and 1 if busy
  --no-idle-gate            do not wait for an idle machine (the accounting stays on)
  --max-external-threads N  limit of CPU used by other processes, in threads (default 1.0)
  --max-gpu-util PCT        limit of GPU utilization at the gate (default 25)
  --gate-timeout MIN        stop the campaign if a run waits MIN minutes at the gate (default: no)
  --max-attempts N          attempts per run and invocation when a run is contaminated (default 3)

Idle gate and load accounting. The machine is shared, and a benchmark that runs beside another
job is worthless, so the runner proves that each run had the machine to itself instead of hoping:

  * Before every run (nothing of ours running) it measures a 5 s window: the CPU load that is not
    ours (busy time from /proc/stat minus this runner's own getrusage, in threads), the CUDA
    compute processes that are not ours (nvidia-smi) and the GPU utilization. The machine is idle
    if external CPU <= --max-external-threads (default 1.0), no foreign compute process exists and
    GPU utilization <= --max-gpu-util (default 25 %). If not, it says why (numbers and the top
    processes) at most once every 10 minutes, waits 60 s and measures again, without limit unless
    --gate-timeout MIN is given (then the campaign stops cleanly, exit status 75, the ledger as
    it was after the last recorded run). After any wait, 3 idle windows in a row are needed.
    --no-idle-gate disables the waiting only; the accounting below stays on.
  * Over the whole of run_campaign.execute (baseline window, timed commands, settle, telemetry
    queries, post commands) a monitor thread samples the same counters every 0.5 s and, every 3 s,
    scans /proc and asks nvidia-smi for foreign compute processes. The record gets `gate` (the
    passing measurement), `external_cpu` (threads_avg, cpu_seconds, raw, wall_s, top_processes and
    the same figures for the timed window t0..t1, snapped outward to the samples) and `external_gpu`
    (foreign compute processes seen, and at the end; polls, polls_failed and blind_s, the longest
    stretch without a query that worked).
  * A run is CONTAMINATED if its external CPU exceeds the limit over the whole call or over the
    timed window, or a foreign compute process appeared, or the GPU was not watched (no nvidia-smi
    query worked for more than 30 s at any point, the two ends of the run included: a run is not
    called clean on polls that proved nothing; at the gate an unreadable nvidia-smi is busy too).
    Its output folder moves to results/local-hw-flow/_contaminated/<id>.<attempt>/ (never
    deleted), it is not recorded, an entry goes to ledger["contaminated"] and the run is gated and
    repeated, at most --max-attempts (3) times per invocation; after the last one it stays undone,
    so the next invocation retries it. A run that failed for another reason, on an attempt that was
    clean, is recorded as failed, as run_campaign does; a failed attempt that was also
    contaminated is repeated.
  * A T7 run that is compared with another run of this campaign (the two KV runs and the CPU run,
    against the f16 CUDA run of the same weights: post_commands) is deferred, not run and not
    recorded, until that run is recorded ok: run_campaign never revisits a recorded run, so one
    recorded without its comparison would stay without it for good. It runs once the first is
    recorded, in this invocation or in a later one.
  * --gate-timeout MIN counts the minutes one run waits at the gate, not the campaign.
  * Exit status: 0 when every run asked for is recorded; 75 (EX_TEMPFAIL, "try again later") when
    the campaign stopped by --gate-timeout or ended with runs undone (contaminated on every
    attempt, or deferred): run it again, it does the runs that are left.

One writer at a time: the campaign, --render and --reextract each hold an exclusive lock on
ledger-flow.lock for as long as they live (the kernel drops it if the process dies). A second one
exits at once and names the holder, instead of saving its copy of the ledger over records the
other saved in the meantime. --dry-run needs no lock. --render is "reports only": it saves the
ledger only if that call changed it (machine info, a guard verdict).

Weights guard: after every model run the JSON it wrote is checked for the fine-tuned weights (the
pinned file name and sha256, `weights: flow`, the served id rizzo-flow-<size>-<quant>). The evidence
goes to stats.weights_check; a mismatch, or no evidence at all, makes the run `failed`.
"""

import argparse
import collections
import contextlib
import csv
import fcntl
import importlib.util
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True  # no .pyc files next to the campaign scripts
CAMPAIGN = Path(__file__).resolve().parent
sys.path.insert(0, str(CAMPAIGN))

import run_campaign as rc  # noqa: E402

ROOT = rc.ROOT
VARIANT = "flow"
FLOW_OUT = ROOT / "results" / "local-hw-flow"
FLOW_LEDGER = CAMPAIGN / "ledger-flow.json"
FLOW_LOG = CAMPAIGN / "campaign-flow.log"
ERRORS_LOG = CAMPAIGN / "errors-flow.log"
BASE_OUT = ROOT / "results" / "local-hw"  # read-only: outputs of the base campaign
BASE_LEDGER = CAMPAIGN / "ledger.json"  # read-only: measurements of the base campaign
MAINTAINERS_FLOW = ROOT / "results" / "semif-compare" / "rizzo-flow-q8_0-v3-llama-cuda"
EVIDENCE_FILE = "weights-evidence.json"  # written by pin_4b_flow_plugin.py into the T1 run dir
MODEL_SUITES = ("T4", "T5", "T6", "T7", "T9")  # the commands that pass --weights
GUARDED = ("T1",) + MODEL_SUITES  # the suites that load a model
# Campaign tools that are not model processes (or are this one): never leftovers. Not a licence to
# run two writers at once: that is what the lock on ledger-flow.lock refuses (acquire_lock).
RUNNERS = ("run_campaign.py", "run_flow.py", "render_flow.py")
GUARD = "weights guard:"
# strict: --render / --reextract (report errors instead of logging them); reextract: --reextract
STATE = {"strict": False, "reextract": False}
TICKS = os.sysconf("SC_CLK_TCK")  # /proc counts CPU time in these units per second
# The idle gate (module docstring); main() sets it from --no-idle-gate, --max-external-threads,
# --max-gpu-util, --gate-timeout and --max-attempts. on: wait for an idle machine; max_ext: threads
# of external CPU; max_gpu: percent of GPU utilization; timeout_s: how long one run may wait, None
# for ever; attempts: per run and invocation; window_s: length of a measurement; retry_s: wait
# between measurements; log_every_s: at most one "not idle" message in this time; passes: idle
# windows in a row that are needed after a wait.
GATE = {"on": True, "max_ext": 1.0, "max_gpu": 25.0, "timeout_s": None, "attempts": 3,
        "window_s": 5.0, "retry_s": 60.0, "log_every_s": 600.0, "passes": 3}
SAMPLE_S = 0.5  # the monitor reads the CPU counters this often while a run executes
POLL_S = 3.0  # ... and scans /proc and asks nvidia-smi for compute processes this often
# A run whose GPU was not watched for longer than this (no nvidia-smi query that succeeded, from the
# start of the watch to its end) is not proven clean. One failed poll costs about 2 * POLL_S; a
# query that hangs until its 20 s timeout still fits, two in a row do not.
BLIND_MAX_S = 30.0
LABEL_CPU_S = 0.2  # a process that used this much CPU between two scans is named at once
CONTAMINATED = "_contaminated"  # results/local-hw-flow/_contaminated/<id>.<attempt>/
EX_TEMPFAIL = 75  # exit status when --gate-timeout stops the campaign (sysexits.h: try again later)

EXPECTED_IDS = """
T0-unit T1-integration-4b
cuda-4b-q8_0-f16-T4 cuda-4b-q8_0-f16-T5 cuda-4b-q8_0-f16-T6 cuda-4b-q8_0-f16-T7 cuda-4b-q8_0-f16-T9
cuda-4b-bf16-f16-T4 cuda-4b-bf16-f16-T5 cuda-4b-bf16-f16-T6 cuda-4b-bf16-f16-T7 cuda-4b-bf16-f16-T9
cuda-4b-q4_k_m-f16-T4 cuda-4b-q4_k_m-f16-T5 cuda-4b-q4_k_m-f16-T6 cuda-4b-q4_k_m-f16-T7
cuda-4b-q4_k_m-f16-T9
cuda-1.7b-q8_0-f16-T4 cuda-1.7b-q8_0-f16-T5 cuda-1.7b-q8_0-f16-T6 cuda-1.7b-q8_0-f16-T7
cuda-1.7b-q8_0-f16-T9
cuda-1.7b-q4_k_m-f16-T4 cuda-1.7b-q4_k_m-f16-T5 cuda-1.7b-q4_k_m-f16-T6 cuda-1.7b-q4_k_m-f16-T7
cuda-1.7b-q4_k_m-f16-T9
cuda-1.7b-bf16-f16-T4 cuda-1.7b-bf16-f16-T5 cuda-1.7b-bf16-f16-T6 cuda-1.7b-bf16-f16-T7
cuda-1.7b-bf16-f16-T9
cuda-4b-q8_0-q8_0-T4 cuda-4b-q8_0-q8_0-T7 cuda-4b-q8_0-q4_0-T4 cuda-4b-q8_0-q4_0-T7
cpu-4b-q8_0-f16-T4 cpu-4b-q8_0-f16-T7 cpu-4b-q8_0-f16-T9
""".split()


def _original(name):
    """run_campaign's own function, whether or not install() has replaced it."""
    return getattr(rc, "_flow_originals", {}).get(name) or getattr(rc, name)


def _require_installed():
    if rc.OUT != FLOW_OUT or rc.LEDGER != FLOW_LEDGER:
        raise RuntimeError("run_flow.install() has not run: run_campaign still points at the "
                           "base campaign (results/local-hw, ledger.json)")


def rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


# ----------------------------------------------------------------------------- base campaign

_BASE = {}


def base_ledger():
    """The base campaign's ledger, read once and never written."""
    if "ledger" not in _BASE:
        try:
            _BASE["ledger"] = json.loads(BASE_LEDGER.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _BASE["ledger"] = {"runs": {}}
    return _BASE["ledger"]


def measured(run):
    """Wall time of the same run id in the base campaign, if it finished ok there."""
    rec = base_ledger()["runs"].get(run["id"])
    return rec["wall_s"] if rec and rec.get("status") == "ok" and rec.get("wall_s") else None


def cpu_threads():
    """The thread count the base campaign found fastest (rc.cpu_threads on the base ledger)."""
    return rc.cpu_threads(base_ledger())


# ----------------------------------------------------------------------------- plan

def plan():
    runs = []

    def add(run):
        run.setdefault("kv", None)
        run.setdefault("extra", {})
        runs.append(run)

    add({"id": "T0-unit", "block": "one-off", "suite": "T0", "device": None})
    # The suite loads the smallest Q8_0 on disk (a 1.7B file); the plugin pins it to the flow 4B.
    add({"id": "T1-integration-4b", "block": "one-off", "suite": "T1", "device": None,
         "pin_4b": True, "weights": VARIANT})

    def config_runs(device, size, quant, block, suites, kv=None):
        for suite in suites:
            ex = {}
            if suite == "T7" and device == "cuda" and size == "4b" and quant in ("q8_0", "bf16") \
                    and kv is None:
                ex["direct_states"] = 37  # as in the base plan: comparable with the published flips
            add({"id": f"{device}-{size}-{quant}-{kv or 'f16'}-{suite}", "block": block,
                 "suite": suite, "device": device, "size": size, "quant": quant, "kv": kv,
                 "weights": VARIANT, "extra": ex})

    for size, quant in rc.WEIGHTS:
        config_runs("cuda", size, quant, "core", ["T4", "T5", "T6", "T7", "T9"])
    for kv in ("q8_0", "q4_0"):  # KV runs: smoke and SemIf only, no extras
        config_runs("cuda", "4b", "q8_0", "kv", ["T4", "T7"], kv=kv)
    config_runs("cpu", "4b", "q8_0", "core(cpu)", ["T4", "T7", "T9"])
    ids = [r["id"] for r in runs]
    assert ids == EXPECTED_IDS, f"plan drifted from the agreed 39 ids: {set(ids) ^ set(EXPECTED_IDS)}"
    return runs


# ----------------------------------------------------------------------------- commands

def variant(run):
    return run.get("weights", VARIANT)


def model_args(run):
    args = ["--size", run["size"], "--quant", run["quant"], "--weights", variant(run),
            "--device", run["device"]]
    if run.get("kv"):
        args += ["--kv-type", run["kv"]]
    return args


def commands(run, rundir, ledger):
    """List of (argv, env_extra, label). run_campaign.commands with --weights flow everywhere."""
    suite, ex = run["suite"], run["extra"]
    rizzo, py = str(rc.VENV / "rizzo"), str(rc.VENV / "python")
    threads = ["--threads", str(cpu_threads())] if run.get("device") == "cpu" else []
    if suite == "T0":
        return [([str(rc.VENV / "pytest"), "-q"], {}, "pytest")]
    if suite == "T1":
        if not run.get("pin_4b"):
            raise ValueError("the flow campaign runs the integration suite pinned to the 4B only")
        # A path relative to the repo (the cwd of every command): the env is copied into reports.
        env = {"RIZZO_REAL": "1", "PYTHONPATH": str(rc.CAMPAIGN),
               "RIZZO_HW_EVIDENCE": rel(rundir / EVIDENCE_FILE)}
        return [([str(rc.VENV / "pytest"), "-q", "-m", "integration", "-p", "pin_4b_flow_plugin"],
                 env, "pytest")]
    if suite in ("T4", "T5"):
        bench = "smoke" if suite == "T4" else "perturbations"
        out = [([rizzo, "evaluate", f"benchmarks/{bench}.jsonl", "--compare-modes",
                 *model_args(run), *threads, "--output", str(rundir / f"{bench}.json")], {}, bench)]
    else:
        # The scripts have no --threads flag: on the CPU a wrapper injects the thread count.
        script_env = {"RIZZO_HW_THREADS": threads[1]} if threads else {}
        script = [py, str(rc.CAMPAIGN / "threads_wrapper.py")] if threads else [py]
        if suite == "T6":  # validate_checkpoint.py and typed_decisions.py have no --kv-type
            assert not run.get("kv"), "no KV runs of T6/T9 in the flow plan"
            argv = [*script, "scripts/validate_checkpoint.py", "--size", run["size"], "--quant",
                    run["quant"], "--weights", variant(run), "--device", run["device"],
                    "--output", str(rundir / "validate")]
            out = [(argv, script_env, "validate")]
        elif suite == "T7":
            argv = [*script, "scripts/semif_compare.py", "--system", "rizzo", "--semif",
                    str(rc.SEMIF), *model_args(run), "--output", str(rundir / "semif")]
            if "batch_size" in ex:
                argv += ["--batch-size", str(ex["batch_size"])]
            if "direct_states" in ex:
                argv += ["--direct-states", str(ex["direct_states"])]
            out = [(argv, script_env, "semif")]
        elif suite == "T9":
            assert not run.get("kv"), "no KV runs of T6/T9 in the flow plan"
            # The pinned flow file by name, not the base runner's --model <path>.
            argv = [*script, "scripts/typed_decisions.py", str(rc.TYPED), "--size", run["size"],
                    "--quant", run["quant"], "--weights", variant(run), "--device",
                    run["device"], "--output", str(rundir / "typed.json")]
            out = [(argv, script_env, "typed")]
        else:
            raise ValueError(f"suite {suite} is not part of the flow campaign")
    for argv, _, _ in out:  # self-check: a model command that forgot the flag would run base
        assert argv[argv.index("--weights") + 1] == VARIANT, argv
    return out


def pairing_base(run):
    """(name, run id) of the run of this campaign that a T7 run is compared with (`--against
    name=<its semif folder>`), None if there is none: the KV runs against the f16 run of the same
    device and weights, the CPU run against the f16 CUDA run. Only the ledger of this campaign
    can hold it; the base campaign and the maintainers' run are fixed files."""
    if run["suite"] != "T7":
        return None
    if run.get("kv"):  # KV effect alone: same device, same weights, f16 KV
        return f"this-{run['device']}-kv-f16", f"{run['device']}-{run['size']}-{run['quant']}-f16-T7"
    if run["device"] != "cuda":  # device effect alone
        return "this-cuda", f"cuda-{run['size']}-{run['quant']}-f16-T7"
    return None


def post_commands(run, rundir, ledger):
    """semif_report after the timed window (no telemetry): held-out numbers and paired differences."""
    if run["suite"] != "T7" or not (rundir / "semif" / "report.json").exists():
        return []
    ours = rundir / "semif"
    argv = [str(rc.VENV / "python"), "scripts/semif_report.py", str(ours), "--semif", str(rc.SEMIF)]
    targets = []
    if (run["device"], run["size"], run["quant"]) == ("cuda", "4b", "q8_0") and not run.get("kv") \
            and "batch_size" not in run["extra"]:  # the maintainers' fine-tuned run, another machine
        targets.append(("maintainers-flow", MAINTAINERS_FLOW))
    targets.append(("this-base", BASE_OUT / run["id"] / "semif"))  # fine-tuning effect alone
    if pairing_base(run):
        name, base_id = pairing_base(run)
        targets.append((name, FLOW_OUT / base_id / "semif"))
    for name, folder in targets:
        if (folder / "report.json").exists() and folder != ours:
            argv += ["--against", f"{name}={folder}"]
    return [(argv, {}, "semif_report")]


# ----------------------------------------------------------------------------- estimate

def estimate(run):
    """Measured wall time of the same run id in the base ledger (+ baseline and settle, the way
    run_campaign.estimate counts them); rc.estimate's model for a run the base never finished."""
    wall = measured(run)
    if wall is not None:
        return wall + rc.BASELINE_S + rc.SETTLE_S
    return _original("estimate")(run)


# ----------------------------------------------------------------------------- weights guard

_SPEC_SCRIPT = """
import json
from rizzo_flow import compat
from rizzo_flow.config import MODELS, QUANTS, gguf_spec
out = {}
for size in MODELS:
    for quant in QUANTS:
        spec = gguf_spec(size, quant, "flow")
        out[f"{size}/{quant}"] = {
            "file": spec.file, "path": str(spec.path), "sha256": spec.sha256, "repo": spec.repo,
            "revision": spec.revision, "checkpoint": MODELS[size].repo,
            "served_id": compat.model_name(
                {"source": MODELS[size].repo, "precision": quant, "weights": "flow"}),
        }
print(json.dumps(out))
"""
_SPECS = {}


def flow_specs():
    """The pinned flow GGUFs as the repo defines them (file, sha256, repo, served id), keyed
    'size/quant'. Read through the venv's interpreter: this runner stays stdlib-only."""
    if not _SPECS:
        proc = subprocess.run([str(rc.VENV / "python"), "-c", _SPEC_SCRIPT], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
        if proc.returncode or not proc.stdout.strip():
            raise RuntimeError("cannot read rizzo_flow.config: " + proc.stderr.strip()[-300:])
        _SPECS.update(json.loads(proc.stdout.strip().splitlines()[-1]))
    return _SPECS


def expected(run):
    size, quant = ("4b", "q8_0") if run.get("pin_4b") else (run["size"], run["quant"])
    return {**flow_specs()[f"{size}/{quant}"], "size": size, "quant": quant}


def _served_id(meta):
    """compat.model_name, which this runner cannot import: rizzo-flow-4b-q8_0."""
    checkpoint = str(meta.get("source", "")).split("/")[-1].lower()  # spark-x2.5-4b
    precision = meta.get("precision", "unknown")
    if meta.get("weights") == VARIANT:
        return f"rizzo-flow-{checkpoint.rsplit('-', 1)[-1]}-{precision}"
    return f"rizzo-{checkpoint}-{precision}"


def check_meta(where, meta, exp, kv_allowed, need_files):
    """Problems found in one backend metadata dict (the `model` block the suites write)."""
    if not isinstance(meta, dict):
        return [f"{where}: no backend metadata"]
    problems = []

    def expect(key, want):
        if meta.get(key) != want:
            problems.append(f"{where}: {key} is {meta.get(key)!r}, expected {want!r}")

    expect("weights", VARIANT)
    expect("source", exp["checkpoint"])
    expect("precision", exp["quant"])
    expect("gguf_source", exp["repo"])
    expect("gguf_revision", exp["revision"])
    files = meta.get("source_files")  # {file name: sha256}; typed.json keeps scalars only
    if (files is not None or need_files) and files != {exp["file"]: exp["sha256"]}:
        problems.append(f"{where}: source_files is {files!r}, expected "
                        f"{{{exp['file']!r}: {exp['sha256']!r}}}")
    if meta.get("kv_cache") not in kv_allowed:
        problems.append(f"{where}: kv_cache is {meta.get('kv_cache')!r}, expected one of "
                        f"{list(kv_allowed)}")
    if _served_id(meta) != exp["served_id"]:
        problems.append(f"{where}: served id would be {_served_id(meta)!r}, expected "
                        f"{exp['served_id']!r}")
    return problems


def evidence_sources(run, rundir, exp):
    """(served id if the suite prints one, [(where, metadata)], problems), read from the files the
    suite wrote: every suite exposes the backend metadata somewhere else."""
    suite = run["suite"]
    if suite in ("T4", "T5"):  # rizzo evaluate: rows[].response.model
        name = "smoke" if suite == "T4" else "perturbations"
        rows = (rc.read_json(rundir / f"{name}.json") or {}).get("rows") or []
        first = next((row for row in rows if rc.g(row, "response.model")), None)
        if first is None:
            raise ValueError(f"{name}.json has no row with response.model")
        sources = [(f"{name}.json rows[].response.model", first["response"]["model"])]
        if rc.g(first, "alternate_mode_response.model"):
            sources.append((f"{name}.json rows[].alternate_mode_response.model",
                            first["alternate_mode_response"]["model"]))
        return None, sources, []
    if suite == "T6":  # validate_checkpoint.py: summary.json and api-example.json
        sources = []
        for fname in ("summary.json", "api-example.json"):
            data = rc.read_json(rundir / "validate" / fname)
            if data is None:
                raise ValueError(f"validate/{fname} is missing or not JSON")
            sources.append((f"validate/{fname} model", data.get("model")))
        return None, sources, []
    if suite == "T7":  # semif_compare.py: report.json
        data = rc.read_json(rundir / "semif" / "report.json")
        if data is None:
            raise ValueError("semif/report.json is missing or not JSON")
        return None, [("semif/report.json model", data.get("model"))], []
    if suite == "T9":  # typed_decisions.py: the served id, and the scalar part of the metadata
        data = rc.read_json(rundir / "typed.json")
        if data is None:
            raise ValueError("typed.json is missing or not JSON")
        served = data.get("model")
        if not isinstance(served, str):
            raise ValueError(f"typed.json model is {served!r}, not a served model id")
        return served, [("typed.json metadata", data.get("metadata"))], []
    if suite == "T1":  # pin_4b_flow_plugin.py: what it pinned and every backend the tests loaded
        data = rc.read_json(rundir / EVIDENCE_FILE)
        if data is None:
            raise ValueError(f"{EVIDENCE_FILE} is missing: the plugin did not write it")
        problems = []
        pinned, loaded = data.get("pinned") or {}, data.get("loaded") or []
        if pinned.get("file") != exp["file"] or pinned.get("sha256") != exp["sha256"]:
            problems.append(f"{EVIDENCE_FILE}: the plugin pinned {pinned.get('file')!r}, expected "
                            f"{exp['file']!r}")
        if not loaded:
            problems.append("no model was loaded (every integration test skipped?)")
        return None, [(f"{EVIDENCE_FILE} loaded[{i}]", meta) for i, meta in enumerate(loaded)], problems
    raise ValueError(f"no weights evidence defined for suite {suite}")


def weights_check(run, rundir):
    """Evidence that the run used the fine-tuned GGUF it asked for; never raises: a run that
    cannot be verified fails closed instead of being recorded as ok."""
    try:
        exp = expected(run)
        served, sources, problems = evidence_sources(run, rundir, exp)
        if run["suite"] == "T1":  # test_llama_real also loads the quantized-KV variants
            kv_allowed = (None, "q8_0", "q4_0")
        else:
            kv_allowed = (run["kv"] if run.get("kv") not in (None, "f16") else None,)
        for where, meta in sources:
            problems += check_meta(where, meta, exp, kv_allowed, need_files=run["suite"] != "T9")
        if served is not None and served != exp["served_id"]:
            problems.append(f"typed.json: served model id is {served!r}, expected "
                            f"{exp['served_id']!r}")
    except Exception as error:  # noqa: BLE001 - see the docstring
        return {"ok": False, "problems": [f"cannot verify the weights: {error}"]}
    first = sources[0][1] if sources and isinstance(sources[0][1], dict) else {}
    files = first.get("source_files") or {}
    return {
        "ok": not problems, "problems": problems,
        "served_id": served or (_served_id(first) if first else None),
        "weights": first.get("weights"),
        "gguf": next(iter(files), None), "gguf_sha256": next(iter(files.values()), None),
        "gguf_source": first.get("gguf_source"), "gguf_revision": first.get("gguf_revision"),
        "precision": first.get("precision"), "fingerprint": first.get("fingerprint"),
        "kv_cache": first.get("kv_cache"),
        "expected_gguf": exp["file"], "expected_served_id": exp["served_id"],
        "backends_loaded": len(sources) if run["suite"] == "T1" else None,
        "from": [where for where, _ in sources],
    }


def apply_guard(record):
    """Turn the verdict in stats.weights_check into the run's status. Idempotent: a run failed
    by the guard is marked (weights_guard) and gets its status back if a later --reextract finds
    the evidence fine; a run that failed for another reason is left alone. True if it changed."""
    check = (record.get("stats") or {}).get("weights_check")
    if not check:
        return False
    before = (record.get("status"), record.get("weights_guard"), list(record.get("stderr_tail", [])))
    tail = [line for line in record.get("stderr_tail", []) if not line.startswith(GUARD)]
    if not check["ok"] and record["status"] == "ok":
        record["status"], record["weights_guard"] = "failed", "failed"
    elif check["ok"] and record.get("weights_guard") == "failed":
        record["status"] = "ok"
        del record["weights_guard"]
    if record.get("weights_guard") == "failed":
        tail.append(f"{GUARD} " + rc.redact("; ".join(check["problems"]))[:700])
    record["stderr_tail"] = tail
    return before != (record["status"], record.get("weights_guard"), tail)


# ----------------------------------------------------------------------------- idle gate

Sample = collections.namedtuple("Sample", "wall clock busy own")  # time.time(), monotonic, CPU s
Proc = collections.namedtuple("Proc", "pid start comm ppid cpu ours")  # start: ticks since boot
LAUNCHERS = re.compile(r"^(python[\d.]*|node(js)?|deno|bun|java|ruby|perl|php|bash|dash|zsh|sh|env"
                       r"|uvx?|npx?|pnpm|yarn)$")
# What describe() may show of a command line. A word must be made of these characters only: no '=' (an
# assignment such as TOKEN=x), no quote, no space, ':' or ','. The script that a launcher runs is
# shown when it is the first argument, or when it is named like a script (or sits in node_modules)
# after options, whose values (`yarn --token SECRET run`) look like any other word.
SAFE_WORD = re.compile(r"[\w@+./~-]+")
SCRIPT_NAME = re.compile(r".*\.(?:[cm]?[jt]s|py[cw]?|sh|bash|zsh|rb|pl|php|jar|lua)", re.I)
CODE_FLAG = re.compile(r"-[A-Za-z]*[ceE]|--(?:eval|print|command)(?:=.*)?")  # program text follows
MODULE_FLAG = re.compile(r"-[A-Za-z]*m")  # python -m module (also -um)


def busy_seconds(stat_line):
    """CPU seconds all CPUs together have been busy since boot, from the first line of /proc/stat:
    user + nice + system + irq + softirq + steal. Idle and iowait are not busy; guest and
    guest_nice are already inside user and nice."""
    f = [int(x) for x in stat_line.split()[1:9]]
    f += [0] * (8 - len(f))
    return (f[0] + f[1] + f[2] + f[5] + f[6] + f[7]) / TICKS


def own_seconds(self_usage, children_usage):
    """CPU seconds (user + system) of this process and of the children it has waited for."""
    return sum(u.ru_utime + u.ru_stime for u in (self_usage, children_usage))


def external_load(first, last):
    """The CPU load that was not ours between two Samples, in threads: (busy delta - own delta) /
    wall. `raw` can come out a hair below 0 (the two counters are read a few microseconds apart and
    /proc/stat counts in ticks): threads_avg is then 0 and raw keeps the value."""
    wall_s = last.clock - first.clock
    busy, own = last.busy - first.busy, last.own - first.own
    raw = (busy - own) / wall_s if wall_s > 0 else 0.0
    return {"threads_avg": max(raw, 0.0), "cpu_seconds": max(busy - own, 0.0), "raw": raw,
            "wall_s": wall_s, "busy_seconds": busy, "own_seconds": own}


def rounded(values, digits=4):
    """The floats of a flat dict rounded (+ 0.0: no `-0.0` in the records)."""
    return {k: round(v, digits) + 0.0 if isinstance(v, float) else v for k, v in values.items()}


def threads(value):
    """A thread limit as text: 1.0, 0.5, 1.5."""
    return f"{value:.1f}" if value == int(value) else f"{value:g}"


def parse_stat(text, pid):
    """A Proc from /proc/<pid>/stat. comm sits in parentheses and may hold spaces and ')'; utime and
    stime are those of the whole process (all its threads, also the ones that have exited)."""
    head, _, tail = text.rpartition(")")
    f = tail.split()
    return Proc(pid, int(f[19]), head.partition("(")[2], int(f[1]),
                (int(f[11]) + int(f[12])) / TICKS, False)


def mark_ours(procs, root):
    """`procs` with ours=True for `root` and everything below it (by parent pid)."""
    kids = {}
    for proc in procs.values():
        kids.setdefault(proc.ppid, []).append(proc.pid)
    ours, todo = set(), [root]
    while todo:
        pid = todo.pop()
        if pid not in ours:
            ours.add(pid)
            todo.extend(kids.get(pid, ()))
    return {pid: proc._replace(ours=pid in ours) for pid, proc in procs.items()}


def script_label(path):
    """Base name of a script; inside node_modules the package too: @stryker-mutator/core/x.js."""
    base = os.path.basename(path.rstrip("/")) or path
    if "node_modules/" in path:
        parts = path.rsplit("node_modules/", 1)[1].split("/")
        if parts[0] != ".bin":
            package = "/".join(parts[:2]) if parts[0].startswith("@") else parts[0]
            if package != base:
                return f"{package}/{base}"
    return base


def first_word(text):
    """The first whitespace-separated word of `text`, "" when there is none."""
    words = text.split(None, 1)
    return words[0] if words else ""


def plain(text, limit=80):
    """`text` as it may be written to a log, the ledger or a report: no '=' (NAME=value), quote,
    backtick or backslash and no control character (each becomes '?'), at most `limit` characters."""
    return re.sub(r"[=\"'`\\\x00-\x1f\x7f]", "?", text)[:limit]


def describe(argv, comm):
    """A name a person recognises: `node @stryker-mutator/core/child-process-proxy-worker.js`,
    `python main.py`, `ugrep`. Only the program and its script are kept, never the other arguments
    (a command line can carry a token): after `env` and its NAME=value words the program that it
    runs; a flag that takes program text (-c, -lc, -e, --eval) is named and its text dropped; a word
    is shown only if it is made of SAFE_WORD characters. A program that rewrites its own title
    (chrome, postgres) has its whole command line in argv[0]: only the first word of it is the
    program. comm, the kernel's name, when there is no command line. At most 80 characters."""
    if not argv or not argv[0].strip():
        return plain(comm)
    exe, args = os.path.basename(first_word(argv[0])).rstrip(":"), argv[1:]
    for _ in range(3):  # env [-u NAME] [NAME=value ...] program ...: the program is what runs
        if not SAFE_WORD.fullmatch(exe):
            return plain(comm)
        if exe != "env":
            break
        i = 0
        while i < len(args) and (args[i].startswith("-") or "=" in args[i]):
            i += 2 if args[i] in ("-u", "-C", "--unset", "--chdir") else 1
        if i >= len(args):
            return exe
        exe, args = os.path.basename(first_word(args[i])).rstrip(":"), args[i + 1:]
    if not SAFE_WORD.fullmatch(exe):
        return plain(comm)
    name = exe
    if LAUNCHERS.match(exe):
        i = 0
        while i < len(args) and args[i].startswith("-"):
            if MODULE_FLAG.fullmatch(args[i]) and i + 1 < len(args):
                module = first_word(args[i + 1])
                return f"{exe} -m {module}"[:80] if re.fullmatch(r"[\w.]+", module) else f"{exe} -m"
            if CODE_FLAG.fullmatch(args[i]):
                return f"{exe} {args[i].split('=', 1)[0]}"[:80]
            i += 1
        word = first_word(args[i]) if i < len(args) else ""
        if SAFE_WORD.fullmatch(word) and (i == 0 or SCRIPT_NAME.fullmatch(word)
                                          or "node_modules/" in word):
            name = f"{exe} {script_label(word)}"
    return plain(name)


def parse_compute_apps(text):
    """[{"pid", "driver_name", "used_mib"}] from `nvidia-smi --query-compute-apps=pid,process_name,
    used_memory --format=csv,noheader,nounits` (a process on two GPUs once); [] for no output.
    None if a row cannot be read (a pid printed as [N/A] in a container or on a vGPU, an error
    text): something is listed that cannot be told from a foreign process, so the GPU is not
    proven free."""
    apps, seen = [], set()
    try:
        for row in csv.reader((text or "").splitlines(), skipinitialspace=True):
            if not any(cell.strip() for cell in row):
                continue  # a blank line
            if len(row) < 3 or not row[0].strip().isdecimal():
                return None
            if int(row[0]) in seen:
                continue
            seen.add(int(row[0]))
            mib = row[-1].strip()
            apps.append({"pid": int(row[0]), "driver_name": ", ".join(row[1:-1]).strip(),
                         "used_mib": int(mib) if mib.isdecimal() else None})
    except csv.Error:  # a field over the csv module's size limit: not nvidia-smi's output
        return None
    return apps


def parse_util(text):
    """Utilization samples (percent) from `nvidia-smi --query-gpu=utilization.gpu -lms 500`."""
    out = []
    for line in (text or "").splitlines():
        try:
            out.append(float(line.strip()))
        except ValueError:
            continue
    return out


class Host:
    """What the gate and the accounting read from the machine. The tests give HOST a fake with the
    same methods, so that none of this needs a busy (or an idle) machine to be checked."""

    def clock(self):
        return time.monotonic()

    def wall(self):
        return time.time()

    def sleep(self, seconds):
        time.sleep(seconds)

    def busy_seconds(self):
        with open("/proc/stat", encoding="ascii") as stat:
            return busy_seconds(stat.readline())

    def own_seconds(self):
        return own_seconds(resource.getrusage(resource.RUSAGE_SELF),
                           resource.getrusage(resource.RUSAGE_CHILDREN))

    def sample(self):
        return Sample(self.wall(), self.clock(), self.busy_seconds(), self.own_seconds())

    def uptime_ticks(self):
        with open("/proc/uptime", encoding="ascii") as uptime:
            return float(uptime.read().split()[0]) * TICKS

    def processes(self):
        """{pid: Proc} of every process (a scan of /proc); ours: this runner and its descendants."""
        procs = {}
        with os.scandir("/proc") as entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry.name}/stat", encoding="utf-8", errors="replace") as f:
                        procs[int(entry.name)] = parse_stat(f.read(), int(entry.name))
                except (OSError, ValueError, IndexError):
                    continue  # gone, or not readable
        return mark_ours(procs, os.getpid())

    def is_descendant(self, pid):
        """Is `pid` this runner or one of its descendants? None if it has gone."""
        seen = set()
        while pid > 0 and pid not in seen:
            if pid == os.getpid():
                return True
            seen.add(pid)
            try:
                with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
                    pid = parse_stat(f.read(), pid).ppid
            except (OSError, ValueError, IndexError):
                return None if len(seen) == 1 else False
        return False

    def name(self, pid, comm):
        """describe() of the process's command line; comm if it cannot be read (gone, kernel)."""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
        except OSError:
            return plain(comm)
        return describe(argv, comm)

    def cwd(self, pid):
        """Base name of a process's working directory ("" if unreadable): a hint for a reader."""
        try:
            return os.path.basename(os.readlink(f"/proc/{pid}/cwd"))
        except OSError:
            return ""

    gpu_error = ""  # why the last gpu_apps() returned None; "" when it did not

    def gpu_apps(self):
        """The CUDA compute processes, [{"pid", "driver_name", "used_mib"}]; None if nvidia-smi
        cannot say (gpu_error says why). Desktop graphics processes are not compute processes and
        do not appear. A name that is not valid UTF-8 is decoded with replacements: it must not
        make the query fail."""
        self.gpu_error = ""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, errors="replace",
                timeout=20, check=False)
        except FileNotFoundError:
            self.gpu_error = "nvidia-smi not found"
            return None
        except subprocess.TimeoutExpired:
            self.gpu_error = "nvidia-smi timed out"
            return None
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.gpu_error = f"nvidia-smi failed ({type(error).__name__})"
            return None
        if out.returncode != 0:
            self.gpu_error = f"nvidia-smi exited with status {out.returncode}"
            return None
        apps = parse_compute_apps(out.stdout)
        if apps is None:
            first = next((line for line in out.stdout.splitlines() if line.strip()), "")
            self.gpu_error = f"unreadable nvidia-smi output {plain(first.strip(), 60)!r}"
        return apps

    def foreign_gpu_apps(self, procs=None):
        """The compute processes that are not ours: [{"pid", "name", "used_mib", "gone"}]; None if
        nvidia-smi cannot say. /proc is scanned first (unless `procs` was scanned just before), so
        that a child of ours that exits while nvidia-smi runs is still known to be ours."""
        procs = self.processes() if procs is None else procs
        apps = self.gpu_apps()
        if apps is None:
            return None
        out = []
        for app in apps:
            proc = procs.get(app["pid"])
            ours = proc.ours if proc is not None else self.is_descendant(app["pid"])
            if not ours:
                driver = os.path.basename(app["driver_name"].rstrip("/")) or app["driver_name"]
                out.append({"pid": app["pid"], "name": self.name(app["pid"], driver),
                            "used_mib": app["used_mib"], "gone": ours is None})
        return out

    def gpu_util_start(self):
        """Start sampling the GPU utilization every 0.5 s; the handle for gpu_util_stop."""
        try:
            return subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                 "-lms", "500"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                errors="replace")
        except OSError:
            return None

    def gpu_util_stop(self, proc):
        """Stop the sampler and reap it (its CPU is ours); the samples, None without nvidia-smi."""
        if proc is None:
            return None
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        return parse_util(out)


HOST = Host()


class Tracker:
    """CPU seconds per process, other than ours, between the first observation and the last.

    A process that started inside the window counts from its start; one that exited between two
    observations counts up to the last time it was seen. The list can therefore explain less than
    the machine-wide external load when short-lived processes did much of the work: `attributed`
    says how much it does explain."""

    def __init__(self, host):
        self.host = host
        self.t0 = host.uptime_ticks()  # read before the first scan: what starts later is new
        self.base, self.last, self.ours, self.label = {}, {}, set(), {}

    def observe(self, procs=None):
        for proc in (self.host.processes() if procs is None else procs).values():
            key = (proc.pid, proc.start)  # the start time tells a pid from its reuse
            if proc.ours:
                self.ours.add(key)
            self.base.setdefault(key, 0.0 if proc.start >= self.t0 else proc.cpu)
            before = self.last[key].cpu if key in self.last else self.base[key]
            if key not in self.label and key not in self.ours and proc.cpu - before >= LABEL_CPU_S:
                self.label[key] = self.host.name(proc.pid, proc.comm)  # while it is still alive
            self.last[key] = proc

    def deltas(self):
        return {key: proc.cpu - self.base[key] for key, proc in self.last.items()
                if key not in self.ours and proc.cpu > self.base[key]}

    def attributed(self):
        return sum(self.deltas().values())

    def _name(self, key):
        return self.label.get(key) or self.host.name(key[0], self.last[key].comm)

    def top(self, n, wall):
        """The n processes with the most CPU: [{"pid", "name", "threads", "cpu_s"}]."""
        rows = sorted(self.deltas().items(), key=lambda item: -item[1])[:n]
        return [{"pid": key[0], "name": self._name(key), "threads": round(cpu / wall, 3),
                 "cpu_s": round(cpu, 3)} for key, cpu in rows] if wall > 0 else []

    def names(self, n, wall):
        """The same by name, [{"name", "procs", "threads", "cpu_s"}]: 33 workers are one row."""
        groups = {}
        for key, cpu in sorted(self.deltas().items(), key=lambda item: -item[1])[:200]:
            group = groups.setdefault(self._name(key), [0.0, 0])
            group[0], group[1] = group[0] + cpu, group[1] + 1
        rows = sorted(groups.items(), key=lambda item: -item[1][0])[:n]
        return [{"name": name, "procs": count, "threads": round(cpu / wall, 3),
                 "cpu_s": round(cpu, 3)} for name, (cpu, count) in rows] if wall > 0 else []


def offenders(load, n=3):
    """`node @stryker-mutator/core/x.js x33 20.1, stryker 3.2` (name, x processes, threads) from the
    top_names of a window or of a run; its top_processes when there are no names."""
    names = load.get("top_names") or [{"name": p["name"], "procs": 1, "threads": p["threads"]}
                                      for p in load.get("top_processes") or []]
    shown = [f"{x['name']}{' x' + str(x['procs']) if x['procs'] > 1 else ''} {x['threads']:.1f}"
             for x in names[:n]]
    return ", ".join(shown) or "no process accounts for it"


def apps_text(apps):
    return ", ".join(f"{a['name']} (pid {a['pid']}, "
                     f"{'?' if a['used_mib'] is None else a['used_mib']} MiB)" for a in apps)


def measure_window(seconds=None, host=None):
    """One window of GATE window_s seconds: the external CPU load, the processes behind it, the
    GPU utilization and the foreign compute processes. The gate calls it with nothing of ours
    running, apart from this measurement itself: its CPU, and that of nvidia-smi, is ours and is
    subtracted. The scans before the first sample and after the last one are outside the window."""
    host = host or HOST
    seconds = GATE["window_s"] if seconds is None else seconds
    tracker = Tracker(host)
    before = host.foreign_gpu_apps()
    error = getattr(host, "gpu_error", "") if before is None else ""
    tracker.observe()
    first = host.sample()
    smi = host.gpu_util_start()  # after the first sample: all its CPU falls inside the window
    try:
        host.sleep(seconds / 2)
        tracker.observe()
        host.sleep(seconds / 2)
    finally:
        util = host.gpu_util_stop(smi)  # reaped before the last sample
    last = host.sample()
    tracker.observe()
    after = host.foreign_gpu_apps()
    if after is None:
        error = getattr(host, "gpu_error", "") or error
    wall = last.clock - first.clock
    apps = None if before is None or after is None else list(  # either failing: not proven free
        {a["pid"]: a for a in [*after, *before]}.values())
    return {
        "window_s": round(wall, 3), "external": rounded(external_load(first, last)),
        "top_processes": tracker.top(5, wall), "top_names": tracker.names(5, wall),
        "attributed_threads": round(tracker.attributed() / wall, 3) if wall > 0 else 0.0,
        "gpu_util_avg": round(sum(util) / len(util), 1) if util else None,
        "gpu_util_max": max(util) if util else None, "gpu_util_samples": len(util or []),
        "gpu_apps": apps, "gpu_error": error if apps is None else "",
    }


def attribution(m):
    """How much of the external CPU the named processes explain: `they account for 1.35 of the
    2.19 threads, the rest is short-lived processes, kernel or softirq time` (the per-pid
    figures miss a process that starts and ends between two looks, and a child that a long-lived
    parent has already reaped) and, when it is under 70 %, that stopping them may not be enough.
    "" when they explain (nearly) all of it, or without the figure."""
    named, load = m.get("attributed_threads"), m["external"]["threads_avg"]
    if not isinstance(named, (int, float)) or load <= 0 or named >= 0.95 * load:
        return ""
    text = (f"; they account for {named:.2f} of the {load:.2f} threads, the rest is "
            "short-lived processes, kernel or softirq time")
    return text + (": stopping them may not be enough" if named < 0.7 * load else "")


def problems(m):
    """[(kind, text)]: what makes a measurement not idle, with the numbers and the top offenders.
    Empty when it is idle. An unreadable GPU counts as busy: nothing proves it is free."""
    out, load = [], m["external"]["threads_avg"]
    if load > GATE["max_ext"]:
        limit = threads(GATE["max_ext"])
        out.append(("cpu", f"external CPU {load:.2f} threads > {limit} "
                           f"(top: {offenders(m)}{attribution(m)})"))
    if m["gpu_apps"] is None:
        why = f": {m['gpu_error']}" if m.get("gpu_error") else ""
        out.append(("gpu-unreadable", f"GPU compute processes unreadable (nvidia-smi{why})"))
    elif m["gpu_apps"]:
        out.append(("gpu-apps", "foreign GPU compute process " + apps_text(m["gpu_apps"])))
    util = m["gpu_util_avg"]
    if util is None:
        out.append(("gpu-unreadable", "GPU utilization unreadable (nvidia-smi)"))
    elif util > GATE["max_gpu"]:
        out.append(("gpu-util", f"GPU utilization {util:.0f} % > {GATE['max_gpu']:g} %"))
    return out


class GateTimeout(SystemExit):
    """--gate-timeout ran out: the campaign stops (exit status EX_TEMPFAIL); the run that was
    waiting is still to do and the ledger holds what was recorded before it."""


def gate_limits():
    return {"max_external_threads": GATE["max_ext"], "max_gpu_util_pct": GATE["max_gpu"]}


def wait_for_idle(label, strict=False, host=None):
    """Block until the machine is idle, then return the record of the passing measurement.

    One idle window is enough if the very first one is idle; after any wait GATE passes windows in
    a row are needed (`strict`: a repeat after a contaminated attempt counts as a wait). The reason
    for a busy window (numbers and top offenders) is printed at most every log_every_s; the next
    measurement comes retry_s later. After timeout_s of waiting, if there is one, GateTimeout. A
    measurement that raises counts as a busy window (up to 5 in a row, then the error is raised)."""
    host = host or HOST
    began, last_said = host.clock(), None
    busy, kinds, first_busy, streak, broken = 0, [], None, [], 0  # broken: raised in a row

    def say(text, force=False):
        nonlocal last_said
        now = host.clock()
        if force or last_said is None or now - last_said >= GATE["log_every_s"]:
            last_said = now
            print(f"[{time.strftime('%H:%M:%S')}] gate {label}: {text}", flush=True)

    while True:
        waited = host.clock() - began
        if busy and GATE["timeout_s"] is not None and waited >= GATE["timeout_s"]:
            say(f"still not idle after {waited / 60:.1f} min (--gate-timeout "
                f"{GATE['timeout_s'] / 60:g}): stopping the campaign; this run is still to do and "
                "the ledger holds the runs recorded before it", force=True)
            raise GateTimeout(EX_TEMPFAIL)
        try:
            m = measure_window(host=host)
            found = problems(m)
            broken = 0
        except Exception as error:  # noqa: BLE001 - a reader of the machine failing is a busy window
            broken += 1
            if broken > 5:  # not a hiccup: a bug, or a machine that cannot be read
                raise
            m, found = None, [("gate-error", f"the measurement failed: {error!r}")]
        if not found:
            streak.append(m)
            need = GATE["passes"] if busy or strict else 1
            if len(streak) >= need:
                break
            say(f"idle window {len(streak)}/{need}, measuring again")
            continue
        streak.clear()
        busy += 1
        for kind, _ in found:
            if kind not in kinds:
                kinds.append(kind)
        text = "; ".join(t for _, t in found)
        first_busy = first_busy or text
        say(f"not idle: {text}; measuring again in {GATE['retry_s']:g} s (waiting "
            f"{waited / 60:.1f} min)")
        delay = GATE["retry_s"]
        if GATE["timeout_s"] is not None:
            delay = max(0.0, min(delay, GATE["timeout_s"] - (host.clock() - began)))
        host.sleep(delay)
    last, waited = streak[-1], host.clock() - began
    say(f"idle: external CPU {last['external']['threads_avg']:.2f} threads, GPU utilization "
        f"{last['gpu_util_avg']:.0f} %, no foreign compute process"
        + (f" (after {waited / 60:.1f} min, {len(streak)} idle windows in a row)" if busy else ""),
        force=True)
    return {"enabled": True, "at": round(host.wall(), 3), "waited_s": round(waited, 1),
            "busy_windows": busy, "busy_kinds": kinds, "first_busy": first_busy,
            "idle_windows": len(streak), "window_s": GATE["window_s"],
            "passes_after_wait": GATE["passes"],
            "external_threads": last["external"]["threads_avg"],
            "external_raw": last["external"]["raw"],
            "gpu_util_avg": last["gpu_util_avg"], "gpu_util_max": last["gpu_util_max"],
            "gpu_apps": last["gpu_apps"], "top_processes": last["top_processes"],
            "windows": [{"external_threads": w["external"]["threads_avg"],
                         "gpu_util_avg": w["gpu_util_avg"]} for w in streak],
            "limits": gate_limits()}


class LoadMonitor:
    """Watches the machine while a run executes, then says how much of the load was not ours.

    A daemon thread takes a Sample every SAMPLE_S and, every POLL_S, scans /proc (who used the
    CPU) and asks nvidia-smi for compute processes that are not ours. start() and stop() bracket
    the run with a sample each; the scan and the query around them fall outside that window.
    Nothing here touches the run: it only reads, and its own CPU (thread, nvidia-smi) is ours."""

    def __init__(self, host=None):
        self.host = host or HOST
        self.samples, self.tracker = [], None
        self.seen, self.at_end, self.polls, self.failed, self.errors = {}, [], 0, 0, []
        self._halt, self._thread, self._polled = threading.Event(), None, 0.0
        self.last_ok, self.blind_s = None, 0.0  # monotonic time of the last good query; longest gap

    def start(self, thread=True):
        self.tracker = Tracker(self.host)
        self.last_ok = self.host.clock()  # the watch starts here: a first query that fails counts
        self.poll()
        self.samples.append(self.host.sample())
        if thread:  # without: the caller calls tick() itself (the tests, on a fake clock)
            self._thread = threading.Thread(target=self._loop, name="load-monitor", daemon=True)
            self._thread.start()

    def _loop(self):
        while not self._halt.wait(SAMPLE_S):
            try:
                self.tick()
            except Exception as error:  # noqa: BLE001 - a hiccup must not end the watching
                self.errors.append(repr(error))
                print(f"load monitor: {error!r}", file=sys.stderr, flush=True)

    def tick(self):
        self.samples.append(self.host.sample())
        if self.host.clock() - self._polled >= POLL_S:
            self.poll()

    def _watched(self, ok=True):
        """The GPU was watched up to now (a query worked, or, with ok=False, the watch ended):
        extend blind_s, the longest stretch since the last query that worked; a query that worked
        also becomes the new last one."""
        now = self.host.clock()
        if self.last_ok is not None:
            self.blind_s = max(self.blind_s, now - self.last_ok)
        if ok:
            self.last_ok = now

    def poll(self):
        """Scan /proc and ask for compute processes. Never raises: the watching must not hurt the
        run (a failed poll is counted, and the run's record says how many there were)."""
        self._polled = self.host.clock()
        self.polls += 1
        try:
            procs = self.host.processes()  # before the query: see Host.foreign_gpu_apps
            self.tracker.observe(procs)
            found = self.host.foreign_gpu_apps(procs)
        except Exception as error:  # noqa: BLE001
            self.errors.append(repr(error))
            found = None
        if found is None:
            self.failed += 1
            self.at_end = None
            return
        self._watched()
        now = self.host.wall()
        self.at_end = found
        for app in found:
            seen = self.seen.setdefault(app["pid"], {
                "pid": app["pid"], "name": app["name"], "used_mib": app["used_mib"],
                "first_seen": round(now, 3), "polls": 0})
            seen["last_seen"], seen["polls"] = round(now, 3), seen["polls"] + 1
            seen["used_mib"] = max(seen["used_mib"] or 0, app["used_mib"] or 0) or None

    def stop(self):
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        self.samples.append(self.host.sample())
        self.poll()
        self._watched(ok=False)  # a last poll that failed leaves the GPU unwatched up to here

    def account(self, t0=None, t1=None):
        """(external_cpu, external_gpu) for the record: the whole watched call and, when the
        run's t0 and t1 are given, the timed window alone, from the last sample at or before t0
        to the first at or after t1 (execute() shows no t0/t1 while it runs, so the window is
        snapped outward to the sampling grid; own and busy are read at the same instants)."""
        s = self.samples
        whole = external_load(s[0], s[-1])
        wall = whole["wall_s"]
        timed = None
        if t0 is not None and t1 is not None and len(s) > 1:
            a = max((i for i, x in enumerate(s) if x.wall <= t0), default=0)
            b = min((i for i, x in enumerate(s) if x.wall >= t1), default=len(s) - 1)
            if b > a:
                timed = {**rounded(external_load(s[a], s[b])),
                         "window": [round(s[a].wall, 3), round(s[b].wall, 3)]}
        attributed = self.tracker.attributed() / wall if wall > 0 else 0.0
        cpu = {**rounded(whole), "limit": GATE["max_ext"],
               "top_processes": self.tracker.top(5, wall), "top_names": self.tracker.names(5, wall),
               "attributed_threads": round(attributed, 3), "timed": timed, "samples": len(s),
               "monitor_errors": len(self.errors)}
        gpu = {"apps_seen": sorted(self.seen.values(), key=lambda a: a["first_seen"]),
               "apps_at_end": self.at_end, "polls": self.polls, "polls_failed": self.failed,
               "blind_s": round(self.blind_s, 1)}
        return cpu, gpu


def gpu_unwatched(gpu):
    """Why the polls do not prove the GPU free of foreign compute processes ("" if they do): no
    query worked, or none for more than BLIND_MAX_S seconds at a stretch (from the start of the
    watch to its end, so a first or a last poll that failed counts). A few failed polls among many
    do not matter; a run is not called clean on polls that proved nothing."""
    polls, failed = gpu.get("polls") or 0, gpu.get("polls_failed") or 0
    blind = gpu.get("blind_s")
    if polls and failed >= polls:
        return f"GPU not watched: none of {polls} nvidia-smi queries worked"
    if isinstance(blind, (int, float)) and blind > BLIND_MAX_S:
        return (f"GPU not watched for {blind:.0f} s (limit {BLIND_MAX_S:.0f} s; {failed} of {polls} "
                "nvidia-smi queries failed)")
    return ""


def verdicts(cpu, gpu):
    """[(text, public text)], one per reason a run cannot be recorded (empty: it was clean): external
    CPU over the limit in the whole call or in the timed window alone (the baseline and settle time
    dilute a burst in it), a foreign compute process at any poll, the last one included, or a GPU
    that was not watched (gpu_unwatched). The text names the processes (terminal, ledger, working
    notes); the public one, for the issue comment, is numbers and counts only."""
    limit, out = GATE["max_ext"], []
    if cpu["threads_avg"] > limit:
        head = (f"external CPU {cpu['threads_avg']:.2f} threads over the whole run "
                f"({cpu['cpu_seconds']:.1f} CPU-s in {cpu['wall_s']:.0f} s; limit {threads(limit)}")
        out.append((f"{head}; top: {offenders(cpu)})", f"{head})"))
    timed = cpu.get("timed")
    if timed and timed["threads_avg"] > limit:
        text = (f"external CPU {timed['threads_avg']:.2f} threads in the timed window "
                f"(limit {threads(limit)})")
        out.append((text, text))
    if gpu["apps_seen"]:
        n = len(gpu["apps_seen"])
        out.append(("foreign GPU compute process " + apps_text(gpu["apps_seen"]),
                    f"{n} foreign GPU compute process{'es' if n > 1 else ''} seen"))
    if gpu_unwatched(gpu):
        out.append((gpu_unwatched(gpu),) * 2)
    return out


def contamination(cpu, gpu):
    """The reasons of verdicts(), with the names: empty when the run was clean."""
    return [text for text, _ in verdicts(cpu, gpu)]


def quarantine(run, attempt):
    """Move the run's output folder to _contaminated/<id>.<attempt>/, the next free number if that
    exists. Never deleted, never overwritten. (attempt, new path or None if there was no folder)."""
    src = FLOW_OUT / run["id"]
    while (FLOW_OUT / CONTAMINATED / f"{run['id']}.{attempt}").exists():
        attempt += 1
    dst = FLOW_OUT / CONTAMINATED / f"{run['id']}.{attempt}"
    if not src.exists():
        return attempt, None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return attempt, dst


def gate_settings():
    """The gate as configured, for ledger["gate"]."""
    return {"enabled": GATE["on"], **gate_limits(), "window_s": GATE["window_s"],
            "passes_after_wait": GATE["passes"], "retry_s": GATE["retry_s"],
            "timeout_min": None if GATE["timeout_s"] is None else GATE["timeout_s"] / 60,
            "max_attempts": GATE["attempts"]}


def gate_rule():
    return (f"external CPU <= {threads(GATE['max_ext'])} threads (this runner's own CPU "
            f"subtracted), no foreign GPU compute process, GPU utilization <= "
            f"{GATE['max_gpu']:g} %")


def gate_text():
    """One line for --dry-run and the campaign header."""
    repeat = f"repeated (at most {GATE['attempts']} attempts)"
    if not GATE["on"]:
        return ("off (--no-idle-gate): no waiting; the external load is still measured and a "
                f"contaminated run is still moved to {CONTAMINATED}/ and {repeat}")
    wait = ("for ever" if GATE["timeout_s"] is None else
            f"{GATE['timeout_s'] / 60:g} min for a run, then stop the campaign")
    return (f"on: {gate_rule()}; a {GATE['window_s']:g} s window before every run; when busy, "
            f"measure again every {GATE['retry_s']:g} s (waiting {wait}) and log at most every "
            f"{GATE['log_every_s'] / 60:g} min; {GATE['passes']} idle windows in a row after a "
            f"wait; a run with external CPU > {threads(GATE['max_ext'])} threads, a foreign "
            f"compute process or a GPU that was not watched (no nvidia-smi answer for "
            f"{BLIND_MAX_S:g} s) is moved to {CONTAMINATED}/ and {repeat}")


def no_leftovers():
    """run_campaign.execute's wait for another model process (up to 60 s, then stop), made before
    the gate: a model process of this checkout that was left running would keep the gate busy for
    good, and execute's own message says what to do about it."""
    for _ in range(12):
        left = leftover_model_processes()
        if not left:
            return
        HOST.sleep(5)
    raise SystemExit(f"another model process is running, stopping: {left}")


def check_idle():
    """--check-idle: one window, what it found, the verdict. Exit status 0 if the machine is idle,
    1 if it is not. No lock, no ledger, nothing written."""
    print(f"idle check: one {GATE['window_s']:g} s window; idle = {gate_rule()}", flush=True)
    m = measure_window()
    found = problems(m)
    busy_kinds = {kind for kind, _ in found}

    def flag(kind):
        return "BUSY" if kind in busy_kinds else "ok"

    e, wall = m["external"], m["external"]["wall_s"]
    print(f"CPU: external {e['threads_avg']:.2f} threads (busy {e['busy_seconds'] / wall:.2f} - "
          f"own {e['own_seconds'] / wall:.2f}, raw {e['raw']:.2f}) over {wall:.1f} s "
          f"[limit {threads(GATE['max_ext'])}: {flag('cpu')}]")
    if m["top_processes"]:
        print(f"  top processes ({m['attributed_threads']:.2f} of {e['threads_avg']:.2f} threads "
              "attributed, per pid):")
        for p in m["top_processes"]:
            where = HOST.cwd(p["pid"])
            print(f"    {p['threads']:6.2f} thr  pid {p['pid']:<8} {p['name']}"
                  + (f"  [{where}]" if where else ""))
        print("  by name: " + "; ".join(
            f"{x['name']} x{x['procs']} {x['threads']:.2f}" for x in m["top_names"]))
    apps = m["gpu_apps"]
    if apps is None:
        print("GPU: compute processes unreadable"
              + (f" ({m['gpu_error']})" if m.get("gpu_error") else "") + " [BUSY]")
    else:
        print(f"GPU: {len(apps)} foreign compute process(es) [{'BUSY' if apps else 'ok'}]")
        for a in apps:
            where = HOST.cwd(a["pid"])
            print(f"    pid {a['pid']:<8} {a['name']}, {a['used_mib']} MiB"
                  + (f"  [{where}]" if where else ""))
    util = m["gpu_util_avg"]
    print("GPU: utilization unreadable [BUSY]" if util is None else
          f"GPU: utilization avg {util:.0f} % (max {m['gpu_util_max']:.0f}, "
          f"{m['gpu_util_samples']} samples) [limit {GATE['max_gpu']:g}: {flag('gpu-util')}]")
    print("verdict: " + ("IDLE" if not found else "BUSY - " + "; ".join(t for _, t in found)),
          flush=True)
    return 1 if found else 0


# ----------------------------------------------------------------------------- overrides

def extract(run, rundir, outputs):
    stats = _original("extract")(run, rundir, outputs)
    if run["suite"] in GUARDED and run.get("weights") == VARIANT:  # flow-plan runs only
        stats["weights_check"] = weights_check(run, rundir)
    return stats


def _run_once(run, ledger):
    """run_campaign.execute, then the weights guard: one attempt at the run."""
    record = _original("execute")(run, ledger)
    if apply_guard(record):  # metrics.json was written before the verdict: keep it in step
        (FLOW_OUT / run["id"] / "metrics.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    if record.get("weights_guard") == "failed":
        problems = record["stats"]["weights_check"]["problems"]
        print(f"[{time.strftime('%H:%M:%S')}] !!! {GUARD} {run['id']} is FAILED: "
              f"{'; '.join(problems)[:400]}", flush=True)
    return record


def _monitored(run, ledger):
    """One attempt under a LoadMonitor: (record, external_cpu, external_gpu). execute() shows t0
    and t1 only in the record it returns, so the timed window is taken from the samples after it."""
    monitor = LoadMonitor()
    monitor.start()
    try:
        record = _run_once(run, ledger)
    finally:
        monitor.stop()
    return (record, *monitor.account(record.get("t0"), record.get("t1")))


def load_line(run_id, cpu, gpu, reasons):
    timed, seen = cpu.get("timed"), len(gpu["apps_seen"])
    polls, failed = gpu.get("polls", 0), gpu.get("polls_failed", 0)
    return (f"[{time.strftime('%H:%M:%S')}] load {run_id}: external CPU {cpu['threads_avg']:.2f} "
            f"threads over the run" + (f", {timed['threads_avg']:.2f} in the timed window"
                                       if timed else "")
            + f" (limit {threads(GATE['max_ext'])}), foreign GPU compute processes "
            + (f"{seen} seen" if seen else "none") + f" in {polls - failed} of {polls} polls"
            + (f" (longest stretch not watched {gpu.get('blind_s', 0):.0f} s)" if failed else "")
            + " -> " + ("CONTAMINATED: " + "; ".join(reasons) if reasons else "clean"))


def unpaired(run, ledger):
    """(id of the comparison base, why it is not there) for a T7 run whose base is not recorded ok
    with the semif/report.json that post_commands reads; None when the run has no base, or it is."""
    pair = pairing_base(run)
    if pair is None:
        return None
    base_id = pair[1]
    rec = (ledger.get("runs") or {}).get(base_id)
    if not isinstance(rec, dict):
        return base_id, "not recorded"
    if rec.get("status") != "ok":
        return base_id, f"recorded as {rec.get('status')}, not ok"
    if not (FLOW_OUT / base_id / "semif" / "report.json").exists():
        return base_id, "recorded ok but its semif/report.json is missing"
    return None


def execute(run, ledger):
    """run_campaign.execute behind the idle gate, with the load that was not ours accounted for.

    A contaminated attempt is moved aside and made again. When every attempt is contaminated the
    run is given up: the record returned is marked _discard, and save_ledger (which run_campaign's
    loop calls right after it stores whatever execute returned) takes it out again, so that the
    run stays undone and the next invocation retries it. A run that is compared with another run
    of this campaign is deferred the same way, without running, while that one is not recorded ok
    (run_campaign never revisits a recorded run: it would stay without its comparison for good)."""
    _require_installed()
    waiting = unpaired(run, ledger)
    if waiting:
        print(f"[{time.strftime('%H:%M:%S')}] deferred {run['id']}: its comparison base "
              f"{waiting[0]} is {waiting[1]}; not run, so that it is not recorded without the "
              "comparison (run this again once the base is recorded ok)", flush=True)
        return {"id": run["id"], "status": "deferred", "wall_s": 0.0, "exit_codes": [],
                "_discard": True}
    ledger["gate"] = gate_settings()
    for made in range(1, GATE["attempts"] + 1):
        no_leftovers()
        gate = wait_for_idle(run["id"], strict=made > 1) if GATE["on"] else {"enabled": False}
        record, cpu, gpu = _monitored(run, ledger)
        found = verdicts(cpu, gpu)
        reasons = [text for text, _ in found]
        record.update(gate={**gate, "limits": gate_limits()}, external_cpu=cpu, external_gpu=gpu)
        (FLOW_OUT / run["id"] / "metrics.json").write_text(  # written before: keep it in step
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(load_line(run["id"], cpu, gpu, reasons), flush=True)
        if not reasons:
            return record  # a failure that was clean is recorded as failed, like run_campaign does
        thrown = ledger.get("contaminated") or []
        attempt, moved = quarantine(run, 1 + sum(1 for c in thrown if c.get("id") == run["id"]))
        ledger.setdefault("contaminated", []).append({
            "id": run["id"], "attempt": attempt, "reason": "; ".join(reasons),
            "reason_public": "; ".join(public for _, public in found), "external_cpu": cpu,
            "external_gpu": gpu, "gate": record["gate"], "t0": record.get("t0"),
            "t1": record.get("t1"), "status": record.get("status"),
            "moved_to": rel(moved) if moved else None,
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S")})
        ledger.get("runs", {}).pop(run["id"], None)  # execute adds nothing; the loop does, later
        rc.save_ledger(ledger)
        try:
            render(ledger, plan())  # the reports show the thrown-away attempt at once
        except Exception as error:  # noqa: BLE001 - a report bug must not stop the campaign
            print(f"warning: render after a contaminated attempt failed ({error!r})",
                  file=sys.stderr)
    print(f"[{time.strftime('%H:%M:%S')}] !!! {run['id']}: {plural(GATE['attempts'], 'attempt')}, "
          f"all contaminated: the run stays undone and the next invocation retries it (output kept in "
          f"{rel(FLOW_OUT / CONTAMINATED)})", flush=True)
    record["status"], record["_discard"] = "contaminated", True
    return record


def save_ledger(ledger):
    """run_campaign.save_ledger, except that a run execute() gave up on (_discard) is never saved
    as a result: it is taken out of the ledger here, before the write."""
    runs = ledger.get("runs")
    if isinstance(runs, dict):
        for rid in [k for k, v in runs.items() if isinstance(v, dict) and v.get("_discard")]:
            del runs[rid]
    _original("save_ledger")(ledger)


def render(ledger, runs):
    """run_campaign.render, with the flow reports: render_flow.render(ledger, runs)."""
    _require_installed()
    before = json.dumps(ledger, sort_keys=True)
    for rec in ledger["runs"].values():  # the guard verdicts also apply after --reextract
        apply_guard(rec)
    try:
        if "machine" not in ledger:
            ledger["machine"] = rc.machine_info()
        try:
            import render_flow
        except ImportError as error:
            print(f"warning: render_flow is not importable ({error}): no reports written; "
                  "run --render once it exists", file=sys.stderr)
        else:
            with base_view():
                render_flow.render(ledger, runs)
    except Exception as error:  # noqa: BLE001 - a report bug must not stop the campaign
        if STATE["strict"]:
            raise
        with open(ERRORS_LOG, "a", encoding="utf-8") as log:
            log.write(f"{time.strftime('%F %T')} render: {error!r}\n")
        print(f"warning: render_flow.render failed ({error!r}); see {ERRORS_LOG.name}",
              file=sys.stderr)
    finally:
        # --render is "reports only": its copy of the ledger was loaded when it started, so an
        # unchanged copy is not written back (it could only overwrite something saved or edited by
        # hand since). The campaign loop and --reextract, whose changes are the point, save as
        # run_campaign.render does.
        reports_only = STATE["strict"] and not STATE["reextract"]
        if not reports_only or json.dumps(ledger, sort_keys=True) != before:
            rc.save_ledger(ledger)


def _refuse(*args, **kwargs):
    raise RuntimeError("HARDWARE-REPORT-RESEARCH.md and HARDWARE-REPORT-ISSUE.md belong to the "
                       "base campaign; the flow reports come from render_flow.render")


def _refuse_writer(*args, **kwargs):
    raise RuntimeError("run_campaign's writers (save_ledger, execute, main, render) are disabled "
                       "while render_flow.render runs: it reads the ledgers and writes the two "
                       "flow reports, nothing else")


def _ancestors():
    pids, pid = set(), os.getpid()
    while pid > 0 and pid not in pids:
        pids.add(pid)
        try:
            stat = (Path("/proc") / str(pid) / "stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])  # field 4: ppid
        except (OSError, ValueError, IndexError):
            break
    return pids


def leftover_model_processes():
    """rc.leftover_model_processes, except that this runner, its ancestors (uv, a shell) and the
    other campaign tools are not leftovers: run_flow.py has 'rizzo' in its path too."""
    mine = _ancestors()
    found = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in mine:
            continue
        try:
            cmd = (Path("/proc") / pid / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if str(rc.VENV) in cmd and not any(name in cmd for name in RUNNERS) and any(
                k in cmd for k in ("rizzo", "scripts/", "pytest")):
            found.append(f"{pid}: {cmd[:120]}")
    return found


def install():
    """Point run_campaign at the flow campaign. Idempotent; nothing is patched at import time."""
    if hasattr(rc, "_flow_originals"):
        return
    assert rc.OUT == BASE_OUT and rc.LEDGER == BASE_LEDGER, "run_campaign is not the base one"
    assert FLOW_OUT != BASE_OUT and FLOW_LEDGER != BASE_LEDGER
    names = ("plan", "model_args", "commands", "post_commands", "estimate", "extract", "execute",
             "render", "leftover_model_processes", "save_ledger")
    rc._flow_originals = {name: getattr(rc, name) for name in (
        *names, "render_research", "render_pr", "OUT", "LEDGER", "RESEARCH_MD", "PR_MD", "__doc__")}
    rc.OUT, rc.LEDGER = FLOW_OUT, FLOW_LEDGER
    rc.RESEARCH_MD = rc.PR_MD = FLOW_OUT / "_base-report-disabled.md"  # never created
    rc.__doc__ = __doc__  # argparse takes its description from the module's __doc__
    for name in names:
        setattr(rc, name, globals()[name])
    rc.render_research = rc.render_pr = _refuse  # the base reports stay untouched, whatever calls


PURE = ("plan", "model_args", "commands", "post_commands", "estimate", "extract")  # pure: no I/O
WRITERS = ("save_ledger", "execute", "main", "render")  # write a ledger, or start a model


@contextlib.contextmanager
def base_view():
    """run_campaign's pure functions as they were before install(), and none of its writers.

    render_flow was written against the pristine module (`python render_flow.py` does not see the
    flow patches), so plan(), commands(), extract()... are the base ones. rc.OUT and rc.LEDGER are
    NOT put back to results/local-hw and ledger.json: render_flow takes the base campaign's paths
    from its own constants, and with them restored a stray rc.save_ledger(...) would overwrite the
    141-run base ledger. As a second lock the writers are refused for the length of the call, the
    way install() refuses the base report writers for good."""
    originals = getattr(rc, "_flow_originals", {})
    swaps = {name: originals[name] for name in PURE if name in originals}
    swaps.update({name: _refuse_writer for name in WRITERS})
    saved = {name: getattr(rc, name) for name in swaps}
    for name, value in swaps.items():
        setattr(rc, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(rc, name, value)


# ----------------------------------------------------------------------------- main

_LOCK = {}  # the open lock file: held until the process exits


def lock_path():
    return FLOW_LEDGER.with_suffix(".lock")  # ledger-flow.lock, next to the ledger it protects


def acquire_lock(role):
    """Become the only process that writes the flow ledger and reports (see the module docstring).

    run_campaign's --render/--reextract, and so ours, load the ledger, work for a while and save
    that copy back: run beside a live campaign they silently drop the records it saved in between
    (and would share the ledger's tmp file). An exclusive flock is atomic, is not fooled by how a
    process was started (a /proc scan is), and the kernel releases it when the holder dies, so it
    cannot go stale. Re-entrant within a process."""
    path = lock_path()
    if _LOCK.get("path") == path and not _LOCK["handle"].closed:
        return
    handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115 - held for the life of the process
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        holder = handle.read().strip() or "another process"
        handle.close()
        raise SystemExit(
            f"{rel(path)} is held by {holder}. Only one process writes the flow ledger and reports "
            f"at a time; the campaign rewrites the reports after every run. Wait for it to finish "
            f"(or stop it), then run this again.") from None
    except OSError as error:  # not "locked": the filesystem cannot lock, say so instead of a traceback
        handle.close()
        raise SystemExit(f"cannot lock {rel(path)}: {error}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"{role}, pid {os.getpid()}, since {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    handle.flush()
    _LOCK.update(path=path, handle=handle)


def _same_file(stream, path):
    try:
        a, b = os.fstat(stream.fileno()), os.stat(path)
        return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    except (OSError, ValueError, AttributeError):
        return False


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno...
        return getattr(self.stream, name)


def pending(args):
    """The runs rc.main will do: the plan minus the ledger, then --only and --limit."""
    done = rc.load_ledger()["runs"]
    todo = [r for r in plan() if r["id"] not in done]
    if args.only:
        keys = args.only.split(",")
        todo = [r for r in todo if any(k in r["id"] for k in keys)]
    if args.limit:
        todo = todo[: args.limit]
    return todo


def wanted(todo):
    """'size/quant' of every flow GGUF the runs to do load, in plan order."""
    keys = []
    for r in todo:
        if r["suite"] in GUARDED:
            key = "4b/q8_0" if r.get("pin_4b") else f"{r['size']}/{r['quant']}"
            if key not in keys:
                keys.append(key)
    return keys


def header(todo, args):
    base, mode = base_ledger()["runs"], "dry-run" if args.dry_run else "campaign"
    from_base = sum(1 for r in todo if measured(r) is not None)
    print(f"# flow follow-up ({mode}): {len(plan())} runs planned, {len(todo)} to do; every model "
          f"command passes --weights {VARIANT}")
    print(f"# output {rel(FLOW_OUT)}, ledger {rel(FLOW_LEDGER)}"
          f"{'' if FLOW_LEDGER.exists() else ' (absent)'}, log {rel(FLOW_LOG)}")
    print(f"# base campaign, read-only: {len(base)} runs in {rel(BASE_LEDGER)}, outputs in "
          f"{rel(BASE_OUT)}; run ids pair with the flow ids")
    print(f"# ETA: measured wall time of the same run id in the base ledger + "
          f"{rc.BASELINE_S + rc.SETTLE_S} s baseline/settle for {from_base} of {len(todo)} runs; "
          f"rc.estimate() model for the other {len(todo) - from_base}")
    print(f"# cpu threads: {cpu_threads()} (rc.cpu_threads on the base ledger)")
    print(f"# idle gate: {gate_text()}")
    pairs = [(r["id"], pairing_base(r)[1]) for r in todo if pairing_base(r)]
    if pairs:  # a run that is compared with another one of this campaign waits for it (execute)
        print(f"# deferral: {', '.join(i for i, _ in pairs)} wait until "
              f"{', '.join(sorted({b for _, b in pairs}))} is recorded ok, to be paired with it")
    try:
        specs = flow_specs()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"# weights: cannot read rizzo_flow.config ({error})")
        return
    for key in wanted(todo):
        path = ROOT / specs[key]["path"]
        state = f"{path.stat().st_size / 1e9:.2f} GB, present" if path.is_file() else "MISSING"
        print(f"# weights {key}: {specs[key]['path']} ({state}) -> served as "
              f"{specs[key]['served_id']}")


def preflight(todo):
    """Fail before the first run, not after an hour of failed runs: files and tools the runs need."""
    problems = []
    try:
        specs = flow_specs()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"flow campaign pre-flight failed: {error}") from error
    for key in wanted(todo):
        path = ROOT / specs[key]["path"]
        if not path.is_file():
            problems.append(f"missing flow GGUF {rel(path)} (rizzo download --only weights)")
        if path.with_name(path.name + ".part").exists():
            problems.append(f"{rel(path)}.part exists: a download did not finish")
    bench = [ROOT / "benchmarks" / "smoke.jsonl", ROOT / "benchmarks" / "perturbations.jsonl"]
    needs = {
        "T0": [rc.VENV / "pytest"],
        "T1": [rc.VENV / "pytest", CAMPAIGN / "pin_4b_flow_plugin.py"],
        "T4": [rc.VENV / "rizzo", bench[0]],
        "T5": [rc.VENV / "rizzo", bench[1]],
        "T6": [rc.VENV / "python", ROOT / "scripts" / "validate_checkpoint.py",
               ROOT / "examples" / "ticket.json", *bench],
        "T7": [rc.VENV / "python", ROOT / "scripts" / "semif_compare.py",
               ROOT / "scripts" / "semif_report.py", rc.SEMIF / "benchmarks" / "data"],
        "T9": [rc.VENV / "python", ROOT / "scripts" / "typed_decisions.py", rc.TYPED],
    }
    wanted_paths = {p for r in todo for p in needs.get(r["suite"], [])}
    if any(r.get("device") == "cpu" and r["suite"] in ("T6", "T7", "T9") for r in todo):
        wanted_paths.add(CAMPAIGN / "threads_wrapper.py")
    problems += [f"missing {rel(p)}" for p in sorted(wanted_paths) if not p.exists()]
    if problems:
        raise SystemExit("flow campaign pre-flight failed:\n  " + "\n  ".join(problems))
    try:
        missing = importlib.util.find_spec("render_flow") is None
    except ValueError:  # already imported (no __spec__): it is there
        missing = False
    if missing:
        print("warning: render_flow.py not found: the campaign will run but no report is written "
              "until it exists (then: run_flow.py --render)", file=sys.stderr)


def summary():
    ledger, runs = rc.load_ledger(), plan()
    done, total, failed, elapsed, _, _ = rc.progress(ledger, runs)
    guard = [rid for rid, rec in ledger["runs"].items() if rec.get("weights_guard") == "failed"]
    thrown = ledger.get("contaminated") or []
    thrown_ids = ", ".join(sorted({str(c.get("id", "?")) for c in thrown}))
    print(f"# flow campaign: {done}/{total} runs recorded, {failed} failed "
          f"({len(guard)} by the weights guard{': ' + ', '.join(guard) if guard else ''}); "
          f"{plural(len(thrown), 'contaminated attempt')} thrown away"
          f"{f' ({thrown_ids})' if thrown else ''}; "
          f"elapsed {rc.hms(elapsed)}", flush=True)


def gate_args():
    """The gate's flags, taken out of sys.argv (run_campaign's parser would refuse them) and put in
    GATE. --check-idle is returned as True. Exits with a message for a value that cannot work."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--check-idle", action="store_true")
    parser.add_argument("--no-idle-gate", action="store_true")
    parser.add_argument("--max-external-threads", type=float)
    parser.add_argument("--max-gpu-util", type=float)
    parser.add_argument("--gate-timeout", type=float)
    parser.add_argument("--max-attempts", type=int)
    args, rest = parser.parse_known_args()
    STATE["argv"], sys.argv[1:] = sys.argv[1:], rest  # the log line keeps the full command line
    if args.max_external_threads is not None and not args.max_external_threads > 0:
        raise SystemExit("--max-external-threads must be greater than 0")
    if args.max_gpu_util is not None and not 0 <= args.max_gpu_util <= 100:
        raise SystemExit("--max-gpu-util is a percentage, 0 to 100")
    if args.gate_timeout is not None and not args.gate_timeout > 0:
        raise SystemExit("--gate-timeout is a number of minutes greater than 0")
    if args.max_attempts is not None and args.max_attempts < 1:
        raise SystemExit("--max-attempts must be at least 1")
    GATE["on"] = not args.no_idle_gate
    timeout = None if args.gate_timeout is None else args.gate_timeout * 60
    for key, value in (("max_ext", args.max_external_threads), ("max_gpu", args.max_gpu_util),
                       ("attempts", args.max_attempts), ("timeout_s", timeout)):
        if value is not None:
            GATE[key] = value
    return args.check_idle


def main():
    if gate_args():  # --check-idle: before install(); no lock, nothing patched, nothing written
        sys.exit(check_idle())
    install()
    parser = argparse.ArgumentParser(add_help=False)  # only to know what rc.main will do
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--reextract", action="store_true")
    parser.add_argument("--only")
    parser.add_argument("--limit", type=int)
    args, _ = parser.parse_known_args()
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        return rc.main()
    reports = args.render or args.reextract
    real = not (args.dry_run or reports)
    if reports and not FLOW_LEDGER.exists():
        raise SystemExit(f"{rel(FLOW_LEDGER)} does not exist: nothing to render or re-extract")
    if not args.dry_run:  # the campaign and the report modes write: one of them at a time
        acquire_lock("campaign" if real else "--reextract" if args.reextract else "--render")
    STATE["strict"], STATE["reextract"] = reports, args.reextract
    if real:  # the base log came from the shell's redirection; this one is written here
        log = open(FLOW_LOG, "a", encoding="utf-8")  # noqa: SIM115 - lives as long as the process
        if not _same_file(sys.stdout, FLOW_LOG):  # unless the shell already sends it there
            sys.stdout = _Tee(sys.stdout, log)
        if not _same_file(sys.stderr, FLOW_LOG):
            sys.stderr = _Tee(sys.stderr, log)
        print(f"\n=== run_flow.py {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} "
              f"args {STATE.get('argv', sys.argv[1:])} ===", flush=True)
    if not reports:
        todo = pending(args)
        header(todo, args)
        if real:
            preflight(todo)
    try:
        rc.main()
    except GateTimeout:  # --gate-timeout: stopped between runs, the ledger as the last run left it
        if real:
            summary()
        raise
    if real:
        summary()
        recorded = rc.load_ledger()["runs"]
        left = [r["id"] for r in todo if r["id"] not in recorded]
        if left:  # contaminated on every attempt, or deferred: not a complete campaign
            print(f"# {plural(len(left), 'run')} left undone: {', '.join(left)}; run this again to do "
                  f"them (exit status {EX_TEMPFAIL})", flush=True)
            sys.exit(EX_TEMPFAIL)


if __name__ == "__main__":
    main()
