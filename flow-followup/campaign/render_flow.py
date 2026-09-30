#!/usr/bin/env python3
"""Report renderer for the fine-tuned-weights follow-up (`--weights flow`) of the hardware campaign.

Companion of run_flow.py (the runner). The follow-up repeats 39 runs of the base campaign of issue
#25 with the same run ids, so every flow run pairs with the base run of the same id. This file
only reads:

  - the flow ledger (ledger-flow.json; may be missing or partial: pending runs print as "—"),
  - the base ledger of issue #25 (ledger.json) and the raw outputs of both campaigns
    (results/local-hw-flow/<id>/ and results/local-hw/<id>/),
  - the maintainers' published numbers, read from the repository: results/semif-compare/rizzo-*,
    results/llama-q8_0-cuda-validation, README.md and docs/training.md (typed-decisions tables),

and writes, in the repository root by default:

  - HARDWARE-REPORT-FLOW.md          working notes: header, checks, tables, per-run log,
  - HARDWARE-REPORT-FLOW-COMMENT.md  paste-ready follow-up comment for issue #25 (English,
                                     GitHub markdown, under 60000 characters).

Both contain the same tables, one per question, and read (stats = ledger record's `stats`;
raw = <out>/<id>/ of that campaign; base and flow are paired by run id):
  a SemIf, flow T7 (6 CUDA f16, 2 KV, CPU) + maintainers' flow row: stats authored144,
    perturbations108, held_out_*, flips_*, missing_evidence_confident, rule_application_pert_*,
    p50_s/p95_s, shared_dps/direct_dps, sd_flips/sd_rows, direct_states, peak_device_bytes,
    against["maintainers-flow"]; raw semif/report.json (missing-evidence accuracy),
    semif/analysis.json (perturbations108 paired difference); reference folder
    results/semif-compare/rizzo-flow-q8_0-v3-llama-cuda/{report,analysis}.json
  b fine-tuning effect: the same keys of flow and base T7, stats.against["this-base"] +
    raw analysis.json; reference row from the maintainers' flow analysis.json, its timing cells
    against their same-day base rerun (README.md, Results so far, footnote 1)
  c smoke / perturbations, T4 and T5: stats accuracy, categorical_rows, nll, brier, ece,
    latency_*, decisions_per_second, changed_argmaxes, mode_decisions, peak_device_bytes;
    reference row results/llama-q8_0-cuda-validation/{smoke,perturbations}.json
  d typed-decisions, T9: stats accuracy, kl, brier, ece, p50_ms, p95_ms, decisions_per_second;
    raw typed.json (cases) + .research/typed-decisions/all/test.jsonl for the paired bootstrap;
    reference rows parsed from README.md and docs/training.md
  e validate_checkpoint, T6: stats smoke_*, perturbations_accuracy, long_*, validation_s,
    model.load_seconds, peak_device_bytes, ledger status and exit_codes; reference row
    results/llama-q8_0-cuda-validation/summary.json
  f KV cache: T4 and T7 of cuda-4b-q8_0-{f16,q8_0,q4_0}, stats.against["this-cuda-kv-f16"] +
    raw analysis.json
  g CPU: T4, T7, T9 of cpu-4b-q8_0-f16, stats.against["this-cuda"] + raw analysis.json, the
    CUDA runs' p50 for the ratios
  h telemetry of the T7 runs: record keys nvidia_smi, prometheus, rusage, wall_s
  i commands: record keys commands and env (a template from run_campaign.commands until run)
  j machine state: run_flow.py's idle gate and load accounting. Record keys gate,
    external_cpu (threads_avg, timed.threads_avg, top_names...) and external_gpu (apps_seen,
    polls, polls_failed, blind_s); ledger keys gate (the settings) and contaminated (the attempts
    that were thrown away: reason, reason_public). Records from before the gate lack them: every
    figure then prints as a dash. The working notes name the processes behind the load; the
    GitHub comment, which goes to a public issue, does not: counts and numbers only (its
    thrown-away attempts come from reason_public).
Also: machine (ledger["machine"]), weights check (raw model metadata against rizzo_flow.config,
T1 through weights-evidence.json), run status and log (record, rc.stats_lines, rc.telemetry_line).

  python .research/hw-campaign/render_flow.py                  # from ledger-flow.json
  python .research/hw-campaign/render_flow.py --out-dir DIR    # write somewhere else (previews)

Optional hand-written text, as for the base report: pr-flow-summary.md (intro) and
pr-flow-notes.md ("Anything odd") in this folder. The data-branch link and the ambient temperature
line are placeholders until given with --data-url / --ambient (or RIZZO_FLOW_DATA_URL /
RIZZO_FLOW_AMBIENT in the environment of the runner).

Generated text states numbers and intervals only. In the tables a run whose status is not ok
prints dashes only: no raw file is read for its numbers (the run log of the working notes lists
what it produced, marked with its status). The comment never exceeds 60000 characters: sections
are left out in OMIT_ORDER, or render() raises when the hand-written intro alone is too long.
Nothing here writes a ledger, anything under results/, run_campaign.py's reports or the base
report; no model is ever loaded.
"""

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # importing the runner must not touch its __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_campaign as rc  # noqa: E402

g = rc.g
ROOT = rc.ROOT
CAMPAIGN = Path(__file__).resolve().parent
# The base campaign's files come from fixed paths, never from rc.OUT / rc.LEDGER: run_flow.py may
# point those module globals at the flow campaign while it reuses run_campaign.execute().
BASE_LEDGER = CAMPAIGN / "ledger.json"
BASE_OUT = ROOT / "results" / "local-hw"
FLOW_OUT = ROOT / "results" / "local-hw-flow"  # raw outputs (run_flow.py OUT)
FLOW_LEDGER = CAMPAIGN / "ledger-flow.json"  # run_flow.py LEDGER, same schema as ledger.json
FLOW_MD = ROOT / "HARDWARE-REPORT-FLOW.md"
COMMENT_MD = ROOT / "HARDWARE-REPORT-FLOW-COMMENT.md"
FLOW_SUMMARY = CAMPAIGN / "pr-flow-summary.md"  # optional hand-written intro
FLOW_NOTES = CAMPAIGN / "pr-flow-notes.md"  # optional hand-written notes for "Anything odd"
COMMENT_LIMIT = 60000  # GitHub accepts 65536 characters per comment
# Bounds of the "Anything odd" bullets that quote a run's stderr or a test name.
ODD_STDERR_CHARS, ODD_MAX_RUNS, ODD_MAX_TESTS = 300, 12, 12

# HARDWARE-REPORT-RESEARCH.md, section 10 ("Published").
ISSUE_URL = "https://github.com/Rizzo-AI-Academy/rizzo-flow/issues/25"
BASE_DATA_URL = "https://github.com/Axiumine/rizzo-flow/tree/hardware-report-issue-25"
# Placeholder until the follow-up data branch exists: --data-url, or RIZZO_FLOW_DATA_URL when the
# runner calls render() (which takes no such argument). Same for the ambient temperature line.
DATA_URL = os.environ.get("RIZZO_FLOW_DATA_URL") or "DATA_BRANCH_URL"
AMBIENT = os.environ.get("RIZZO_FLOW_AMBIENT") or (
    f"not recorded for this follow-up. Base campaign: {rc.AMBIENT}")

MAINT = ROOT / "results" / "semif-compare"
MAINT_FLOW = MAINT / "rizzo-flow-q8_0-v3-llama-cuda"  # maintainers' flow 4B Q8_0 CUDA (T7)
MAINT_BASE = ROOT / rc.PUBLISHED[("cuda", "4b", "q8_0")]  # their base run, same config
MAINT_VALIDATION = ROOT / "results" / "llama-q8_0-cuda-validation"  # their base smoke/validate
REF_OS = "Windows 10"  # CLAUDE.md, "Passaggio a llama.cpp": "Windows 10 + RTX 5060 Ti"
MAINT_PAIR_ID = "cuda-4b-q8_0-f16-T7"  # the only run with a maintainers-flow counterpart
KV_PREFIX = {"f16": "cuda-4b-q8_0-f16", "q8_0": "cuda-4b-q8_0-q8_0", "q4_0": "cuda-4b-q8_0-q4_0"}
# SemIf authored144 rows with provenance.variant == "missing" (the right answer is `insufficient`);
# run_campaign.stats_lines prints the same "/36".
MISSING_ROWS = 36
# Paired bootstrap over typed-decisions cases: same draws and seed as SemIf's analysis.json blocks.
TYPED_SAMPLES, TYPED_SEED = 1000, 217

FLOW_IDS = [
    "T0-unit",
    "T1-integration-4b",
    *[
        f"cuda-{size}-{quant}-f16-{suite}"
        for size, quant in rc.WEIGHTS
        for suite in ("T4", "T5", "T6", "T7", "T9")
    ],
    "cuda-4b-q8_0-q8_0-T4",
    "cuda-4b-q8_0-q8_0-T7",
    "cuda-4b-q8_0-q4_0-T4",
    "cuda-4b-q8_0-q4_0-T7",
    "cpu-4b-q8_0-f16-T4",
    "cpu-4b-q8_0-f16-T7",
    "cpu-4b-q8_0-f16-T9",
]
assert len(FLOW_IDS) == 39 and len(set(FLOW_IDS)) == 39

# Row labels of the published typed-decisions tables (the numbers are parsed from the files).
# README.md, "Training and calibration > Results on typed-decisions" (table).
README_TYPED_ROWS = {
    ("4b", "q8_0", "base"): "Spark-X2.5-4B, base",
    ("4b", "q8_0", "flow"): "Rizzo Flow 4B (default)",
    ("4b", "q4_k_m", "flow"): "Rizzo Flow 4B, Q4_K_M",
    ("1.7b", "q8_0", "base"): "Spark-X2.5-1.7B, base",
    ("1.7b", "q8_0", "flow"): "Rizzo Flow 1.7B",
    ("1.7b", "q4_k_m", "flow"): "Rizzo Flow 1.7B, Q4_K_M",
}
# docs/training.md, section 10 ("Numeri di partenza da battere", table): the base BF16 row.
TRAINING_TYPED_ROWS = {("4b", "bf16", "base"): "Spark 4B BF16"}

DASH = "—"


# ----------------------------------------------------------------------------- plan and files

def flow_plan():
    """The 39 flow runs from run_campaign.plan(): configuration and extras (T7 direct states) are
    the base runs' own, id for id."""
    by_id = {r["id"]: r for r in rc.plan()}
    missing = [i for i in FLOW_IDS if i not in by_id]
    if missing:
        raise SystemExit(f"run ids not in run_campaign.plan(): {missing}")
    return [by_id[i] for i in FLOW_IDS]


def load_ledger_file(path):
    """A ledger dict; {} when the file is missing or unreadable (nothing has run yet)."""
    data = rc.read_json(path)
    return data if isinstance(data, dict) else {}


_RAW = {}


def rkey(path):
    """Identity of a file's current content for caching: (path, mtime, size), None if absent."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return str(path), stat.st_mtime_ns, stat.st_size


def rj(path):
    """rc.read_json with a small cache keyed by path, mtime and size (the runner renders after
    every run, and most raw files do not change)."""
    key = rkey(path)
    if key is None:
        return None
    path = Path(path)
    if key not in _RAW:
        if len(_RAW) > 600:
            _RAW.clear()
        _RAW[key] = rc.read_json(path)
    return _RAW[key]


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def optional_text(path):
    text = read_text(path).strip() if path else ""
    return text or None


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=15)
    except Exception:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


# ----------------------------------------------------------------------------- formatting

def flt(value, digits=3):
    """run_campaign.fmt for rates and scores: a whole number from JSON (accuracy 1) still prints
    with decimals."""
    if isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    return rc.fmt(value, digits)


def sflt(value, digits=3):
    """Signed difference, +0.033 / -0.033 (never -0.000)."""
    if value is None:
        return DASH
    return f"{round(value, digits) + 0.0:+.{digits}f}"


def cnt(value):
    return rc.fmt(value)


def frac(n, d):
    return DASH if n is None or d is None else f"{n}/{d}"


def join(*cells, sep=" / "):
    return DASH if all(c == DASH for c in cells) else sep.join(cells)


def arrow(before, after):
    return DASH if before == DASH and after == DASH else f"{before} → {after}"


def kilo(value):
    return None if value is None else value / 1e3


def times(num, den):
    """Ratio of two timings as `×12.3`."""
    if not num or not den:
        return DASH
    return f"×{num / den:.1f}"


def ci_text(ci):
    """`[lo, hi]`, plus "includes 0" when the interval contains 0 at the printed precision."""
    if not (isinstance(ci, (list, tuple)) and len(ci) == 2
            and all(isinstance(x, (int, float)) for x in ci)):
        return DASH
    lo, hi = ci
    mark = " · includes 0" if round(lo, 3) <= 0 <= round(hi, 3) else ""
    return f"[{sflt(lo)}, {sflt(hi)}]{mark}"


def diff_cell(pair):
    """(difference, ci) -> `+0.033 [-0.033, +0.105] · includes 0`."""
    difference, ci = pair
    return DASH if difference is None else f"{sflt(difference)} {ci_text(ci)}"


def acc_nn(stats):
    """Accuracy as `19/20 (0.950)` from the evaluate summary."""
    acc, rows = stats.get("accuracy"), stats.get("categorical_rows")
    if acc is None or not rows:
        return DASH
    return f"{round(acc * rows)}/{rows} ({flt(acc)})"


def esc(text):
    return str(text).replace("|", "\\|")


def T(header, rows):
    """run_campaign.table (separator row |---|---|) with escaped cells; a missing cell is —."""
    cells = []
    for row in rows:
        row = [DASH if c is None else esc(c) for c in row]
        if len(row) != len(header):
            raise ValueError(f"row of {len(row)} cells under {len(header)} headers: {row[:2]}")
        cells.append(row)
    return rc.table(header, cells)


def details(summary, lines):
    return [f"<details><summary>{summary}</summary>", "", *lines, "", "</details>", ""]


# ----------------------------------------------------------------------------- one run

class Entry:
    """One run on one side of a pair (flow or base): its ledger record and its raw folder."""

    def __init__(self, run, rec, out):
        self.run, self.rec, self.out = run, rec, Path(out)

    @property
    def id(self):
        return self.run["id"]

    @property
    def stats(self):
        """The run's statistics for the tables: nothing from a run that is not ok (the runner's
        weights guard also fails runs that produced numbers, and those must not be shown)."""
        stats = self.rec.get("stats") if self.ok else None
        return stats if isinstance(stats, dict) else {}

    @property
    def against(self):
        """The `--against` blocks of a T7 record, {name: block}."""
        blocks = self.stats.get("against")
        if not isinstance(blocks, dict):
            return {}
        return {k: v for k, v in blocks.items() if isinstance(v, dict)}

    @property
    def status(self):
        return "pending" if self.rec is None else str(self.rec.get("status"))

    @property
    def ok(self):
        return self.rec is not None and self.rec.get("status") == "ok"

    def raw(self, *parts):
        """A raw output file of this run, whatever the run's status: the weights check needs the
        metadata of failed runs too. Whatever prints numbers from it must test `ok` first, as
        view(), paired() and typed_paired() do."""
        return rj(self.out / self.id / Path(*parts))

    def raw_key(self, *parts):
        return rkey(self.out / self.id / Path(*parts))


def cfg(entry):
    """Configuration label; a run that has not produced numbers says so."""
    note = "" if entry.ok else " (pending)" if entry.rec is None else " (failed)"
    return esc(rc.label(entry.run)) + note


def mem(entry):
    if not entry.ok:
        return DASH
    try:
        return rc.mem_cell(entry.rec)
    except Exception:
        return DASH


def pick(runs, suite, device=None, kv=None):
    """Plan entries of one suite; kv=False keeps the default KV cache, kv=True the others."""
    out = []
    for run in runs:
        if run["suite"] != suite or (device and run.get("device") != device):
            continue
        if (kv is False and run.get("kv")) or (kv is True and not run.get("kv")):
            continue
        out.append(run)
    return out


def view(entry):
    """T7 stats of a ledger entry plus the missing-evidence accuracy, which only report.json has.
    A run that is not ok has no numbers: its raw files are not read for them either."""
    v = dict(entry.stats)
    v["missing_acc"] = (g(entry.raw("semif", "report.json"), "stability.missing_evidence.accuracy")
                        if entry.ok else None)
    return v


def block_pairs(ledger_block, raw_block):
    """One `--against` block -> {fixture: (difference, ci)}, different argmax and row count.
    The ledger holds authored144 only; analysis.json holds both fixtures."""
    ledger_block, raw_block = ledger_block or {}, raw_block or {}
    diffs = raw_block.get("paired_difference_ours_minus_theirs") or {}
    out = {}
    for fixture in ("authored144", "perturbations108"):
        d = diffs.get(fixture) or {}
        out[fixture] = (d.get("difference"), d.get("paired_source_group_bootstrap_95"))
    if ledger_block.get("difference_authored144") is not None:
        out["authored144"] = (ledger_block["difference_authored144"],
                              ledger_block.get("ci95_authored144"))
    for key in ("different_argmax", "rows"):
        out[key] = ledger_block.get(key, raw_block.get(key))
    out["found"] = bool(ledger_block or raw_block)
    out["method"] = next((d for d in diffs.values() if isinstance(d, dict) and d.get("method")), {})
    return out


def paired(entry, name):
    """The paired comparison of a T7 run against `name` (maintainers-flow, this-base,
    this-cuda-kv-f16, this-cuda): ledger stats first, analysis.json for what the ledger lacks.
    Nothing for a run that is not ok (a weights-guard failure leaves its analysis.json on disk)."""
    if not entry.ok:
        return block_pairs(None, None)
    raw_blocks = (entry.raw("semif", "analysis.json") or {}).get("against")
    raw_block = raw_blocks.get(name) if isinstance(raw_blocks, dict) else None
    return block_pairs(entry.against.get(name), raw_block if isinstance(raw_block, dict) else None)


def expected_against(run):
    """The `--against` block that run_flow.py's post_commands adds to a KV or CPU T7 run (the f16
    run of the same weights, the CUDA run); None for the T7 runs that have no such comparison."""
    if run.get("suite") != "T7":
        return None
    if run.get("kv"):
        return f"this-{run.get('device')}-kv-f16"
    return "this-cuda" if run.get("device") != "cuda" else None


# ----------------------------------------------------------------------------- weights check

def model_meta(entry):
    """The model metadata block the run's own tool wrote (weights, file hashes, KV type, device)."""
    suite = entry.run["suite"]
    if suite == "T7":
        return (entry.raw("semif", "report.json") or {}).get("model")
    if suite in ("T4", "T5"):
        name = "smoke.json" if suite == "T4" else "perturbations.json"
        rows = (entry.raw(name) or {}).get("rows")
        return g(rows[0], "response.model") if rows else None
    if suite == "T6":
        return (entry.raw("validate", "summary.json") or {}).get("model")
    if suite == "T3":
        return (entry.raw("decide-ticket.json") or {}).get("model")
    if suite == "T9":
        return (entry.raw("typed.json") or {}).get("metadata")
    return None


EVIDENCE_FILE = "weights-evidence.json"  # written by run_flow's pin_4b_flow_plugin.py (T1)


def check_t1(entry, specs):
    """The integration run is pinned to the flow 4B Q8_0 by the runner's plugin, which records
    what it pinned and the metadata of every backend the tests loaded."""
    data = entry.raw(EVIDENCE_FILE)
    if not isinstance(data, dict):
        return "unknown", f"no {EVIDENCE_FILE}"
    spec = (specs or {}).get(("4b", "q8_0"))
    problems = []
    pinned = data.get("pinned") if isinstance(data.get("pinned"), dict) else {}
    if spec and pinned.get("sha256") != spec.sha256:
        problems.append(f"pinned file {pinned.get('file')!r}")
    loaded = [m for m in (data.get("loaded") or []) if isinstance(m, dict)]
    if not loaded:
        problems.append("no model loaded")
    problems += [f"loaded weights={m.get('weights')!r}" for m in loaded
                 if m.get("weights") != "flow"][:3]
    if problems:
        return "mismatch", "; ".join(problems)
    return "ok", f"pinned {pinned.get('file')}, {len(loaded)} backends loaded, all weights flow"


def check_weights(entry, base, specs):
    """('ok' | 'mismatch' | 'unknown' | 'n/a', detail): does the raw output show the pinned flow
    GGUF, the requested precision, device and KV type, and a fingerprint different from the base
    run's?"""
    if entry.run["suite"] == "T0":
        return "n/a", "no model in a unit-test run"
    if entry.rec is None:
        return "unknown", "pending"
    if entry.run["suite"] == "T1":
        return check_t1(entry, specs)
    meta = model_meta(entry)
    if not isinstance(meta, dict):
        return "unknown", "no raw model metadata"
    run = entry.run
    spec = (specs or {}).get((run["size"], run["quant"]))
    problems = []
    if meta.get("weights") != "flow":
        problems.append(f"weights={meta.get('weights')!r}")
    if spec:
        if meta.get("gguf_source") != spec.repo or meta.get("gguf_revision") != spec.revision:
            problems.append(f"GGUF {meta.get('gguf_source')}@{str(meta.get('gguf_revision'))[:8]}")
        files = meta.get("source_files")  # by hash: the file may have been renamed
        if isinstance(files, dict) and spec.sha256 not in files.values():
            problems.append(f"file hash {sorted(files)}")
    if meta.get("precision") != run["quant"]:
        problems.append(f"precision={meta.get('precision')}")
    if run.get("device") == "cuda" and meta.get("backend") != "cuda":
        problems.append(f"backend={meta.get('backend')}")
    if run.get("device") == "cpu" and meta.get("device") != "cpu":
        problems.append(f"device={meta.get('device')}")
    wanted_kv = run.get("kv") if run.get("kv") not in (None, "f16") else None
    if meta.get("kv_cache") != wanted_kv:
        problems.append(f"kv_cache={meta.get('kv_cache')}")
    other = model_meta(base) if base is not None else None
    if isinstance(other, dict) and meta.get("fingerprint") \
            and meta.get("fingerprint") == other.get("fingerprint"):
        problems.append("same fingerprint as the base run")
    if problems:
        return "mismatch", "; ".join(problems)
    sha = ""
    files = meta.get("source_files")
    if isinstance(files, dict) and files:
        sha = ", sha256 " + next(iter(files.values()))[:12] + "…"
    return "ok", f"weights flow{sha}"


def flow_specs():
    """The six pinned flow GGUF files, {(size, quant): GgufSpec}, from rizzo_flow.config."""
    try:
        from rizzo_flow import config
    except Exception:
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from rizzo_flow import config
        except Exception as error:
            return None, repr(error)
    try:
        return {(s, q): config.gguf_spec(s, q, "flow") for s, q in rc.WEIGHTS}, None
    except Exception as error:
        return None, repr(error)


# ----------------------------------------------------------------------------- references

class _Flat:
    """Path-like adapter so run_campaign.extract() reads a folder that has no `semif/` (or
    `validate/`) level, like the maintainers' results/semif-compare/* folders."""

    def __init__(self, folder, level):
        self.folder, self.level = Path(folder), level

    def __truediv__(self, name):
        return self.folder if name == self.level else self.folder / name


def folder_view(folder):
    """T7 view of a semif_compare folder, derived exactly as for the campaign's own runs."""
    report = rj(Path(folder) / "report.json")
    if not report:
        return None
    v = rc.extract({"suite": "T7"}, _Flat(folder, "semif"), {})
    v["missing_acc"] = g(report, "stability.missing_evidence.accuracy")
    return v


def md_table_after(text, pattern):
    """Rows (lists of cell strings) of the first markdown table after the first line matching."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if re.search(pattern, line)), None)
    if start is None:
        return []
    rows = []
    for line in lines[start + 1:]:
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                continue
            rows.append(cells)
        elif rows:
            break
    return rows


def first_number(cell):
    match = re.search(r"[-+−]?\d+(?:\.\d+)?", cell.replace("*", ""))
    return float(match.group(0).replace("−", "-")) if match else None


def typed_rows(text, pattern, labels, source):
    """{(size, quant, variant): {accuracy, kl, brier, ece, p50_ms, source}} from one table."""
    rows = md_table_after(text, pattern)
    if not rows:
        return {}
    head = [c.lower() for c in rows[0]]

    def col(word):
        return next((i for i, h in enumerate(head) if word in h), None)

    index = {k: col(w) for k, w in
             (("accuracy", "accuracy"), ("kl", "kl"), ("brier", "brier"), ("ece", "ece"),
              ("p50_ms", "p50"))}
    out = {}
    for key, label in labels.items():
        for cells in rows[1:]:
            if cells and cells[0].replace("*", "").strip() == label:
                row = {k: (first_number(cells[i]) if i is not None and i < len(cells) else None)
                       for k, i in index.items()}
                out[key] = {**row, "source": source}
    return out


def load_typed_refs():
    readme = read_text(ROOT / "README.md")
    training = read_text(ROOT / "docs" / "training.md")
    out = typed_rows(training, r"Numeri di partenza da battere", TRAINING_TYPED_ROWS,
                     "docs/training.md §10")
    out.update(typed_rows(readme, r"^### Results on typed-decisions", README_TYPED_ROWS,
                          "README.md"))
    text = re.sub(r"\s+", " ", readme)
    match = re.search(r"Accuracy ([+\-−]\d\.\d+), 95% interval \[([+\-−]\d\.\d+), "
                      r"([+\-−]\d\.\d+)\]", text)
    if match:  # README.md, Results on typed-decisions, first bullet: 4B base -> flow
        out["paired_4b_q8_0"] = tuple(float(x.replace("−", "-")) for x in match.groups())
    return out


def load_same_day():
    """The maintainers' base rerun on the day of their flow run, from README.md, Results so far,
    footnote 1 ("the base weights run again right after it on the same machine gave 66 / 73 ms and
    15.64 decisions/s"): {p50_s, p95_s, shared_dps}, None when that sentence is not there."""
    text = re.sub(r"\s+", " ", read_text(ROOT / "README.md"))
    match = re.search(r"base weights run again[^.]*?gave (\d+) / (\d+) ms and (\d+(?:\.\d+)?) "
                      r"decisions/s", text)
    if not match:
        return None
    p50, p95, dps = match.groups()
    return {"p50_s": int(p50) / 1e3, "p95_s": int(p95) / 1e3, "shared_dps": float(dps)}


def load_refs(errors):
    """Everything the maintainers published that the tables quote, read from the repository. A
    piece that cannot be read is left empty (its rows print as —) and reported in `errors`."""
    ref = {"flow": None, "base": None, "analysis": {}, "flow_vs_base": None, "smoke": None,
           "perturbations": None, "validate": None, "typed": {}, "same_day": None,
           # fallback label: README.md, "Results so far": "Windows + RTX 5060 Ti 16 GB"
           "label": REF_OS + " + RTX 5060 Ti"}

    def attempt(name, fn):
        try:
            return fn()
        except Exception as error:
            errors.append(f"reference `{name}` unreadable: {error!r}")
            return ref[name]

    ref["flow"] = attempt("flow", lambda: folder_view(MAINT_FLOW))
    ref["base"] = attempt("base", lambda: folder_view(MAINT_BASE))
    ref["analysis"] = attempt("analysis", lambda: rj(MAINT_FLOW / "analysis.json") or {})

    def flow_vs_base():
        for block in (ref["analysis"].get("against") or {}).values():
            folder = str(block.get("folder", "")).replace("\\", "/").rstrip("/")
            if folder.endswith(MAINT_BASE.name):
                return block_pairs(None, block)

    ref["flow_vs_base"] = attempt("flow_vs_base", flow_vs_base)

    def label():
        name = g(rj(MAINT_FLOW / "report.json"), "model.device_name") or "RTX 5060 Ti"
        return f"{REF_OS} + {name.replace('NVIDIA GeForce ', '')}"

    ref["label"] = attempt("label", label)
    ref["smoke"] = attempt("smoke", lambda: rc.evaluate_stats(rj(MAINT_VALIDATION / "smoke.json"))
                           if rj(MAINT_VALIDATION / "smoke.json") else None)
    ref["perturbations"] = attempt(
        "perturbations", lambda: rc.evaluate_stats(rj(MAINT_VALIDATION / "perturbations.json"))
        if rj(MAINT_VALIDATION / "perturbations.json") else None)
    ref["validate"] = attempt(
        "validate", lambda: rc.extract({"suite": "T6"}, _Flat(MAINT_VALIDATION, "validate"), {})
        if rj(MAINT_VALIDATION / "summary.json") else None)
    ref["typed"] = attempt("typed", load_typed_refs)
    ref["same_day"] = attempt("same_day", load_same_day)
    if ref["same_day"] is None:
        errors.append("the maintainers' same-day base rerun (README.md, Results so far, footnote "
                      "1) was not found: the timing cells of the ref row of the effect table are "
                      "left empty")
    absent = [k for k in (*README_TYPED_ROWS, *TRAINING_TYPED_ROWS) if k not in ref["typed"]]
    if absent:
        errors.append("typed-decisions reference rows not found in README.md / docs/training.md: "
                      + ", ".join("/".join(k) for k in absent))
    return ref


# ----------------------------------------------------------------------------- typed-decisions

_TYPED = {}


def _typed_paired(a, b, samples, seed):
    """See typed_paired; a and b are the two parsed typed.json reports."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import typed_decisions as td

    rows = td.load(rc.TYPED)

    def flags(report):
        by_id = {c["id"]: c["answers"] for c in report["cases"]}
        out = {}
        for row in rows:
            gold, answers = json.loads(row["gold"]), by_id[row["id"]]
            out[row["id"]] = [
                td.argmax(answers[name], td.labels(question)) == gold[name]["label"]
                for name, question in json.loads(row["questions"]).items()]
        return out

    fa, fb = flags(a), flags(b)
    for flag, report in ((fa, a), (fb, b)):  # must reproduce the accuracy the tool reported
        total = sum(len(v) for v in flag.values())
        reported = g(report, "metrics.overall.accuracy")
        if reported is None or abs(sum(map(sum, flag.values())) / total - reported) > 1e-9:
            return None
    cases = [(sum(fa[r["id"]]), sum(fb[r["id"]]), len(fa[r["id"]])) for r in rows]
    rng, n = random.Random(seed), len(cases)
    draws = []
    for _ in range(samples):
        chosen = [cases[rng.randrange(n)] for _ in range(n)]
        draws.append((sum(c[0] for c in chosen) - sum(c[1] for c in chosen))
                     / sum(c[2] for c in chosen))
    draws.sort()
    point = (sum(c[0] for c in cases) - sum(c[1] for c in cases)) / sum(c[2] for c in cases)
    return {"difference": point,
            "ci": [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]],
            "cases": n, "samples": samples, "seed": seed}


def typed_paired(flow, base, samples=TYPED_SAMPLES, seed=TYPED_SEED):
    """Accuracy difference flow - base on the typed-decisions test cases with a paired bootstrap
    over cases (the maintainers' method, README.md "Results on typed-decisions"), computed from the
    two raw reports and the test split. None when anything is missing or does not add up to the
    accuracy the tool reported."""
    if not (flow.ok and base.ok):
        return None
    a, b = flow.raw("typed.json"), base.raw("typed.json")
    if not a or not b:
        return None
    key = (flow.raw_key("typed.json"), base.raw_key("typed.json"), rkey(rc.TYPED), samples, seed)
    if key not in _TYPED:
        try:
            _TYPED[key] = _typed_paired(a, b, samples, seed)
        except Exception:
            _TYPED[key] = None
    return _TYPED[key]


def typed_counts():
    """(cases, decisions) of the typed-decisions test split, or (None, None)."""
    try:
        rows = [json.loads(line) for line in read_text(rc.TYPED).splitlines() if line.strip()]
        return len(rows), sum(len(json.loads(r["questions"])) for r in rows)
    except Exception:
        return None, None


# ----------------------------------------------------------------------------- context

class Ctx:
    """Everything the sections need: both ledgers as Entry maps, references, machine, checks."""

    def __init__(self, flow_ledger, runs, base_ledger, flow_out, base_out, data_url, ambient):
        self.runs = runs
        frecs = (flow_ledger or {}).get("runs")
        brecs = (base_ledger or {}).get("runs")
        frecs = {k: v for k, v in frecs.items() if isinstance(v, dict)} if isinstance(
            frecs, dict) else {}
        brecs = {k: v for k, v in brecs.items() if isinstance(v, dict)} if isinstance(
            brecs, dict) else {}
        self.F = {r["id"]: Entry(r, frecs.get(r["id"]), flow_out) for r in runs}
        self.B = {r["id"]: Entry(r, brecs.get(r["id"]), base_out) for r in runs}
        self.extra_ids = sorted(set(frecs) - {r["id"] for r in runs})
        self.base_total = len(brecs)  # runs of the base campaign (141 for issue #25)
        self.machine = flow_ledger.get("machine") if isinstance(flow_ledger, dict) else None
        if not isinstance(self.machine, dict) or not self.machine:
            self.machine = rc.machine_info()
            if isinstance(flow_ledger, dict):
                flow_ledger["machine"] = self.machine  # like run_campaign: the caller may save it
        base_machine = (base_ledger or {}).get("machine")
        self.base_machine = base_machine if isinstance(base_machine, dict) else {}
        gate = flow_ledger.get("gate") if isinstance(flow_ledger, dict) else None
        self.gate = gate if isinstance(gate, dict) else {}  # run_flow.py's settings, last invocation
        thrown = flow_ledger.get("contaminated") if isinstance(flow_ledger, dict) else None
        self.contaminated = [c for c in thrown if isinstance(c, dict)] if isinstance(
            thrown, list) else []  # attempts that were not alone on the machine (not results)
        starts = sorted(str(r["started"]) for r in brecs.values() if r.get("started"))
        self.base_span = f"{starts[0][:10]} to {starts[-1][:10]}" if starts else "earlier"
        self.data_url, self.ambient = data_url or DATA_URL, ambient or AMBIENT
        keys = ("cpu", "ram", "gpu", "os", "cuda", "python", "uv")
        self.machine_changes = [f"{k}: `{self.base_machine.get(k)}` → `{self.machine.get(k)}`"
                                for k in keys if self.base_machine.get(k) and self.machine.get(k)
                                and self.base_machine.get(k) != self.machine.get(k)]
        self.same_machine = bool(self.base_machine) and not self.machine_changes
        self.semif_rows = next(
            (b["rows"] for e in [*self.F.values(), *self.B.values()]
             for b in e.against.values() if b.get("rows")), None)
        self.errors = []
        try:  # the thread count of the base campaign's CPU runs (fastest of its --threads sweep)
            self.cpu_threads = rc.cpu_threads(base_ledger if isinstance(base_ledger, dict)
                                              else {"runs": {}})
        except Exception:
            self.cpu_threads = None
        self.ref = load_refs(self.errors)
        self.specs, self.specs_error = flow_specs()
        self.checks = {}
        for run in runs:
            try:
                self.checks[run["id"]] = check_weights(self.F[run["id"]], self.B[run["id"]],
                                                       self.specs)
            except Exception as error:
                self.checks[run["id"]] = ("unknown", f"check failed: {error!r}")
        self.mismatches = [i for i, (state, _) in self.checks.items() if state == "mismatch"]
        self.notes = self.pair_notes()
        self.release = next((e.stats.get("model", {}).get("llama_cpp_release")
                             for e in [*self.F.values(), *self.B.values()]
                             if isinstance(e.stats.get("model"), dict)
                             and e.stats["model"].get("llama_cpp_release")), None)

    def pair_notes(self):
        """Differences between a flow run and its base run that make the pair not like-for-like."""
        notes = []
        for run in self.runs:
            f, b = self.F[run["id"]], self.B[run["id"]]
            if not (f.ok and b.ok):
                continue
            if run["suite"] == "T7":
                for key in ("direct_states", "shared_states"):
                    if f.stats.get(key) != b.stats.get(key):
                        notes.append(f"`{run['id']}`: {key} {f.stats.get(key)} (flow) against "
                                     f"{b.stats.get(key)} (base), so shared-vs-direct counts "
                                     "are not comparable")
            if f.rec.get("threads") != b.rec.get("threads"):
                notes.append(f"`{run['id']}`: threads {f.rec.get('threads')} (flow) against "
                             f"{b.rec.get('threads')} (base)")
        for run in self.runs:  # run_flow.py defers these until their base is recorded: a safety net
            need, f = expected_against(run), self.F[run["id"]]
            if need and f.ok and not paired(f, need)["found"]:
                notes.append(f"`{run['id']}`: recorded without its paired comparison (`--against "
                             f"{need}`, the run it is compared with was not recorded when it ran): "
                             "its difference cells are dashes")
        return notes

    @property
    def done(self):
        return [e for e in self.F.values() if e.rec is not None]

    @property
    def rows_text(self):
        if self.semif_rows:
            return f"{self.semif_rows} rows"
        return "authored144 and perturbations108 rows"


# ----------------------------------------------------------------------------- sections

def python_note(version):
    """`Python 3.14.7`, plus a note when the repository pins another minor version (.python-version
    at HEAD), as the base report did."""
    pinned = git("show", "HEAD:.python-version")
    found = re.search(r"(\d+\.\d+)", str(version or ""))
    if pinned and found and not pinned.startswith(found.group(1)):
        return (f"{version} (the repository pins {pinned}; `.venv` was created with "
                f"{found.group(1)})")
    return version


def sec_machine(ctx):
    m, b = ctx.machine, ctx.base_machine
    release, changed = ctx.release, ctx.machine_changes
    L = T(["Item", "Value"], [
        ("CPU", m.get("cpu")),
        ("RAM", f"{m.get('ram')} usable — {rc.RAM_DETAIL}"),
        ("GPU", f"{m.get('gpu')} — also drives the desktop display"),
        ("Ambient temperature", ctx.ambient),
        ("Driver / CUDA", m.get("cuda")), ("OS", m.get("os")),
        ("Python", python_note(m.get("python"))), ("uv", m.get("uv")),
        ("Commit", f"`{m.get('commit')}` (issue #25 base campaign: `{b.get('commit')}`)"),
        ("llama.cpp", f"release `{release}`, official prebuilt packages "
                      f"`llama-{release}-linux-x64-cuda` and `-cpu`, as in issue #25"
         if release else "as in issue #25"),
    ])
    if not b:
        same = "No machine record of the base campaign to compare with."
    elif changed:
        same = "Differences from the base campaign's machine: " + "; ".join(changed) + "."
    else:
        same = ("Same machine, driver, OS and toolchain as the base campaign of "
                f"[issue #25]({ISSUE_URL}).")
    L += ["", same, ""]
    changed_files = None
    base_commit, head = b.get("commit"), m.get("commit") or "HEAD"
    if base_commit and head:
        listing = git("diff", "--name-only", f"{base_commit}..{head}", "--", "src", "scripts",
                      "benchmarks", "examples")
        if listing is not None:
            changed_files = [x for x in listing.splitlines() if x]
    if changed_files is not None:
        prompt_same = "src/rizzo_flow/prompts.py" not in changed_files
        fixtures = [x for x in changed_files if x.startswith(("benchmarks/", "examples/"))]
        L += [f"Code between the two commits (`{base_commit}` → `{head}`), `src/`, `scripts/`, "
              f"`benchmarks/`, `examples/`: {len(changed_files)} files changed"
              + (f" ({', '.join(f'`{x}`' for x in changed_files)})" if changed_files else "")
              + f"; `prompts.py` {'unchanged' if prompt_same else 'CHANGED'}; benchmark and "
              f"example fixtures {'unchanged' if not fixtures else 'CHANGED'}.", ""]
    L += ["Weights: the fine-tuned GGUF files, `--weights flow` (the default at this commit), "
          "pinned in `rizzo_flow.config`. The base numbers are XHToken's original GGUF files "
          "(`--weights base`), from the base campaign.", ""]
    if ctx.specs:
        L += T(["Size", "Quant", "Repository", "Revision", "File", "sha256"], [
            (s.size.upper(), s.quant, s.repo, s.revision, s.file, s.sha256)
            for s in ctx.specs.values()])
    else:
        L += [f"(the pinned flow files could not be read: {ctx.specs_error})"]
    return L


def num(value):
    """A JSON number (not a bool), else None."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def plural(n, word, many=None):
    return f"{n} {word}" if n == 1 else f"{n} {many or word + 's'}"


def limit_text(value):
    """A limit as text: 1.0, 0.5, 25."""
    if value is None:
        return DASH
    return f"{value:.1f}" if value == int(value) and value < 10 else f"{value:g}"


def spread(values):
    """`median 0.45, max 0.61 (39 runs)` over the numbers in `values`; a dash if there are none."""
    nums = [v for v in map(num, values) if v is not None]
    if not nums:
        return DASH
    return (f"median {flt(statistics.median(nums), 2)}, max {flt(max(nums), 2)} "
            f"({plural(len(nums), 'run')})")


BUSY_KINDS = {"cpu": "external CPU", "gpu-apps": "a foreign GPU process",
              "gpu-util": "GPU utilization", "gpu-unreadable": "GPU state unreadable"}


def state_gate(ctx, recs):
    """The gate's thresholds as text, from what each record says it was gated with (the ledger's
    settings when no record says); a dash when neither does."""
    pairs = []
    for r in recs:
        lim = g(r, "gate.limits")
        if isinstance(lim, dict):
            pair = (num(lim.get("max_external_threads")), num(lim.get("max_gpu_util_pct")))
            if pair not in pairs:
                pairs.append(pair)
    if not pairs and ctx.gate:
        pairs = [(num(ctx.gate.get("max_external_threads")), num(ctx.gate.get("max_gpu_util_pct")))]
    if not pairs:
        return DASH
    text = " / ".join(
        f"external CPU ≤ {limit_text(a)} threads, GPU utilization ≤ {limit_text(b)} %, no foreign "
        "CUDA compute process" for a, b in pairs)
    window = num(g(ctx.gate, "window_s")) or next(
        (num(g(r, "gate.window_s")) for r in recs if num(g(r, "gate.window_s"))), None)
    passes = num(g(ctx.gate, "passes_after_wait")) or next(
        (num(g(r, "gate.passes_after_wait")) for r in recs if num(g(r, "gate.passes_after_wait"))),
        None)
    if window:
        text += f"; a {window:g} s window before every run" + (
            f", {passes:g} idle windows in a row after a wait" if passes else "")
    off = sum(1 for r in recs if g(r, "gate.enabled") is False)
    return text + (f" (waiting was off for {plural(off, 'run')})" if off else "")


def state_waits(recs):
    waits = [(r, num(g(r, "gate.waited_s"))) for r in recs]
    waits = [(r, w) for r, w in waits if w is not None]
    if not waits:
        return DASH
    waited = [(r, w) for r, w in waits if (num(g(r, "gate.busy_windows")) or 0) > 0]
    if not waited:
        return f"none of {plural(len(waits), 'run')} had to wait"
    kinds = {}
    for r, _ in waited:
        for kind in g(r, "gate.busy_kinds") or []:
            name = BUSY_KINDS.get(kind, str(kind))
            kinds[name] = kinds.get(name, 0) + 1
    why = ", ".join(f"{name} ({n})" for name, n in kinds.items())
    longest = max(w for _, w in waited) / 60
    return (f"{len(waited)} of {len(waits)} runs waited, longest {longest:.1f} min"
            + (f"; because of {why}" if why else ""))


def count(rec, path):
    """A JSON integer of a record (not a bool), else 0."""
    value = g(rec, path)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def state_gpu(ctx, recs, public=False):
    """Were foreign GPU compute processes seen? In the recorded runs (external_gpu.apps_seen: none
    by construction; the polls are the proof, so the ones that worked are counted and the failed
    ones said: a run whose GPU went unwatched is not recorded), in the attempts that were thrown
    away, and at the gate (the reasons it had to wait, in both). `public` (the issue comment):
    counts, no process names."""
    tried = [*recs, *ctx.contaminated]
    polls = sum(count(r, "external_gpu.polls") for r in recs)
    failed = sum(count(r, "external_gpu.polls_failed") for r in recs)
    if not polls and not any(isinstance(g(r, "gate.busy_kinds"), list) for r in tried):
        return DASH

    def seen(records):
        apps = [a for r in records for a in g(r, "external_gpu.apps_seen") or []
                if isinstance(a, dict)]
        if not apps:
            return "none"
        if public:
            return plural(len({a.get("pid") for a in apps}), "process", "processes")
        return ", ".join(sorted({str(a.get("name")) for a in apps}))

    at_gate = sum(1 for r in tried if "gpu-apps" in (g(r, "gate.busy_kinds") or []))
    worked = (f"{polls - failed} of {plural(polls, 'poll')} worked" if failed
              else plural(polls, "poll"))
    return (f"recorded runs: {seen(recs)} ({worked}); thrown-away attempts: "
            f"{seen(ctx.contaminated)}; at the gate: "
            + (f"seen while waiting before {plural(at_gate, 'attempt')}" if at_gate else "none"))


def thrown_reason(c, public):
    """Why an attempt was thrown away. Working notes: the runner's text, with the names of the
    processes behind the load. Issue comment (`public`): the same reasons as numbers and counts
    only (`reason_public`: no process name, pid or size, which belong to the machine's owner), or a
    bare statement for an entry that has none."""
    if public:
        reason = c.get("reason_public")
        reason = reason if isinstance(reason, str) and reason.strip() else "not a clean run"
    else:
        reason = c.get("reason") or DASH
    reason = rc.redact(str(reason))
    if len(reason) > ODD_STDERR_CHARS:
        reason = reason[:ODD_STDERR_CHARS].rstrip() + "…"
    return reason


def sec_state(ctx, public=False):
    """Machine state during the runs: the idle gate, the load that was not the runner's, the
    attempts that were thrown away (run_flow.py's gate and accounting). `public`: the issue
    comment's version, without process names, pids or sizes."""
    recs = [e.rec for e in ctx.done]
    measured = any(isinstance(r.get("external_cpu"), dict) for r in recs)
    if ctx.contaminated:
        thrown = f"{len(ctx.contaminated)} (listed below)"
    else:
        thrown = "none" if measured or ctx.gate else DASH
    L = T(["Item", "Value"], [
        ("Idle gate", state_gate(ctx, recs)),
        ("External CPU per run, whole run (threads)", spread(g(r, "external_cpu.threads_avg")
                                                             for r in recs)),
        ("External CPU per run, timed window (threads)",
         spread(g(r, "external_cpu.timed.threads_avg") for r in recs)),
        ("Runs that waited at the gate", state_waits(recs)),
        ("Contaminated attempts", thrown),
        ("Foreign GPU compute processes", state_gpu(ctx, recs, public)),
    ])
    L += ["", "External CPU is the machine's busy CPU time (`/proc/stat`) minus the CPU time of "
          "the runner and its children (`getrusage`), in threads, from just before a run to after "
          "its telemetry queries; the timed window is the commands alone. A run over the limit, "
          "with a foreign CUDA compute process, or whose GPU could not be watched (`nvidia-smi` "
          "gave no answer for too long), is not recorded: its output is kept aside and the run "
          "is repeated once the machine is idle. Desktop graphics processes are not compute "
          "processes."]
    if ctx.contaminated:
        L += ["", "Contaminated attempts (not results; output kept in "
              "`results/local-hw-flow/_contaminated/`):"]
        for c in ctx.contaminated[:ODD_MAX_RUNS]:
            L.append(f"- `{c.get('id')}` attempt {c.get('attempt', DASH)}: "
                     f"{thrown_reason(c, public)}")
        if len(ctx.contaminated) > ODD_MAX_RUNS:
            L.append(f"- {len(ctx.contaminated) - ODD_MAX_RUNS} more (working notes: ledger).")
    return L


def gate_line(rec):
    """The run log's gate line: what the measurement before the run found."""
    gate = rec.get("gate")
    if not isinstance(gate, dict):
        return f"Idle gate: {DASH}"
    if gate.get("enabled") is False:
        return "Idle gate: off (--no-idle-gate)"
    lim = gate.get("limits") if isinstance(gate.get("limits"), dict) else {}
    apps = gate.get("gpu_apps")
    window = num(gate.get("window_s"))
    cpu_limit = limit_text(num(lim.get("max_external_threads")))
    gpu_limit = limit_text(num(lim.get("max_gpu_util_pct")))
    return (f"Idle gate: passed after {flt(gate.get('waited_s'), 1)} s of waiting "
            f"({cnt(gate.get('busy_windows'))} busy windows), {cnt(gate.get('idle_windows'))} idle "
            f"window(s) of {DASH if window is None else f'{window:g}'} s: external CPU "
            f"{flt(gate.get('external_threads'), 2)} threads (limit {cpu_limit}), GPU utilization "
            f"{flt(gate.get('gpu_util_avg'), 0)} % (limit {gpu_limit}), foreign compute "
            f"processes {len(apps) if isinstance(apps, list) else DASH}")


def load_line(rec):
    """The run log's external-load line: what the accounting over the whole run found."""
    cpu = rec.get("external_cpu")
    if not isinstance(cpu, dict):
        return f"External load: {DASH}"
    timed = cpu.get("timed") if isinstance(cpu.get("timed"), dict) else {}
    tops = [t for t in (cpu.get("top_names") or cpu.get("top_processes") or [])
            if isinstance(t, dict)]
    apps = g(rec, "external_gpu.apps_seen")
    text = (f"External load: {flt(cpu.get('threads_avg'), 2)} threads over the run "
            f"({flt(cpu.get('cpu_seconds'), 1)} CPU-s in {flt(cpu.get('wall_s'), 1)} s), "
            f"{flt(timed.get('threads_avg'), 2)} in the timed window (limit "
            f"{limit_text(num(cpu.get('limit')))})")
    if tops:
        text += "; top: " + ", ".join(f"{rc.redact(str(t.get('name')))} {flt(t.get('threads'), 2)}"
                                      for t in tops[:3])
    polls, failed = count(rec, "external_gpu.polls"), count(rec, "external_gpu.polls_failed")
    proof = "" if not polls else (f" ({polls - failed} of {polls} polls worked, longest gap "
                                  f"{flt(g(rec, 'external_gpu.blind_s'), 0)} s)" if failed
                                  else f" ({polls} polls)")
    return text + "; foreign GPU compute processes: " + (
        DASH if not isinstance(apps, list) else "none" + proof if not apps else ", ".join(
            str(a.get("name")) for a in apps if isinstance(a, dict)))


def sec_tests(ctx):
    rows = []
    for run in ctx.runs:
        if run["suite"] not in ("T0", "T1"):
            continue
        name = rc.SUITE_NAMES[run["suite"]] + (" (pinned to the 4B Q8_0)" if run.get("pin_4b")
                                               or run["id"].endswith("-4b") else "")
        for side, e in (("base", ctx.B[run["id"]]), ("flow", ctx.F[run["id"]])):
            s = e.stats
            rows.append((name, side, e.status, cnt(s.get("passed")), cnt(s.get("failed")),
                         cnt(s.get("skipped")),
                         f"`{s.get('summary_line')}`" if s.get("summary_line") else DASH))
    if not rows:
        return []
    return ["The unit tests load no weights. The integration run loads the 4B Q8_0 through a "
            "pytest plugin of the campaign: the fine-tuned file for the flow row (the plugin "
            "records the file and hash it loaded), the original file for the base row, which ran "
            "at the earlier commit.", ""] + T(
        ["Suite", "Campaign", "Status", "Passed", "Failed", "Skipped", "Summary"], rows)


def semif_cells(v):
    return {
        "held": join(flt(v.get("held_out_authored144")), flt(v.get("held_out_perturbations108"))),
        "flips": join(cnt(v.get("flips_reversal")), cnt(v.get("flips_wrapper")),
                      cnt(v.get("flips_context"))),
        "missing": (DASH if v.get("missing_acc") is None else
                    f"{round(v['missing_acc'] * MISSING_ROWS)}/{MISSING_ROWS} "
                    f"({flt(v['missing_acc'])})"),
        "confident": frac(v.get("missing_evidence_confident"), MISSING_ROWS),
        "rule": (DASH if v.get("rule_application_pert_bacc") is None
                 and v.get("rule_application_pert_nll") is None else
                 f"{flt(v.get('rule_application_pert_bacc'))} "
                 f"({flt(v.get('rule_application_pert_nll'), 2)})"),
        "latency": join(rc.ms(v.get("p50_s")), rc.ms(v.get("p95_s"))),
        "dps": join(flt(v.get("shared_dps"), 2), flt(v.get("direct_dps"), 2)),
        "sd": frac(v.get("sd_flips"), v.get("sd_rows")),
    }


def sec_semif(ctx):
    t7 = pick(ctx.runs, "T7")
    ref = ctx.ref["flow"]
    label = f"*ref* maintainers, flow 4B Q8_0 CUDA ({ctx.ref['label']})"
    quality, speed, seen = [], [], set()
    for run in t7:
        e = ctx.F[run["id"]]
        v, c = view(e), semif_cells(view(e))
        seen.add((v.get("direct_states"), v.get("sd_rows")))
        quality.append((cfg(e), flt(v.get("authored144")), flt(v.get("perturbations108")),
                        c["held"], c["flips"], c["missing"], c["confident"], c["rule"]))
        speed.append((cfg(e), c["latency"], c["dps"], c["sd"], cnt(v.get("direct_states")), mem(e)))
    if ref:
        c = semif_cells(ref)
        seen.add((ref.get("direct_states"), ref.get("sd_rows")))
        quality.append((label, flt(ref.get("authored144")), flt(ref.get("perturbations108")),
                        c["held"], c["flips"], c["missing"], c["confident"], c["rule"]))
        speed.append((label, c["latency"], c["dps"], c["sd"], cnt(ref.get("direct_states")),
                      rc.gib(ref.get("peak_device_bytes"))))
    L = ["SemIf's fixtures (`scripts/semif_compare.py`, SemIf's own `evaluate.py`), flow weights. "
         "Balanced accuracy is the mean over families; held-out = the odd source groups "
         "(`semif_report.py`); flips = argmax changes under option reversal, criterion wrapper and "
         f"irrelevant context; missing evidence = the {MISSING_ROWS} rows where `insufficient` is "
         "the right answer, confident = a different answer with p ≥ 0.8; rule_application "
         "perturbed = balanced accuracy (NLL). Probabilities are not calibrated.", "",
         "**Quality**", ""]
    L += T(["Config", "authored144", "perturbations108",
            "Held-out (authored144 / perturbations108)", "Flips (reversal / wrapper / context)",
            "Missing evidence: accuracy", "Confident wrong (p ≥ 0.8)",
            "rule_application perturbed"], quality)
    sizes = sorted(((d, n) for d, n in seen if d is not None and n is not None), reverse=True)
    sizes_text = ", ".join(f"{d} states = {n} decisions" for d, n in sizes)
    L += ["", "**Speed and memory**", "",
          "Direct mode ran on the number of states in the *Direct states* column"
          + (f" ({sizes_text})" if sizes else "")
          + ": changed-argmax counts with different denominators are not comparable. Peak memory "
          "is rizzo's `peak_device_bytes` on the GPU (the drop in free GPU memory since before the load, so "
          "desktop use shows: the column is the shape777 phase, and the same run read lower in "
          "earlier phases, e.g. the 4B Q8_0 f16 flow run 5.61 GiB in authored144) and the process's "
          "peak RSS on the CPU.", ""]
    L += T(["Config", "p50 / p95 ms", "shape777 shared / direct dec/s",
            "Shared vs direct changed argmaxes", "Direct states", "Peak memory"], speed)
    e = ctx.F.get(MAINT_PAIR_ID)
    if e is not None:
        p = paired(e, "maintainers-flow")
        mine = view(e).get("direct_states")
        theirs = (ref or {}).get("direct_states")
        clause = ""
        if mine is not None and theirs is not None and mine != theirs:
            clause = (f"; direct states: {mine} here, {theirs} there, not part of the difference, "
                      f"which uses the {ctx.rows_text} of the quality fixtures")
        L += ["", "**Same weights, other machine** (this run minus the maintainers' flow run, "
              f"`--against maintainers-flow`; their machine: {ctx.ref['label']}{clause})", ""]
        L += T(["Config", "authored144 difference [95 % CI]",
                "perturbations108 difference [95 % CI]", "Rows with a different argmax"],
               [(cfg(e), diff_cell(p["authored144"]), diff_cell(p["perturbations108"]),
                 frac(p["different_argmax"], p["rows"]))])
    return L


def direct_clause(ctx, id_a, id_b, what_a, what_b):
    """Sentence saying that two T7 runs used different direct-state counts, and that the count
    does not enter a difference computed on the quality rows only; empty when they match."""
    def states(run_id):
        for entries in (ctx.F, ctx.B):
            e = entries.get(run_id)
            if e is not None and e.stats.get("direct_states") is not None:
                return e.stats["direct_states"]
        return None

    a, b = states(id_a), states(id_b)
    if a is None or b is None or a == b:
        return ""
    return (f" The {what_a} run used {a} direct states and the {what_b} run {b}; the count does "
            f"not enter these differences, which use only the {ctx.rows_text} of the quality "
            "fixtures.")


def method_line(ctx):
    """How the paired intervals are built, quoted from the maintainers' own analysis.json."""
    analysis = ctx.ref.get("analysis") or {}
    for block in (analysis.get("against") or {}).values():
        for d in (block.get("paired_difference_ours_minus_theirs") or {}).values():
            if isinstance(d, dict) and d.get("method"):
                method = str(d["method"])
                return (f"95 % interval: {method[:1].lower() + method[1:]} ({d.get('samples')} "
                        f"draws, seed {d.get('seed')}, {d.get('source_groups')} source groups).")
    return "95 % interval: paired source-group bootstrap (SemIf's `evaluate.py`)."


def sec_effect(ctx):
    t7 = pick(ctx.runs, "T7")
    one, two = [], []
    for run in t7:
        f, b = ctx.F[run["id"]], ctx.B[run["id"]]
        fv, bv = view(f), view(b)
        fc, bc = semif_cells(fv), semif_cells(bv)
        p = paired(f, "this-base")
        one.append((cfg(f), arrow(flt(bv.get("authored144")), flt(fv.get("authored144"))),
                    diff_cell(p["authored144"]),
                    arrow(flt(bv.get("perturbations108")), flt(fv.get("perturbations108"))),
                    diff_cell(p["perturbations108"]), frac(p["different_argmax"], p["rows"])))
        two.append((cfg(f), arrow(bc["missing"], fc["missing"]),
                    arrow(bc["confident"], fc["confident"]), arrow(bc["flips"], fc["flips"]),
                    arrow(bc["rule"], fc["rule"]),
                    arrow(rc.ms(bv.get("p50_s")), rc.ms(fv.get("p50_s"))),
                    arrow(flt(bv.get("shared_dps"), 2), flt(fv.get("shared_dps"), 2))))
    ref_f, ref_b, pair = ctx.ref["flow"], ctx.ref["base"], ctx.ref.get("flow_vs_base")
    ref_note = ""
    if ref_f and ref_b and pair:
        label = f"*ref* maintainers, 4B Q8_0 CUDA ({ctx.ref['label']})"
        rf, rb = semif_cells(ref_f), semif_cells(ref_b)
        # Their published base and flow runs are from different days and the machine was slower on
        # the second (README.md, Results so far, footnote 1), so the timing cells pair the flow run
        # with the base run they repeated right after it, not with the published one.
        day = ctx.ref.get("same_day")
        published = (f"{rc.ms(ref_b.get('p50_s'))} ms, {flt(ref_b.get('shared_dps'), 2)} dec/s "
                     "shared")
        if day:
            p50 = arrow(rc.ms(day.get("p50_s")), rc.ms(ref_f.get("p50_s")))
            dps = arrow(flt(day.get("shared_dps"), 2), flt(ref_f.get("shared_dps"), 2))
            ref_note = (" Its timing cells pair their flow run with the base weights run they "
                        "repeated right after it on the same machine (README.md, Results so far, "
                        f"footnote 1: {rc.ms(day.get('p50_s'))} / {rc.ms(day.get('p95_s'))} ms, "
                        f"{flt(day.get('shared_dps'), 2)} dec/s shared), not with their published "
                        f"base run ({published}): the footnote notes that the machine's speed was "
                        "not the same on the two days, so those two runs are not a like-for-like "
                        "timing pair.")
        else:
            p50 = dps = DASH
            ref_note = (" Its timing cells are empty: the maintainers' published base run "
                        f"({published}) and their flow run were taken on different days "
                        "(README.md, Results so far, footnote 1), so they are not a like-for-like "
                        "timing pair.")
        one.append((label, arrow(flt(ref_b.get("authored144")), flt(ref_f.get("authored144"))),
                    diff_cell(pair["authored144"]),
                    arrow(flt(ref_b.get("perturbations108")), flt(ref_f.get("perturbations108"))),
                    diff_cell(pair["perturbations108"]),
                    frac(pair["different_argmax"], pair["rows"])))
        two.append((label, arrow(rb["missing"], rf["missing"]),
                    arrow(rb["confident"], rf["confident"]), arrow(rb["flips"], rf["flips"]),
                    arrow(rb["rule"], rf["rule"]), p50, dps))
    L = [f"Each pair is the same run id in the base campaign ({ctx.base_span}, XHToken's GGUF) "
         "and in this one (fine-tuned GGUF): same configuration and direct-state count, other "
         f"weights. Difference = flow − base, paired on SemIf's {ctx.rows_text} "
         "(`--against this-base`). "
         + method_line(ctx) + " An interval that contains 0 at the printed precision is marked "
         "*includes 0*; no difference is claimed for it. The timing columns of the second table "
         "come from different days (ambient temperature and desktop activity were not "
         "controlled), so they are two measurements, not a controlled pair. Commits: base "
         f"`{ctx.base_machine.get('commit')}`, flow `{ctx.machine.get('commit')}` (see "
         "*Machine and weights* for the files that differ).", "",
         "**Accuracy on SemIf's fixtures: base → flow**", ""]
    L += T(["Config", "authored144", "Difference [95 % CI]", "perturbations108",
            "Difference [95 % CI]", "Rows with a different argmax"], one)
    L += ["", "**Missing evidence, stability, latency and throughput: base → flow**", ""]
    L += T(["Config", "Missing evidence: accuracy", "Confident wrong (p ≥ 0.8)",
            "Flips (reversal / wrapper / context)", "rule_application perturbed",
            "p50 ms", "Shared dec/s"], two)
    L += ["", "The *ref* row is the maintainers' own base and flow runs on their machine "
          "(`analysis.json` of the flow run)." + ref_note]
    return L


def sec_native(ctx):
    L = ["`rizzo evaluate benchmarks/{smoke,perturbations}.jsonl --compare-modes`, same command "
         "for both weights. Accuracy is n of N labelled decisions; NLL, Brier and ECE are "
         "computed on the model's own probabilities (not calibrated); *shared vs direct* = argmax "
         "changes between the prefix-sharing and the fresh-prefill mode.", ""]
    for suite, title, key in (("T4", "Smoke (`benchmarks/smoke.jsonl`)", "smoke"),
                              ("T5", "Perturbations (`benchmarks/perturbations.jsonl`)",
                               "perturbations")):
        rows = []
        for run in pick(ctx.runs, suite):
            for side, e in (("base", ctx.B[run["id"]]), ("flow", ctx.F[run["id"]])):
                s = e.stats
                rows.append((cfg(e), side, acc_nn(s), flt(s.get("nll")), flt(s.get("brier")),
                             flt(s.get("ece")),
                             join(rc.ms(s.get("latency_median_s")), rc.ms(s.get("latency_p95_s"))),
                             flt(s.get("decisions_per_second"), 2),
                             frac(s.get("changed_argmaxes"), s.get("mode_decisions")), mem(e)))
        r = ctx.ref.get(key)
        if r:
            rows.append((f"*ref* maintainers, 4B Q8_0 CUDA ({ctx.ref['label']})", "base",
                         acc_nn(r), flt(r.get("nll")), flt(r.get("brier")), flt(r.get("ece")),
                         join(rc.ms(r.get("latency_median_s")), rc.ms(r.get("latency_p95_s"))),
                         flt(r.get("decisions_per_second"), 2),
                         frac(r.get("changed_argmaxes"), r.get("mode_decisions")),
                         rc.gib(r.get("peak_device_bytes"))))
        n = next((e.stats.get("categorical_rows") for e in [*ctx.F.values(), *ctx.B.values()]
                  if e.run["suite"] == suite and e.stats.get("categorical_rows")), None)
        if rows:
            L += [f"**{title}**" + (f" — N = {n} labelled decisions, one decision = {1 / n:.3f} "
                                    "of accuracy" if n else ""), ""]
            L += T(["Config", "Weights", "Accuracy", "NLL", "Brier", "ECE", "Median / p95 ms",
                    "dec/s", "Shared vs direct changed argmaxes", "Peak memory"], rows)
            L.append("")
    L.append("The *ref* rows are the maintainers' published base-weight run "
             "(`results/llama-q8_0-cuda-validation`), not a flow measurement.")
    return L


def stat_delta(after, before, key):
    """Point difference of one statistic, `+0.033`; — when either side is missing."""
    x, y = after.get(key), before.get(key)
    return DASH if x is None or y is None else sflt(x - y)


def sec_typed(ctx):
    cases, decisions = typed_counts()
    size = f" ({cases} cases, {decisions:,} decisions)" if cases else ""
    refs = ctx.ref["typed"]
    rows, deltas = [], []
    for run in pick(ctx.runs, "T9"):
        f, b = ctx.F[run["id"]], ctx.B[run["id"]]
        for side, e in (("base", b), ("flow", f)):
            s = e.stats
            rows.append((cfg(e), f"{side}, this machine", flt(s.get("accuracy")), flt(s.get("kl")),
                         flt(s.get("brier")), flt(s.get("ece")),
                         join(flt(s.get("p50_ms"), 0), flt(s.get("p95_ms"), 0)),
                         flt(s.get("decisions_per_second"), 2)))
        if run.get("device") == "cuda":
            for variant in ("base", "flow"):
                r = refs.get((run["size"], run["quant"], variant))
                if r:
                    rows.append((esc(rc.label(run)),
                                 f"*ref* {variant}, maintainers ({r['source']})",
                                 flt(r.get("accuracy")), flt(r.get("kl")), flt(r.get("brier")),
                                 flt(r.get("ece")), join(flt(r.get("p50_ms"), 0), DASH), DASH))
        tp = typed_paired(f, b) if f.ok and b.ok else None
        deltas.append((cfg(f), diff_cell((tp["difference"], tp["ci"])) if tp else DASH,
                       *[stat_delta(f.stats, b.stats, k) for k in ("kl", "brier", "ece")]))
    L = [f"Test split of `LocalLLaMA/typed-decisions`, config `all`{size}, replayed as "
         "`/v1/systemone` request bodies, in process (no HTTP layer), by "
         "`scripts/typed_decisions.py`, the same command for both "
         "weights. Base = zero-shot original GGUF; flow = the fine-tune. The *ref* rows are the "
         f"maintainers' published numbers ({ctx.ref['label']}), parsed from the file named in the "
         "row (README.md, Results on typed-decisions; docs/training.md, section 10) and not "
         "re-measured here; where no row is shown, nothing is published for that configuration. "
         "KL, Brier and ECE are computed on uncalibrated probabilities.", ""]
    L += T(["Config", "Weights / source", "Accuracy", "KL", "Brier", "ECE",
            "p50 / p95 ms per case", "dec/s"], rows)
    L += ["", "**Difference flow − base on this machine**", "",
          f"Accuracy: paired bootstrap over cases, {TYPED_SAMPLES} draws, seed {TYPED_SEED}, "
          "computed from the two raw reports and the test split (`—` when either run is missing). "
          "KL, Brier and ECE: point differences, no interval.", ""]
    L += T(["Config", "Accuracy difference [95 % CI]", "KL", "Brier", "ECE"], deltas)
    paired_ref = refs.get("paired_4b_q8_0")
    if paired_ref:
        L += ["", f"Maintainers' paired difference, 4B Q8_0, base → flow ({ctx.ref['label']}): "
              f"accuracy {sflt(paired_ref[0])} [{sflt(paired_ref[1])}, {sflt(paired_ref[2])}] "
              "(README.md, Results on typed-decisions)."]
    return L


def validate_smoke(s):
    if s.get("smoke_accuracy") is None:
        return DASH
    return (f"{flt(s.get('smoke_accuracy'))}, NLL {flt(s.get('smoke_nll'))}, "
            f"{cnt(s.get('smoke_changed_argmaxes'))} changed")


def validate_long(s):
    if s.get("long_changed_argmaxes") is None:
        return DASH
    return (f"{cnt(s.get('long_changed_argmaxes'))} "
            f"(max Δp {flt(s.get('long_max_probability_delta'), 6)})")


def sec_validate(ctx):
    rows = []
    for run in pick(ctx.runs, "T6"):
        for side, e in (("base", ctx.B[run["id"]]), ("flow", ctx.F[run["id"]])):
            s = e.stats
            status = e.status
            if e.rec is not None and any(e.rec.get("exit_codes") or []):
                status += f" (exit {e.rec['exit_codes']})"
            rows.append((cfg(e), side, status, validate_smoke(s),
                         flt(s.get("perturbations_accuracy")),
                         join(rc.ms(s.get("long_shared_s")), rc.ms(s.get("long_direct_s"))),
                         validate_long(s),
                         flt(s.get("validation_s"), 1), flt(g(s, "model.load_seconds"), 1),
                         rc.gib(s.get("peak_device_bytes"))))
    r = ctx.ref.get("validate")
    if r:
        rows.append((f"*ref* maintainers, 4B Q8_0 CUDA ({ctx.ref['label']})", "base", "published",
                     validate_smoke(r), flt(r.get("perturbations_accuracy")),
                     join(rc.ms(r.get("long_shared_s")), rc.ms(r.get("long_direct_s"))),
                     validate_long(r), flt(r.get("validation_s"), 1),
                     flt(g(r, "model.load_seconds"), 1), rc.gib(r.get("peak_device_bytes"))))
    if not rows:
        return []
    return ["`scripts/validate_checkpoint.py`: smoke, perturbations and a long shared state "
            "(crossing Spark's 512-token attention window) evaluated in process on the engine, "
            "in shared and direct mode, plus one example request through the HTTP app; the "
            "timings have no HTTP layer. Status is the runner's: `ok` = the tool produced its "
            "summary.", ""] + T(
        ["Config", "Weights", "Status", "Smoke accuracy, NLL, changed argmaxes shared vs direct",
         "Perturbations accuracy", "Long state shared / direct ms",
         "Long state changed argmaxes", "Validation s", "Load s", "Peak memory"], rows)


def smoke_cell(s):
    """`19/20 (0.950), NLL 0.450` from an evaluate summary."""
    return DASH if acc_nn(s) == DASH else f"{acc_nn(s)}, NLL {flt(s.get('nll'))}"


def sec_kv(ctx):
    def side_rows(side, entries):
        rows = []
        for kv, prefix in KV_PREFIX.items():
            t4, t7 = entries.get(f"{prefix}-T4"), entries.get(f"{prefix}-T7")
            if t4 is None or t7 is None:
                continue
            p = paired(t7, "this-cuda-kv-f16") if kv != "f16" else None
            v = t7.stats
            note = "" if t7.ok else " (pending)" if t7.rec is None else " (failed)"
            rows.append((side, kv + note, smoke_cell(t4.stats),
                         flt(v.get("authored144")),
                         diff_cell(p["authored144"]) if p else "reference",
                         flt(v.get("perturbations108")),
                         diff_cell(p["perturbations108"]) if p else "reference",
                         frac(p["different_argmax"], p["rows"]) if p else "reference",
                         mem(t7), flt(v.get("shared_dps"), 2)))
        return rows

    rows = side_rows("flow", ctx.F) + side_rows("base", ctx.B)
    if not rows:
        return []
    clause = direct_clause(ctx, KV_PREFIX["f16"] + "-T7", KV_PREFIX["q8_0"] + "-T7", "f16",
                           "quantized-KV")
    return ["4B Q8_0 on CUDA, `--kv-type f16` (the default) / `q8_0` / `q4_0`, same weights within "
            "a block. Differences are against the f16 run of the same weights "
            f"(`--against this-cuda-kv-f16`, SemIf's {ctx.rows_text}).{clause} "
            "The base block is the base campaign's own KV runs.", "",
            *T(["Weights", "KV type", "Smoke accuracy, NLL", "authored144",
                "Difference vs f16 [95 % CI]", "perturbations108", "Difference vs f16 [95 % CI]",
                "Rows with a different argmax vs f16", "Peak memory", "Shared dec/s"], rows)]


def sec_cpu(ctx):
    cpu = pick(ctx.runs, "T7", device="cpu")
    threads = next((e.rec.get("threads") for e in ctx.F.values() if e.rec
                    and e.run.get("device") == "cpu" and e.rec.get("threads")), None)
    cpu_name = ctx.machine.get("cpu") or "CPU"
    L = [f"{cpu_name}, `cpu` package, `--threads {threads or ctx.cpu_threads or '?'}` (the "
         "count of the base "
         "campaign's CPU runs; T7 and T9 get it through the campaign's `threads_wrapper.py`, since "
         "those scripts have no `--threads` flag). Differences in the T7 table are against the "
         "CUDA run of the same weights and precision on the same side of the pair (flow rows: the "
         "flow CUDA run; base rows: the base campaign's CUDA run; `--against this-cuda`, SemIf's "
         f"{ctx.rows_text}); the *p50 vs CUDA* columns are the ratio of the CPU run's median to "
         "that CUDA run's, also per side."
         + direct_clause(ctx, "cuda-4b-q8_0-f16-T7", "cpu-4b-q8_0-f16-T7", "CUDA", "CPU"), ""]
    t4 = []
    for run in pick(ctx.runs, "T4", device="cpu"):
        for side, e in (("base", ctx.B[run["id"]]), ("flow", ctx.F[run["id"]])):
            s = e.stats
            t4.append((cfg(e), side, smoke_cell(s),
                       join(rc.ms(s.get("latency_median_s")), rc.ms(s.get("latency_p95_s"))),
                       flt(s.get("decisions_per_second"), 2),
                       frac(s.get("changed_argmaxes"), s.get("mode_decisions")), mem(e)))
    if t4:
        L += ["**T4 smoke**", ""] + T(["Config", "Weights", "Accuracy, NLL", "Median / p95 ms",
                                       "dec/s", "Shared vs direct changed argmaxes",
                                       "Peak memory"], t4) + [""]
    t7 = []
    for run in cpu:
        cuda_id = f"cuda-{run['size']}-{run['quant']}-f16-T7"
        for side, ents in (("base", ctx.B), ("flow", ctx.F)):
            e, ref = ents[run["id"]], ents.get(cuda_id)
            v, c, p = view(e), semif_cells(view(e)), paired(e, "this-cuda")
            t7.append((cfg(e), side, flt(v.get("authored144")), flt(v.get("perturbations108")),
                       diff_cell(p["authored144"]), diff_cell(p["perturbations108"]),
                       frac(p["different_argmax"], p["rows"]), c["latency"],
                       times(v.get("p50_s"), (ref.stats.get("p50_s") if ref else None)),
                       c["dps"], c["sd"], mem(e)))
    if t7:
        L += ["**T7 SemIf fixtures**", ""] + T(
            ["Config", "Weights", "authored144", "perturbations108",
             "authored144 difference vs CUDA [95 % CI]",
             "perturbations108 difference vs CUDA [95 % CI]",
             "Rows with a different argmax vs CUDA", "p50 / p95 ms", "p50 vs CUDA",
             "shape777 shared / direct dec/s", "Shared vs direct changed argmaxes",
             "Peak memory"],
            t7) + [""]
    t9 = []
    for run in pick(ctx.runs, "T9", device="cpu"):
        cuda_id = f"cuda-{run['size']}-{run['quant']}-f16-T9"
        for side, ents in (("base", ctx.B), ("flow", ctx.F)):
            e, ref = ents[run["id"]], ents.get(cuda_id)
            s = e.stats
            t9.append((cfg(e), side, flt(s.get("accuracy")), flt(s.get("kl")), flt(s.get("brier")),
                       flt(s.get("ece")), join(flt(s.get("p50_ms"), 0), flt(s.get("p95_ms"), 0)),
                       times(s.get("p50_ms"), (ref.stats.get("p50_ms") if ref else None)),
                       flt(s.get("decisions_per_second"), 2)))
    if t9:
        L += ["**T9 typed-decisions**", ""] + T(
            ["Config", "Weights", "Accuracy", "KL", "Brier", "ECE", "p50 / p95 ms per case",
             "p50 vs CUDA", "dec/s"], t9) + [""]
    return L[:-1] if L and L[-1] == "" else L


def sec_telemetry(ctx):
    rows = []
    for run in pick(ctx.runs, "T7"):
        for side, e in (("base", ctx.B[run["id"]]), ("flow", ctx.F[run["id"]])):
            if not e.ok:  # pending, or failed (its measurements are not shown either)
                rows.append((cfg(e), side) + (DASH,) * 8)
                continue
            n = e.rec.get("nvidia_smi") or {}
            p = e.rec.get("prometheus") or {}
            r = e.rec.get("rusage") or {}
            cpu = run.get("device") == "cpu"
            rows.append((cfg(e), side, flt(e.rec.get("wall_s"), 1),
                         join(flt(n.get("power_avg_w"), 1), flt(n.get("power_max_w"), 1)),
                         flt(n.get("power_baseline_w"), 1),
                         "— (GPU idle)" if cpu else join(flt(kilo(n.get("net_energy_j")), 3),
                                                         flt(kilo(p.get("gpu_net_energy_j")), 3)),
                         flt(p.get("gpu_sm_active_avg"), 2),
                         flt(p.get("gpu_power_cap_throttle_s"), 1),
                         flt(n.get("temp_max_c"), 0),
                         join(flt(r.get("avg_cores"), 2), flt(r.get("max_rss_gib"), 2))))
    if not rows:
        return []
    return ["GPU: `nvidia-smi` at 100 ms (power, energy, temperature) and DCGM at 1 s (SM "
            "activity, power-cap throttling); CPU: the child's rusage. Net energy = integral of "
            "GPU power over the run − 10 s idle baseline power × duration (nvidia-smi, then DCGM); "
            "each "
            "window covers the whole command, model loading included. The GPU also drives the "
            "desktop, so the baseline is not zero. Base and flow runs are from different days.",
            "", *T(["Config", "Weights", "Wall s", "GPU avg / peak W", "Idle baseline W",
                    "Net GPU energy kJ (nvidia-smi / DCGM)", "SM active (DCGM)",
                    "Power-cap throttle s", "Temp max °C", "CPU cores avg / peak RSS GiB"], rows)]


def env_prefix(env):
    """`K=V K=V` in front of a command. Paths inside the repository are printed relative to it
    (PYTHONPATH=.research/hw-campaign), like the arguments, so the line pastes from the root."""
    return " ".join(f"{k}={rc.redact(str(v))}" for k, v in (env or {}).items())


def recorded(rec):
    """A run's recorded commands, environment variables in front."""
    cmds = rec.get("commands") or []
    envs = rec.get("env") or [{}] * len(cmds)
    out = []
    for cmd, env in zip(cmds, envs, strict=False):
        out.append(f"{env_prefix(env)} {cmd}".strip())
    return out


def template(run, base_ledger):
    """What run_campaign.commands() runs for this plan entry, with the fine-tuned weights."""
    out = []
    for argv, env, _ in rc.commands(run, FLOW_OUT / run["id"], base_ledger or {"runs": {}}):
        args = [str(a) for a in argv]
        if "--weights" in args:  # already the flow campaign's own command
            pass
        elif "--model" in args:  # T9 pins the base GGUF by path: select the flow file instead
            i = args.index("--model")
            args[i:i + 2] = ["--size", run["size"], "--quant", run["quant"], "--weights", "flow"]
        elif "--quant" in args:
            i = args.index("--quant")
            args[i + 2:i + 2] = ["--weights", "flow"]
        if run["suite"] == "T1" and (CAMPAIGN / "pin_4b_flow_plugin.py").exists():
            # the base plugin indexes config.GGUF by (size, quant); the flow campaign has its own
            args = ["pin_4b_flow_plugin" if a == "pin_4b_plugin" else a for a in args]
            env = {**env, "RIZZO_HW_EVIDENCE": f"results/local-hw-flow/{run['id']}/{EVIDENCE_FILE}"}
        out.append(f"{env_prefix(env)} {' '.join(rc._short(a) for a in args)}".strip())
    return out


def sec_commands(ctx, base_ledger):
    def group(run):
        return run["suite"], "kv" if run.get("kv") else run.get("device") or "-"

    first, order = {}, []
    for run in ctx.runs:
        if group(run) not in first:
            first[group(run)] = run
            order.append(group(run))
    L = ["```bash",
         "uv sync --extra test --locked",
         "uv run rizzo download --weights flow           # CUDA runtime + 4B Q8_0 flow GGUF",
         "uv run rizzo download --weights flow --only weights --quant q4_k_m   # and bf16",
         "uv run rizzo download --weights flow --only weights --size 1.7b --quant q8_0   # and "
         "q4_k_m, bf16"]
    if any(r.get("device") == "cpu" for r in ctx.runs):  # `--device cpu` needs the cpu package
        L.append("uv run rizzo download --only runtime --runtime cpu   # CPU-only package, for "
                 "--device cpu")
    L.append("uv run rizzo devices")
    templated = False
    for suite, klass in order:
        run = first[(suite, klass)]
        rec = ctx.F[run["id"]].rec
        what = {"kv": "KV cache runs (--kv-type)", "cuda": "CUDA", "cpu": "CPU-only"}.get(klass, "")
        head = f"# {rc.SUITE_NAMES.get(suite, suite)}" + (f", {what}" if what else "")
        if rec is not None and rec.get("commands"):
            L += [f"{head} — as run for {run['id']}", *recorded(rec)]
        else:
            templated = True
            L += [f"{head} — template for {run['id']}, not run yet", *template(run, base_ledger)]
    L += ["# After every T7 run the runner calls semif_report.py; --against blocks per run:",
          f"#   maintainers-flow  {MAINT_PAIR_ID} only: {MAINT_FLOW.relative_to(ROOT)}",
          "#   this-base         every flow T7 run: the base run of the same id "
          "(results/local-hw/ID/semif)",
          "#   this-cuda-kv-f16  the KV runs: results/local-hw-flow/cuda-4b-q8_0-f16-T7/semif",
          "#   this-cuda         the CPU run: results/local-hw-flow/cuda-4b-q8_0-f16-T7/semif",
          "uv run python scripts/semif_report.py results/local-hw-flow/ID/semif "
          "--semif .research/SemIf \\",
          "  --against this-base=results/local-hw/ID/semif",
          "```", ""]
    direct = sorted({(r["id"], r["extra"]["direct_states"]) for r in ctx.runs
                     if r["suite"] == "T7" and "direct_states" in (r.get("extra") or {})})
    text = ("Every command loads the model again; runs were strictly sequential (one model process "
            "at a time). Other configurations differ only in `--size`, `--quant`, `--device`, "
            "`--kv-type` and `--threads`.")
    helpers = []
    if any(r.get("pin_4b") for r in ctx.runs):
        helpers.append("`.research/hw-campaign/pin_4b_flow_plugin.py` (pytest plugin: pins the "
                       "integration suite to the 4B Q8_0 and records what it loaded)")
    if any(r.get("device") == "cpu" and r["suite"] in ("T6", "T7", "T9") for r in ctx.runs):
        helpers.append("`.research/hw-campaign/threads_wrapper.py` (sets `--threads` for the "
                       "CPU-only SemIf and typed-decisions runs, whose scripts have no such flag)")
    if helpers:
        text += (" Some commands use campaign scripts that are not part of the repository "
                 "(`.research/` is git-ignored): " + " and ".join(helpers) + ".")
    if direct:
        text += (" T7 runs with an explicit `--direct-states`: "
                 + ", ".join(f"`{i}` ({n})" for i, n in direct)
                 + "; the others use the script default, as in the base campaign.")
    if templated:
        text += (" Commands marked *template* are what the runner will execute (built from the "
                 "base plan); they are replaced by the recorded ones as runs finish.")
    placeholder = " (placeholder: fill in when the branch is pushed)" if (
        ctx.data_url == "DATA_BRANCH_URL") else ""
    L += [text, "",
          f"Data and raw outputs of this follow-up: {ctx.data_url}{placeholder}. Base campaign "
          f"data: {BASE_DATA_URL}."]
    return L


def ids_text(ids, limit=6):
    shown = ", ".join(f"`{i}`" for i in ids[:limit])
    return shown + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


def sec_odd(ctx, notes_path):
    L = []
    for bad in ctx.mismatches[:6]:
        L.append(f"- **Weights check failed** for `{bad}`: {ctx.checks[bad][1]}. Its numbers "
                 "are not flow numbers.")
    if len(ctx.mismatches) > 6:
        L.append(f"- **Weights check failed** for {len(ctx.mismatches) - 6} more runs "
                 f"({ids_text(ctx.mismatches[6:], 4)}).")
    L += [f"- {n}." for n in ctx.notes]
    tests = [f"- {rc.SUITE_NAMES[e.run['suite']]} (`{e.id}`): `{test}` failed."
             for e in ctx.done for test in g(e.rec, "stats.failed_tests") or []]
    L += tests[:ODD_MAX_TESTS]
    if len(tests) > ODD_MAX_TESTS:
        L.append(f"- {len(tests) - ODD_MAX_TESTS} more failed tests (run log of the working "
                 "notes).")
    bad_runs = [e for e in ctx.done if e.rec.get("status") != "ok"
                or (any(e.rec.get("exit_codes") or []) and e.run["suite"] not in ("T0", "T1"))]
    for e in bad_runs[:ODD_MAX_RUNS]:
        rec = e.rec
        tail = rc.redact(str((rec.get("stderr_tail") or [""])[-1] or ""))
        if len(tail) > ODD_STDERR_CHARS:
            tail = tail[:ODD_STDERR_CHARS].rstrip() + "…"
        L.append(f"- `{e.id}`: status {rec.get('status')}, exit codes {rec.get('exit_codes')}"
                 + (f"; stderr: `{tail}`" if tail else ""))
    if len(bad_runs) > ODD_MAX_RUNS:
        L.append(f"- {len(bad_runs) - ODD_MAX_RUNS} more runs with a status other than ok or a "
                 f"non-zero exit code ({ids_text([e.id for e in bad_runs[ODD_MAX_RUNS:]], 4)}); "
                 "stderr in the run log of the working notes.")
    if ctx.extra_ids:
        L.append("- Ledger records that are not in the 39-run plan (ignored): "
                 + ", ".join(f"`{i}`" for i in ctx.extra_ids) + ".")
    extra = optional_text(notes_path)
    if extra:
        L.append(extra)
    return L or ["- Nothing recorded so far."]


# ----------------------------------------------------------------------------- assembly

def guarded(ctx, name, fn, *args):
    """A broken section must not stop the others (or the campaign loop that calls render)."""
    try:
        return fn(ctx, *args)
    except Exception as error:
        ctx.errors.append(f"section `{name}`: {error!r}")
        return [f"> Section `{name}` could not be rendered: {error!r}"]


def progress_line(ctx):
    recs = [e.rec for e in ctx.done]
    failed = sum(1 for r in recs if r.get("status") != "ok")
    text = f"Progress: **{len(recs)}/{len(ctx.runs)}** runs, {failed} failed"
    try:
        _, _, _, elapsed, remaining, _ = rc.progress({"runs": {r["id"]: r for r in recs}}, ctx.runs)
        text += f"; elapsed {rc.hms(elapsed)}, estimated remaining {rc.hms(remaining)}"
    except Exception:  # the estimate is a courtesy
        pass
    try:
        starts = sorted(str(r["started"]) for r in recs if r.get("started"))
        ends = [r["t1"] for r in recs if isinstance(r.get("t1"), (int, float))]
        if starts:
            text += f"; first run started {starts[0]}"
        if ends:
            last = time.strftime("%Y-%m-%d %H:%M", time.localtime(max(ends)))
            text += f", last one finished {last}"
    except (OverflowError, OSError, ValueError):
        pass
    return text + f". Updated {time.strftime('%Y-%m-%d %H:%M')}."


def sections(ctx, base_ledger, notes_path):
    """(key, title, lines) for every table section, shared by both files."""
    return [
        ("machine", "Machine and weights", guarded(ctx, "machine", sec_machine)),
        ("state", "Machine state", guarded(ctx, "state", sec_state)),
        ("tests", "Test suites", guarded(ctx, "tests", sec_tests)),
        ("semif", "SemIf fixtures, fine-tuned weights", guarded(ctx, "semif", sec_semif)),
        ("effect", "Fine-tuning effect on this machine", guarded(ctx, "effect", sec_effect)),
        ("native", "Native benchmarks: smoke and perturbations",
         guarded(ctx, "native", sec_native)),
        ("typed", "typed-decisions", guarded(ctx, "typed", sec_typed)),
        ("validate", "`scripts/validate_checkpoint.py`", guarded(ctx, "validate", sec_validate)),
        ("kv", "KV cache on the fine-tuned 4B Q8_0", guarded(ctx, "kv", sec_kv)),
        ("cpu", "CPU-only", guarded(ctx, "cpu", sec_cpu)),
        ("telemetry", "Telemetry of the SemIf runs", guarded(ctx, "telemetry", sec_telemetry)),
        ("commands", "Commands and data", guarded(ctx, "commands", sec_commands, base_ledger)),
        ("odd", "Anything odd", guarded(ctx, "odd", sec_odd, notes_path)),
    ]


def trimmed(lines):
    lines = list(lines)
    while lines and lines[-1] == "":
        lines.pop()
    return lines


COLLAPSED = {"telemetry"}  # collapsed in the GitHub comment, open in the working notes


# Sections left out of the GitHub comment, first to last, when it would exceed COMMENT_LIMIT.
OMIT_ORDER = ("telemetry", "tests", "validate", "commands", "kv", "cpu", "native", "typed",
              "effect", "semif", "machine", "state", "odd")


def scope_text(ctx):
    """Which part of the base campaign this follow-up repeats. Said in the text itself: the
    status line goes away when the campaign is complete, the fact that it is a subset does not."""
    of = f"{len(ctx.runs)} of the {ctx.base_total} runs" if ctx.base_total else (
        f"{len(ctx.runs)} runs")
    return (f"Scope: {of} of that report, with the same run ids: the unit and integration "
            "suites (the integration one pinned to the 4B Q8_0); on CUDA, smoke, perturbations, "
            "`validate_checkpoint`, SemIf and typed-decisions for the 4B and the 1.7B at Q8_0, "
            "Q4_K_M and BF16; the `--kv-type q8_0` and `q4_0` runs (smoke, SemIf) of the 4B Q8_0; "
            "and its CPU-only smoke, SemIf and typed-decisions runs. Not repeated: Vulkan, "
            "`rizzo decide`, the device listing, the unpinned integration run, the batch-size and "
            "thread-count sweeps, the KV-cache runs of other weights and on the CPU, and the "
            "other CPU-only runs.")


def comment_text(ctx, secs, summary_path, omit):
    """The GitHub comment; `omit` lists section keys left out to fit the size limit."""
    where = "on the same machine" if ctx.same_machine else "on this machine"
    L = [f"## Follow-up: the fine-tuned weights (`--weights flow`) {where}", ""]
    done, total = len(ctx.done), len(ctx.runs)
    if ctx.mismatches:
        L += ["> **Do not post yet:** the raw outputs of " + ids_text(ctx.mismatches)
              + " do not show the pinned flow GGUF (see *Anything odd*).", ""]
    if done < total:
        L += [f"> Status: **{done}/{total}** runs done — this file is regenerated after every "
              "run; the campaign is still in progress.", ""]
    intro = optional_text(summary_path)
    if intro:
        L += [intro, ""]
    lead = "**Reading notes.** " if intro else ""
    L += [f"{lead}Follow-up to [this report]({ISSUE_URL}), run with the fine-tuned GGUF files "
          "(*flow*) instead of XHToken's original ones (*base*, `--weights base`, the numbers of "
          f"that report). {scope_text(ctx)} A flow run and the base run of the same id are "
          "paired: same configuration and commands, other weights. Differences are flow − base; "
          "where an interval is shown, one that contains 0 is marked *includes 0*. The "
          "probabilities are not calibrated.", ""]
    for key, title, lines in secs:
        if not lines:
            continue
        lines = trimmed(lines)
        if key in omit:
            L += [f"*{title}: left out of this comment to fit GitHub's size limit; see the "
                  "data branch.*", ""]
        elif key in COLLAPSED:
            L += details(f"{title}", lines)
        else:
            L += [f"## {title}", "", *lines, ""]
    return "\n".join(L).rstrip("\n") + "\n"


def render_comment(ctx, secs, summary_path):
    """The GitHub comment, under COMMENT_LIMIT characters: when it is longer, sections are left
    out one at a time in OMIT_ORDER (each replaced by a pointer to the data branch). Nothing over
    the limit is ever returned."""
    left = [key for key in OMIT_ORDER if any(k == key and lines for k, _, lines in secs)]
    omit = []
    text = comment_text(ctx, secs, summary_path, omit)
    while len(text) > COMMENT_LIMIT and len(omit) < len(left):
        omit.append(left[len(omit)])
        text = comment_text(ctx, secs, summary_path, omit)
    if len(text) > COMMENT_LIMIT:
        raise ValueError(f"the comment is {len(text)} characters with every section left out "
                         f"(limit {COMMENT_LIMIT}): shorten {FLOW_SUMMARY.name}")
    if omit:
        ctx.errors.append(f"the comment would exceed {COMMENT_LIMIT} characters: left out of it: "
                          + ", ".join(omit))
    return text


def status_row(ctx, i, run):
    e = ctx.F[run["id"]]
    state, detail = ctx.checks[run["id"]]
    if e.rec is None:
        return (i, f"`{run['id']}`", e.status, DASH, DASH, DASH, DASH)
    return (i, f"`{run['id']}`", e.status, e.rec.get("exit_codes"), flt(e.rec.get("wall_s"), 1),
            e.rec.get("started"), f"{state}: {detail}")


def run_log(ctx, i, run):
    """One run of the working notes' log: commands, statistics, telemetry, checks."""
    e, base = ctx.F[run["id"]], ctx.B[run["id"]]
    rec = e.rec
    L = [f"### {i}. `{run['id']}` — {rc.SUITE_NAMES.get(run['suite'], run['suite'])} — "
         f"{rec.get('status')}", ""]
    threads = f" · {rec['threads']} threads" if rec.get("threads") else ""
    L.append(f"- {rc.label(run)}{threads}"
             + f" · started {rec.get('started')} · wall {flt(rec.get('wall_s'), 1)} s"
             + f" · exit {rec.get('exit_codes')}")
    L += [f"- `{c}`" for c in recorded(rec)]
    try:
        L += [f"- {line}" for line in rc.stats_lines(rec)]
    except Exception as error:
        L.append(f"- (stats unavailable: {error!r})")
    try:
        L.append(f"- Telemetry: {rc.telemetry_line(rec)}")
    except Exception as error:
        L.append(f"- (telemetry unavailable: {error!r})")
    try:
        L += [f"- {gate_line(rec)}", f"- {load_line(rec)}"]
    except Exception as error:
        L.append(f"- (machine state unavailable: {error!r})")
    state, detail = ctx.checks[run["id"]]
    L.append(f"- Weights check: {state} — {detail}")
    # The runner's own guard (run_flow.py), same evidence. Read from the record, not from e.stats:
    # a run the guard failed has status `failed`, so e.stats is empty for it, and that is exactly
    # the run whose verdict this line has to show.
    stats = rec.get("stats")
    guard = stats.get("weights_check") if isinstance(stats, dict) else None
    if isinstance(guard, dict):
        L.append("- Runner's weights guard: " + ("ok" if guard.get("ok") else "FAILED: "
                 + "; ".join(str(x) for x in (guard.get("problems") or []))[:400]))
    if base.rec is not None:
        L.append(f"- Base run of the same id: {short_stats(base)}")
    if rec.get("status") != "ok" or any(rec.get("exit_codes") or []):
        L += [f"- stderr: `{rc.redact(str(t))}`" for t in (rec.get("stderr_tail") or [])[-5:]]
    L.append("")
    return L


def render_notes(ctx, secs, runs):
    m, b = ctx.machine, ctx.base_machine
    L = ["# Hardware follow-up: fine-tuned weights (`--weights flow`) — working notes", "",
         "Generated by `.research/hw-campaign/render_flow.py` at "
         f"{time.strftime('%Y-%m-%d %H:%M')}, repo at `{m.get('commit')}`. Follow-up to "
         f"[issue #25]({ISSUE_URL}) (base campaign at `{b.get('commit')}`): 39 runs with the "
         "run ids of the base plan, `--weights flow`. Runner `run_flow.py`, ledger "
         "`ledger-flow.json`, "
         "raw outputs `results/local-hw-flow/<id>/`; base ledger `ledger.json`, raw outputs "
         "`results/local-hw/<id>/`.", "", progress_line(ctx), ""]
    warnings = []
    if ctx.mismatches:
        warnings.append(f"WEIGHTS CHECK FAILED for {len(ctx.mismatches)} runs: "
                        + ids_text(ctx.mismatches, 8))
    warnings += ctx.notes + ctx.errors
    if warnings:
        L += ["**Warnings**", "", *[f"- {w}" for w in warnings], ""]
    verified = sum(1 for s, _ in ctx.checks.values() if s == "ok")
    unknown = sum(1 for i, (s, _) in ctx.checks.items()
                  if s == "unknown" and ctx.F[i].rec is not None)
    L += [f"Weights check (raw model metadata against `rizzo_flow.config`): {verified} runs "
          f"verified as the pinned flow file, {len(ctx.mismatches)} mismatches, {unknown} runs "
          "without raw metadata.", ""]
    n = 0
    for _key, title, lines in secs:
        if not lines:
            continue
        n += 1
        L += [f"## {n}. {title}", "", *trimmed(lines), ""]
    n += 1
    L += [f"## {n}. Run status", ""]
    L += T(["#", "Run", "Status", "Exit", "Wall s", "Started", "Weights check"],
           [status_row(ctx, i, run) for i, run in enumerate(runs, 1)])
    n += 1
    L += ["", f"## {n}. Run log", ""]
    pending = []
    for i, run in enumerate(runs, 1):
        if ctx.F[run["id"]].rec is None:
            pending.append(run["id"])
            continue
        try:
            L += run_log(ctx, i, run)
        except Exception as error:
            L += [f"### {i}. `{run['id']}` — log entry unavailable: {error!r}", ""]
    if pending:
        L += [f"Pending: {len(pending)} of {len(runs)} runs (see the status table)."
              if len(pending) > 8 else
              f"Pending ({len(pending)}): " + ", ".join(f"`{i}`" for i in pending), ""]
    n += 1
    L += [f"## {n}. Sources", "",
          "- flow ledger and raw outputs: the paths above; base ledger and raw outputs: "
          "`.research/hw-campaign/ledger.json`, `results/local-hw/`",
          f"- maintainers' flow run: `{MAINT_FLOW.relative_to(ROOT)}` (`report.json`, "
          f"`analysis.json`); their base run: `{MAINT_BASE.relative_to(ROOT)}`; their base "
          f"smoke/validation: `{MAINT_VALIDATION.relative_to(ROOT)}`",
          "- typed-decisions references: README.md (Results on typed-decisions) and "
          "docs/training.md (section 10), parsed at render time; test split "
          "`.research/typed-decisions/all/test.jsonl` for the paired accuracy interval",
          "- pinned flow files: `rizzo_flow.config.GGUF`", ""]
    return "\n".join(L).rstrip("\n") + "\n"


def short_stats(e):
    s, suite = e.stats, e.run["suite"]
    if suite in ("T0", "T1"):
        return f"passed {cnt(s.get('passed'))}, failed {cnt(s.get('failed'))}"
    if suite == "T7":
        return (f"authored144 {flt(s.get('authored144'))}, perturbations108 "
                f"{flt(s.get('perturbations108'))}, p50 {rc.ms(s.get('p50_s'))} ms, shared "
                f"{flt(s.get('shared_dps'), 2)} dec/s")
    if suite in ("T4", "T5"):
        return (f"accuracy {acc_nn(s)}, NLL {flt(s.get('nll'))}, median "
                f"{rc.ms(s.get('latency_median_s'))} ms")
    if suite == "T6":
        return (f"smoke {flt(s.get('smoke_accuracy'))} (NLL {flt(s.get('smoke_nll'))}), long state "
                f"{rc.ms(s.get('long_shared_s'))} / {rc.ms(s.get('long_direct_s'))} ms")
    if suite == "T9":
        return (f"accuracy {flt(s.get('accuracy'))}, KL {flt(s.get('kl'))}, Brier "
                f"{flt(s.get('brier'))}, ECE {flt(s.get('ece'))}, p50 {flt(s.get('p50_ms'), 0)} ms")
    return DASH


def render(flow_ledger, runs=None, out_dir=None, *, base_ledger=None, flow_out=None,
           base_out=None, summary=None, notes=None, data_url=None, ambient=None):
    """Write HARDWARE-REPORT-FLOW.md and HARDWARE-REPORT-FLOW-COMMENT.md (repo root by default);
    returns the two paths. `flow_ledger` may be None, {} or partial. Sets flow_ledger["machine"]
    when it is missing, as run_campaign does, so the caller can save it."""
    runs = runs or flow_plan()
    flow_ledger = flow_ledger if isinstance(flow_ledger, dict) else {}
    if base_ledger is None:
        base_ledger = load_ledger_file(BASE_LEDGER)
    ctx = Ctx(flow_ledger, runs, base_ledger, flow_out or FLOW_OUT, base_out or BASE_OUT,
              data_url, ambient)
    notes_path = FLOW_NOTES if notes is None else notes
    summary_path = FLOW_SUMMARY if summary is None else summary
    secs = sections(ctx, base_ledger, notes_path)
    # The comment goes to a public issue: its state section names no process, pid or size.
    public = [(key, title, guarded(ctx, "state (comment)", sec_state, True) if key == "state"
               else lines) for key, title, lines in secs]
    out = Path(out_dir) if out_dir else ROOT
    paths = (out / FLOW_MD.name, out / COMMENT_MD.name)
    failure = None
    for path, build in ((paths[1], lambda: render_comment(ctx, public, summary_path)),
                        (paths[0], lambda: render_notes(ctx, secs, runs))):
        try:  # one file failing must not prevent the other
            write_text(path, build())
        except Exception as error:
            failure = failure or error
            print(f"render_flow: {path.name} not written: {error!r}", file=sys.stderr)
    for problem in ctx.errors:
        print(f"render_flow: {problem}", file=sys.stderr)
    if failure:
        raise failure
    return paths


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-ledger", type=Path, default=FLOW_LEDGER)
    parser.add_argument("--base-ledger", type=Path, default=BASE_LEDGER)
    parser.add_argument("--flow-out", type=Path, default=FLOW_OUT,
                        help="raw outputs of the flow runs (default results/local-hw-flow)")
    parser.add_argument("--base-out", type=Path, default=BASE_OUT,
                        help="raw outputs of the base runs (default results/local-hw)")
    parser.add_argument("--out-dir", type=Path, default=ROOT,
                        help="where the two markdown files go (default: repository root)")
    parser.add_argument("--summary", type=Path, default=FLOW_SUMMARY)
    parser.add_argument("--notes", type=Path, default=FLOW_NOTES)
    parser.add_argument("--data-url", default=DATA_URL, help="link to the follow-up data branch")
    parser.add_argument("--ambient", help="ambient temperature line for the machine table")
    args = parser.parse_args()
    ledger = load_ledger_file(args.flow_ledger)
    paths = render(ledger, flow_plan(), args.out_dir,
                   base_ledger=load_ledger_file(args.base_ledger), flow_out=args.flow_out,
                   base_out=args.base_out, summary=args.summary, notes=args.notes,
                   data_url=args.data_url, ambient=args.ambient)
    done = len(ledger.get("runs") or {})
    print(f"{done}/{len(FLOW_IDS)} runs in the flow ledger; wrote " + ", ".join(map(str, paths)))


if __name__ == "__main__":
    main()
