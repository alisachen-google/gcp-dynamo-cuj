#!/usr/bin/env python3
"""Collect actual native results and render the C480 flag comparison."""

import csv
import gzip
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
DATA = STUDY / "sim-results/agentx_disagg_c480_native_20260919"
ROOT = Path("/tmp/agentx-disagg-c480-runs")
REPORT = STUDY / "reports/agentx-disagg-c480-kv-flags"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def valid_request(record):
    meta = record.get("metadata", {})
    if meta.get("benchmark_phase") != "profiling":
        return None, "outside_profiling"
    if (
        record.get("error")
        or meta.get("was_cancelled")
        or meta.get("context_overflow_skip")
    ):
        return None, "error_or_cancelled_or_overflow"
    metrics = record.get("metrics", {})

    def metric(name):
        return metrics.get(name, {}).get("value")

    osl = metric("usage_completion_tokens")
    latency = metric("request_latency")
    ttft = metric("time_to_first_token")
    if (
        not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in [osl, latency, ttft]
        )
        or osl <= 0
        or latency <= 0
        or ttft < 0
    ):
        return None, "invalid_metric"
    if (
        metrics["request_latency"].get("unit") != "ms"
        or metrics["time_to_first_token"].get("unit") != "ms"
    ):
        raise ValueError("Unexpected latency unit")
    return {
        "source_trace_id": meta.get("source_trace_id"),
        "root_correlation_id": meta.get("root_correlation_id"),
        "turn_index": meta.get("turn_index"),
        "agent_depth": meta.get("agent_depth"),
        "input_tokens": metric("usage_prompt_tokens"),
        "output_tokens": osl,
        "request_start_ns": meta.get("request_start_ns"),
        "request_end_ns": meta.get("request_end_ns"),
        "ttft_s": ttft / 1000,
        "e2e_s": latency / 1000,
        "normalized_e2e_s_per_token": latency / 1000 / osl,
    }, None


def collect(job, run):
    destination = DATA / "runs" / job["name"]
    output = destination / "result.json"
    if output.exists():
        return read(output)
    status = read(run / "status.json")
    if status["state"] != "completed":
        raise ValueError("Cannot collect an incomplete run")
    destination.mkdir(parents=True, exist_ok=True)
    rows, exclusions = [], Counter()
    source = run / "artifacts/profile_export.jsonl"
    with source.open() as stream:
        for line in stream:
            row, reason = valid_request(json.loads(line))
            if reason:
                exclusions[reason] += 1
            else:
                rows.append(row)
    if not rows:
        raise ValueError("No eligible per-request profiling metrics")
    compact = destination / "request-metrics.csv.gz"
    with gzip.open(compact, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = read(run / "result.json")
    core = read(run / "server-environment.json")["core_sha256"]
    tuning = read(run / "router-flags-confirmed.json")
    expected = {
        "prefill_load_scale": job["load_scale"],
        "overlap_score_credit": job["overlap_credit"],
        "overlap_score_credit_decay": job["credit_decay"],
        "router_temperature": 0.0,
        "router_queue_policy": "fcfs",
    }
    if tuning != expected:
        raise ValueError("Run's resolved flags disagree with the declared sweep point")
    for role in ["prefill", "decode"]:
        if read(run / role / "engine.json") != read(
            DATA / "configs" / (role + "-engine.json")
        ):
            raise ValueError("Engine config changed during the flag sweep")
    ratios = [r["normalized_e2e_s_per_token"] for r in rows]
    request_count = result.get("requests") or len(rows)
    errors = result.get("errors") or 0
    result.update(
        name=job["name"],
        repeat=job.get("repeat", False),
        load_scale=job["load_scale"],
        overlap_credit=job["overlap_credit"],
        credit_decay=job["credit_decay"],
        native_core_sha256=core,
        p95_ttft_s=result["p95_ttft_ms"] / 1000,
        e2e_normalized_interactivity_p90=1 / percentile(ratios, 0.9),
        e2e_latency_p95_s=percentile([r["e2e_s"] for r in rows], 0.95),
        error_fraction=errors / (request_count + errors),
        errors=errors,
        eligible_requests=len(rows),
        excluded_records=dict(exclusions),
        isl_p50=percentile([r["input_tokens"] for r in rows], 0.5),
        osl_p90=percentile([r["output_tokens"] for r in rows], 0.9),
        turn_index_p50=percentile([r["turn_index"] for r in rows], 0.5),
        source_trace_ids=sorted({r["source_trace_id"] for r in rows}),
        source_export_sha256=sha(source),
        request_metrics_sha256=sha(compact),
        source_run=str(run),
        collected_utc=datetime.now(timezone.utc).isoformat(),
    )
    for filename in [
        "provenance.json",
        "server-environment.json",
        "frontend-resolved.json",
        "router-flags-confirmed.json",
        "warmup-inputs.json",
        "client-config.json",
        "topology.json",
        "calibration-at-launch.json",
    ]:
        shutil.copy2(run / filename, destination / filename)
    shutil.copy2(
        run / "artifacts/profile_export_aiperf.json",
        destination / "aiperf-summary.json",
    )
    write(output, result)
    return result


def assessed_results():
    plan = read(DATA / "plan.json")
    rows = [read(path) for path in sorted((DATA / "runs").glob("*/result.json"))]
    baseline = next((r for r in rows if r["name"] == "baseline"), None)
    for row in rows:
        row["warmup_matches_baseline"] = bool(
            baseline and row["warmup_input_sha256"] == baseline["warmup_input_sha256"]
        )
        row["quality_pass"] = bool(
            row["valid_agentx_submission"]
            and row["native_core_sha256"] == plan["native_core_sha256"]
            and row["warmup_matches_baseline"]
            and row["error_fraction"] <= plan["quality"]["maximum_error_fraction"]
            and row["excluded_records"].get("invalid_metric", 0) == 0
        )
        row["ttft_slo_pass"] = row["p95_ttft_s"] <= plan["slo"]["p95_ttft_seconds"]
        row["interactivity_slo_pass"] = (
            row["e2e_normalized_interactivity_p90"]
            >= plan["slo"]["e2e_normalized_interactivity_p90_tokens_per_second"]
        )
        row["both_slos_pass"] = row["ttft_slo_pass"] and row["interactivity_slo_pass"]
        row["throughput_change_pct"] = (
            100 * (row["total_tok_s_per_gpu"] / baseline["total_tok_s_per_gpu"] - 1)
            if baseline
            else None
        )
    return rows


def choose_finalist(rows):
    candidates = [
        r
        for r in rows
        if r["quality_pass"] and not r["repeat"] and r["name"] != "baseline"
    ]
    feasible = [r for r in candidates if r["both_slos_pass"]]
    if feasible:
        return max(feasible, key=lambda r: r["total_tok_s_per_gpu"])
    if candidates:
        return max(candidates, key=lambda r: r["e2e_normalized_interactivity_p90"])
    return None


def render():
    plan = read(DATA / "plan.json")
    state = (
        read(ROOT / "queue-state.json")
        if (ROOT / "queue-state.json").exists()
        else {"jobs": {}}
    )
    jobs = read(DATA / "jobs.json")["jobs"]
    rows = assessed_results()
    by_name = {r["name"]: r for r in rows}
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    assessed = {
        "updated_utc": timestamp,
        "plan": plan,
        "state": state,
        "completed": len(rows),
        "rows": rows,
        "interactivity_definition": "1 / P90(E2E_seconds / server_output_tokens), linear percentile; successful profiling requests only",
    }
    write(DATA / "results.json", assessed)
    columns = [
        "name",
        "load_scale",
        "overlap_credit",
        "credit_decay",
        "total_tok_s_per_gpu",
        "output_tok_s_per_gpu",
        "p95_ttft_s",
        "e2e_normalized_interactivity_p90",
        "errors",
        "error_fraction",
        "quality_pass",
        "both_slos_pass",
        "throughput_change_pct",
    ]
    with (DATA / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# AgentX disagg KV flag sweep at concurrency 480",
        "",
        (
            f"Updated: {timestamp}. Completed: **{len(rows)}/{len(jobs)}** scheduled runs. "
            f"Queue: **{state.get('state', 'starting')}**."
        ),
        "",
        (
            "64 GPUs, 8 prefill + 8 decode TP4 workers; 480 live AgentX sessions. "
            "Native DynoSim V11 disaggregation extension with frozen V10 AIC timing coefficients. "
            "These are simulation forecasts; disaggregated TTFT is not calibrated to hardware."
        ),
        "",
        (
            "The initial grid uses scale 1, credit {0.6, 0.8, 1.0}, and decay {0, 0.5, 1.0}. "
            "Two controls use scales 2 and 3 at credit 0.8 / decay 0. Temperature stays 0 and "
            "FCFS stays fixed. Baseline and finalist repeats check reproducibility."
        ),
        "",
        "| Run | Scale | Credit | Decay | Total tok/s/GPU | Δ vs baseline | P95 TTFT | E2E I90 | Errors | Quality / both SLOs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for job in jobs:
        row = by_name.get(job["name"])
        lead = f"| {job['name']} | {job['load_scale']:g} | {job['overlap_credit']:g} | {job['credit_decay']:g} |"
        if row:
            delta = row["throughput_change_pct"]
            lines.append(
                lead
                + f" {row['total_tok_s_per_gpu']:,.1f} | "
                + (f"{delta:+.2f}%" if delta is not None else "—")
                + f" | {row['p95_ttft_s']:.3f} s | {row['e2e_normalized_interactivity_p90']:.2f} tok/s | "
                + f"{row['errors']:g} | {'pass' if row['quality_pass'] else 'FAIL'} / "
                + f"{'pass' if row['both_slos_pass'] else 'fail'} |"
            )
        else:
            status = state["jobs"].get(job["name"], {}).get("state", "pending")
            lines.append(lead + f" — | — | — | — | — | {status} |")
    lines += [
        "",
        "## Interpretation and controls",
        "",
        (
            "Select the highest observed total input+output throughput/GPU satisfying P95 TTFT ≤10 s "
            "and E2E-normalized interactivity I90 ≥20 output tok/s. I90 is computed as "
            "`1 / P90(E2E_seconds / output_tokens)` from individual successful profiling requests; "
            "it is not inverse TPOT. Pending runs provide no evidence of a win."
        ),
        "",
        (
            "Quality requires valid AgentX replay, the frozen native build, matching baseline warmup, "
            "valid per-request latency metrics, and error rate ≤0.1%. Errors remain visible. "
            "Different completed turn mixes remain possible in a time-bounded, closed-loop replay."
        ),
        "",
        (
            "Load-dependent credit is `credit / (1 + decay × excess_prefill_load / request_size)`. "
            "Decay 1 halves credit at one request-equivalent of excess backlog. It does not expire "
            "cache entries. In this pure-prefill deterministic configuration, multiplying all "
            "candidate prefill costs by the same positive scale is expected to preserve their "
            "ordering; scale controls test whether another cost or runtime effect matters."
        ),
        "",
        (
            "Every arm uses the same 393-root dataset, tokenizer, seed 42, cache-bust namespace, "
            "25–75% trajectory starts, trajectory warmup, 3600-second profile, 60-second grace, "
            "1200-second request timeout, engine configs, and 64 GB/s per-rank transfer assumption. "
            "Each starts with new workers and empty caches. No GPU hardware is provisioned."
        ),
        "",
        (
            "The native handoff implementation has finite session admission; overload errors must "
            "be considered when comparing flags. Transfer contention and prefill cache-capacity "
            "uncertainty remain simulation limitations."
        ),
        "",
        (
            "[Plan](../sim-results/agentx_disagg_c480_native_20260919/plan.json) · "
            "[Results JSON](../sim-results/agentx_disagg_c480_native_20260919/results.json) · "
            "[CSV](../sim-results/agentx_disagg_c480_native_20260919/results.csv) · "
            "[Native flag propagation check](../sim-results/agentx_disagg_c480_native_20260919/preflight.json)"
        ),
        "",
    ]
    baseline = by_name.get("baseline")
    repeat = by_name.get("baseline-repeat")
    if baseline and repeat:
        changes = {
            key: 100 * (repeat[key] / baseline[key] - 1)
            for key in [
                "total_tok_s_per_gpu",
                "p95_ttft_s",
                "e2e_normalized_interactivity_p90",
            ]
        }
        lines += [
            "Baseline repeat deltas (one pair, not a confidence interval): "
            + "; ".join(f"{key}: {value:+.2f}%" for key, value in changes.items())
            + ".",
            "",
        ]
    markdown = "\n".join(lines)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.with_suffix(".md").write_text(markdown)
    from markdown_it import MarkdownIt

    body = MarkdownIt().enable("table").render(markdown)
    REPORT.with_suffix(".html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>AgentX disagg C480 KV flag sweep</title><style>"
        "body{font:16px/1.55 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#162337}"
        "table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}"
        "td,th{padding:9px 12px;border-bottom:1px solid #ccd5df;text-align:right;white-space:nowrap}"
        "th{background:#eaf0f6}td:first-child,th:first-child{text-align:left}"
        "a{color:#075aad}code{background:#edf1f5;padding:2px 4px}</style>"
        + body
        + "</html>"
    )
    return assessed


if __name__ == "__main__":
    result = render()
    print(
        json.dumps(
            {"completed": result["completed"], "report": str(REPORT.with_suffix(".md"))}
        )
    )
