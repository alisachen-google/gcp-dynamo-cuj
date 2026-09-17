#!/usr/bin/env python3
"""Audit and archive full AgentX replays, keeping hardware references explicit."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
import statistics
from collections import Counter
from pathlib import Path

from audit_aiperf_replay import audit, barriers, lanes, records

METRICS = {
    "total_tokens_s_gpu": ("total_token_throughput", "avg"),
    "output_tokens_s_gpu": ("output_token_throughput", "avg"),
    "requests_s": ("request_throughput", "avg"),
    "goodput_requests_s": ("goodput", "avg"),
    "ttft_p95_s": ("time_to_first_token", "p95"),
    "ttft_p50_s": ("time_to_first_token", "p50"),
    "ttft_p99_s": ("time_to_first_token", "p99"),
    "itl_mean_ms": ("inter_token_latency", "avg"),
    "itl_p50_ms": ("inter_token_latency", "p50"),
    "cached_input_percent": ("overall_usage_prompt_cache_read_pct", "avg"),
    "mean_input_tokens": ("input_sequence_length", "avg"),
    "mean_output_tokens": ("output_sequence_length", "avg"),
}


def metrics(summary, gpus):
    result = {}
    for name, (key, stat) in METRICS.items():
        divisor = (
            gpus if name.endswith("_gpu") else 1000 if name.startswith("ttft_") else 1
        )
        result[name] = summary[key][stat] / divisor
    result["successful_requests"] = int(summary["request_count"]["avg"])
    result["request_errors"] = int(summary.get("error_request_count", {}).get("avg", 0))
    result["goodput_percent"] = (
        100 * result["goodput_requests_s"] / result["requests_s"]
    )
    return result


def percentile(values, percent):
    """Linear sample percentile for queue diagnostics (not summary metric replacement)."""
    values = sorted(values)
    if not values:
        return None
    location = (len(values) - 1) * percent / 100
    lower = int(location)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (location - lower)


def queue_diagnostics(run, dispatch, phase):
    # AIPerf stores epoch nanoseconds; serving records store virtual loop seconds.
    # Recover their shared offset from an identical initial warmup request.
    with (run / "profile_export.jsonl").open() as stream:
        metadata = json.loads(next(stream))["metadata"]
    first = next(
        r
        for r in dispatch
        if r["root_correlation_id"] == metadata["root_correlation_id"]
        and r["conversation_id"] == metadata["conversation_id"]
        and r["turn_index"] == metadata["turn_index"]
        and r["phase"] == metadata["benchmark_phase"]
    )
    epoch = metadata["request_start_ns"] / 1e9 - first["start_s"]
    start = int(phase[1]) / 1e9 - epoch
    end = int(phase[2]) / 1e9 - epoch
    halves = []
    for left, right in ((start, (start + end) / 2), ((start + end) / 2, end)):
        arrivals = [
            r
            for r in dispatch
            if r["phase"] == "profiling" and left <= r["start_s"] < right
        ]
        completed = [r for r in arrivals if not r.get("cancelled")]
        halves.append(
            {
                "dispatches": len(arrivals),
                "successful_requests": len(completed),
                "ttft_p50_successful_s": percentile(
                    [r["ttft_s"] for r in completed], 50
                ),
                "ttft_p95_successful_s": percentile(
                    [r["ttft_s"] for r in completed], 95
                ),
                "prefill_queue_p95_all_dispatches_s": percentile(
                    [r["prefill_queue_s"] for r in arrivals], 95
                ),
                "zero_prefill_queue_dispatch_fraction": sum(
                    r["prefill_queue_s"] < 1e-6 for r in arrivals
                )
                / len(arrivals)
                if arrivals
                else None,
            }
        )
    return {
        "halves": halves,
        "note": "Split by dispatch time at the exact profile midpoint. Successful-request TTFT and all-dispatch modeled queue waits are separate; finite closed-loop and changing request mix prevent a stationarity proof from these two samples alone.",
    }


def internal_audit(point, run):
    summary = json.loads((run / "profile_export_aiperf.json").read_text())
    stats = json.loads((run / "simulation-stats.json").read_text())
    provenance = json.loads((run / "simulation-provenance.json").read_text())
    dispatch = records(run / "simulation-dispatch.jsonl")
    log = (run / "logs/aiperf.log").read_text()
    phase = re.search(
        r"PhaseRecordsStats\(phase=CreditPhase.PROFILING[^\n]*?start_ns=(\d+), sent_end_ns=(\d+)",
        log,
    )
    assert phase, point["id"]
    seconds = (int(phase[2]) - int(phase[1])) / 1e9
    assert abs(seconds - 3600) < 0.01, (point["id"], seconds)
    assert not summary["was_cancelled"]
    assert summary["metadata"]["submission_valid"]
    assert stats["requests"] == len(dispatch)
    assert stats["tokenization_full_parity_checks"] >= 100
    assert len(lanes(log)) == point["clients"]
    fingerprint = barriers(log)
    assert fingerprint["requests"] == 68266 and fingerprint["gated_turns"] == 34015
    assert all(
        e["error_details"]["type"] == "TimeoutError" for e in summary["error_summary"]
    )
    assert summary["input_config"]["phases"][0]["concurrency"] == point["clients"]
    assert summary["benchmark_id"] == point["benchmark_id"]
    assert provenance["serving"] == point["serving"]
    max_elapsed = max(r["end_s"] - r["start_s"] for r in dispatch)
    assert max_elapsed <= 1200.01
    warmup = [r for r in dispatch if r["phase"] == "warmup"]
    assert all(r["output_tokens"] == 1 and not r.get("cancelled") for r in warmup)
    profile = [
        r for r in dispatch if r["phase"] == "profiling" and not r.get("cancelled")
    ]
    assert len(profile) == int(summary["request_count"]["avg"])
    assignment = {}
    for tier, workers in [
        ("prefill", point["serving"]["prefill_workers"]),
        (
            "decode",
            point["serving"]["prefill_workers"]
            if point["serving"]["agg"]
            else point["serving"]["decode_workers"],
        ),
    ]:
        counts = Counter(r[f"{tier}_worker"] for r in dispatch)
        assert set(counts) <= set(range(workers)), (point["id"], tier, counts)
        values = [counts[i] for i in range(workers)]
        if point["policy"] == "rr":
            assert max(values) - min(values) <= 1, (point["id"], tier, counts)
        assignment[tier] = values
    placement = []
    for worker in range(len(assignment["decode"])):
        selected = [r for r in profile if r["decode_worker"] == worker]
        placement.append(
            {
                "decode_worker": worker,
                "profile_successes": len(selected),
                "itl_mean_ms": sum(r["tpot_ms"] for r in selected)
                / max(1, len(selected)),
                "max_admitted_requests": max(
                    (
                        r["decode_admitted_at_dispatch"]
                        for r in dispatch
                        if r["decode_worker"] == worker
                    ),
                    default=0,
                ),
            }
        )
    checks = {
        "profiling_send_window_s": seconds,
        "initial_lanes": len(lanes(log)),
        "dependency_fingerprint": fingerprint,
        "trajectory_warmup_requests": len(warmup),
        "warmup_one_output_token_and_completed": True,
        "cached_tokenization_full_array_checks": stats[
            "tokenization_full_parity_checks"
        ],
        "dispatch_counts_including_warmup_and_cancelled": assignment,
        "rr_balance_checked": point["policy"] == "rr",
        "max_request_elapsed_s": max_elapsed,
        "profile_cancelled_dispatches": sum(
            r["phase"] == "profiling" and bool(r.get("cancelled")) for r in dispatch
        ),
    }
    profile_dispatches = [r for r in dispatch if r["phase"] == "profiling"]
    decode_limit = point.get("recipe_serving_settings", {}).get(
        "decode_max_running_requests"
    )
    diagnostics = {
        "within_run_queue_diagnostics": queue_diagnostics(run, dispatch, phase),
        "mean_prefill_queue_s": statistics.mean(r["prefill_queue_s"] for r in profile),
        "mean_prefill_service_s": statistics.mean(
            r["prefill_service_s"] for r in profile
        ),
        "mean_decode_admitted_at_dispatch": statistics.mean(
            r["decode_admitted_at_dispatch"] for r in profile
        ),
        "max_decode_admitted_at_dispatch": max(
            r["decode_admitted_at_dispatch"] for r in profile_dispatches
        ),
        "recipe_decode_max_running_requests": decode_limit,
        "profile_dispatches_above_recipe_decode_running_limit": sum(
            r["decode_admitted_at_dispatch"] > decode_limit for r in profile_dispatches
        )
        if decode_limit
        else None,
        "admission_note": "Admitted count includes requests waiting for prefill; this is not the physical running GPU batch. Exceeding the recipe limit flags unsupported scheduling pressure, not a measured limit violation.",
    }
    return {
        "id": point["id"],
        "arm": point["arm"],
        "policy": point["policy"],
        "clients": point["clients"],
        "gpus": point["gpus"],
        "recipe_point": point.get("recipe_point"),
        "hardware_comparison_source": point["hardware_comparison_source"],
        "simulation": metrics(summary, point["gpus"]),
        "checks": checks,
        "decode_worker_metrics": placement,
        "serving_diagnostics": diagnostics,
        "stats": stats,
        "provenance": provenance,
    }


def archive(run, destination, compact=False):
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "profile_export_aiperf.json",
        "simulation-provenance.json",
        "simulation-stats.json",
        "replay-config.json",
    ):
        shutil.copy2(run / name, destination / name)
    record_names = (
        "profile_export.jsonl",
        "simulation-dispatch.jsonl",
        "logs/aiperf.log",
    )
    if compact:
        hashes = {
            name: {
                "sha256": hashlib.file_digest(
                    (run / name).open("rb"), "sha256"
                ).hexdigest(),
                "bytes": (run / name).stat().st_size,
            }
            for name in record_names
        }
        (destination / "full-records-manifest.json").write_text(
            json.dumps(
                {
                    "local_run_directory": str(run),
                    "files": hashes,
                    "note": "Full records audited locally; compact archive retains summaries, config and provenance. Rerun commands reproduce the experiment; temporary local files are not permanent storage.",
                },
                indent=2,
            )
            + "\n"
        )
        return
    for name in record_names:
        with (
            (run / name).open("rb") as source,
            gzip.GzipFile(
                filename=str(destination / (Path(name).name + ".gz")),
                mode="wb",
                mtime=0,
            ) as target,
        ):
            shutil.copyfileobj(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", help="Optional comma-separated subset")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Archive summaries, provenance, and record hashes; keep full records in the local run directory",
    )
    args = parser.parse_args()
    targets = json.loads(args.targets.read_text())
    points = targets["points"]
    if args.points:
        selected = set(args.points.split(","))
        assert selected <= {p["id"] for p in points}
        points = [p for p in points if p["id"] in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"metric_basis": targets["metric_basis"], "points": []}
    rows = []
    for point in points:
        candidates = [
            p / point["id"] for p in args.run_root if (p / point["id"]).is_dir()
        ]
        assert len(candidates) == 1, (point["id"], candidates)
        run = candidates[0]
        report = internal_audit(point, run)
        if point["hardware_comparison_source"]:
            parity = audit(point, args.real_dir, run)
            report["hardware_replay_audit"] = parity["replay_checks"]
            report["performance_comparison"] = parity["performance"]
            real_summary = json.loads(
                (args.real_dir / point["id"] / "summary.json").read_text()
            )
            report["real"] = metrics(real_summary, point["gpus"])
        elif point["id"] == "disagg12-6-rr-c192" or point.get("input_replay_audit_id"):
            # The source is a KV hardware run: check ONLY its identical replay
            # inputs, never use its performance as a real RR measurement.
            source_point = point | {
                "id": point.get("input_replay_audit_id") or "disagg12-6-kv-c192",
                "policy": "kv",
            }
            parity = audit(source_point, args.real_dir, run)
            report["config_source_replay_audit"] = {
                "source_policy": "kv",
                "source": point["replay_config_source"],
                "checks": parity["replay_checks"],
            }
        archive(run, args.output / point["id"], args.compact)
        result["points"].append(report)
        row = {
            k: report[k]
            for k in ("id", "arm", "policy", "clients", "gpus", "recipe_point")
        }
        row.update({"sim_" + k: v for k, v in report["simulation"].items()})
        row.update({"real_" + k: v for k, v in report.get("real", {}).items()})
        if report.get("real"):
            row["total_throughput_error_pct"] = 100 * (
                report["simulation"]["total_tokens_s_gpu"]
                / report["real"]["total_tokens_s_gpu"]
                - 1
            )
        rows.append(row)
        print(
            point["id"],
            "PASS",
            f"{report['simulation']['total_tokens_s_gpu']:.1f} total tok/s/GPU",
            flush=True,
        )
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    hashes = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.rglob("*"))
        if p.is_file() and p.name != "sha256.json"
    }
    (args.output / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
