#!/usr/bin/env python3
"""Compare an actual-aiperf simulation with the corresponding hardware artifacts.

Checks replay invariants separately from performance: response timing changes
which requests a closed-loop AgentX run reaches within its profiling window.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def lanes(log):
    return re.findall(r"lane=\d+[^\n]*?trace_id=[0-9a-f]+", log)


def barriers(log):
    match = re.search(
        r"Replay interval barriers active: (\d+) requests, (\d+) gated turns, join-widths=(\{[^}]+\})",
        log,
    )
    if not match:
        raise ValueError("Missing replay dependency fingerprint")
    return {
        "requests": int(match[1]),
        "gated_turns": int(match[2]),
        "join_widths": match[3],
    }


def metric(row, key):
    return row["metrics"].get(key, {}).get("value")


def audit(point, real_dir, run_dir):
    if (real_dir / point["id"]).is_dir():
        real_records_path = real_dir / point["id"] / "records.jsonl"
        real_log_path = real_dir / point["id"] / "aiperf.log"
        real_summary_path = real_dir / point["id"] / "summary.json"
    else:
        short = f"{point['policy']}{point['clients']}"
        real_records_path = real_dir / f"real-{short}.jsonl"
        real_log_path = real_dir / f"real-{short}.log"
        real_summary_path = real_dir / f"real-{short}-summary.json"
    real = records(real_records_path)
    sim = records(run_dir / "simulation-dispatch.jsonl")
    real_log = real_log_path.read_text()
    sim_log = (run_dir / "logs/aiperf.log").read_text()
    real_warm = {
        (r["metadata"]["conversation_id"], r["metadata"]["turn_index"]): r
        for r in real
        if r["metadata"]["benchmark_phase"] == "warmup"
    }
    sim_warm = {
        (r["conversation_id"], r["turn_index"]): r
        for r in sim
        if r["phase"] == "warmup"
    }
    common = real_warm.keys() & sim_warm.keys()
    warm_errors = Counter(
        (
            sim_warm[k]["input_tokens"] - metric(real_warm[k], "input_sequence_length"),
            sim_warm[k]["output_tokens"]
            - metric(real_warm[k], "output_sequence_length"),
        )
        for k in common
    )
    # The UUIDs differ but the same warmed trajectory tree identifies the
    # initial play on both sides. Recycled plays intentionally are not matched.
    roots = {
        sim_warm[k]["root_correlation_id"]: real_warm[k]["metadata"][
            "root_correlation_id"
        ]
        for k in common
    }
    real_profile = {
        (
            r["metadata"].get("root_correlation_id"),
            r["metadata"]["conversation_id"],
            r["metadata"]["turn_index"],
        ): r
        for r in real
        if r["metadata"]["benchmark_phase"] == "profiling"
    }
    input_errors, output_errors = Counter(), Counter()
    for s in sim:
        if s["phase"] != "profiling" or s.get("cancelled"):
            continue
        key = (
            roots.get(s["root_correlation_id"]),
            s["conversation_id"],
            s["turn_index"],
        )
        r = real_profile.get(key)
        if r is not None and metric(r, "output_sequence_length") is not None:
            input_errors[s["input_tokens"] - metric(r, "input_sequence_length")] += 1
            output_errors[s["output_tokens"] - metric(r, "output_sequence_length")] += 1
    checks = {
        "initial_lanes": len(lanes(sim_log)),
        "all_initial_lane_snapshots_match": lanes(real_log) == lanes(sim_log),
        "dependency_fingerprint": barriers(sim_log),
        "dependency_fingerprint_matches": barriers(real_log) == barriers(sim_log),
        "real_warmup_requests": len(real_warm),
        "simulation_warmup_requests": len(sim_warm),
        "warmup_conversations_and_turns_match": real_warm.keys() == sim_warm.keys(),
        "warmup_input_and_output_tokens_match": warm_errors
        == Counter({(0, 0): len(real_warm)}),
        "warmup_token_error_counts": {
            str(k): v for k, v in sorted(warm_errors.items())
        },
        "matched_initial_tree_profile_requests": sum(input_errors.values()),
        "initial_tree_profile_input_token_error_counts": dict(
            sorted(input_errors.items())
        ),
        "initial_tree_profile_output_token_error_counts": dict(
            sorted(output_errors.items())
        ),
    }
    for name in (
        "all_initial_lane_snapshots_match",
        "dependency_fingerprint_matches",
        "warmup_conversations_and_turns_match",
        "warmup_input_and_output_tokens_match",
    ):
        assert checks[name], (point["id"], name, checks)
    assert checks["initial_lanes"] == point["clients"]
    assert checks["dependency_fingerprint"]["requests"] == 68266
    assert checks["dependency_fingerprint"]["gated_turns"] == 34015
    assert sum(input_errors.values()) > 0, "No common profiling requests to audit"
    assert set(input_errors) == {0}, (point["id"], "input token mismatch", input_errors)
    assert set(output_errors) == {0}, (
        point["id"],
        "output token mismatch",
        output_errors,
    )

    phase_times = re.search(
        r"PhaseRecordsStats\(phase=CreditPhase.PROFILING[^\n]*?start_ns=(\d+), sent_end_ns=(\d+)",
        sim_log,
    )
    assert phase_times, "Missing completed profiling phase"
    checks["actual_profiling_send_window_s"] = (
        int(phase_times[2]) - int(phase_times[1])
    ) / 1e9
    assert abs(checks["actual_profiling_send_window_s"] - 3600) < 0.01
    checks["max_request_elapsed_s"] = max(s["end_s"] - s["start_s"] for s in sim)
    assert checks["max_request_elapsed_s"] <= 1200.01

    summary = json.loads((run_dir / "profile_export_aiperf.json").read_text())
    assert not summary["was_cancelled"]
    assert all(
        e["error_details"]["type"] == "TimeoutError" for e in summary["error_summary"]
    ), summary["error_summary"]
    real_summary = json.loads(real_summary_path.read_text())
    specs = {
        "total_tokens_s_gpu": ("total_token_throughput", "avg", point["gpus"]),
        "output_tokens_s_gpu": ("output_token_throughput", "avg", point["gpus"]),
        "requests_s": ("request_throughput", "avg", 1),
        "ttft_p95_s": ("time_to_first_token", "p95", 1000),
        "itl_mean_ms": ("inter_token_latency", "avg", 1),
        "itl_p50_ms": ("inter_token_latency", "p50", 1),
        "mean_input_tokens": ("input_sequence_length", "avg", 1),
        "mean_output_tokens": ("output_sequence_length", "avg", 1),
        "cached_input_percent": ("overall_usage_prompt_cache_read_pct", "avg", 1),
    }
    comparison = {}
    for name, (key, stat, divisor) in specs.items():
        real_value = real_summary[key][stat] / divisor
        sim_value = summary[key][stat] / divisor
        comparison[name] = {
            "real": real_value,
            "simulation": sim_value,
            "error_pct": 100 * (sim_value / real_value - 1),
        }
    kinds = {
        "real": Counter(
            r["metadata"].get("source_kind")
            for r in real
            if r["metadata"]["benchmark_phase"] == "profiling"
            and metric(r, "output_sequence_length") is not None
        ),
        "simulation": Counter(
            s["source_kind"]
            for s in sim
            if s["phase"] == "profiling" and not s.get("cancelled")
        ),
    }
    return {
        "id": point["id"],
        "replay_checks": checks,
        "performance": comparison,
        "profile_request_kinds": kinds,
        "profile_duration_config_s": summary["input_config"]["phases"][0]["duration"],
        "reported_benchmark_duration": summary["benchmark_duration"],
        "simulation_error_summary": summary["error_summary"],
        "simulation_branch_stats": summary["branch_stats"],
        "simulation_provenance": json.loads(
            (run_dir / "simulation-provenance.json").read_text()
        ),
        "simulation_stats": json.loads((run_dir / "simulation-stats.json").read_text()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-suffix", default="-verified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    targets = json.loads(args.targets.read_text())
    result = {
        "metric_basis": targets["metric_basis"],
        "points": [
            audit(p, args.real_dir, args.run_root / (p["id"] + args.run_suffix))
            for p in targets["points"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
