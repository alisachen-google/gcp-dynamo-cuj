#!/usr/bin/env python3
"""Preserve the eight D88 simulations and audit their matched hardware inputs.

Numeric records, configuration files, log evidence and hashes are published.
This command only reads completed jobs; it never launches or alters a run.
"""

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from import_native_disagg_report_data import copy, project, read, sha, write
from native_disagg_flags.run_path1_topology import warmup_signature

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
STUDY = ROOT / "sim-results/agentx_d88_matched_20260920"
DEST = REPORTS / "agentx-native-d88-matched-data"
FILES = [
    "status.json",
    "result.json",
    "server-environment.json",
    "provenance.json",
    "topology.json",
    "client-config.json",
    "warmup-inputs.json",
    "calibration-at-launch.json",
    "aic-callback-confirmation.json",
    "router-flags-confirmed.json",
    "frontend-resolved.json",
    "startup-health.json",
    "prefill/engine.json",
    "decode/engine.json",
    "workload-source.json",
]


def hardware_inputs(plan, raw_roots, destination):
    inventory = read(REPORTS / "agentx-serving-perf-data/manifest.json")
    hardware = {r["id"]: r for r in inventory["runs"]}
    for point in plan["points"]:
        target = destination / "hardware-warmup" / f"{point['hardware_id']}.json"
        item = hardware[point["hardware_id"]]
        folder = REPORTS / item["input_root"]
        entry = read(folder / "request-metrics/manifest.json")["runs"][item["id"]]
        assert sha(folder / item["summary"]) == sha(STUDY / point["hardware_summary"])
        if target.exists():
            saved = read(target)
            assert saved["source_sha256"] == entry["source_sha256"]
            assert saved["summary_sha256"] == sha(folder / item["summary"])
            continue
        candidates = [r / f"{item['id']}.jsonl" for r in raw_roots]
        source = next((p for p in candidates if p.exists()), None)
        if source is None:
            raise FileNotFoundError(f"Supply the original export: {entry['source']}")
        assert sha(source) == entry["source_sha256"], source
        signature = warmup_signature(source)
        assert signature["all_successful_one_token"]
        write(
            target,
            signature
            | {
                "source": entry["source"],
                "source_sha256": entry["source_sha256"],
                "source_bytes": source.stat().st_size,
                "summary_sha256": sha(folder / item["summary"]),
                "hardware_id": item["id"],
            },
        )
        print(
            f"Preserved {item['id']}: {signature['requests']} warmup inputs", flush=True
        )


def log_evidence(source, target):
    # Counts are log mentions, not request-error counts. Keep representative
    # lines and complete-log hashes so errors cannot disappear in a summary.
    phrases = [
        "mocker handoff session limit reached",
        "handoff session timed out",
        "handoff timeout",
        "panicked at",
        "Traceback (most recent call last)",
    ]
    logs = {}
    for path in sorted(source.glob("*.log")):
        if not re.fullmatch(
            r"(?:prefill-\d+|decode-\d+|frontend|client)\.log", path.name
        ):
            continue
        data = path.read_bytes()
        lines = data.decode(errors="replace").splitlines()
        examples = [
            re.sub(r"\x1b\[[0-9;]*m", "", line)
            for line in lines
            if any(phrase in line for phrase in phrases)
        ]
        logs[path.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "mentions": {phrase: data.count(phrase.encode()) for phrase in phrases},
            "examples": examples[:5],
        }
        if path.name == "client.log":
            phases = re.findall(
                r"Phase profiling \(profiling\) complete \| "
                r"completed=([\d,]+), cancelled=([\d,]+), errors=([\d,]+)"
                r".*elapsed=([\d.]+)s \| grace_period_timeout=(True|False)",
                data.decode(errors="replace"),
            )
            assert len(phases) == 1, (source, phases)
            completed, cancelled, errors, elapsed, timed_out = phases[0]
            write(
                target / "client-phase-audit.json",
                {
                    "completed_credits": int(completed.replace(",", "")),
                    "cancelled_credits": int(cancelled.replace(",", "")),
                    "error_credits": int(errors.replace(",", "")),
                    "elapsed_s_including_drain": float(elapsed),
                    "grace_period_timeout": timed_out == "True",
                    "source_log_sha256": logs[path.name]["sha256"],
                    "scope": "Profiling phase credits from the AIPerf client log; drain cancellations are separate from exported request errors.",
                },
            )
    write(target / "log-evidence.json", logs)
    return sum(v["mentions"][phrases[0]] for v in logs.values())


def preserve_host_observations(source, target, projection):
    with gzip.open(projection, "rt", newline="") as stream:
        requests = [r for r in csv.DictReader(stream) if r["phase"] == "profiling"]
    begin = min(int(r["request_start_ns"]) for r in requests) / 1e9
    end = max(int(r["request_end_ns"]) for r in requests) / 1e9
    path = source / "host-observations.jsonl"
    with (
        path.open("rb") as incoming,
        (target / "host-observations.jsonl.gz").open("wb") as outgoing,
        gzip.GzipFile(fileobj=outgoing, mode="wb", filename="", mtime=0) as zipped,
    ):
        shutil.copyfileobj(incoming, zipped)
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    profile = [s for s in samples if begin <= s["time"] <= end]
    assert profile
    memory = [
        int(re.search(r"MemAvailable:\s+(\d+)", s["meminfo"])[1]) / 1024**2
        for s in profile
    ]
    cpu = [list(map(int, s["proc_stat"].splitlines()[0].split()[1:9])) for s in profile]
    ticks = [b - a for a, b in zip(cpu[0], cpu[-1])]
    audit = {
        "source_sha256": sha(path),
        "samples": len(samples),
        "profiling_samples": len(profile),
        "profiling_start_unix": begin,
        "profiling_end_unix": end,
        "minimum_available_memory_gib": min(memory),
        "maximum_one_minute_loadavg": max(s["loadavg"][0] for s in profile),
        "host_cpu_busy_pct": 100 * (1 - (ticks[3] + ticks[4]) / sum(ticks)),
        "scope": "Shared host counters across the profiling window, including other processes; not modeled GPU utilization.",
    }
    write(target / "host-audit.json", audit)


def model_audit(source, target):
    result = {}
    for role in ["prefill", "decode"]:
        path = source / role / "forward-passes.jsonl.gz"
        peaks = {
            k: 0
            for k in [
                "num_prefill_requests",
                "sum_prefill_tokens",
                "num_decode_requests",
            ]
        }
        events, workers = 0, set()
        with gzip.open(path, "rt") as stream:
            for line in stream:
                event = json.loads(line)["fpm"]
                events += 1
                workers.add(event["worker_id"])
                for key, previous in peaks.items():
                    peaks[key] = max(previous, event["scheduled_requests"][key])
        assert len(workers) == 8, (source, role, workers)
        assert peaks["num_prefill_requests"] <= (8 if role == "prefill" else 0)
        assert peaks["num_decode_requests"] <= (64 if role == "decode" else 0)
        assert peaks["sum_prefill_tokens"] <= (16384 if role == "prefill" else 0)
        result[role] = {
            "events": events,
            "workers": len(workers),
            "peaks": peaks,
            "source": str(path),
            "source_sha256": sha(path),
            "source_bytes": path.stat().st_size,
        }
    write(target / "model-audit.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--destination", type=Path, default=DEST)
    parser.add_argument(
        "--hardware-raw-root",
        type=Path,
        action="append",
        default=[Path("/tmp/agentx-d88-import"), Path("/tmp/agentx-concrete-import")],
    )
    parser.add_argument("--prepare-hardware-only", action="store_true")
    args = parser.parse_args()
    plan = read(STUDY / "plan.json")
    for name, expected in plan["config_sha256"].items():
        assert sha(STUDY / name) == expected, name
    dest = args.destination.resolve()
    hardware_inputs(plan, args.hardware_raw_root, dest)
    if args.prepare_hardware_only:
        return
    run_root = args.run_root or Path(plan["run_root"])
    copy(run_root / "state.json", dest / "controller-state.json")
    copy(run_root / "jobs.json", dest / "controller-jobs.json")
    adjustment = run_root / "controller-resource-adjustment.json"
    if adjustment.exists():
        copy(adjustment, dest / adjustment.name)
    requests, runs = {}, []
    for point in plan["points"]:
        name = point["name"]
        source, target = run_root / name, dest / "native" / name
        status = read(source / "status.json")
        assert status["state"] == "completed" and status["valid_agentx_submission"], (
            name
        )
        assert (
            read(source / "server-environment.json")["core_sha256"]
            == plan["native_core_sha256"]
        )
        for filename in FILES:
            copy(source / filename, target / filename)
        copy(source / "artifacts/profile_export_aiperf.json", target / "summary.json")
        requests[name] = project(
            source / "artifacts/profile_export.jsonl",
            dest / "request-metrics" / f"{name}.csv.gz",
            destination=dest,
        )
        preserve_host_observations(source, target, dest / requests[name]["file"])
        model_audit(source, target)
        # Check independent exports, not just the seed or summary validity stamp.
        simulation = warmup_signature(source / "artifacts/profile_export.jsonl")
        hardware = read(dest / "hardware-warmup" / f"{point['hardware_id']}.json")
        assert simulation == read(source / "warmup-inputs.json"), name
        assert simulation["all_successful_one_token"]
        assert simulation["inputs"] == hardware["inputs"], name
        limits = log_evidence(source, target)
        runs.append(
            {
                "id": name,
                "kind": "simulation",
                "policy": point["policy"],
                "clients": point["concurrency"],
                "gpus": 64,
                "topology": "p8d8",
                "role": "holdout",
                "hardware_id": point["hardware_id"],
                "build": plan["method"],
                "source_directory": str(source),
                "summary": (target / "summary.json").relative_to(dest).as_posix(),
                "handoff_limit_mentions": limits,
                "warmup_inputs_match": True,
            }
        )
        print(
            f"Preserved {name}: {requests[name]['rows_by_phase_and_status']}",
            flush=True,
        )
    write(dest / "request-metrics/manifest.json", {"runs": requests})
    write(
        dest / "manifest.json",
        {
            "collected_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": plan["source_commit"],
            "native_core_sha256": plan["native_core_sha256"],
            "native_base_sha256": plan["native_base_sha256"],
            "plan": "../../sim-results/agentx_d88_matched_20260920/plan.json",
            "plan_sha256": sha(STUDY / "plan.json"),
            "request_metrics_manifest": "request-metrics/manifest.json",
            "native_runs": runs,
            "files": {
                p.relative_to(dest).as_posix(): {
                    "sha256": sha(p),
                    "bytes": p.stat().st_size,
                }
                for p in sorted(dest.rglob("*"))
                if p.is_file() and p != dest / "manifest.json"
            },
            "scope": "Eight new 64-GPU D88 simulations at existing hardware points; frozen agg V10 timing, no D88 fitting.",
            "limitations": [
                "Native V11 disaggregation extension plus explicit bounded handoff queues, not the unchanged agg V10 binary.",
                "Per-worker protocol queue capacity is 4096; scheduled batches remain P8/D64 and state capacity is unchanged.",
                "Transfer bandwidth, shared-link contention and prefill cache sizing remain modeling assumptions.",
                "One run per point; closed-loop completion mixes can differ between simulation and hardware.",
                "Raw payload duplication is disabled; numeric per-request records and all replay controls are retained.",
            ],
        },
    )


if __name__ == "__main__":
    main()
