#!/usr/bin/env python3
"""Preserve completed native disagg runs and their two legacy hardware references.

This imports existing exports. It never starts a simulation or hardware job.
Raw text stays at the source; numeric request projections and original hashes
make the report's latency, error and workload checks reproducible.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports/agentx-native-disagg-data"
CORE = "39e41354104490981f35f76850bd17973369fc8222dab9f7813e25f50146e36e"
FIELDS = [
    "phase",
    "status",
    "request_latency_ms",
    "output_sequence_length",
    "ttft_ms",
    "input_sequence_length",
    "request_start_ns",
    "request_end_ns",
    "source_trace_id",
    "conversation_id",
    "source_kind",
    "source_outer_idx",
    "turn_index",
    "agent_depth",
    "root_correlation_id",
    "error_code",
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def project(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    digest = hashlib.sha256()
    with (
        source.open("rb") as stream,
        target.open("wb") as output,
        gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
    ):
        writer = csv.DictWriter(text, fieldnames=FIELDS)
        writer.writeheader()
        for line in stream:
            digest.update(line)
            record = json.loads(line)
            meta, metrics = (
                record.get("metadata", {}),
                record.get("metrics", {}),
            )
            error = record.get("error")
            failed = bool(
                error or meta.get("was_cancelled") or meta.get("context_overflow_skip")
            )
            value = lambda name, metrics=metrics: metrics.get(name, {}).get("value")
            row = {k: meta.get(k) for k in FIELDS}
            row.update(
                phase=meta.get("benchmark_phase"),
                status="error" if failed else "success",
                request_latency_ms=value("request_latency"),
                ttft_ms=value("time_to_first_token"),
                output_sequence_length=value("output_sequence_length"),
                input_sequence_length=value("input_sequence_length"),
                error_code=error.get("code") if isinstance(error, dict) else None,
            )
            if not failed:
                assert metrics["request_latency"]["unit"] == "ms"
                assert metrics["time_to_first_token"]["unit"] == "ms"
                assert row["output_sequence_length"] == value("usage_completion_tokens")
            counts[f"{row['phase']}/{row['status']}"] += 1
            writer.writerow(row)
    return {
        "file": target.relative_to(DEST).as_posix(),
        "source": str(source),
        "source_sha256": digest.hexdigest(),
        "source_bytes": source.stat().st_size,
        "rows_by_phase_and_status": dict(counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, default=Path("/tmp/n3u-path1-topology-runs")
    )
    parser.add_argument(
        "--hardware-root", type=Path, default=Path("/tmp/n3u-path1-hardware")
    )
    args = parser.parse_args()
    requests, runs, paired = {}, [], []
    names = [f"topo64-v11-p8d8-{p}-c{c}" for p in ["kv", "rr"] for c in [16, 64, 256]]
    names += [f"disagg-p12d6-kv-c{c}-calibrated-v11" for c in [192, 384]]
    for name in names:
        source = args.run_root / name
        target = DEST / "native" / name
        status, env, provenance = [
            read(source / p)
            for p in ["status.json", "server-environment.json", "provenance.json"]
        ]
        assert status["state"] == "completed" and status["valid_agentx_submission"]
        assert env["core_sha256"] == CORE
        assert provenance["speedup"] == 1 and provenance["clock"] == "wall"
        for filename in [
            "status.json",
            "result.json",
            "server-environment.json",
            "provenance.json",
            "topology.json",
            "client-config.json",
            "warmup-inputs.json",
            "calibration-at-launch.json",
            "aic-callback-confirmation.json",
            "prefill/engine.json",
            "decode/engine.json",
            "workload-source.json",
        ]:
            copy(source / filename, target / filename)
        copy(source / "artifacts/profile_export_aiperf.json", target / "summary.json")
        requests[name] = project(
            source / "artifacts/profile_export.jsonl",
            DEST / "request-metrics" / f"{name}.csv.gz",
        )
        logs = {}
        for path in sorted(source.glob("*.log")):
            if not re.match(r"(?:prefill|decode)-\d+\.log$", path.name):
                continue
            data = path.read_bytes()
            count = data.count(b"mocker handoff session limit reached")
            if count:
                logs[path.name] = {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "handoff_limit_mentions": count,
                }
        write(target / "handoff-error-evidence.json", logs)
        entry = {
            "id": name,
            "kind": "simulation",
            "build": "Native V11 disaggregation extension; frozen V10 timing",
            "policy": "rr" if provenance["router"] == "round-robin" else "kv",
            "clients": provenance["concurrency"],
            "gpus": provenance["gpu_count"],
            "topology": "p8d8" if provenance["gpu_count"] == 64 else "p12d6",
            "summary": (target / "summary.json").relative_to(DEST).as_posix(),
            "source_directory": str(source),
            "handoff_limit_mentions": sum(
                v["handoff_limit_mentions"] for v in logs.values()
            ),
            "hardware_id": None,
        }
        if entry["gpus"] == 72:
            c = entry["clients"]
            hardware_id = f"n3u-mnnvl-126-agentx-kv-c{c}"
            hardware_source = args.hardware_root / hardware_id
            hw_target = DEST / "hardware" / hardware_id
            assert (
                sha(hardware_source / "summary.json")
                == provenance["workload_source_sha256"]
            )
            copy(hardware_source / "summary.json", hw_target / "summary.json")
            requests[hardware_id] = project(
                hardware_source / "records.jsonl",
                DEST / "request-metrics" / f"{hardware_id}.csv.gz",
            )
            for filename in [
                "hardware-comparison-throughput.json",
                "hardware-input-audit.json",
            ]:
                copy(source / filename, target / filename)
            hw = read(hw_target / "summary.json")
            artifact = re.search(r"/perf/([^/]+)/", hw["run_info"]["cli_command"])[1]
            paired.append(
                {
                    "id": hardware_id,
                    "kind": "hardware",
                    "policy": "kv",
                    "clients": c,
                    "gpus": 72,
                    "topology": "p12d6",
                    "artifact": artifact,
                    "summary": (hw_target / "summary.json")
                    .relative_to(DEST)
                    .as_posix(),
                    "gcs_console": f"https://console.cloud.google.com/storage/browser/alisachen-models/perf/{artifact}",
                }
            )
            entry["hardware_id"] = hardware_id
        runs.append(entry)
        print(f"Preserved {name}", flush=True)
    write(DEST / "request-metrics/manifest.json", {"runs": requests})
    files = {
        p.relative_to(DEST).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(DEST.rglob("*"))
        if p.is_file() and p != DEST / "manifest.json"
    }
    write(
        DEST / "manifest.json",
        {
            "collected_utc": datetime.now(timezone.utc).isoformat(),
            "native_core_sha256": CORE,
            "native_runs": runs,
            "legacy_hardware": paired,
            "request_metrics_manifest": "request-metrics/manifest.json",
            "files": files,
            "scope": "Eight existing September 17–18 native simulations, two matched legacy 72-GPU KV hardware references; no new executions.",
            "limitations": [
                "P8D8 simulation C16/C64/C256 do not coincide with the report's hardware concurrency points.",
                "P8D8 RR256 has 3710 errors and is a diagnostic only; KV256 has 11 errors.",
                "The original P12D6 topology file contains a stale 64-GPU prose note; structured pools and launch commands specify 12P+6D TP4 = 72 GPUs.",
                "Transfer bandwidth and prefill cache capacity remain unmeasured assumptions.",
            ],
        },
    )


if __name__ == "__main__":
    main()
