#!/usr/bin/env python3
"""Preserve the D88 report's hardware summaries and numeric request records.

Read-only GCS import. No benchmark jobs are launched. Full JSONL exports remain
in --scratch; only numeric/identity fields and their source hashes enter Git.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "request_latency_ms": ("request_latency", "ms"),
    "output_sequence_length": ("output_sequence_length", "tokens"),
    "input_sequence_length": ("input_sequence_length", "tokens"),
    "ttft_ms": ("time_to_first_token", "ms"),
}
META = [
    "source_trace_id",
    "root_correlation_id",
    "turn_index",
    "agent_depth",
    "request_start_ns",
    "request_end_ns",
]
FIELDS = ["source_line", "phase", "status", *METRICS, *META]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project(source, target):
    counts = Counter()
    checksum = hashlib.sha256()
    with (
        source.open("rb") as incoming,
        target.open("wb") as outgoing,
        gzip.GzipFile(fileobj=outgoing, mode="wb", filename="", mtime=0) as zipped,
        io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text,
    ):
        writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for number, line in enumerate(incoming, 1):
            checksum.update(line)
            record = json.loads(line)
            md = record["metadata"]
            phase = md["benchmark_phase"]
            status = (
                "error"
                if record.get("error")
                else "cancelled"
                if md.get("was_cancelled")
                else "context_overflow_skip"
                if md.get("context_overflow_skip")
                else "success"
            )
            row = {"source_line": number, "phase": phase, "status": status}
            row.update({key: md.get(key, "") for key in META})
            for column, (key, unit) in METRICS.items():
                metric = record.get("metrics", {}).get(key)
                if metric is not None:
                    assert metric["unit"] == unit, (source, number, key)
                    row[column] = metric["value"]
            writer.writerow(row)
            counts[f"{phase}/{status}"] += 1
    return {
        "sha256": digest(target),
        "source_sha256": checksum.hexdigest(),
        "source_bytes": source.stat().st_size,
        "rows_by_phase_and_status": dict(sorted(counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "reports/agentx-disagg-kv-rr-data"
    )
    args = parser.parse_args()
    data = args.data_dir
    for sub in ["hardware", "request-metrics", "source"]:
        (data / sub).mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_paths = [
        "AGENTX_D88_RESULTS.md",
        "sim-results/measured_agentx_d88.json",
        "sim-results/dynosim_n3u_agentx_d64_v5.csv",
        "scripts/gen_agentx_d88_report.py",
        "scripts/agentx_runner_flags.sh",
        "scripts/agentx_runner.sh",
        "scripts/pick_agentx_flag_winner.py",
        "scripts/run_agentx_88_rest11.sh",
        "../sglang/manifests/n3u-mnnvl-88.yaml",
        "../manifests/perf/sgl-d72-agentx.yaml",
        "../sglang/scripts/knee_check.py",
    ]
    files = {}
    for relative in source_paths:
        src = (ROOT / relative).resolve()
        dst = data / "source" / (src.name + ".txt")
        shutil.copyfile(src, dst)
        repo_path = src.relative_to(ROOT.parents[1]).as_posix()
        files[dst.relative_to(data).as_posix()] = {
            "sha256": digest(dst),
            "source": f"https://github.com/alisachen-google/gcp-dynamo-cuj/blob/{commit}/{repo_path}",
        }
    cells = json.loads((ROOT / "sim-results/measured_agentx_d88.json").read_text())

    def fetch(cell):
        run_id = f"n3u-mnnvl-88-agentx-{cell['pol']}-c{cell['clients']}"
        base = f"gs://alisachen-models/perf/{cell['art']}/"
        dirs = subprocess.check_output(["gsutil", "ls", base], text=True).splitlines()
        candidates = [p for p in dirs if f"_trace_c{cell['clients']}_" in p]
        assert len(candidates) == 1, (base, candidates)
        prefix = candidates[0]
        targets = {
            "profile_export_aiperf.json": data / "hardware" / f"{run_id}.json",
            "server_metrics_export.csv": data / "hardware" / f"{run_id}-server.csv",
            "profile_export.jsonl": args.scratch / f"{run_id}.jsonl",
        }
        for name, target in targets.items():
            if not target.exists():
                partial = target.with_suffix(target.suffix + ".partial")
                subprocess.run(
                    ["gsutil", "-q", "cp", prefix + name, str(partial)], check=True
                )
                partial.replace(target)
        summary = json.loads(targets["profile_export_aiperf.json"].read_text())
        print(
            run_id,
            "summary P10 E2E speed",
            summary.get("e2e_output_token_throughput", {}).get("p10"),
            flush=True,
        )
        target = data / "request-metrics" / f"{run_id}.csv.gz"
        projection = project(targets["profile_export.jsonl"], target)
        projection.update(
            file=target.relative_to(data).as_posix(),
            source=prefix + "profile_export.jsonl",
        )
        item = {
            "id": run_id,
            "policy": cell["pol"],
            "clients": cell["clients"],
            "summary": targets["profile_export_aiperf.json"]
            .relative_to(data)
            .as_posix(),
            "server_metrics": targets["server_metrics_export.csv"]
            .relative_to(data)
            .as_posix(),
            "gcs_summary": prefix + "profile_export_aiperf.json",
            "gcs_console": f"https://console.cloud.google.com/storage/browser/alisachen-models/perf/{cell['art']}",
            "source_cell": cell,
        }
        entry_files = {
            projection["file"]: {
                "sha256": projection["sha256"],
                "source": projection["source"],
            }
        }
        for name in ["profile_export_aiperf.json", "server_metrics_export.csv"]:
            path = targets[name]
            entry_files[path.relative_to(data).as_posix()] = {
                "sha256": digest(path),
                "source": prefix + name,
            }
        print(run_id, projection["rows_by_phase_and_status"], flush=True)
        return item, projection, entry_files

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(fetch, cells))
    request_manifest = {
        "schema_version": 1,
        "extraction": "Numeric metrics and trace identity/timing fields only; every exported phase/status retained. No prompt or response text. Source line is 1-based.",
        "runs": {item["id"]: projection for item, projection, _ in outputs},
    }
    path = data / "request-metrics/manifest.json"
    path.write_text(json.dumps(request_manifest, indent=2) + "\n")
    files["request-metrics/manifest.json"] = {
        "sha256": digest(path),
        "source": "Projection provenance and original export hashes",
    }
    for _, _, entry_files in outputs:
        files.update(entry_files)
    manifest = {
        "schema_version": 1,
        "source_commit": commit,
        "gpus": 64,
        "source_report": files["source/AGENTX_D88_RESULTS.md.txt"]["source"],
        "slo_ttft_p95_s": 10,
        "slo_interactivity_p90_tok_s": 20,
        "hardware": [item for item, _, _ in outputs],
        "request_metrics_manifest": "request-metrics/manifest.json",
        "files": files,
    }
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Preserved {len(outputs)} D88 runs in {data}", flush=True)


if __name__ == "__main__":
    main()
