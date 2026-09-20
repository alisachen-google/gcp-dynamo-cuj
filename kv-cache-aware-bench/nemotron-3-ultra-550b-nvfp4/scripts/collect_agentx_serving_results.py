#!/usr/bin/env python3
"""Collect new AgentX hardware results while reusing the audited report inputs.

This only reads GCS and local reports. It never starts or changes a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from import_agentx_d88 import project

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
NAME = "agentx-serving-perf-data"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scratch", type=Path, default=Path("/tmp/agentx-concrete-import")
    )
    parser.add_argument("--inventory", type=Path)
    parser.add_argument(
        "--previous-manifest",
        type=Path,
        help="Prior published inventory, used to identify additions in this update",
    )
    args = parser.parse_args()
    folder = REPORTS / NAME
    previous_path = args.previous_manifest or folder / "manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    previous_artifacts = (
        {r["artifact"] for r in previous["runs"]} if previous else set()
    )
    for part in ["source", "hardware", "request-metrics"]:
        (folder / part).mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    stamp = datetime.now(timezone.utc).isoformat()
    inventory = (
        args.inventory.read_text()
        if args.inventory
        else subprocess.check_output(
            ["gsutil", "ls", "gs://alisachen-models/perf/"], text=True
        )
    )
    selected = [
        s
        for s in inventory.splitlines()
        if "agentx" in s and ("n3u-agg" in s or "n3u-mnnvl-88" in s)
    ]
    (folder / "source/gcs-inventory.txt").write_text("\n".join(selected) + "\n")
    source_files = [
        "AGENTX_AGG_RESULTS.md",
        "AGENTX_D88_RESULTS.md",
        "sim-results/measured_agentx_d88.json",
        "scripts/agentx_runner_flags.sh",
        "scripts/run_agentx_agg_decay_np2_v2.sh",
        "scripts/run_agentx_88_followup.sh",
        "../sglang/manifests/n3u-mnnvl-88.yaml",
        "../sglang/manifests/n3u-agg-newstack-np2.yaml",
        "../sglang/manifests/n3u-agg-newstack2-np2.yaml",
        "../manifests/perf/sgl-d72-agentx.yaml",
        "../sglang/scripts/knee_check.py",
        "reports/agentx-disagg-c480-kv-flags.md",
    ]
    files = {}
    for name in source_files:
        path = (ROOT / name).resolve()
        target = folder / "source" / (path.name + ".txt")
        shutil.copyfile(path, target)
        files[target.relative_to(folder).as_posix()] = {
            "sha256": sha(target),
            "source": f"https://github.com/alisachen-google/gcp-dynamo-cuj/blob/{commit}/{path.relative_to(ROOT.parents[1]).as_posix()}",
        }
    inventory_path = folder / "source/gcs-inventory.txt"
    files["source/gcs-inventory.txt"] = {
        "sha256": sha(inventory_path),
        "source": "GCS listing at " + stamp,
    }
    cells = json.loads((ROOT / "sim-results/measured_agentx_d88.json").read_text())
    by_art = {c["art"]: c for c in cells}
    entries = {}
    seen = set()
    for architecture, input_root, campaign in [
        ("agg", "agentx-agg-kv-rr-data", "agg-20260916"),
        ("disagg", "agentx-disagg-kv-rr-data", "d88-20260917-19"),
    ]:
        manifest = json.loads((REPORTS / input_root / "manifest.json").read_text())
        for item in manifest["hardware"]:
            item = dict(item)
            art = item["gcs_console"].rstrip("/").split("/")[-1]
            item.update(
                architecture=architecture,
                input_root=input_root,
                campaign=campaign,
                artifact=art,
                is_new=False,
            )
            if architecture == "disagg":
                item["source_cell"] = by_art[art]
            entries[item["id"]] = item
            seen.add(art)
    work = []
    unavailable = []
    for uri in selected:
        art = uri.rstrip("/").split("/")[-1]
        if art in seen:
            continue
        if int(art.split("_")[0]) < 1789500000:
            unavailable.append(
                {
                    "artifact": art,
                    "reason": "Historical failed smoke; predates the completed AgentX ladder",
                }
            )
            continue
        match = re.search(r"-agentx-(\w+)-c(\d+)$", art)
        assert match, art
        architecture = "agg" if "n3u-agg" in art else "disagg"
        item = {
            "id": art,
            "artifact": art,
            "policy": match[1],
            "clients": int(match[2]),
            "architecture": architecture,
            "campaign": "agg-np2-20260919"
            if architecture == "agg"
            else "d88-20260917-19",
            "is_new": True,
            "gcs_console": "https://console.cloud.google.com/storage/browser/alisachen-models/perf/"
            + art,
        }
        if architecture == "disagg" and art in by_art:
            item["source_cell"] = by_art[art]
        work.append(item)
    # Extend these two old projections with timestamps to audit warmup drift.
    for item in entries.values():
        if (
            item["architecture"] == "agg"
            and item["clients"] == 192
            and item["policy"] in {"kv", "kvs3c08"}
        ):
            work.append(dict(item))

    def fetch(item):
        art = item["artifact"]
        prefix = "gs://alisachen-models/perf/" + art + "/"
        paths = subprocess.check_output(
            ["gsutil", "ls", prefix], text=True
        ).splitlines()
        candidates = [s for s in paths if f"_trace_c{item['clients']}_" in s]
        if len(candidates) != 1:
            return None, {
                "artifact": art,
                "reason": "No unique AgentX profiling artifact directory",
            }
        prefix = candidates[0]
        summary_path = folder / "hardware" / f"{item['id']}.json"
        if not summary_path.exists():
            cp = subprocess.run(
                [
                    "gsutil",
                    "-q",
                    "cp",
                    prefix + "profile_export_aiperf.json",
                    str(summary_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if cp.returncode:
                return None, {
                    "artifact": art,
                    "reason": "No completed AIPerf summary available at collection time",
                    "source": prefix,
                }
        summary = json.loads(summary_path.read_text())
        print(
            art,
            "summary",
            {
                "TTFT_p95_s": summary["time_to_first_token"]["p95"] / 1000,
                "E2E_rate_P10": summary["e2e_output_token_throughput"]["p10"],
                "errors": summary.get("error_request_count", {}).get("avg", 0),
            },
            flush=True,
        )
        raw = args.scratch / f"{art}.jsonl"
        if not raw.exists():
            partial = raw.with_suffix(".jsonl.partial")
            subprocess.run(
                ["gsutil", "-q", "cp", prefix + "profile_export.jsonl", str(partial)],
                check=True,
            )
            partial.replace(raw)
        target = folder / "request-metrics" / f"{item['id']}.csv.gz"
        projection = project(raw, target)
        projection.update(
            file=target.relative_to(folder).as_posix(),
            source=prefix + "profile_export.jsonl",
        )
        item.update(
            input_root=NAME,
            summary=summary_path.relative_to(folder).as_posix(),
            gcs_summary=prefix + "profile_export_aiperf.json",
        )
        print(art, projection["rows_by_phase_and_status"], flush=True)
        return (item, projection), None

    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(fetch, work))
    request_entries = {}
    for result, missing in fetched:
        if missing:
            unavailable.append(missing)
            continue
        item, projection = result
        entries[item["id"]] = item
        request_entries[item["id"]] = projection
        for name, source in [
            (item["summary"], item["gcs_summary"]),
            (projection["file"], projection["source"]),
        ]:
            files[name] = {"sha256": sha(folder / name), "source": source}
    for item in entries.values():
        item["new_since_previous_report"] = item["artifact"] not in previous_artifacts
    request_manifest = folder / "request-metrics/manifest.json"
    request_manifest.write_text(
        json.dumps({"schema_version": 1, "runs": request_entries}, indent=2) + "\n"
    )
    files["request-metrics/manifest.json"] = {
        "sha256": sha(request_manifest),
        "source": "Numeric projections with original full-export hashes",
    }
    native_root = Path("/tmp/agentx-disagg-c480-runs")
    native = None
    if (native_root / "queue-state.json").exists():
        status = json.loads((native_root / "queue-state.json").read_text())
        native = {
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "state": status["state"],
            "runs": [],
        }
        for key, row in status["jobs"].items():
            evidence = []
            for log in sorted((native_root / key).glob("prefill-*.log")):
                for number, line in enumerate(
                    log.read_text(errors="replace").splitlines(), 1
                ):
                    if "mocker handoff session limit reached" in line:
                        evidence.append(
                            {
                                "file": str(log),
                                "sha256": sha(log),
                                "line": number,
                                "message": "mocker handoff session limit reached for DP rank 0",
                            }
                        )
                        break
            native["runs"].append(
                {
                    "run": key,
                    "state": row["state"],
                    "failure": row.get("failure"),
                    "handoff_limit_evidence": evidence,
                }
            )
        path = folder / "source/native-c480-status.json"
        path.write_text(json.dumps(native, indent=2) + "\n")
        files[path.relative_to(folder).as_posix()] = {
            "sha256": sha(path),
            "source": "Local queue state and immutable hashes of failure logs, observed "
            + native["observed_utc"],
        }
    manifest = {
        "schema_version": 1,
        "collected_utc": stamp,
        "source_commit": commit,
        "previous_snapshot": {
            "collected_utc": previous["collected_utc"],
            "source_commit": previous["source_commit"],
            "manifest_sha256": sha(previous_path),
            "hardware_runs": len(previous["runs"]),
        }
        if previous
        else None,
        "files": files,
        "runs": list(entries.values()),
        "unavailable": unavailable,
        "request_metrics_manifest": "request-metrics/manifest.json",
        "native_c480_status": "source/native-c480-status.json" if native else None,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "completed_hardware": len(entries),
                "new_hardware": sum(x["is_new"] for x in entries.values()),
                "unavailable": unavailable,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
