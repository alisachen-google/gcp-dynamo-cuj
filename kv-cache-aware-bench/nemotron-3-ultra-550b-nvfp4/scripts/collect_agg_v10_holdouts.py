#!/usr/bin/env python3
"""Audit and preserve the eight frozen-V10 agg holdouts for the performance report."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from import_agentx_request_metrics import extract


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_sha(path):
    # The original calibration matrix hashes sorted JSON, not file formatting.
    return hashlib.sha256(json.dumps(read(path), sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.data_dir / "manifest.json")
    original = read(args.data_dir / "native-v10/matrix.json")
    core = {p["native_core_sha256"] for p in original["points"]}
    engine = {p["shared_engine_sha256"] for p in original["points"]}
    assert len(core) == len(engine) == 1
    hardware = {p["id"]: p for p in manifest["hardware"]}
    jobs = read(args.runs / "jobs.json")
    expected = {
        (p["policy"], p["clients"])
        for p in hardware.values()
        if p["policy"] not in {"kv", "rr"} or p["clients"] in {48, 96}
    }
    assert {(j["policy"], j["clients"]) for j in jobs} == expected
    # Refuse to publish partial output or turn replay failures into data points.
    for job in jobs:
        status = read(args.runs / job["tag"] / "status.json")
        assert status["state"] == "completed" and status["valid_agentx_submission"], (
            job["tag"]
        )
    request_manifest_path = args.data_dir / "request-metrics/manifest.json"
    requests = read(request_manifest_path)
    points = []
    for job in jobs:
        run = args.runs / job["tag"]
        status = read(run / "status.json")
        assert status["warmup_valid"] and status["warmup_inputs_match"]
        server = read(run / "server-environment.json")
        assert server["core_sha256"] in core
        assert engine_sha(run / "engine.json") in engine, job["tag"]
        assert server["versions"] == {
            "ai-dynamo": "1.4.2",
            "ai-dynamo-runtime": "1.4.2",
            "aiconfigurator-core": "0.11.0",
        }
        entry = hardware[job["hardware_id"]]
        source = read(args.data_dir / entry["summary"])
        assert read(run / "hardware-summary.json") == source
        summary_path = Path(status["summary"])
        summary = read(summary_path)
        assert summary["benchmark_id"] == source["benchmark_id"]
        assert summary["metadata"]["submission_valid"]
        provenance = read(run / "provenance.json")
        assert provenance["hardware_sha256"] == sha(args.data_dir / entry["summary"])
        config = read(run / "client-config.json")["benchmark"]
        expected_config = json.loads(json.dumps(source["input_config"]))
        expected_config["endpoint"]["urls"] = config["endpoint"]["urls"]
        expected_config["artifacts"]["dir"] = config["artifacts"]["dir"]
        expected_config["gpu_telemetry"]["enabled"] = False
        expected_config["runtime"]["ui"] = "simple"
        assert config == expected_config, job["tag"]
        assert provenance["gpu_count"] == 24 and provenance["speedup"] == 1
        command = provenance["commands"]["frontend"]
        start = command.index("--router-mode")
        actual_flags = command[start : command.index("--request-plane")]
        assert actual_flags == shlex.split(job["router_flags"])
        for name, checksum in provenance["launcher_sha256"].items():
            assert sha(run / name) == checksum
        dest = args.data_dir / "native-v10" / job["tag"]
        dest.mkdir(exist_ok=True)
        for name in [
            "status.json",
            "comparison.json",
            "engine.json",
            "server-environment.json",
            "provenance.json",
            "client-config.json",
        ]:
            shutil.copyfile(run / name, dest / name)
        shutil.copyfile(summary_path, dest / "simulation_summary.json")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "native_v10_agg/check_path1_inputs.py"),
                "--hardware-records",
                provenance["hardware_records"],
                "--run-dir",
                str(run),
                "--output",
                str(dest / "input_identity.json"),
            ],
            check=True,
        )
        assert read(dest / "input_identity.json")["warmup"]["all_checks_pass"]
        for path in dest.iterdir():
            name = str(path.relative_to(args.data_dir))
            if path.name == "simulation_summary.json":
                source_location = str(summary_path)
            elif path.name == "input_identity.json":
                source_location = (
                    f"Recomputed with check_path1_inputs.py from {run} and "
                    + provenance["hardware_records"]
                )
            else:
                source_location = str(run / path.name)
            manifest["files"][name] = {
                "sha256": sha(path),
                "source": source_location,
            }
        request_entry = extract(
            {
                "id": job["tag"],
                "records": str(summary_path.parent / "profile_export.jsonl"),
                "source": str(summary_path.parent / "profile_export.jsonl"),
            },
            args.data_dir / "request-metrics",
        )
        requests["runs"][job["tag"]] = request_entry
        manifest["files"][request_entry["file"]] = {
            "sha256": request_entry["sha256"],
            "source": request_entry["source"],
        }
        points.append(
            {
                "run": job["tag"],
                "policy": job["policy"],
                "router": "rr" if job["policy"] == "rr" else "kv",
                "clients": job["clients"],
                "hardware_id": job["hardware_id"],
                "state": "completed",
                "replay_valid": True,
                "hardware_settings_match": True,
                "hardware_serving_shape_matches": True,
                "native_core_sha256": server["core_sha256"],
                "shared_engine_sha256": engine_sha(run / "engine.json"),
                "metrics": status["metrics"],
                "role": "holdout",
                "router_flags": job["router_flags"],
            }
        )
    # Earlier custom-model measurements are not part of the Native V10 comparison.
    for name in list(requests["runs"]):
        if name.startswith("agg6-"):
            old = requests["runs"].pop(name)
            manifest["files"].pop(old["file"], None)
    write(request_manifest_path, requests)
    manifest["files"][manifest["request_metrics_manifest"]]["sha256"] = sha(
        request_manifest_path
    )
    holdout_path = args.data_dir / "native-v10/holdouts.json"
    write(
        holdout_path,
        {
            "all_runs_complete": True,
            "replay_valid": True,
            "calibration_frozen": True,
            "original_calibration_matrix": "matrix.json",
            "points": points,
        },
    )
    manifest["files"]["native-v10/holdouts.json"] = {
        "sha256": sha(holdout_path),
        "source": str(args.runs / "jobs.json"),
    }
    manifest["native_holdout_date"] = "2026-09-18"
    write(args.data_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {"preserved_native_holdouts": len(points), "all_replay_checks_passed": True}
        )
    )


if __name__ == "__main__":
    main()
