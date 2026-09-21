#!/usr/bin/env python3
"""Run the eight selected D88 hardware-matched native simulations, resumably."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def alive(pid, run):
    try:
        return str(run).encode() in Path(f"/proc/{pid}/cmdline").read_bytes().split(
            b"\0"
        )
    except OSError:
        return False


def resources(root):
    fields = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return {
        "available_memory_gib": int(fields["MemAvailable"].split()[0]) / 1024**2,
        "available_disk_gib": shutil.disk_usage(root).free / 1024**3,
        "loadavg": os.getloadavg(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--server-python", required=True)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--minimum-free-memory-gib", type=float, default=70)
    args = parser.parse_args()
    data = args.study.resolve()
    plan = read(data / "plan.json")
    preflight = read(data / "preflight.json")
    assert preflight["passed"]
    assert preflight["queued"]["native_core_sha256"] == plan["native_core_sha256"]
    for name, expected in plan["config_sha256"].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
    root = Path(plan["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "state.json"
    state = read(state_path) if state_path.exists() else {"jobs": {}}
    state.update(
        controller_pid=os.getpid(),
        state="running",
        started_unix=state.get("started_unix", time.time()),
        minimum_free_memory_gib=args.minimum_free_memory_gib,
    )
    owned = {}
    # Start longer warmups first; exact commands and hardware pairing are saved.
    jobs = sorted(plan["points"], key=lambda p: p["concurrency"], reverse=True)
    for index, job in enumerate(jobs):
        name = job["name"]
        row = state["jobs"].setdefault(
            name, {"state": "pending", "run_dir": str(root / name)}
        )
        row["hardware_id"] = job["hardware_id"]
        row["command"] = [
            sys.executable,
            str(Path(__file__).with_name("run_path1_topology.py")),
            "--workload-summary",
            str(data / job["hardware_summary"]),
            "--topology-config",
            str(data / "configs/p8d8.json"),
            "--concurrency",
            str(job["concurrency"]),
            "--benchmark-id",
            job["benchmark_id"],
            "--calibration-matrix",
            str(data / "configs/matrix_v10_throughput.json"),
            "--run-dir",
            row["run_dir"],
            "--http-port",
            str(26000 + index),
            "--metrics-port",
            str(27000 + 32 * index),
            "--bootstrap-port",
            str(28000 + 16 * index),
            "--namespace",
            "matched-20260920-" + name,
            "--router",
            "kv" if job["policy"] == "kv" else "round-robin",
            "--server-python",
            args.server_python,
            "--expected-native-core-sha256",
            plan["native_core_sha256"],
            "--no-raw-payloads",
            "--purpose",
            "matched_d88_validation",
        ]
    write(root / "jobs.json", state["jobs"])
    while True:
        for proc in owned.values():
            proc.poll()
        active = 0
        for name, row in state["jobs"].items():
            run = Path(row["run_dir"])
            if row.get("pid") and alive(row["pid"], run):
                active += 1
                row["state"] = "running"
                if (run / "status.json").exists():
                    row["run_state"] = read(run / "status.json")["state"]
                continue
            if row["state"] != "running":
                continue
            status = (
                read(run / "status.json")
                if (run / "status.json").exists()
                else {"state": "orphaned"}
            )
            row.update(
                state="completed" if status["state"] == "completed" else "failed",
                finished_unix=time.time(),
                result=status,
            )
            print(
                json.dumps({"event": row["state"], "job": name, "result": status}),
                flush=True,
            )
        info = resources(root)
        pending = [row for row in state["jobs"].values() if row["state"] == "pending"]
        if (
            pending
            and active < args.max_parallel
            and info["available_memory_gib"] > args.minimum_free_memory_gib
            and info["available_disk_gib"] > 8 + 2 * (active + 1)
            and info["loadavg"][0] < 32
        ):
            row = pending[0]
            run = Path(row["run_dir"])
            if run.exists():
                raise RuntimeError(f"Refusing to overwrite {run}")
            with (root / (run.name + ".controller.log")).open("x") as log:
                proc = subprocess.Popen(
                    row["command"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
                )
            owned[run.name] = proc
            row.update(state="running", pid=proc.pid, started_unix=time.time())
            active += 1
            print(
                json.dumps(
                    {
                        "event": "started",
                        "job": run.name,
                        "pid": proc.pid,
                        "resources": info,
                    }
                ),
                flush=True,
            )
        state.update(updated_unix=time.time(), resources=info)
        if not pending and not active:
            state["state"] = (
                "completed"
                if all(r["state"] == "completed" for r in state["jobs"].values())
                else "completed_with_failures"
            )
        write(state_path, state)
        if state["state"] != "running":
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
