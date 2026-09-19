#!/usr/bin/env python3
"""Resume, monitor, collect and archive the explicitly authorized native sweep."""

import fcntl
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

from analyze import (
    DATA,
    ROOT,
    assessed_results,
    choose_finalist,
    collect,
    read,
    render,
    sha,
    write,
)


def alive(pid, run):
    try:
        parts = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        return str(run).encode() in parts
    except OSError:
        return False


def resources():
    fields = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    memory = int(fields["MemAvailable"].split()[0]) / 1024**2
    other = 0
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            parts = path.read_bytes().split(b"\0")
        except OSError:
            continue
        if any(
            Path(os.fsdecode(part)).name
            in ["run_path1_topology.py", "run_path1_agentx.py"]
            for part in parts
            if part
        ) and not any(str(ROOT).encode() in part for part in parts):
            other += 1
    return {
        "available_memory_gib": memory,
        "available_disk_gib": shutil.disk_usage("/tmp").free / 1024**3,
        "other_controllers": other,
        "loadavg": os.getloadavg(),
    }


def final_repeat(jobs, candidate):
    source = next(j for j in jobs if j["name"] == candidate["name"])
    job = dict(source)
    job["name"] = "finalist-repeat-" + source["name"]
    job["repeat"] = True
    job["repeat_of"] = source["name"]
    job["index"] = len(jobs)
    job["run_dir"] = str(ROOT / job["name"])
    job["command"] = list(source["command"])
    for flag, value in {
        "--run-dir": job["run_dir"],
        "--namespace": "disagg-c480-" + job["name"],
        "--http-port": str(29000 + job["index"]),
        "--metrics-port": str(30000 + 32 * job["index"]),
        "--bootstrap-port": str(32000 + 16 * job["index"]),
    }.items():
        job["command"][job["command"].index(flag) + 1] = value
    return job


def refresh(state, state_path):
    write(state_path, state)
    assessment = render()
    publication_key = [
        assessment["completed"],
        state["state"],
        sorted(name for name, row in state["jobs"].items() if row["state"] == "failed"),
    ]
    if state.get("published_key") != publication_key:
        try:
            from publish import publish

            publication = publish(
                f"Update native disagg C480 flag sweep: {assessment['completed']} completed"
            )
            state["published_key"] = publication_key
            state["publication"] = publication
            state.pop("publication_error", None)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            state["publication_error"] = str(error)
            print(traceback.format_exc(), flush=True)
        write(state_path, state)
    return state.get("published_key") == publication_key


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(DATA / "plan.json")
    preflight = read(DATA / "preflight.json")
    assert (
        preflight["passed"]
        and preflight["native_environment"]["core_sha256"] == plan["native_core_sha256"]
    )
    for relative, expected in plan["config_sha256"].items():
        assert sha(DATA / relative) == expected, relative
    state_path = ROOT / "queue-state.json"
    state = read(state_path) if state_path.exists() else {"jobs": {}}
    state.update(controller_pid=os.getpid(), state="running")
    owned = {}
    while True:
        for process in owned.values():
            process.poll()
        jobs = read(DATA / "jobs.json")["jobs"]
        active = 0
        for job in jobs:
            row = state["jobs"].setdefault(
                job["name"], {"state": "pending", "run_dir": job["run_dir"]}
            )
            run = Path(row["run_dir"])
            if row.get("controller_pid") and alive(row["controller_pid"], run):
                active += 1
                row["state"] = "running"
                status_path = run / "status.json"
                if status_path.exists():
                    row["run_state"] = read(status_path)["state"]
                continue
            if row["state"] == "pending":
                continue
            status_path = run / "status.json"
            status = (
                read(status_path) if status_path.exists() else {"state": "orphaned"}
            )
            if status["state"] == "completed":
                try:
                    if not row.get("collected"):
                        result = collect(job, run)
                        row.update(
                            state="completed",
                            collected=True,
                            result=str(DATA / "runs" / job["name"] / "result.json"),
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "collected",
                                    "name": job["name"],
                                    "total_tok_s_per_gpu": result[
                                        "total_tok_s_per_gpu"
                                    ],
                                    "p95_ttft_s": result["p95_ttft_s"],
                                    "interactivity": result[
                                        "e2e_normalized_interactivity_p90"
                                    ],
                                    "errors": result["errors"],
                                }
                            ),
                            flush=True,
                        )
                    if (
                        not row.get("archived")
                        and time.time() - status.get("updated_unix", 0) >= 30
                    ):
                        from archive_path1_topology import archive

                        archive(run)
                        row["archived"] = True
                except Exception as error:
                    row["collection_errors"] = row.get("collection_errors", 0) + 1
                    row["last_error"] = str(error)
                    print(traceback.format_exc(), flush=True)
                    if row["collection_errors"] >= 3:
                        state.update(
                            state="needs_attention",
                            reason="Repeated collection/archive failure",
                        )
                        refresh(state, state_path)
                        raise
            else:
                row.update(
                    state="failed",
                    failure=status,
                    last_error=status.get(
                        "error", "Controller exited without a completed result"
                    ),
                )
        pending = [j for j in jobs if state["jobs"][j["name"]]["state"] == "pending"]
        if not pending and active == 0 and not state.get("finalist_scheduled"):
            rows = assessed_results()
            candidate = choose_finalist(rows)
            if candidate:
                job = final_repeat(jobs, candidate)
                jobs.append(job)
                write(DATA / "jobs.json", {"jobs": jobs})
                state["jobs"][job["name"]] = {
                    "state": "pending",
                    "run_dir": job["run_dir"],
                }
                pending = [job]
                state["finalist_scheduled"] = job["name"]
            else:
                state["finalist_scheduled"] = "none: no candidate passed quality checks"
        info = resources()
        reason = None
        if active + info["other_controllers"] >= plan["max_parallel"]:
            reason = "parallel_limit"
        elif info["available_memory_gib"] < 90:
            reason = "memory_headroom"
        elif info["available_disk_gib"] < 16 * (active + 1) + 12:
            reason = "artifact_storage_headroom"
        elif not pending:
            reason = "no_pending_jobs"
        if reason is None:
            job = pending[0]
            run = Path(job["run_dir"])
            if run.exists():
                raise RuntimeError(f"Refusing to overwrite an existing run: {run}")
            log = (ROOT / (job["name"] + ".controller.log")).open("x")
            process = subprocess.Popen(
                job["command"],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=ROOT,
                env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
            )
            log.close()
            owned[job["name"]] = process
            state["jobs"][job["name"]].update(
                state="running",
                controller_pid=process.pid,
                started_unix=time.time(),
                attempt=1,
            )
            active += 1
            print(
                json.dumps(
                    {"event": "launched", "name": job["name"], "pid": process.pid}
                ),
                flush=True,
            )
        state.update(
            updated_unix=time.time(), active=active, resources=info, wait_reason=reason
        )
        if (
            not pending
            and not active
            and all(
                row.get("archived") or row["state"] == "failed"
                for row in state["jobs"].values()
            )
        ):
            failures = [
                name for name, row in state["jobs"].items() if row["state"] == "failed"
            ]
            state.update(
                state="completed_with_failures" if failures else "completed",
                failed_runs=failures,
            )
            if refresh(state, state_path):
                print(
                    json.dumps({"event": "sweep_finished", "failures": failures}),
                    flush=True,
                )
                break
        else:
            refresh(state, state_path)
        time.sleep(30)


if __name__ == "__main__":
    main()
