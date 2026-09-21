#!/usr/bin/env python3
"""Compare a bounded burst against legacy and queued native handoff admission.

This is a correctness fixture, not an AgentX performance point. Both builds keep
two running requests per engine; only the protocol queue capacity changes.
"""

import argparse
import asyncio
import gzip
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import aiohttp


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def port_block(count):
    while True:
        sockets = []
        try:
            first = socket.socket()
            # Dynamo 1.4.2's system port parser accepts signed 16-bit values.
            first.bind(
                ("127.0.0.1", random.SystemRandom().randrange(12000, 32767 - count))
            )
            sockets.append(first)
            port = first.getsockname()[1]
            for offset in range(1, count):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", port + offset))
            return port
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()


def fixture(args, label, server_python, queue_capacity):
    root = args.output / label
    configs = args.output / (label + "-configs")
    configs.mkdir(parents=True, exist_ok=False)
    for role in ["prefill", "decode"]:
        engine = json.loads((args.configs / f"{role}-engine.json").read_text())
        engine["max_num_seqs"] = 2
        if queue_capacity is None:
            engine.pop("handoff_max_sessions", None)
        else:
            engine["handoff_max_sessions"] = queue_capacity
        write(configs / f"{role}-engine.json", engine)
    topology = json.loads((args.configs / "p8d8.json").read_text())
    topology.update(name="p1d1-burst-correctness", gpu_count=8)
    for pool in topology["pools"]:
        pool["workers"] = 1
    write(configs / "topology.json", topology)
    probe = subprocess.check_output(
        [
            server_python,
            "-c",
            (
                "import dynamo._core,hashlib;from pathlib import Path;"
                "print(hashlib.sha256(Path(dynamo._core.__file__).read_bytes()).hexdigest())"
            ),
        ],
        text=True,
    ).strip()
    http, metrics, bootstrap = port_block(1), port_block(2), port_block(1)
    namespace = f"d88-admission-{label}-{os.getpid()}"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("run_path1_topology.py")),
            "--workload-summary",
            str(args.configs / "hardware-rr-c72.json"),
            "--topology-config",
            str(configs / "topology.json"),
            "--concurrency",
            "32",
            "--benchmark-id",
            "d88-queue-preflight",
            "--calibration-matrix",
            str(args.configs / "matrix_v10_throughput.json"),
            "--run-dir",
            str(root),
            "--http-port",
            str(http),
            "--metrics-port",
            str(metrics),
            "--bootstrap-port",
            str(bootstrap),
            "--namespace",
            namespace,
            "--router",
            "round-robin",
            "--server-python",
            server_python,
            "--expected-native-core-sha256",
            probe,
            "--purpose",
            "handoff_queue_correctness",
            "--prepare-only",
        ],
        check=True,
    )
    provenance = json.loads((root / "provenance.json").read_text())
    commands = provenance["commands"]
    env = os.environ | provenance["env_overrides"]
    url = f"http://127.0.0.1:{http}"
    model = json.loads((args.configs / "hardware-rr-c72.json").read_text())[
        "input_config"
    ]["models"]["items"][0]["name"]
    processes, logs = {}, []

    def state(value):
        write(root / "status.json", {"state": value})

    def launch(name):
        log = (root / f"{name}.log").open("w")
        logs.append(log)
        overrides = {}
        if name in provenance["worker_metrics_ports"]:
            overrides["DYN_SYSTEM_PORT"] = str(provenance["worker_metrics_ports"][name])
        processes[name] = subprocess.Popen(
            commands[name],
            env=env | overrides,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    async def burst():
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        ) as session:

            async def request(i):
                outputs = 1 if i % 2 == 0 else 32
                body = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Fixture {i}: " + "handoff pressure " * 4096,
                        }
                    ],
                    "max_tokens": outputs,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "ignore_eos": True,
                }
                started = time.monotonic()
                try:
                    async with session.post(
                        url + "/v1/chat/completions", json=body
                    ) as response:
                        raw = await response.text()
                        events = [
                            json.loads(line[6:])
                            for line in raw.splitlines()
                            if line.startswith("data: ") and line[6:] != "[DONE]"
                        ]
                        usages = [e["usage"] for e in events if e.get("usage")]
                        success = (
                            response.status == 200
                            and bool(usages)
                            and usages[-1]["completion_tokens"] == outputs
                        )
                        return {
                            "request": i,
                            "success": success,
                            "requested_tokens": outputs,
                            "usage": usages[-1] if usages else None,
                            "elapsed_s": time.monotonic() - started,
                            "status": response.status,
                            "error": None if success else raw[:1000],
                        }
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    return {"request": i, "success": False, "error": str(exc)}

            return await asyncio.gather(*(request(i) for i in range(32)))

    try:
        state("starting_server")
        for name in ["frontend", "prefill-0", "decode-0"]:
            launch(name)
        for _ in range(180):
            if any(p.poll() is not None for p in processes.values()):
                raise RuntimeError(f"{label}: server exited; see {root}")
            try:
                health = json.loads(
                    urllib.request.urlopen(url + "/health", timeout=2).read()
                )
                counts = Counter(
                    r.get("component")
                    for r in health.get("instances", [])
                    if r.get("namespace") == namespace
                    and r.get("endpoint") == "generate"
                )
                models = json.loads(
                    urllib.request.urlopen(url + "/v1/models", timeout=2).read()
                )
                if counts["prefill"] == counts["backend"] == 1 and models.get("data"):
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Fixture pool readiness failed")
        state("running")
        launch("observe-prefill")
        launch("observe-decode")
        time.sleep(3)
        results = asyncio.run(burst())
        state("completed")
        for role in ["prefill", "decode"]:
            processes[f"observe-{role}"].wait(timeout=15)
        forward = {}
        for role in ["prefill", "decode"]:
            with gzip.open(root / role / "forward-passes.jsonl.gz", "rt") as stream:
                events = [json.loads(line)["fpm"] for line in stream]
            batches = [e["scheduled_requests"] for e in events]
            forward[role] = {
                "computed_prefill_tokens": sum(
                    b["sum_prefill_tokens"] for b in batches
                ),
                "scheduled_decode_tokens": sum(
                    b["num_decode_requests"] for b in batches
                ),
                "max_decode_batch": max(
                    (b["num_decode_requests"] for b in batches), default=0
                ),
            }
        result = {
            "label": label,
            "native_core_sha256": probe,
            "queue_capacity": queue_capacity,
            "running_limit": 2,
            "requests": results,
            "successes": sum(r["success"] for r in results),
            "forward_passes": forward,
        }
        assert forward["decode"]["computed_prefill_tokens"] == 0
        assert forward["decode"]["max_decode_batch"] <= 2
        write(root / "result.json", result)
        print(
            json.dumps(
                {
                    "fixture": label,
                    "successes": result["successes"],
                    "requests": len(results),
                    "forward": forward,
                }
            ),
            flush=True,
        )
        return result
    finally:
        state("completed" if (root / "result.json").exists() else "failed")
        for process in reversed(list(processes.values())):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-python", required=True)
    parser.add_argument("--queued-python", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = fixture(args, "legacy", args.baseline_python, None)
    queued = fixture(args, "queued", args.queued_python, 128)
    assert baseline["successes"] < 32, (
        "Fixture did not reproduce legacy admission failure"
    )
    assert queued["successes"] == 32, queued
    assert queued["forward_passes"]["decode"]["scheduled_decode_tokens"] == 16 * 31
    write(
        args.output / "preflight.json",
        {"passed": True, "baseline": baseline, "queued": queued},
    )


if __name__ == "__main__":
    main()
