#!/usr/bin/env python3
"""Exercise native disagg flag propagation; this is not a performance result."""

import asyncio
import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

import aiohttp


def main():
    scripts = Path(__file__).resolve().parent
    study = scripts.parents[1]
    config_dir = study / "sim-results/agentx_disagg_c480_native_20260919/configs"
    root = Path("/tmp/agentx-disagg-c480-runs/preflight-v2")
    topology = json.loads((config_dir / "p8d8.json").read_text())
    topology.update(name="flag-preflight-p2d2", gpu_count=16)
    for pool in topology["pools"]:
        pool["workers"] = 2
        pool["engine_config"] = str(config_dir / pool["engine_config"])
    topology_path = root.parent / "preflight-topology.json"
    topology_path.write_text(json.dumps(topology, indent=2) + "\n")
    subprocess.run(
        [
            "/tmp/n3u-path1-venv/bin/python",
            str(scripts / "run_path1_topology.py"),
            "--workload-summary",
            str(config_dir / "workload-source.json"),
            "--topology-config",
            str(topology_path),
            "--concurrency",
            "4",
            "--benchmark-id",
            "c480flagpreflight",
            "--calibration-matrix",
            str(config_dir / "matrix_v10_throughput.json"),
            "--run-dir",
            str(root),
            "--http-port",
            "28480",
            "--metrics-port",
            "28500",
            "--bootstrap-port",
            "28600",
            "--namespace",
            "disagg-c480-flag-preflight",
            "--router",
            "kv",
            "--load-scale",
            "3",
            "--overlap-credit",
            "0.8",
            "--credit-decay",
            "1",
            "--server-python",
            "/tmp/n3u-path1-disagg-venv/bin/python",
            "--routing-debug",
            "--prepare-only",
        ],
        check=True,
    )
    provenance = json.loads((root / "provenance.json").read_text())
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("DYN_ROUTER_", "DYN_KV_OVERLAP_"))
    }
    env.update(provenance["env_overrides"])
    processes, logs = [], []
    url = "http://127.0.0.1:28480"
    try:
        for name in ["frontend", "prefill-0", "prefill-1", "decode-0", "decode-1"]:
            log = (root / (name + ".log")).open("w")
            logs.append(log)
            worker_env = env.copy()
            if name in provenance["worker_metrics_ports"]:
                worker_env["DYN_SYSTEM_PORT"] = str(
                    provenance["worker_metrics_ports"][name]
                )
            processes.append(
                subprocess.Popen(
                    provenance["commands"][name],
                    env=worker_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            time.sleep(0.3)
        for _ in range(120):
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("Preflight server exited")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    health = json.load(response)
                counts = Counter(
                    i.get("component")
                    for i in health.get("instances", [])
                    if i.get("endpoint") == "generate"
                )
                with urllib.request.urlopen(url + "/v1/models", timeout=2) as response:
                    models = json.load(response)
                if counts["prefill"] == counts["backend"] == 2 and models.get("data"):
                    model = models["data"][0]["id"]
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Preflight pools did not become ready")
        resolved = json.loads((root / "frontend-resolved.json").read_text())
        (root / "status.json").write_text(json.dumps({"state": "running_fixture"}))

        async def exercise():
            results = []
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=180)
            ) as session:

                async def send(index, length=8000):
                    # Repeated prefixes plus concurrent long misses create cache and backlog signals.
                    prompt = (
                        "native disaggregation routing flag check " * length
                    ) + str(index)
                    body = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 32,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "ignore_eos": True,
                    }
                    async with session.post(
                        url + "/v1/chat/completions", json=body
                    ) as response:
                        response.raise_for_status()
                        raw = await response.text()
                    events = [
                        json.loads(line[6:])
                        for line in raw.splitlines()
                        if line.startswith("data: ") and line[6:] != "[DONE]"
                    ]
                    usage = [event["usage"] for event in events if event.get("usage")][
                        -1
                    ]
                    assert usage["completion_tokens"] == 32, usage
                    results.append(usage)

                await send(0)
                await asyncio.gather(
                    *(send(i, 8000 + (i % 3) * 3000) for i in range(8))
                )
            return results

        usages = asyncio.run(exercise())
        time.sleep(1)
        text = re.sub(r"\x1b\[[0-9;]*m", "", (root / "frontend.log").read_text())
        formula_rows = [
            line for line in text.splitlines() if "Formula for worker_id=" in line
        ]
        prefill = [
            line
            for line in formula_rows
            if 'worker_type="prefill"' in line or "worker_type=prefill" in line
        ]
        decays = [
            float(m.group(1))
            for line in prefill
            if (m := re.search(r"overlap_credit_decay: ([0-9.]+)", line))
        ]
        if not prefill or not decays or min(decays) >= 1:
            raise RuntimeError(
                "Did not observe credit decay under native prefill backlog"
            )
        if not all("= 3.000 * " in line for line in prefill):
            raise RuntimeError("Load scale did not reach the prefill selector")
        report = {
            "passed": True,
            "scope": "functional fixture, not AgentX performance",
            "router_tuning": provenance["router_tuning"],
            "requests": len(usages),
            "prefill_formula_rows": len(prefill),
            "minimum_observed_credit_multiplier": min(decays),
            "sampled_formula_rows": prefill[:3]
            + [line for line in prefill if "overlap_credit_decay: 1.000" not in line][
                :3
            ],
            "native_environment": json.loads(
                (root / "server-environment.json").read_text()
            ),
            "resolved_frontend": resolved,
        }
        (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        (config_dir.parent / "preflight.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ["sampled_formula_rows", "resolved_frontend"]
                }
            ),
            flush=True,
        )
    finally:
        (root / "status.json").write_text(
            json.dumps(
                {"state": "completed" if (root / "result.json").exists() else "failed"}
            )
        )
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
