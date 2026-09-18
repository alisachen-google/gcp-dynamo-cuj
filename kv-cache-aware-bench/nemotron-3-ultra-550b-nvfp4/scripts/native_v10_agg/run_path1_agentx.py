#!/usr/bin/env python3
"""Run a saved hardware AgentX workload against live NVIDIA Mocker + AIC.

External local etcd/NATS must already be running. This controller owns only its
frontend, worker and AIPerf process groups and saves their exact configuration.
It is orchestration, not an inference or replay simulator.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def warmup_inputs(path):
    """Compare source/branch identities and prompt sizes, not completion order."""
    inputs = Counter()
    with path.open() as records:
        for line in records:
            record = json.loads(line)
            metadata = record.get("metadata", {})
            if metadata.get("benchmark_phase") != "warmup":
                continue
            metrics = record.get("metrics", {})
            inputs[(metadata.get("source_trace_id"), metadata.get("conversation_id"),
                    metadata.get("source_kind"), metadata.get("source_outer_idx"),
                    metadata.get("turn_index"),
                    metrics.get("usage_prompt_tokens", {}).get("value"))] += 1
    return inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hardware-summary", type=Path, required=True)
    p.add_argument("--hardware-records", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--http-port", type=int, required=True)
    p.add_argument("--metrics-port", type=int, required=True)
    p.add_argument("--namespace", required=True)
    p.add_argument("--router", choices=["round-robin", "kv"], required=True)
    p.add_argument("--engine-config", type=Path, required=True)
    p.add_argument("--router-args-file", type=Path, help="JSON list of the exact hardware router arguments")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--gpus", type=int, default=24)
    p.add_argument("--server-python", default="/tmp/n3u-path1-venv/bin/python")
    p.add_argument("--client-python", default="/tmp/n3u-faithful-venv/bin/python")
    p.add_argument("--tokenizer", default="/tmp/n3u-faithful-data/tokenizer")
    p.add_argument("--hf-home", default="/tmp/n3u-path1-hf")
    p.add_argument("--dataset-cache", default="/tmp/n3u-path1-data/hf-datasets")
    p.add_argument("--mmap-cache", default="/tmp/n3u-path1-data/mmap-cache")
    p.add_argument("--nats", default="nats://127.0.0.1:18222")
    p.add_argument("--etcd", default="http://127.0.0.1:18279")
    a = p.parse_args()
    if a.run_dir.exists():
        raise ValueError("Use a new run directory; previous attempts are retained")
    a.run_dir.mkdir(parents=True)
    source = json.loads(a.hardware_summary.read_text())
    cfg = copy.deepcopy(source["input_config"])
    assert cfg["scenario"] == "inferencex-agentx-mvp"
    assert cfg["phases"][0]["duration"] == 3600.0
    assert cfg["phases"][0]["timing_mode"] == "agentic_replay"
    engine = json.loads(a.engine_config.read_text())
    assert engine["engine_type"] == "sglang"
    assert engine["speedup_ratio"] == engine["decode_speedup_ratio"] == 1.0
    assert engine["aic_backend"] == "sglang"
    assert engine["worker_type"] == "aggregated", "Disaggregation needs two pools"
    assert a.gpus == a.workers * engine["aic_tp_size"]
    model = cfg["models"]["items"][0]["name"]
    url = f"http://127.0.0.1:{a.http_port}"
    cfg["endpoint"]["urls"] = [url]
    cfg["artifacts"]["dir"] = str(a.run_dir / "artifacts")
    cfg["gpu_telemetry"]["enabled"] = False
    cfg["runtime"]["ui"] = "simple"
    client_config = {
        "benchmark": cfg,
        "random_seed": source["run_info"]["random_seed"],
    }
    write_json(a.run_dir / "client-config.json", client_config)
    write_json(a.run_dir / "engine.json", engine)
    write_json(a.run_dir / "hardware-summary.json", source)
    env_delta = dict(
        HF_HOME=a.hf_home,
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_DATASETS_CACHE=a.dataset_cache,
        AIPERF_DATASET_MMAP_CACHE_DIR=a.mmap_cache,
        AIPERF_DATASET_MMAP_BASE_PATH=str(a.run_dir / "mmap"),
        PATH1_BENCHMARK_ID=source["benchmark_id"],
        PATH1_CONTEXT_LENGTH="262144",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        NATS_SERVER=a.nats,
        ETCD_ENDPOINTS=a.etcd,
        DYN_NAMESPACE=a.namespace,
        # Saved hardware raw requests have no Dynamo session/parent headers.
        AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID="0",
        RUST_LOG="info",
    )
    env = os.environ.copy()
    env.update(env_delta)
    scripts = Path(__file__).resolve().parent
    common = ["--request-plane", "nats", "--discovery-backend", "etcd",
              "--event-plane", "zmq"]
    router_args = ["--router-mode", a.router]
    if a.router == "kv":
        router_args += ["--router-temperature", "0.0", "--router-queue-policy", "fcfs"]
    if a.router_args_file is not None:
        router_args = json.loads(a.router_args_file.read_text())
        assert router_args[:2] == ["--router-mode", a.router]
    front = [a.server_python, "-m", "dynamo.frontend", "--namespace", a.namespace,
             "--http-host", "127.0.0.1", "--http-port", str(a.http_port),
             "--router-min-initial-workers", str(a.workers)] + router_args + common
    worker = [a.server_python, str(scripts / "path1_mocker.py"), "--model-path",
              a.tokenizer, "--model-name", model, "--endpoint",
              f"dyn://{a.namespace}.backend.generate", "--num-workers", "1",
              "--extra-engine-args", str(a.run_dir / "engine.json")] + common
    client = [a.client_python, str(scripts / "path1_agentx_client.py"), "profile",
              "--config", str(a.run_dir / "client-config.json")]
    workers = {f"worker-{i}": worker for i in range(a.workers)}
    commands = dict(frontend=front, **workers, client=client)
    commands["observer"] = [a.server_python, str(scripts / "observe_path1.py"), str(a.run_dir)]
    server_names = ["frontend", *workers]
    provenance = dict(method="path1_live_aiperf_dynamo_mocker_aic", clock="wall",
                      speedup=1, gpu_count=a.gpus, hardware_source=str(a.hardware_summary),
                      hardware_sha256=hashlib.sha256(a.hardware_summary.read_bytes()).hexdigest(),
                      hardware_records=str(a.hardware_records),
                      env_overrides=env_delta, commands=commands,
                      worker_metrics_port=a.metrics_port,
                      launcher_sha256={s.name: hashlib.sha256(s.read_bytes()).hexdigest()
                                       for s in (Path(__file__), scripts / "path1_mocker.py",
                                                 scripts / "path1_agentx_client.py",
                                                 scripts / "observe_path1.py")},
                      targets=["total_token_throughput_per_gpu", "time_to_first_token_p95"],
                      relative_error_limit=0.20, started_unix=time.time())
    write_json(a.run_dir / "provenance.json", provenance)
    for name in provenance["launcher_sha256"]:
        (a.run_dir / name).write_bytes((scripts / name).read_bytes())
    environment_probe = (
        "import json,hashlib,importlib.metadata as m,dynamo._core;from pathlib import Path;"
        "p=Path(dynamo._core.__file__);"
        "print(json.dumps(dict(versions={n:m.version(n) for n in "
        "['ai-dynamo','ai-dynamo-runtime','aiconfigurator-core']},"
        "core_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))"
    )
    write_json(a.run_dir / "server-environment.json", json.loads(subprocess.check_output(
        [a.server_python, "-c", environment_probe], env=env, text=True)))
    owned = {}
    logs = []

    def launch(name):
        log = (a.run_dir / f"{name}.log").open("w")
        logs.append(log)
        proc_env = env.copy()
        if name.startswith("worker-"):
            proc_env["DYN_SYSTEM_PORT"] = str(a.metrics_port + int(name.split("-")[1]))
        process = subprocess.Popen(commands[name], stdout=log, stderr=subprocess.STDOUT,
                                   env=proc_env, cwd=a.run_dir, start_new_session=True)
        owned[name] = process
        write_json(a.run_dir / "pids.json", {k: v.pid for k, v in owned.items()})
        return process

    def update(state, **extra):
        write_json(a.run_dir / "status.json", dict(state=state, updated_unix=time.time(), **extra))

    def check_server_tasks():
        # A Rust background task can panic while its containing Python process
        # and advertised health endpoint remain alive. Such a run is invalid.
        for name in server_names:
            path = a.run_dir / f"{name}.log"
            if path.exists() and b"panicked at" in path.read_bytes():
                raise RuntimeError(f"Native server task panicked; see {path}")

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        update("starting_server")
        launch("frontend")
        for worker_name in workers:
            launch(worker_name)
            # Reduce concurrent dynamic-ZMQ port allocation during startup.
            time.sleep(0.5)
        for _ in range(180):
            check_server_tasks()
            if any(v.poll() is not None for v in owned.values()):
                raise RuntimeError("Server process exited; see logs")
            try:
                models = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=2).read())
                health = json.loads(urllib.request.urlopen(url + "/health", timeout=2).read())
                ready = [m for m in models.get("data", []) if m["id"] == model]
                instances = [i for i in health.get("instances", [])
                             if i.get("namespace") == a.namespace
                             and i.get("component") == "backend"
                             and i.get("endpoint") == "generate"]
                initialized = all(b"Metrics background tasks started" in
                                  (a.run_dir / f"{name}.log").read_bytes()
                                  for name in workers)
                if (ready and ready[0].get("context_window") == 262144
                        and len(instances) == a.workers and initialized):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Server did not advertise model, context and all workers")
        update("running_aiperf")
        launch("observer")
        proc = launch("client")
        started = time.monotonic()
        with (a.run_dir / "host-observations.jsonl").open("w") as host:
            while proc.poll() is None:
                check_server_tasks()
                host.write(json.dumps(dict(time=time.time(), loadavg=os.getloadavg(),
                                           meminfo=Path('/proc/meminfo').read_text(),
                                           proc_stat=Path('/proc/stat').read_text())) + "\n")
                host.flush()
                if any(owned[n].poll() is not None for n in server_names):
                    raise RuntimeError("Server exited during AIPerf run")
                if time.monotonic() - started > 10800:
                    raise RuntimeError("AIPerf exceeded three-hour controller limit")
                time.sleep(10)
        exports = list((a.run_dir / "artifacts").rglob("profile_export_aiperf.json"))
        if proc.returncode != 0 or len(exports) != 1:
            raise RuntimeError(f"AIPerf exit={proc.returncode}, summary exports={len(exports)}")
        result = json.loads(exports[0].read_text())
        comparison = {}
        for name, hw, sim in [
            ("total_tok_s_per_gpu", source["total_token_throughput"]["avg"] / a.gpus,
             result["total_token_throughput"]["avg"] / a.gpus),
            ("p95_ttft_ms", source["time_to_first_token"]["p95"], result["time_to_first_token"]["p95"]),
        ]:
            error = sim / hw - 1
            comparison[name] = dict(hardware=hw, simulation=sim, relative_error=error,
                                    within_20_percent=abs(error) <= 0.20)
        warmup = result.get("warmup_metrics", {})
        expected_warmup_count = source["warmup_metrics"]["request_count"]["avg"]
        warmup_valid = (warmup.get("request_count", {}).get("avg") == expected_warmup_count
                        and warmup.get("error_request_count", {}).get("avg", 0) == 0
                        and warmup.get("output_sequence_length", {}).get("min") == 1
                        and warmup.get("output_sequence_length", {}).get("max") == 1)
        valid = (result.get("metadata", {}).get("submission_valid") is True
                 and not result.get("was_cancelled") and warmup_valid)
        replay_inputs = warmup_inputs(exports[0].parent / "profile_export.jsonl")
        reference_inputs = warmup_inputs(a.hardware_records)
        warmup_inputs_match = replay_inputs == reference_inputs
        valid = valid and warmup_inputs_match
        report = dict(metrics=comparison, valid_agentx_submission=valid,
                      warmup_valid=warmup_valid,
                      warmup_inputs_match=warmup_inputs_match,
                      warmup_missing_inputs=list((reference_inputs - replay_inputs).elements()),
                      warmup_unexpected_inputs=list((replay_inputs - reference_inputs).elements()),
                      all_targets_pass=valid and all(x["within_20_percent"] for x in comparison.values()),
                      summary=str(exports[0]))
        write_json(a.run_dir / "comparison.json", report)
        update("completed", **report)
    except BaseException as exc:
        update("failed", error=str(exc))
        raise
    finally:
        # Let the passive observer see the terminal status and close its gzip
        # streams before terminating the servers it is watching.
        observer = owned.get("observer")
        if observer is not None and observer.poll() is None:
            try:
                observer.wait(timeout=12)
            except subprocess.TimeoutExpired:
                pass
        for proc in reversed(list(owned.values())):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in reversed(list(owned.values())):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
