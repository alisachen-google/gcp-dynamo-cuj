#!/usr/bin/env python3
"""Run a topology forecast with the live AIPerf/Dynamo/Mocker/AIC Path 1 stack.

The saved hardware summary supplies only the workload configuration. This runner
does not manufacture a hardware reference for a new concurrency or GPU count.
It changes no serving, cache, routing, timing, or AgentX scheduling behavior.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def warmup_signature(path):
    counts = Counter()
    failures = []
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            metadata = record.get("metadata", {})
            if metadata.get("benchmark_phase") != "warmup":
                continue
            metrics = record.get("metrics", {})
            count = lambda name: metrics.get(name, {}).get("value")
            identity = tuple(metadata.get(k) for k in (
                "source_trace_id", "conversation_id", "source_kind",
                "source_outer_idx", "turn_index"))
            counts[identity + (count("usage_prompt_tokens"),)] += 1
            if (record.get("error") or metadata.get("was_cancelled")
                    or count("usage_completion_tokens") != 1):
                failures.append(identity)
    rows = sorted(([list(key), value] for key, value in counts.items()), key=str)
    serialized = json.dumps(rows, sort_keys=True).encode()
    return dict(requests=sum(counts.values()), inputs=rows, failures=failures,
                sha256=hashlib.sha256(serialized).hexdigest(),
                all_successful_one_token=bool(rows) and not failures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload-summary", type=Path, required=True)
    p.add_argument("--topology-config", type=Path, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--benchmark-id", required=True)
    p.add_argument("--calibration-matrix", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--http-port", type=int, required=True)
    p.add_argument("--metrics-port", type=int, required=True)
    p.add_argument("--bootstrap-port", type=int, required=True)
    p.add_argument("--namespace", required=True)
    p.add_argument("--router", choices=["round-robin", "kv"], required=True)
    p.add_argument("--load-scale", type=float, default=1.0)
    p.add_argument("--overlap-credit", type=float, default=1.0)
    p.add_argument("--credit-decay", type=float, default=0.0)
    p.add_argument("--routing-debug", action="store_true")
    p.add_argument("--server-python", required=True)
    p.add_argument("--expected-native-core-sha256", default="39e41354104490981f35f76850bd17973369fc8222dab9f7813e25f50146e36e")
    p.add_argument("--client-python", default="/tmp/n3u-faithful-venv/bin/python")
    p.add_argument("--tokenizer", default="/tmp/n3u-faithful-data/tokenizer")
    p.add_argument("--hf-home", default="/tmp/n3u-path1-hf")
    p.add_argument("--dataset-cache", default="/tmp/n3u-path1-data/hf-datasets")
    p.add_argument("--mmap-cache", default="/tmp/n3u-path1-data/mmap-cache")
    p.add_argument("--nats", default="nats://127.0.0.1:18222")
    p.add_argument("--etcd", default="http://127.0.0.1:18279")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--no-raw-payloads", action="store_true",
                   help="Retain numeric per-request exports without the large duplicate request/response payload file")
    p.add_argument("--purpose", default="disagg_kv_flag_sweep")
    a = p.parse_args()
    if a.concurrency < 1:
        raise ValueError("Concurrency must be positive")
    tuning = dict(prefill_load_scale=a.load_scale,
                  overlap_score_credit=a.overlap_credit,
                  overlap_score_credit_decay=a.credit_decay,
                  router_temperature=0.0, router_queue_policy="fcfs")
    if not math.isfinite(a.load_scale) or a.load_scale <= 0:
        raise ValueError("Load scale must be finite and positive")
    if any(not math.isfinite(x) or x < 0 for x in [a.overlap_credit, a.credit_decay]):
        raise ValueError("Credit and decay must be finite and nonnegative")
    source = json.loads(a.workload_summary.read_text())
    topology = json.loads(a.topology_config.read_text())
    calibration = json.loads(a.calibration_matrix.read_text())
    cfg = copy.deepcopy(source["input_config"])
    assert cfg["scenario"] == "inferencex-agentx-mvp"
    assert len(cfg["phases"]) == 1
    assert cfg["phases"][0]["duration"] == 3600.0
    assert cfg["phases"][0]["timing_mode"] == "agentic_replay"
    assert cfg["phases"][0]["grace_period"] == 60.0
    assert cfg["endpoint"]["timeout"] == 1200.0
    assert source["run_info"]["random_seed"] == 42
    assert cfg["datasets"][0]["entries"] == 393
    roles = [pool["role"] for pool in topology["pools"]]
    assert roles == ["aggregated"] or sorted(roles) == ["decode", "prefill"]
    pools = []
    gpu_count = 0
    for pool in topology["pools"]:
        assert pool["workers"] > 0
        config_path = a.topology_config.parent / pool["engine_config"]
        engine = json.loads(config_path.read_text())
        assert engine["engine_type"] == engine["aic_backend"] == "sglang"
        assert engine["speedup_ratio"] == engine["decode_speedup_ratio"] == 1.0
        assert engine["worker_type"] == pool["role"]
        assert engine["aic_tp_size"] == 4
        assert engine["aic_moe_ep_size"] == 4
        assert engine["aic_moe_tp_size"] == 1
        if pool["role"] == "decode":
            assert engine["enable_prefix_caching"] is False
            assert engine["sglang"]["mamba_state_capacity"] == 64
            assert engine["sglang"]["disagg_reserved_decode_tokens"] == 512
            assert engine["sglang"]["disagg_prefill_first_token"] is True
        if pool["role"] != "aggregated":
            assert engine["kv_transfer_bandwidth"] > 0
            assert engine["kv_bytes_per_token"] == 3072
            assert engine["sglang"]["recurrent_state_transfer_bytes"] == 101990400
        gpu_count += pool["workers"] * engine["aic_tp_size"]
        pools.append((pool, engine))
    assert gpu_count == topology["gpu_count"]
    a.run_dir = a.run_dir.resolve()
    a.run_dir.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    url = f"http://127.0.0.1:{a.http_port}"
    cfg["phases"][0]["concurrency"] = a.concurrency
    cfg["endpoint"]["urls"] = [url]
    cfg["artifacts"]["dir"] = str(a.run_dir / "artifacts")
    if a.no_raw_payloads:
        cfg["artifacts"]["raw"] = False
    cfg["gpu_telemetry"]["enabled"] = False
    cfg["runtime"]["ui"] = "simple"
    model = cfg["models"]["items"][0]["name"]
    write_json(a.run_dir / "client-config.json", dict(benchmark=cfg, random_seed=42))
    write_json(a.run_dir / "topology.json", topology)
    write_json(a.run_dir / "workload-source.json", source)
    write_json(a.run_dir / "calibration-at-launch.json", calibration)
    env_delta = dict(
        HF_HOME=a.hf_home, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_DATASETS_CACHE=a.dataset_cache, AIPERF_DATASET_MMAP_CACHE_DIR=a.mmap_cache,
        AIPERF_DATASET_MMAP_BASE_PATH=str(a.run_dir / "mmap"),
        PATH1_BENCHMARK_ID=a.benchmark_id, PATH1_CONTEXT_LENGTH="262144",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false", NATS_SERVER=a.nats, ETCD_ENDPOINTS=a.etcd,
        DYN_NAMESPACE=a.namespace,
        AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID="0",
        DYN_LOG=("info,dynamo_kv_router::scheduling::selector=debug"
                  if a.routing_debug else "info"))
    # Keep ambient router settings from silently changing one sweep arm.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("DYN_ROUTER_", "DYN_KV_OVERLAP_"))} | env_delta
    common = ["--request-plane", "nats", "--discovery-backend", "etcd",
              "--event-plane", "zmq"]
    backend_workers = sum(pool["workers"] for pool, _ in pools
                          if pool["role"] != "prefill")
    front = [a.server_python, "-m", "dynamo.frontend", "--namespace", a.namespace,
             "--http-host", "127.0.0.1", "--http-port", str(a.http_port),
             "--router-mode", a.router, "--router-min-initial-workers", str(backend_workers),
             "--dump-config-to", str(a.run_dir / "frontend-resolved.json")]
    if a.router == "kv":
        front += ["--router-temperature", "0.0", "--router-queue-policy", "fcfs",
                  "--router-prefill-load-scale", str(a.load_scale),
                  "--router-kv-overlap-score-credit", str(a.overlap_credit),
                  "--router-kv-overlap-score-credit-decay", str(a.credit_decay)]
    commands = dict(frontend=front + common)
    workers, observers, metrics_ports, components = [], [], {}, {}
    metric_offset = 0
    for pool, engine in pools:
        role = pool["role"]
        component = "prefill" if role == "prefill" else "backend"
        components[component] = pool["workers"]
        folder = a.run_dir / role
        folder.mkdir()
        write_json(folder / "engine.json", engine)
        pool_commands = {}
        for i in range(pool["workers"]):
            name = f"{role}-{i}"
            worker = [a.server_python, str(scripts / "path1_mocker.py"),
                      "--model-path", a.tokenizer, "--model-name", model,
                      "--endpoint", f"dyn://{a.namespace}.{component}.generate",
                      "--num-workers", "1", "--extra-engine-args", str(folder / "engine.json")]
            if role != "aggregated":
                worker += ["--disaggregation-mode", role, "--kv-bytes-per-token", "3072"]
            if role == "prefill":
                worker += ["--bootstrap-ports", str(a.bootstrap_port + i)]
            commands[name] = worker + common
            pool_commands[f"worker-{i}"] = commands[name]
            metrics_ports[name] = a.metrics_port + metric_offset + i
            workers.append(name)
        write_json(folder / "provenance.json", dict(
            env_overrides=env_delta, commands=pool_commands,
            worker_metrics_port=a.metrics_port + metric_offset,
            fpm_endpoint=f"{a.namespace}.{component}.generate"))
        (folder / "status.json").symlink_to(a.run_dir / "status.json")
        observer = f"observe-{role}"
        commands[observer] = [a.server_python, str(scripts / "observe_path1.py"), str(folder)]
        observers.append(observer)
        metric_offset += pool["workers"]
    commands["client"] = [a.client_python, str(scripts / "path1_agentx_client.py"),
                          "profile", "--config", str(a.run_dir / "client-config.json")]
    probe = (
        "import json,hashlib,importlib.metadata as m,dynamo._core;from pathlib import Path;"
        "p=Path(dynamo._core.__file__);"
        "print(json.dumps(dict(versions={n:m.version(n) for n in "
        "['ai-dynamo','ai-dynamo-runtime','aiconfigurator-core']},"
        "core_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))")
    server_environment = json.loads(subprocess.check_output(
        [a.server_python, "-c", probe], env=env, text=True))
    if server_environment["core_sha256"] != a.expected_native_core_sha256:
        raise RuntimeError("Native core changed; refusing to mix builds in the flag sweep")
    write_json(a.run_dir / "server-environment.json", server_environment)
    launcher_files = [Path(__file__), scripts / "path1_mocker.py",
                      scripts / "path1_agentx_client.py", scripts / "observe_path1.py"]
    for path in launcher_files:
        (a.run_dir / path.name).write_bytes(path.read_bytes())
    write_json(a.run_dir / "provenance.json", dict(
        method="path1_live_aiperf_dynamo_mocker_aic", clock="wall", speedup=1,
        purpose=a.purpose, topology=topology["name"], gpu_count=gpu_count,
        raw_payload_export=cfg["artifacts"].get("raw", False),
        router_tuning=tuning,
        concurrency=a.concurrency, router=a.router, benchmark_id=a.benchmark_id,
        workload_source=str(a.workload_summary), workload_source_sha256=sha256(a.workload_summary),
        calibration_matrix=str(a.calibration_matrix),
        calibration_matrix_sha256=sha256(a.calibration_matrix),
        calibration_scope="See the immutable calibration-at-launch.json; topology is extrapolation",
        env_overrides=env_delta, commands=commands, worker_metrics_ports=metrics_ports,
        launcher_sha256={path.name: sha256(path) for path in launcher_files},
        started_unix=time.time()))
    owned, logs = {}, []

    def update(state, **extra):
        write_json(a.run_dir / "status.json", dict(state=state, updated_unix=time.time(), **extra))

    if a.prepare_only:
        update("prepared")
        return

    def launch(name):
        log = (a.run_dir / f"{name}.log").open("w")
        logs.append(log)
        local_env = env | ({"DYN_SYSTEM_PORT": str(metrics_ports[name])}
                           if name in metrics_ports else {})
        process = subprocess.Popen(commands[name], stdout=log, stderr=subprocess.STDOUT,
                                   env=local_env, cwd=a.run_dir, start_new_session=True)
        owned[name] = process
        write_json(a.run_dir / "pids.json", {k: v.pid for k, v in owned.items()})
        return process

    def check_servers():
        for name in ["frontend", *workers]:
            if owned[name].poll() is not None:
                raise RuntimeError(f"Server {name} exited")
            if b"panicked at" in (a.run_dir / f"{name}.log").read_bytes():
                raise RuntimeError(f"Native task panicked in {name}")

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        update("starting_server")
        launch("frontend")
        for name in workers:
            launch(name)
            time.sleep(0.5)
        for _ in range(180):
            check_servers()
            try:
                models = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=2).read())
                health = json.loads(urllib.request.urlopen(url + "/health", timeout=2).read())
                ready = [m for m in models.get("data", []) if m["id"] == model]
                counts = Counter(i.get("component") for i in health.get("instances", [])
                                 if i.get("namespace") == a.namespace and i.get("endpoint") == "generate")
                initialized = all(b"Metrics background tasks started" in
                                  (a.run_dir / f"{name}.log").read_bytes() for name in workers)
                if (ready and ready[0].get("context_window") == 262144
                        and all(counts[k] == n for k, n in components.items()) and initialized):
                    resolved = json.loads((a.run_dir / "frontend-resolved.json").read_text())
                    def values_for(obj, key):
                        values = []
                        if isinstance(obj, dict):
                            for k, value in obj.items():
                                if k == key:
                                    values.append(value)
                                values.extend(values_for(value, key))
                        elif isinstance(obj, list):
                            for value in obj:
                                values.extend(values_for(value, key))
                        return values
                    for key, expected in (tuning.items() if a.router == "kv" else []):
                        values = values_for(resolved, key)
                        if not values or any(value != expected for value in values):
                            raise RuntimeError(f"Resolved router flag mismatch: {key}={values}, expected {expected}")
                    write_json(a.run_dir / "router-flags-confirmed.json", tuning)
                    write_json(a.run_dir / "startup-health.json", health)
                    callbacks = {}
                    for name in workers:
                        lines = [line for line in (a.run_dir / f"{name}.log").read_text().splitlines()
                                 if "AIC: using pure-Rust RustAicCallback" in line]
                        if len(lines) != 1:
                            raise RuntimeError(f"Native AIC callback not confirmed for {name}")
                        callbacks[name] = lines[0]
                    write_json(a.run_dir / "aic-callback-confirmation.json", callbacks)
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Model, context, or complete P/D pool readiness failed")
        update("running_aiperf")
        for observer in observers:
            launch(observer)
        proc = launch("client")
        started = time.monotonic()
        with (a.run_dir / "host-observations.jsonl").open("w") as host:
            while proc.poll() is None:
                check_servers()
                host.write(json.dumps(dict(time=time.time(), loadavg=os.getloadavg(),
                    meminfo=Path("/proc/meminfo").read_text(),
                    proc_stat=Path("/proc/stat").read_text())) + "\n")
                host.flush()
                if time.monotonic() - started > 14400:
                    raise RuntimeError("AIPerf exceeded four-hour controller limit")
                time.sleep(10)
        exports = list((a.run_dir / "artifacts").rglob("profile_export_aiperf.json"))
        if proc.returncode != 0 or len(exports) != 1:
            raise RuntimeError(f"AIPerf exit={proc.returncode}, exports={len(exports)}")
        result = json.loads(exports[0].read_text())
        warmup = warmup_signature(exports[0].parent / "profile_export.jsonl")
        write_json(a.run_dir / "warmup-inputs.json", warmup)
        valid = (result.get("metadata", {}).get("submission_valid") is True
                 and not result.get("was_cancelled") and warmup["all_successful_one_token"])
        avg = lambda name: result.get(name, {}).get("avg")
        report = dict(
            topology=topology["name"], router=a.router, concurrency=a.concurrency,
            router_tuning=tuning,
            gpu_count=gpu_count, benchmark_id=a.benchmark_id,
            total_tok_s_per_gpu=avg("total_token_throughput") / gpu_count,
            output_tok_s_per_gpu=avg("output_token_throughput") / gpu_count,
            p95_ttft_ms=result["time_to_first_token"]["p95"],
            p95_itl_ms=result.get("inter_token_latency", {}).get("p95"),
            request_throughput=avg("request_throughput"),
            requests=avg("request_count"), errors=avg("error_request_count"),
            error_summary=result.get("error_summary"),
            effective_concurrency=result.get("effective_concurrency"),
            valid_agentx_submission=valid, warmup_input_sha256=warmup["sha256"],
            summary=str(exports[0]), measured_hardware=False)
        write_json(a.run_dir / "result.json", report)
        update("completed", **report)
    except BaseException as exc:
        update("failed", error=str(exc))
        raise
    finally:
        for name in observers:
            observer = owned.get(name)
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
