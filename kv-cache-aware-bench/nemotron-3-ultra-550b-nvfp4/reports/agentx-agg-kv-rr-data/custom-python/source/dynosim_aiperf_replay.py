#!/usr/bin/env python3
"""Run aiperf 0.12.0 AgentX itself against a simulated serving transport.

Requires a matching aiperf source checkout for its in-process service harness.
Dataset loading, synthesis, trajectory selection, credits, dependency barriers,
child spawning/joins, warmup, and metric export remain aiperf implementations.
Only HF download, inter-service IPC, serving, and the clock are adapted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import Counter
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch


class CachedChatTokenizer:
    """Cache exact encodings at explicit, non-stripping special-token boundaries.

    BPE cannot merge across the added special token. Full-prompt tokenization
    verifies the first 100 requests and every 100th request thereafter.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        separator = tokenizer.added_tokens_decoder[
            tokenizer.convert_tokens_to_ids("<|im_start|>")
        ]
        assert separator.special and not separator.lstrip and not separator.rstrip
        self.requests = 0
        self.validations = 0
        self.encode_segment = lru_cache(maxsize=8192)(
            lambda text: tuple(tokenizer.encode(text, add_special_tokens=False))
        )

    def encode(self, body):
        kwargs = {"add_generation_prompt": True, "tools": body.get("tools")}
        rendered = self.tokenizer.apply_chat_template(
            body["messages"], tokenize=False, **kwargs
        )
        parts = rendered.split("<|im_start|>")
        tokens = list(self.encode_segment(parts[0]))
        for part in parts[1:]:
            tokens.extend(self.encode_segment("<|im_start|>" + part))
        self.requests += 1
        if self.requests <= 100 or self.requests % 100 == 0:
            expected = self.tokenizer.apply_chat_template(
                body["messages"], tokenize=True, **kwargs
            )
            assert tokens == expected, "Cached tokenization changed prompt token IDs"
            self.validations += 1
        return tokens


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiperf-source", type=Path, required=True)
    parser.add_argument("--arrow-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--point", required=True, help="Point ID in --targets")
    parser.add_argument(
        "--engine", choices=["original", "measured"], default="original"
    )
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--wall-clock", action="store_true")
    parser.add_argument("--entries", type=int, default=393)
    parser.add_argument(
        "--serving-model-config",
        type=Path,
        help="Experimental shared-GPU agg RR scheduler configuration; original model when omitted",
    )
    parser.add_argument(
        "--cache-capacity-tokens",
        type=int,
        help="Per-worker prefix-cache capacity; default retains the legacy 100M-token pool",
    )
    parser.add_argument(
        "--kv-decode-block-cost",
        action="store_true",
        help="Controlled agg KV ablation: charge the projected active unique prompt-block footprint",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.artifact_dir.exists() and any(args.artifact_dir.iterdir()):
        raise ValueError("Use a new, empty artifact directory for each invocation")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.arrow_dir.parent
    os.environ.update(
        {
            "HF_HOME": str(data_root / "hf"),
            "MPLCONFIGDIR": str(data_root / "matplotlib"),
            "AIPERF_DATASET_MMAP_CACHE_DIR": str(data_root / "mmap-cache"),
            "AIPERF_DATASET_MMAP_BASE_PATH": str(args.artifact_dir / "mmap"),
            "AIPERF_SERVICE_DISABLE_UVLOOP": "true",
            "AIPERF_TOKENIZER_SKIP_PRELOAD": "1",
            "AIPERF_DATASET_CONFIGURATION_TIMEOUT": "3600",
            "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT": "3600",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # Append, so the installed PyPI wheel remains the aiperf implementation.
    sys.path.append(str(args.aiperf_source))
    import aiperf
    import aiperf.cli_runner
    import dynosim_agentx as da
    import looptime
    import numpy as np
    import orjson
    import psutil
    import tests.harness.fake_service_manager  # noqa: F401 -- Registers the manager.
    import xxhash
    from aiperf.cli import app
    from aiperf.common.models import RequestRecord, SSEMessage
    from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
    from datasets import Dataset, concatenate_datasets
    from tests.harness.fake_communication import FakeCommunication, FakeCommunicationBus
    from tests.harness.fake_transport import FakeTransport
    from transformers import AutoTokenizer

    assert aiperf.__version__ == "0.12.0", aiperf.__version__
    wheel_root = Path(aiperf.__file__).parent
    parity_files = [
        "timing/strategies/agentic_replay.py",
        "timing/trajectory_source.py",
        "timing/branch_orchestrator.py",
        "timing/replay_dependencies.py",
        "timing/session_tree.py",
        "dataset/loader/weka_trace.py",
        "dataset/loader/weka_agent_chains.py",
    ]
    code_hashes = {}
    for name in parity_files:
        content = (wheel_root / name).read_bytes()
        assert content == (args.aiperf_source / "src/aiperf" / name).read_bytes(), name
        code_hashes[name] = hashlib.sha256(content).hexdigest()
    arrow_files = sorted(args.arrow_dir.glob("*.arrow"))
    assert len(arrow_files) == 2
    dataset = concatenate_datasets([Dataset.from_file(str(p)) for p in arrow_files])
    assert len(dataset) == 393
    arrow_hashes = {
        p.name: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
        for p in arrow_files
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), trust_remote_code=True
    )
    chat_tokenizer = CachedChatTokenizer(tokenizer)
    target = next(
        p
        for p in json.loads(args.targets.read_text())["points"]
        if p["id"] == args.point
    )
    serving = target.get(
        "serving", {"agg": True, "prefill_workers": 6, "decode_workers": 0}
    )
    assert target["gpus"] == target["tp"] * (
        serving["prefill_workers"] + serving["decode_workers"]
    )
    assert serving["agg"] or serving["decode_workers"] > 0
    assert target["policy"] in ("rr", "kv")
    scheduled_config = (
        json.loads(args.serving_model_config.read_text())
        if args.serving_model_config
        else None
    )
    if scheduled_config is not None:
        assert serving["agg"] and target["policy"] == "rr"
        assert not args.kv_decode_block_cost and args.engine == "original"
        assert args.cache_capacity_tokens in (
            None,
            scheduled_config["cache_capacity_tokens"],
        ), "Conflicting cache capacities"
    config = target["input_config"]
    assert config["phases"][0]["concurrency"] == target["clients"]
    config["endpoint"]["urls"] = ["http://simulated-serving:8000"]
    config["tokenizer"]["name"] = str(args.tokenizer)
    config["artifacts"]["dir"] = str(args.artifact_dir)
    config["artifacts"]["raw"] = False
    config["gpu_telemetry"] = {"enabled": False}
    config["server_metrics"] = {"enabled": False}
    config["phases"][0]["duration"] = args.duration
    config["datasets"][0]["entries"] = args.entries
    config["unsafe_override"] = args.duration < 900
    config_path = args.artifact_dir / "replay-config.json"
    config_path.write_text(
        json.dumps({"random_seed": 42, "benchmark": config}, indent=2) + "\n"
    )
    provenance = {
        "aiperf_version": aiperf.__version__,
        "aiperf_source_sha256": code_hashes,
        "dataset_snapshot": "8fecd2fc56694469f758f0afbbb6335ad3043740",
        "arrow_sha256": arrow_hashes,
        "tokenizer_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.tokenizer.iterdir()
            if p.is_file()
        },
        "point": args.point,
        "engine": args.engine,
        "scheduled_serving_config": scheduled_config,
        "kv_decode_block_cost_ablation": args.kv_decode_block_cost,
        "cache_capacity_tokens": scheduled_config["cache_capacity_tokens"]
        if scheduled_config
        else args.cache_capacity_tokens or 100_000_000,
        "serving": serving,
        "hardware_comparison_source": target.get(
            "hardware_comparison_source", target.get("source")
        ),
        "replay_config_source": target.get(
            "replay_config_source", target.get("source")
        ),
        "virtual_clock": not args.wall_clock,
        "benchmark_id": target["benchmark_id"],
        "engine_source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "dynosim_agentx.py",
                "dynosim_pd.py",
                "dynosim_aiperf_replay.py",
            )
            + (("agentx_scheduled_engine.py",) if scheduled_config else ())
        },
        "duration_s": args.duration,
        "entries": args.entries,
        "adaptations": [
            "cached Arrow download source",
            "in-process service manager and message bus",
            "simulated serving transport",
            "unified virtual asyncio/time clock",
            "logical workers report zero host CPU load",
            "pin benchmark ID from target for repeatable cache-bust markers",
            "memoize tokenization at explicit special-token boundaries, with full-encoding parity checks",
        ],
        "initial_cache": target.get(
            "initial_cache",
            "empty before aiperf's actual trajectory warmup; no completed separate prewarm exists for the two original hardware artifacts; persistent GPU cache state from earlier jobs is not captured",
        ),
        "request_timeout_seconds": config["endpoint"]["timeout"],
    }
    (args.artifact_dir / "simulation-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    da.dp.apply_n3u_constants()
    if args.cache_capacity_tokens is not None:
        assert args.cache_capacity_tokens >= da.dp.BLOCK_TOKENS
        da.dp.KV_CAPACITY_TOKENS = args.cache_capacity_tokens
    if args.engine == "measured":
        assert serving["agg"], "The optional measured fit only supports agg"
        da.dp.agg_tpot_ms = (
            (lambda b: 5.5 + 1.45 * b)
            if target["policy"] == "rr"
            else (lambda b: 5.8 + 1.33 * b)
        )
    engine = da.Engine(
        serving["prefill_workers"],
        serving["decode_workers"],
        target["policy"],
        agg=serving["agg"],
        decode_policy=serving.get("decode_routing", "least_inflight")
        if not serving["agg"]
        else "least_inflight",
    )
    scheduled = None
    if scheduled_config is not None:
        from agentx_scheduled_engine import ScheduledAggEngine

        scheduled = ScheduledAggEngine(
            serving["prefill_workers"], target["policy"], scheduled_config
        )
    # A single-factor diagnostic for the term omitted by the legacy KV score.
    # This is not a complete implementation of Dynamo's scheduler/sequence state.
    active_prompt_blocks = [Counter() for _ in engine.D]
    incoming_blocks = set()
    if args.kv_decode_block_cost:
        assert serving["agg"] and target["policy"] == "kv"
        for index, worker in enumerate(engine.P):
            old_load = worker.load_blocks
            counts = active_prompt_blocks[index]

            def projected_load(now, old_load=old_load, counts=counts):
                return old_load(now) + len(counts.keys() | incoming_blocks)

            worker.load_blocks = projected_load
    log_file = (args.artifact_dir / "simulation-dispatch.jsonl").open("w", buffering=1)
    real_clock = time.perf_counter
    started = real_clock()
    stats = {
        "requests": 0,
        "completed": 0,
        "tokenization_seconds": 0.0,
        "max_cached_blocks_per_prefill_worker": [0] * len(engine.P),
    }

    # The cache and service time use the actual tokenizer/chat-template payload.
    # Each 64-token block receives a stable ID; prefix matching remains in DynoSim.
    async def perform_request(
        self, request_info, payload, *, first_token_callback=None
    ):
        body = payload if isinstance(payload, dict) else orjson.loads(payload)
        t_encode = real_clock()
        tokens = chat_tokenizer.encode(body)
        stats["tokenization_seconds"] += real_clock() - t_encode
        array = np.asarray(tokens, dtype=np.uint32)
        # Hash the entire prefix, not isolated block contents: KV state also
        # depends on every preceding token (including the tree's cache marker).
        prefix_hash = xxhash.xxh3_64()
        hashes = []
        for i in range(0, len(array) - 63, 64):
            prefix_hash.update(memoryview(array[i : i + 64]))
            hashes.append(prefix_hash.intdigest())
        out_len = int(
            body.get("max_tokens")
            or body.get("max_completion_tokens")
            or request_info.max_tokens
        )
        now = time.perf_counter()
        start_ns = time.perf_counter_ns()
        wall_ns = time.time_ns()
        hit_before = engine.hits
        if args.kv_decode_block_cost:
            incoming_blocks.clear()
            incoming_blocks.update(hashes)
        ticket = None
        if scheduled is not None:
            ticket = scheduled.submit(hashes, len(tokens), out_len, now)
            worker = prefill_worker = ticket.worker
            ttft = tpot = 0.0
        else:
            ttft, tpot, _done, worker = engine.serve(hashes, out_len, now)
            prefill_worker = engine.last_prefill_worker
        if args.kv_decode_block_cost:
            active_prompt_blocks[worker].update(hashes)
        if serving["agg"]:
            prefill_worker = worker
        if prefill_worker is not None and scheduled is None:
            stats["max_cached_blocks_per_prefill_worker"][prefill_worker] = max(
                stats["max_cached_blocks_per_prefill_worker"][prefill_worker],
                len(engine.P[prefill_worker].cache),
            )
        cached_tokens = 64 * (engine.hits - hit_before)
        # Match OSL-1 inter-token intervals; old Engine.done includes OSL.
        latency = ttft + max(0, out_len - 1) * tpot
        stats["requests"] += 1
        row = {
            "phase": str(request_info.credit_phase),
            "credit_num": request_info.credit_num,
            "conversation_id": request_info.conversation_id,
            "turn_index": request_info.turn_index,
            "source_trace_id": request_info.source_trace_id,
            "source_outer_idx": request_info.source_outer_idx,
            "source_inner_idx": request_info.source_inner_idx,
            "source_kind": request_info.source_kind,
            "agent_depth": request_info.agent_depth,
            "root_correlation_id": request_info.root_correlation_id,
            "cache_bust_marker": request_info.cache_bust_marker,
            "start_s": now,
            "input_tokens": len(tokens),
            "output_tokens": out_len,
            "cached_tokens": cached_tokens,
            "worker": worker,
            "prefill_worker": prefill_worker,
            "decode_worker": worker,
            "decode_admitted_at_dispatch": (
                len(scheduled.workers[worker].running)
                + len(scheduled.workers[worker].pending)
                if scheduled is not None
                else engine.D[worker]
            ),
            "prefill_queue_s": 0.0 if ticket else engine.last[0],
            "prefill_service_s": 0.0 if ticket else engine.last[1],
            "ttft_s": ttft,
            "tpot_ms": tpot * 1000,
            "expected_latency_s": latency,
        }

        def update_scheduled_row():
            if ticket is None:
                return
            end = ticket.end_s if ticket.end_s is not None else time.perf_counter()
            first = ticket.first_s if ticket.first_s is not None else end
            row.update(
                cached_tokens=ticket.cached_tokens,
                prefill_queue_s=(
                    ticket.admitted_s if ticket.admitted_s is not None else end
                )
                - now,
                prefill_service_s=ticket.prefill_service_s,
                ttft_s=first - now,
                tpot_ms=1000 * (end - first) / max(1, out_len - 1),
                expected_latency_s=end - now,
                active_at_admission=ticket.active_at_admission,
                serving_model="shared_gpu_scheduled",
            )

        def chunk(content, finish=None, usage=None):
            obj = {
                "id": f"sim-{request_info.credit_num}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body["model"],
                "choices": [
                    {"index": 0, "delta": {"content": content}, "finish_reason": finish}
                ],
            }
            if usage is not None:
                obj["usage"] = usage
            return SSEMessage.parse(
                b"data: " + orjson.dumps(obj), time.perf_counter_ns()
            )

        try:
            if ticket is not None:
                await ticket.first
                cached_tokens = ticket.cached_tokens
                update_scheduled_row()
            else:
                await asyncio.sleep(ttft)
            first = chunk("x")
            if first_token_callback:
                await first_token_callback(time.perf_counter_ns() - start_ns, first)
            if ticket is not None:
                await ticket.done
                update_scheduled_row()
                stats["max_cached_blocks_per_prefill_worker"] = [
                    w.cache.peak_blocks for w in scheduled.workers
                ]
            else:
                await asyncio.sleep(max(0, out_len - 1) * tpot)
            usage = {
                "prompt_tokens": len(tokens),
                "completion_tokens": out_len,
                "total_tokens": len(tokens) + out_len,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            }
            last = chunk(" x" * max(0, out_len - 1), "length", usage)
            stats["completed"] += 1
            row["end_s"] = time.perf_counter()
            log_file.write(json.dumps(row) + "\n")
            if stats["completed"] % 500 == 0:
                print(
                    f"SIM_PROGRESS completed={stats['completed']} virtual_s={row['end_s']:.1f} real_s={real_clock() - started:.1f}",
                    flush=True,
                )
            return RequestRecord(
                start_perf_ns=start_ns,
                end_perf_ns=time.perf_counter_ns(),
                timestamp_ns=wall_ns,
                status=200,
                responses=[first, last],
            )
        except asyncio.CancelledError:
            update_scheduled_row()
            row.update(end_s=time.perf_counter(), cancelled=True)
            log_file.write(json.dumps(row) + "\n")
            raise
        finally:
            if ticket is not None:
                if ticket.end_s is None:
                    scheduled.cancel(ticket)
            else:
                engine.release(worker, hashes)
            if args.kv_decode_block_cost:
                counts = active_prompt_blocks[worker]
                for block in hashes:
                    counts[block] -= 1
                    assert counts[block] >= 0
                    if counts[block] == 0:
                        del counts[block]

    async def send_request(self, request_info, payload, *, first_token_callback=None):
        # HTTP normally owns this timeout. Keep it in the simulated transport
        # so long generations fail and release their branch/credit normally.
        async with asyncio.timeout(config["endpoint"]["timeout"]):
            return await perform_request(
                self, request_info, payload, first_token_callback=first_token_callback
            )

    # Disable payload capture only; keep all actual message delivery/handlers.
    class DiscardCapture(list):
        def append(self, value):
            pass

    bus = FakeCommunicationBus()
    bus.sent_payloads = DiscardCapture()
    bus.received_payloads = DiscardCapture()
    FakeCommunication.set_shared_bus(bus)
    originals = {
        name: getattr(time, name)
        for name in (
            "time",
            "time_ns",
            "perf_counter",
            "perf_counter_ns",
            "monotonic",
            "monotonic_ns",
        )
    }
    epoch = originals["time"]()

    def fake_clock(name):
        def clock():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return originals[name]()
            value = loop.time() + (epoch if name in ("time", "time_ns") else 0)
            return int(value * 1e9) if name.endswith("_ns") else value

        return clock

    def virtual_run(coro, **kwargs):
        # AIPerf schedules nanosecond deadlines. The looptime default (1 us)
        # can round a positive sub-microsecond wait to zero and spin forever.
        loop = looptime.new_event_loop(start=0, resolution=1e-9, _enabled=True)
        loop._clock_resolution = 2e-9
        runner = asyncio.Runner(loop_factory=lambda: loop)
        try:
            with ExitStack() as stack:
                for name in originals:
                    stack.enter_context(patch.object(time, name, fake_clock(name)))
                return runner.run(coro)
        finally:
            # Thread-pool teardown waits on real threads, outside simulated time.
            loop.setup_looptime(_enabled=False)
            runner.close()

    make_run = aiperf.cli_runner._make_benchmark_run

    def make_matching_run(config, **kwargs):
        return make_run(config, **(kwargs | {"benchmark_id": target["benchmark_id"]}))

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(aiperf.cli_runner, "_make_benchmark_run", make_matching_run)
        )
        stack.enter_context(
            patch.object(BaseHFDatasetLoader, "_load_hf_dataset", lambda self: dataset)
        )
        stack.enter_context(patch.object(FakeTransport, "send_request", send_request))
        stack.enter_context(patch.object(os, "_exit", lambda code: None))
        # All logical workers share one host process; its aggregate CPU use is
        # not the load of any one simulated worker and must not throttle replay.
        stack.enter_context(
            patch.object(psutil.Process, "cpu_percent", lambda self, *a, **kw: 0.0)
        )
        if not args.wall_clock:
            stack.enter_context(patch.object(asyncio, "run", virtual_run))
        cli_args = ["profile", "--config", str(config_path)]
        if args.duration < 900:
            cli_args.append("--unsafe-override")
        try:
            app(cli_args)
        finally:
            log_file.close()
            stats["wall_seconds"] = real_clock() - started
            stats["tokenization_full_parity_checks"] = chat_tokenizer.validations
            stats["tokenization_cache"] = (
                chat_tokenizer.encode_segment.cache_info()._asdict()
            )
            if scheduled is not None:
                stats["scheduled_workers"] = scheduled.statistics()
            (args.artifact_dir / "simulation-stats.json").write_text(
                json.dumps(stats, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
