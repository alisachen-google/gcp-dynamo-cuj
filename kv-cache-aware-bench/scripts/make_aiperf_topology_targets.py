#!/usr/bin/env python3
"""Create counterfactual TP4 topology targets from a saved real AgentX config."""

import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-targets", type=Path, required=True)
    parser.add_argument("--source-point", default="disagg12-6-rr-c192")
    parser.add_argument("--splits", default="4:12,8:8,12:4")
    parser.add_argument("--clients", default="16,48,96,192,384")
    parser.add_argument("--policies", default="kv,rr")
    parser.add_argument("--gpus", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = next(
        p
        for p in json.loads(args.source_targets.read_text())["points"]
        if p["id"] == args.source_point
    )
    points = []
    for clients in map(int, args.clients.split(",")):
        assert clients > 0
        for split in args.splits.split(","):
            prefill, decode = map(int, split.split(":"))
            assert min(prefill, decode) > 0
            assert (prefill + decode) * source["tp"] == args.gpus
            for policy in args.policies.split(","):
                assert policy in ("kv", "rr")
                point = copy.deepcopy(source)
                point.update(
                    id=f"disagg{prefill}-{decode}-{policy}-c{clients}",
                    arm=split,
                    policy=policy,
                    clients=clients,
                    gpus=args.gpus,
                    recipe_point=False,
                    hardware_comparison_source=None,
                    initial_cache="empty before actual aiperf trajectory warmup; counterfactual topology",
                    replay_config_overrides={"concurrency": clients},
                    serving={
                        "agg": False,
                        "prefill_workers": prefill,
                        "decode_workers": decode,
                        "decode_routing": "active_blocks"
                        if policy == "kv"
                        else "round_robin",
                    },
                )
                point["input_config"]["phases"][0]["concurrency"] = clients
                point["recipe_serving_settings"].update(
                    prefill_workers=prefill,
                    decode_workers=decode,
                    frontend_router_mode="kv" if policy == "kv" else "round-robin",
                )
                point["input_replay_audit_id"] = (
                    "disagg12-6-kv-c192" if clients == 192 else None
                )
                points.append(point)
    result = {
        "date": "2026-09-17",
        "status": "simulation predictions; no matching 64-GPU hardware runs",
        "metric_basis": "aiperf input+output tokens/s / all 64 GPUs; 3600s profiling plus actual trajectory warmup and 60s grace",
        "capacity_assumption": {
            "prefill_prefix_cache_tokens_per_worker": 809406 * 64,
            "source": source["replay_config_source"].replace(
                "profile_export_aiperf.json", "server_metrics_export.csv"
            ),
            "note": "Decode-facing model metadata reports 809406 blocks/worker. Used as a prefill-cache capacity proxy, pending per-prefill-worker measurement. Not a measured Mamba cache capacity.",
        },
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(points)} targets to {args.output}")


if __name__ == "__main__":
    main()
