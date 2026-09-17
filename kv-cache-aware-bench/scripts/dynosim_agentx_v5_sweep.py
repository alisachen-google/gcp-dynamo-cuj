#!/usr/bin/env python3
"""Historical v5 sweep; superseded by sweep_aiperf_replay.py for AgentX performance.

This handwritten scheduler recycles at sampled trajectory boundaries, uses a
900-second simulated prewarm, and does not implement actual AIPerf branch/join
semantics. Its rates, cache capacity and decode routing also differ from the
current AIPerf experiments. Retained only to reproduce historical CSVs.

Current workflow: nemotron-3-ultra-550b-nvfp4/reports/agentx-replay-howto.md
usage: dynosim_agentx_v5_sweep.py <stream_trace.jsonl> <out.csv> <splits e.g. 12:4,10:6> <policies kv,rr,kv-tuned> <clients 192,576>
"""
import sys, csv, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dynosim_agentx_v5 as v5, dynosim_agentx as da
dp = da.dp
print("HISTORICAL v5 workload model: use sweep_aiperf_replay.py for current AgentX performance.", file=sys.stderr)
trace, out, splits, pols, clients = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(","), [int(c) for c in sys.argv[5].split(",")]
traces = v5.load_stream_trace(trace)
ROUTERS = {"kv": None, "rr": None, "kv-tuned": {"prefill_load_scale": 3.0, "overlap_credit": 0.8}}
cols = ["pd", "gpus", "policy", "clients", "clients_per_gpu", "req_per_s", "total_tok_s_gpu", "out_tok_s_gpu", "hit_rate", "ttft_p50_s", "ttft_p95_s", "tpot_mean_ms", "tpot_p90_ms", "p90_interactivity", "in_tok_per_req", "out_per_req", "sub_share", "wait_p95_s", "lat_mean_s", "n"]
new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(cols)
    for sp in splits:
        n_p, n_d = (int(x) for x in sp.split(":")); gpus = (n_p + n_d) * 4
        for pol in pols:
            for c in clients:
                t0 = time.time(); dp.apply_n3u_constants(); dp.PREFILL_TOKRATE = 24000; dp.TPOT_BASE_MS = 6.9; dp.TPOT_SLOPE_MS = 0.44; dp.KV_CAPACITY_TOKENS = 40_000_000  # bounds sim memory; live working set at 1,152 lanes is ~4.4M blocks of the 10M kept
                router = dict(dp.ROUTER, **ROUTERS[pol]) if ROUTERS.get(pol) else None
                m = v5.simulate_v5(traces, n_p, n_d, "rr" if pol == "rr" else "kv", c, router=router)
                row = [sp, gpus, pol, c, round(c / gpus, 2), round(m["req_per_s"], 3), round(m["total_tok_s"] / gpus, 1), round(m["throughput_tok_s"] / gpus, 2), round(m["hit_rate"], 3),
                       round(m["ttft_p50_s"], 3), round(m["ttft_p95_s"], 3), round(m["tpot_mean_ms"], 2), round(m["tpot_p90_ms"], 2), round(1000 / m["tpot_p90_ms"], 1), round(m["in_tok_per_req"]), round(m["out_per_req"]),
                       round(m["sub_share"], 3), round(m["wait_p95_s"], 3), round(m["lat_mean_s"], 2), m["n"]]
                w.writerow(row); f.flush()
                print(f"{sp:>5} {pol:8s} c={c:5d} tot/GPU={row[6]:8.1f} ttft95={row[10]:7.2f}s P90={row[13]:6.1f} hit={row[8]:.3f} req/s={row[5]:6.2f}  ({time.time()-t0:.0f}s)", flush=True)
