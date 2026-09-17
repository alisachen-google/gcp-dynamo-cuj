#!/usr/bin/env python3
"""Attribute the simulator's prefix-hit-rate shortfall (12:6 KV) to its steps, one substitution at a time:
   A. as published (4 k-request slice, per-play salt, KV capacity 100 M tokens/worker)
   B. + infinite KV capacity (no eviction)
   C. + full trace instead of the 4 k slice (sessions no longer truncated)
   D. + both
Prints hit rate, req/s, total tok/s/GPU and TTFT p95 per variant at the requested client counts."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dynosim_agentx as da
dp = da.dp
slice_path, full_path = sys.argv[1], sys.argv[2]
clients = [int(c) for c in (sys.argv[3] if len(sys.argv) > 3 else "192,768").split(",")]
S4k = da.load_sessions(slice_path)
SFull = da.load_sessions(full_path, limit=10**9)
def run(sess, c, cap=None):
    dp.apply_n3u_constants()
    if cap: dp.KV_CAPACITY_TOKENS = cap
    m = da.simulate_agentx(sess, 12, 6, "kv", c, 3600.0)
    return m
print(f"{'variant':44s} {'clients':>7s} {'hit':>6s} {'req/s':>6s} {'tot/GPU':>8s} {'ttft p95':>8s} {'tpot ms':>7s}")
for c in clients:
    for name, sess, cap in [("A. as published (4k slice, cap 100M)", S4k, None),
                            ("B. + infinite KV capacity", S4k, 10**12),
                            ("C. + full trace (28k requests)", SFull, None),
                            ("D. + full trace + infinite capacity", SFull, 10**12)]:
        m = run(sess, c, cap)
        print(f"{name:44s} {c:7d} {m['hit_rate']:6.3f} {m['req_per_s']:6.2f} {m['total_tok_s']/72:8.0f} {m['ttft_p95_s']:8.2f} {m['tpot_mean_ms']:7.1f}", flush=True)
