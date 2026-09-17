#!/usr/bin/env python3
"""Re-simulate 12:6 KV on the FULL trace with an aiperf-style warm-up before the window; prints hit, req/s, total/GPU,
TTFT p50/p95 and the TTFT decomposition into prefill-queue wait and prefill service (p50/p95) plus mean uncached tokens."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dynosim_agentx as da
dp = da.dp
trace = sys.argv[1]; clients = [int(c) for c in sys.argv[2].split(",")]; warm = float(sys.argv[3]); limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
sess = da.load_sessions(trace, limit=limit)
lab = ("4k slice" if limit < 10**9 else "full trace") + f" + {warm:.0f} s warm-up"
print(f"{'variant':34s} {'clients':>7s} {'hit':>6s} {'req/s':>6s} {'tot/GPU':>8s} {'ttft p50':>8s} {'ttft p95':>8s} {'wait p50':>8s} {'wait p95':>8s} {'svc p50':>8s} {'svc p95':>8s} {'unc/req':>8s} {'tpot':>6s}")
for c in clients:
    dp.apply_n3u_constants()
    m = da.simulate_agentx(sess, 12, 6, "kv", c, 3600.0, warm_s=warm)
    print(f"{lab:34s} {c:7d} {m['hit_rate']:6.3f} {m['req_per_s']:6.2f} {m['total_tok_s']/72:8.0f} {m['ttft_p50_s']:8.2f} {m['ttft_p95_s']:8.2f} {m['wait_p50_s']:8.2f} {m['wait_p95_s']:8.2f} {m['svc_p50_s']:8.2f} {m['svc_p95_s']:8.2f} {m['unc_mean']:8.0f} {m['tpot_mean_ms']:6.1f}", flush=True)
