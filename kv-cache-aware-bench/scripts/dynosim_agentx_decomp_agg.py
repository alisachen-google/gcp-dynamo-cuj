#!/usr/bin/env python3
"""Apple-to-apple drift decomposition for the AgentX-semantics simulator, aggregated 24-GPU arm (agg6 KV, 48/96 clients).
Substitutes measured quantities into dynosim_agentx one at a time and reports req/s, output tok/s,
total tok/s per GPU (input = len(hash_ids)*64 + output, per measured request), TTFT p50, TPOT."""
import sys, pathlib, copy, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dynosim_agentx as da
dp = da.dp  # the simulator loads dynosim_pd via importlib: patch ITS module object, not a second import
trace = sys.argv[1]; GPUS = 24
# TTFT standard for sim-vs-real: **p95** (ttft = measured TTFT p95 in seconds)
# measured agg KV AgentX (profiling phase of the per-request records): 48 → 0.78 req/s, 780 out tok/s, 80,564 total tok/s;
# 96 → 1.92 req/s, 1,811 out tok/s, 165,043 total tok/s.  TTFT = p95, itl = ITL p50 (ms)
MEAS = {48: dict(req_s=0.78, out=780, tot_gpu=3357, ttft=3.83, itl=7.3, out_per_req=997, in_per_req=102_000),
        96: dict(req_s=1.92, out=1811, tot_gpu=6877, ttft=5.36, itl=11.9, out_per_req=945, in_per_req=85_200)}
sessions = da.load_sessions(trace)

def run(sess, clients, floor=0.0):
    # wrap Engine.serve to add a fixed TTFT floor when requested
    orig = da.Engine.serve
    if floor:
        def serve(self, hid, out_len, now):
            ttft, tpot, done, d = orig(self, hid, out_len, now)
            return ttft + floor, tpot, done + floor, d
        da.Engine.serve = serve
    try:
        dp.apply_n3u_constants()
        for k, v in OVERRIDE.items(): setattr(dp, k, v)
        m = da.simulate_agentx(sess, 6, 0, "kv", clients, 3600.0, agg=True)
    finally:
        da.Engine.serve = orig
    return m

def with_total(sess, clients, floor=0.0):
    # re-run capturing per-request input tokens: monkeypatch serve to record len(hid)
    tot = {"in": 0, "n": 0}
    orig = da.Engine.serve
    def serve(self, hid, out_len, now):
        r = orig(self, hid, out_len, now); tot["in"] += len(hid) * dp.BLOCK_TOKENS; tot["n"] += 1; return r
    da.Engine.serve = serve
    try: m = run(sess, clients, floor)
    finally: da.Engine.serve = orig
    if m: m["in_per_req"] = tot["in"] / max(1, tot["n"]); m["tot_gpu"] = (m["req_per_s"] * m["in_per_req"] + m["throughput_tok_s"]) / GPUS
    return m

def scaled(sess, f):
    out = []
    for s in sess:
        out.append([dict(r, output_length=max(1, int(r["output_length"] * f))) for r in s])
    return out

OVERRIDE = {}
variants = []
variants.append(("v1 (AIC-seeded, as published)", lambda c: with_total(sessions, c)))
OSL_F = {48: 997 / 1218, 96: 945 / 1205}
def v_osl(c): return with_total(scaled(sessions, OSL_F[c]), c)
variants.append(("+ output length = measured (×0.81)", v_osl))
def v_ttft(c): return with_total(scaled(sessions, OSL_F[c]), c, floor=0.19)
variants.append(("+ TTFT floor 0.19 s (transfer + scheduling)", v_ttft))
def v_dec(c):
    # apply_n3u_constants() rebinds agg_tpot_ms on every run, so the override must go through OVERRIDE (applied after it)
    global OVERRIDE; OVERRIDE = {"agg_tpot_ms": (lambda b: 5.8 + 1.33 * b)}
    try: return with_total(scaled(sessions, OSL_F[c]), c, floor=0.19)
    finally: OVERRIDE = {}
variants.append(("+ decode = measured ITL (5.8 + 1.33·bs, no cliff)", v_dec))

print(f"{'variant':46s} {'clients':>7s} {'req/s':>6s} {'out tok/s':>9s} {'out/GPU':>7s} {'tot/GPU':>7s} {'ttft p95':>8s} {'tpot ms':>7s} | sim/real: req/s  out  total  ttft")
for name, fn in variants:
    for c in (48, 96):
        m = fn(c); M = MEAS[c]
        if not m: print(name, c, "no result"); continue
        print(f"{name:46s} {c:7d} {m['req_per_s']:6.2f} {m['throughput_tok_s']:9.0f} {m['throughput_tok_s']/GPUS:7.1f} {m['tot_gpu']:7.0f} {m['ttft_p95_s']:8.2f} {m['tpot_mean_ms']:7.1f} | {m['req_per_s']/M['req_s']:5.2f}x {m['throughput_tok_s']/M['out']:5.2f}x {m['tot_gpu']/M['tot_gpu']:5.2f}x {m['ttft_p95_s']/M['ttft']:5.2f}x")
print(f"{'measured (agg KV, AgentX mode)':46s} {'48':>7s} {MEAS[48]['req_s']:6.2f} {MEAS[48]['out']:9.0f} {MEAS[48]['out']/GPUS:7.1f} {MEAS[48]['tot_gpu']:7.0f} {MEAS[48]['ttft']:8.2f} {MEAS[48]['itl']:7.1f}")
print(f"{'measured (agg KV, AgentX mode)':46s} {'96':>7s} {MEAS[96]['req_s']:6.2f} {MEAS[96]['out']:9.0f} {MEAS[96]['out']/GPUS:7.1f} {MEAS[96]['tot_gpu']:7.0f} {MEAS[96]['ttft']:8.2f} {MEAS[96]['itl']:7.1f}")
