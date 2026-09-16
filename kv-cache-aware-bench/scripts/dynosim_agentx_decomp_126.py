#!/usr/bin/env python3
"""Apple-to-apple drift decomposition for the AgentX-semantics simulator (9:9 KV, 48/96 clients).
Substitutes measured quantities into dynosim_agentx one at a time and reports req/s, output tok/s,
total tok/s per GPU (input = len(hash_ids)*64 + output, per measured request), TTFT p50, TPOT."""
import sys, pathlib, copy, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dynosim_agentx as da
dp = da.dp  # the simulator loads dynosim_pd via importlib: patch ITS module object, not a second import
trace = sys.argv[1]; GPUS = 72
# TTFT standard for sim-vs-real: **p95** (ttft = measured TTFT p95 in seconds)
# measured 12:6 KV AgentX (profiling-phase records); ttft = p95 (s), itl = ITL p50 (ms)
MEAS = {96: dict(req_s=2.05, out=1998, tot_gpu=2493, ttft=1.37, itl=8.2, out_per_req=973, in_per_req=86_500),
        192: dict(req_s=3.45, out=3231, tot_gpu=4530, ttft=1.77, itl=10.0, out_per_req=936, in_per_req=93_600),
        384: dict(req_s=6.63, out=6276, tot_gpu=8669, ttft=2.58, itl=13.7, out_per_req=947, in_per_req=93_200),
        480: dict(req_s=7.98, out=7566, tot_gpu=10464, ttft=3.52, itl=16.4, out_per_req=948, in_per_req=93_500)}
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
        m = da.simulate_agentx(sess, 12, 6, "kv", clients, 3600.0)
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
OSL_F = {96: 973 / 1205, 192: 936 / 1200, 384: 947 / 1200, 480: 948 / 1200}
def v_osl(c): return with_total(scaled(sessions, OSL_F[c]), c)
variants.append(("+ output length = measured (×0.81)", v_osl))
def v_ttft(c): return with_total(scaled(sessions, OSL_F[c]), c, floor=0.19)
variants.append(("+ TTFT floor 0.19 s (transfer + scheduling)", v_ttft))
def v_dec(c):
    global OVERRIDE; OVERRIDE = {"TPOT_BASE_MS": 6.9, "TPOT_SLOPE_MS": 0.44}
    try: return with_total(scaled(sessions, OSL_F[c]), c, floor=0.19)
    finally: OVERRIDE = {}
variants.append(("+ decode = measured ITL (6.9 + 0.44·bs ms)", v_dec))

print(f"{'variant':46s} {'clients':>7s} {'req/s':>6s} {'out tok/s':>9s} {'out/GPU':>7s} {'tot/GPU':>7s} {'ttft p95':>8s} {'tpot ms':>7s} | sim/real: req/s  out  total  ttft")
for name, fn in variants:
    for c in (96, 192, 384, 480):
        m = fn(c); M = MEAS[c]
        if not m: print(name, c, "no result"); continue
        print(f"{name:46s} {c:7d} {m['req_per_s']:6.2f} {m['throughput_tok_s']:9.0f} {m['throughput_tok_s']/GPUS:7.1f} {m['tot_gpu']:7.0f} {m['ttft_p95_s']:8.2f} {m['tpot_mean_ms']:7.1f} | {m['req_per_s']/M['req_s']:5.2f}x {m['throughput_tok_s']/M['out']:5.2f}x {m['tot_gpu']/M['tot_gpu']:5.2f}x {m['ttft_p95_s']/M['ttft']:5.2f}x")
for c in (96, 192, 384, 480): print(f"{'measured (12:6 KV, AgentX mode)':46s} {c:7d} {MEAS[c]['req_s']:6.2f} {MEAS[c]['out']:9.0f} {MEAS[c]['out']/GPUS:7.1f} {MEAS[c]['tot_gpu']:7.0f} {MEAS[c]['ttft']:8.2f} {MEAS[c]['itl']:7.1f}")
