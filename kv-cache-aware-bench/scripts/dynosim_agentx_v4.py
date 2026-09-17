#!/usr/bin/env python3
"""AgentX simulator v4: trajectory replay calibrated to aiperf's agentic_replay semantics (inferencex-agentx-mvp).
Differences from v3 (dynosim_agentx.simulate_agentx):
  * a lane holds S concurrent streams (root + subagent chains) instead of one sequential session
  * a stream = a trace session sampled uniformly, started at a uniform trajectory point t* (turn k = floor(u*n)) with
    its turn k-1 prefix PRIMED into the router-chosen worker's cache (aiperf's warm-up), not replayed back-to-back
  * inter-turn gaps are sampled from the measured replayed think-gap CDF (sim-results/agentx_gap_cdf.json) instead of
    the raw trace timestamps (whose gaps average 618 s and are not what aiperf replays)
  * the window opens warm_s after the first profiled request (aiperf warm-up), whole-system idle cap kept
Engine (prefill/decode/cache/router) is unchanged: dynosim_agentx.Engine with dynosim_pd constants."""
import sys, json, random, heapq, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dynosim_agentx as da
dp = da.dp
GAPQ = json.load(open(pathlib.Path(__file__).resolve().parents[1] / "nemotron-3-ultra-550b-nvfp4/sim-results/agentx_gap_cdf.json"))["quantiles_0_to_1_step_0.001"]

def simulate_v4(sessions, n_prefill, n_decode, policy, clients, streams_per_lane=2, window=3600.0, warm_s=900.0, idle_cap=10.0, agg=False, seed=42, router=None):
    rnd = random.Random(seed); eng = da.Engine(n_prefill, n_decode, policy, agg, router)
    ev = []; play = [0]; T0 = None; recs = []; inflight = 0; now = 0.0
    def gap(): return GAPQ[min(1000, int(rnd.random() * 1000))]
    def new_stream(sid, t):
        s = sessions[rnd.randrange(len(sessions))]; n = len(s); k = min(n - 1, int(rnd.random() * n)); salt = play[0]; play[0] += 1
        st = {"s": s, "i": k, "salt": salt}
        if k > 0:  # prime the prefix of turn k-1 on the worker the router would choose (aiperf warm-up)
            hid = [(salt, h) for h in s[k - 1]["hash_ids"]]
            w = eng.rr % len(eng.P) if policy == "rr" else max(range(len(eng.P)), key=lambda i: -eng.P[i].load_blocks(t))
            eng.P[w].insert(hid)
        heapq.heappush(ev, (t + rnd.random() * 30.0, "issue", sid, None))  # spread stream starts over 30 s
        return st
    streams = {}
    for L in range(clients):
        for j in range(streams_per_lane):
            sid = L * streams_per_lane + j; streams[sid] = new_stream(sid, 0.0)
    while ev:
        t, kind, sid, pl = heapq.heappop(ev)
        if kind == "issue" and inflight == 0 and t - now > idle_cap and now > 0:
            shift = (t - now) - idle_cap; t -= shift
            ev = [(tt - shift if kk == "issue" else tt, kk, ss, pp) for (tt, kk, ss, pp) in ev]; heapq.heapify(ev)
        now = max(now, t)
        if kind == "done":
            eng.release(pl["d"]); inflight -= 1
            if pl["measured"]: recs.append(pl)
            st = streams[sid]; st["i"] += 1
            if st["i"] >= len(st["s"]): streams[sid] = new_stream(sid, now); continue
            heapq.heappush(ev, (now + gap(), "issue", sid, None)); continue
        st = streams[sid]; r = st["s"][st["i"]]
        if T0 is None: T0 = now + warm_s
        if now > T0 + window: break
        hid = [(st["salt"], h) for h in r["hash_ids"]]
        ttft, tpot, done, d = eng.serve(hid, r["output_length"], now); inflight += 1
        heapq.heappush(ev, (done, "done", sid, {"ttft": ttft, "tpot": tpot, "out": r["output_length"], "inp": len(hid) * dp.BLOCK_TOKENS, "done": done, "start": now, "measured": T0 is not None and now >= T0, "d": d, "wait": eng.last[0], "svc": eng.last[1], "unc": eng.last[2]}))
    win = [x for x in recs if T0 <= x["start"] <= T0 + window]
    if not win: return None
    dur = window; tt = sorted(x["ttft"] for x in win); tp = sorted(x["tpot"] for x in win); out = sum(x["out"] for x in win); inp = sum(x["inp"] for x in win)
    P = lambda a, p: a[min(len(a) - 1, int(len(a) * p))]
    return {"n": len(win), "req_per_s": len(win) / dur, "throughput_tok_s": out / dur, "total_tok_s": (inp + out) / dur, "in_tok_per_req": inp / len(win), "out_per_req": out / len(win),
            "ttft_p50_s": P(tt, .5), "ttft_p95_s": P(tt, .95), "tpot_mean_ms": sum(tp) / len(tp) * 1000, "tpot_p90_ms": P(tp, .9) * 1000, "hit_rate": eng.hits / max(1, eng.blocks),
            "wait_p95_s": P(sorted(x["wait"] for x in win), .95), "svc_p95_s": P(sorted(x["svc"] for x in win), .95), "unc_mean": sum(x["unc"] for x in win) / len(win)}

if __name__ == "__main__":
    trace = sys.argv[1]; cl = [int(c) for c in sys.argv[2].split(",")]; S_list = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "1,2").split(",")]
    sess = da.load_sessions(trace, limit=10**9)
    print(f"{'variant':30s} {'clients':>7s} {'hit':>6s} {'req/s':>6s} {'ISL/req':>7s} {'OSL':>5s} {'tot/GPU':>8s} {'out/GPU':>7s} {'ttft p50':>8s} {'ttft p95':>8s} {'wait p95':>8s} {'tpot':>6s} {'P90':>6s}")
    for c in cl:
        for S in S_list:
            dp.apply_n3u_constants(); m = simulate_v4(sess, 12, 6, "kv", c, streams_per_lane=S)
            print(f"{'v4 S=%d streams/lane' % S:30s} {c:7d} {m['hit_rate']:6.3f} {m['req_per_s']:6.2f} {m['in_tok_per_req']:7.0f} {m['out_per_req']:5.0f} {m['total_tok_s']/72:8.0f} {m['throughput_tok_s']/72:7.1f} {m['ttft_p50_s']:8.2f} {m['ttft_p95_s']:8.2f} {m['wait_p95_s']:8.2f} {m['tpot_mean_ms']:6.1f} {1000/m['tpot_p90_ms']:6.1f}", flush=True)
