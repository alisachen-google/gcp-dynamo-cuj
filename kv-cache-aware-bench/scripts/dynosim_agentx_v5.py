#!/usr/bin/env python3
"""AgentX simulator v5: trajectory-TREE replay on the stream-level trace (agentx_stream_trace.py output), following
aiperf agentic_replay (inferencex-agentx-mvp as launched by manifests/perf/sgl-d72-agentx.yaml):
  * lane = one trace sampled uniformly; t* = U(start_min, start_max) x trace duration  (job flags: 0.25-0.75)
  * every stream (root + each subagent chain) whose turns extend past t* is live: turns before t* are history; the
    last one before t* is PRIMED into the router-chosen worker's cache (warm-up); the first turn at/after t* fires at
    its recorded offset from t*; later turns fire at previous completion + recorded end-to-start delay
  * subagent streams spawned after t* start at spawn offset; when the root finishes and all children drained, the lane
    recycles a fresh trace at turn 0 (cold, salted = cache-bust)
  * output length = trace 'out' scaled by OSL_SCALE (engine-counted 0.81x of the trace field, from the records)
  * whole-system idle cap 10 s; window opens warm_s after the first profiled request
Engine unchanged: dynosim_agentx.Engine with dynosim_pd constants."""
import sys, json, random, heapq, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dynosim_agentx as da
dp = da.dp
OSL_SCALE = 1.0  # raw dataset out field matches the engine count (root 1,131 vs 1,240 measured; subagent 667 vs 694)

def load_stream_trace(path): return [json.loads(l) for l in open(path)]

def simulate_v5(traces, n_prefill, n_decode, policy, clients, window=3600.0, warm_s=900.0, idle_cap=10.0, start=(0.25, 0.75), agg=False, seed=42, router=None):
    rnd = random.Random(seed); eng = da.Engine(n_prefill, n_decode, policy, agg, router)
    ev = []; play = [0]; T0 = None; recs = []; inflight = 0; now = 0.0; lanes = {}
    def prime(hid, t):
        w = eng.rr % len(eng.P) if policy == "rr" else max(range(len(eng.P)), key=lambda i: -eng.P[i].load_blocks(t))
        eng.P[w].insert(hid)
    def new_play(L, t):
        tr = traces[rnd.randrange(len(traces))]; salt = play[0]; play[0] += 1
        # first play of a lane is a trajectory snapshot at t*; every recycle starts a fresh trace at turn 0 (aiperf: "lane recycles: a fresh session starting at turn 0")
        ts = (start[0] + rnd.random() * (start[1] - start[0])) * tr["duration_s"] if L not in lanes else 0.0
        if not lanes.get(L, {}).get("first_done", False): pass
        st = {"tr": tr, "salt": salt, "ts": ts, "streams": [], "live": 0, "first_done": True, "started": set(), "children_running": 0}
        lanes[L] = st
        for si, s in enumerate(tr["streams"]):
            turns = s["turns"]; k = next((i for i, q in enumerate(turns) if q["t"] >= ts), None)
            if k is None: continue                       # whole stream is history
            if k > 0: prime([(salt, h) for h in turns[k - 1]["hash_ids"]], t)
            sid = (L, si); st["streams"].append(sid); st["live"] += 1; st["started"].add(sid)
            if s["kind"] == "subagent": st["children_running"] += 1
            lanes[L] = st
            heapq.heappush(ev, (t + (turns[k]["delay"] or 0.0), "issue", sid, k))  # handoff carries the stream's next-turn delay, not its absolute offset
        if st["live"] == 0:  # degenerate sample: retry with a fresh trace from turn 0
            tr2 = traces[rnd.randrange(len(traces))]; st = {"tr": tr2, "salt": salt, "ts": 0.0, "streams": [], "live": 0, "first_done": True, "started": set(), "children_running": 0}
            for si, s in enumerate(tr2["streams"]):
                if not s["turns"]: continue
                sid = (L, si); st["streams"].append(sid); st["live"] += 1; lanes[L] = st
                heapq.heappush(ev, (t + s["turns"][0]["t"], "issue", sid, 0))
        lanes[L] = st
    for L in range(clients): new_play(L, rnd.random() * 30.0)
    while ev:
        t, kind, sid, k = heapq.heappop(ev)
        if kind == "issue" and inflight == 0 and t - now > idle_cap and now > 0:
            shift = (t - now) - idle_cap; t -= shift
            ev = [(tt - shift if kk == "issue" else tt, kk, ss, kk2) for (tt, kk, ss, kk2) in ev]; heapq.heapify(ev)
        now = max(now, t); L, si = sid; st = lanes[L]
        if kind == "done":
            pl = k; eng.release(pl["d"]); inflight -= 1
            if pl["measured"]: recs.append(pl)
            turns = st["tr"]["streams"][si]["turns"]; nxt = pl["k"] + 1; kind = st["tr"]["streams"][si]["kind"]
            if nxt < len(turns):
                t_prev, t_next = turns[pl["k"]]["t"], turns[nxt]["t"]
                if kind == "root":
                    # children whose recorded spawn falls inside this gap start now (spawn on parent completion) ...
                    spawned = False
                    for cj, cs in enumerate(st["tr"]["streams"]):
                        if cs["kind"] == "subagent" and t_prev <= cs["spawn_t"] < t_next and (L, cj) not in st["started"]:
                            st["started"].add((L, cj)); st["live"] += 1
                            # blocking child: its recorded end precedes the parent's next turn (the parent waited for it);
                            # background child (async_launched, ends after the parent's next turn) does not gate the join
                            c_end = cs["turns"][-1]["t"] + (cs["turns"][-1].get("api_time") or 0.0)
                            cs["_blocking"] = c_end <= t_next + 1e-3
                            if cs["_blocking"]: st["children_running"] += 1; spawned = True
                            heapq.heappush(ev, (now + (cs["turns"][0]["delay"] or 0.0), "issue", (L, cj), 0))
                    if spawned or st.get("children_running", 0) > 0:
                        st["pending_join"] = (sid, nxt)          # ... and the parent join turn fires when they drain
                    else:
                        heapq.heappush(ev, (now + (turns[nxt]["delay"] or 0.0), "issue", sid, nxt))
                else:
                    heapq.heappush(ev, (now + (turns[nxt]["delay"] or 0.0), "issue", sid, nxt))
            else:
                st["live"] -= 1
                if kind == "subagent" and st["tr"]["streams"][si].get("_blocking", True):
                    st["children_running"] = max(0, st.get("children_running", 1) - 1)
                    if st["children_running"] == 0 and st.get("pending_join"):
                        psid, pk = st.pop("pending_join"); heapq.heappush(ev, (now, "issue", psid, pk))
                if st["live"] <= 0 and not st.get("pending_join"): new_play(L, now)
            continue
        if st["tr"]["streams"][si]["turns"] is None: continue
        q = st["tr"]["streams"][si]["turns"][k]
        if T0 is None: T0 = now + warm_s
        if now > T0 + window: break
        hid = [(st["salt"], h) for h in q["hash_ids"]]; out_len = max(1, int(q["out"] * OSL_SCALE))
        ttft, tpot, done, d = eng.serve(hid, out_len, now); inflight += 1
        heapq.heappush(ev, (done, "done", sid, {"ttft": ttft, "tpot": tpot, "out": out_len, "inp": len(hid) * dp.BLOCK_TOKENS, "done": done, "start": now, "measured": now >= T0, "d": d, "k": k, "kind": st["tr"]["streams"][si]["kind"], "wait": eng.last[0], "svc": eng.last[1], "unc": eng.last[2]}))
    win = [x for x in recs if T0 <= x["start"] <= T0 + window]
    if not win: return None
    dur = window; tt = sorted(x["ttft"] for x in win); tp = sorted(x["tpot"] for x in win); out = sum(x["out"] for x in win); inp = sum(x["inp"] for x in win)
    P = lambda a, p: a[min(len(a) - 1, int(len(a) * p))]
    return {"n": len(win), "req_per_s": len(win) / dur, "throughput_tok_s": out / dur, "total_tok_s": (inp + out) / dur, "in_tok_per_req": inp / len(win), "out_per_req": out / len(win),
            "sub_share": sum(1 for x in win if x["kind"] == "subagent") / len(win), "ttft_p50_s": P(tt, .5), "ttft_p95_s": P(tt, .95), "tpot_mean_ms": sum(tp) / len(tp) * 1000, "tpot_p90_ms": P(tp, .9) * 1000,
            "hit_rate": eng.hits / max(1, eng.blocks), "wait_p95_s": P(sorted(x["wait"] for x in win), .95), "svc_p95_s": P(sorted(x["svc"] for x in win), .95), "unc_mean": sum(x["unc"] for x in win) / len(win)}

if __name__ == "__main__":
    traces = load_stream_trace(sys.argv[1]); cl = [int(c) for c in sys.argv[2].split(",")]
    print(f"{'variant':26s} {'clients':>7s} {'hit':>6s} {'req/s':>6s} {'ISL/req':>7s} {'OSL':>5s} {'sub%':>5s} {'tot/GPU':>8s} {'out/GPU':>7s} {'ttft p50':>8s} {'ttft p95':>8s} {'wait p95':>8s} {'tpot':>6s} {'P90':>6s}")
    for c in cl:
        dp.apply_n3u_constants(); m = simulate_v5(traces, 12, 6, "kv", c)
        print(f"{'v5d recycle at turn 0':26s} {c:7d} {m['hit_rate']:6.3f} {m['req_per_s']:6.2f} {m['in_tok_per_req']:7.0f} {m['out_per_req']:5.0f} {m['sub_share']*100:5.0f} {m['total_tok_s']/72:8.0f} {m['throughput_tok_s']/72:7.1f} {m['ttft_p50_s']:8.2f} {m['ttft_p95_s']:8.2f} {m['wait_p95_s']:8.2f} {m['tpot_mean_ms']:6.1f} {1000/m['tpot_p90_ms']:6.1f}", flush=True)
