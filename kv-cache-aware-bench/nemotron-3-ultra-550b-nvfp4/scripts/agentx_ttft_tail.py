#!/usr/bin/env python3
"""Decompose a measured AgentX TTFT tail from the aiperf per-request records (profile_export.jsonl):
  - cold first turns (turn_index == 0, cache-busted): linear fit TTFT = a + ISL / rate  -> real uncached prefill rate
  - warm turns split by in-flight-at-issue quartile                                    -> queue-driven or size-driven?
  - composition of the p95+ tail: cold share, median ISL, median in-flight                -> what the tail is made of
usage: agentx_ttft_tail.py <records.jsonl> [...]"""
import sys, json, bisect, statistics as st
def v(m, k):
    x = m.get(k); return x.get("value") if isinstance(x, dict) else x
def pct(a, p): a = sorted(a); return a[min(len(a) - 1, int(len(a) * p))]
for f in sys.argv[1:]:
    recs = [json.loads(l) for l in open(f)]; recs = [r for r in recs if r["metadata"].get("benchmark_phase") == "profiling"]
    for r in recs:
        r["_t"] = (v(r["metrics"], "time_to_first_token") or 0) / 1000; r["_isl"] = v(r["metrics"], "usage_prompt_tokens") or 0
        r["_turn"] = r["metadata"].get("turn_index", -1); r["_s"] = r["metadata"]["request_start_ns"]; r["_e"] = r["metadata"]["request_end_ns"]
    starts = sorted(r["_s"] for r in recs); ends = sorted(r["_e"] for r in recs)
    for r in recs: r["_infl"] = bisect.bisect_left(starts, r["_s"]) - bisect.bisect_right(ends, r["_s"])
    cold = [r for r in recs if r["_turn"] == 0]; warm = [r for r in recs if r["_turn"] > 0]
    xs = [r["_isl"] for r in cold]; ys = [r["_t"] for r in cold]; n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs); ic = my - sl * mx
    big = [r for r in cold if r["_isl"] > 100000]
    q = sorted((r["_infl"], r["_t"]) for r in warm); lo = [t for i, t in q[:len(q) // 4]]; hi = [t for i, t in q[-len(q) // 4:]]
    thr = pct([r["_t"] for r in recs], .95); tail = [r for r in recs if r["_t"] >= thr]
    print(f"{f.split('/')[-1]}: n={len(recs)} cold={len(cold)} warm={len(warm)}")
    print(f"  cold turns: TTFT = {ic:.2f} s + ISL / {1/sl:,.0f} tok/s ; cold >100k tok: n={len(big)} p50 {pct([r['_t'] for r in big],.5):.2f} s p95 {pct([r['_t'] for r in big],.95):.2f} s")
    print(f"  warm turns: p50 {pct([r['_t'] for r in warm],.5):.2f} p95 {pct([r['_t'] for r in warm],.95):.2f} | lowest in-flight quartile p95 {pct(lo,.95):.2f} | highest quartile p95 {pct(hi,.95):.2f}")
    print(f"  p95 tail (>= {thr:.2f} s): {len(tail)} req, {sum(1 for r in tail if r['_turn']==0)/len(tail):.0%} cold, median ISL {st.median([r['_isl'] for r in tail]):,.0f}, median in-flight {st.median([r['_infl'] for r in tail]):.0f} (overall {st.median([r['_infl'] for r in recs]):.0f})")
