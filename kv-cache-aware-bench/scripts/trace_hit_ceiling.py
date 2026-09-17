#!/usr/bin/env python3
"""Prefix-hit-rate ceiling of a hash-id trace: the best hit rate any router + cache can reach, assuming an infinite
cache and perfect routing, with the same prefix-stop semantics as the simulator (dynosim_pd.PrefillWorker.overlap_blocks)
and the engine's cached-token counter.  Sessions are grouped exactly as dynosim_agentx.load_sessions does (by the first
hash id), each session's turns are replayed in timestamp order, and a turn's hit is the longest prefix of its hash ids
already inserted by earlier turns of the same session (the first turn of every session is cold: the real scenario
cache-busts it and the simulator salts every replay).

usage: trace_hit_ceiling.py <trace.jsonl> [<trace.jsonl> ...] [--limit N]
Rows are {"timestamp","input_length","output_length","hash_ids":[...]} (64-token blocks)."""
import sys, json
from collections import defaultdict

def ceiling(path, limit=None):
    rows = [json.loads(l) for l in open(path)]
    if limit: rows = rows[:limit]
    sess = defaultdict(list)
    for r in rows: sess[r["hash_ids"][0] if r["hash_ids"] else -1].append(r)
    tot = hit = cold_tok = 0; turns = []
    for v in sess.values():
        v = sorted(v, key=lambda r: r["timestamp"]); seen = set(); turns.append(len(v))
        for i, r in enumerate(v):
            h = r["hash_ids"]; ov = 0
            for b in h:
                if b in seen: ov += 1
                else: break                      # prefix property: stop at the first miss
            tot += len(h); hit += ov
            if i == 0: cold_tok += len(h)
            seen.update(h)
    turns.sort()
    return dict(requests=len(rows), sessions=len(sess), turns_mean=sum(turns)/len(turns), turns_median=turns[len(turns)//2],
                cold_token_share=cold_tok/tot, hit_ceiling=hit/tot, mean_isl=tot*64/len(rows))

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit="): limit = int(a.split("=")[1])
    print(f"{'trace':40s} {'req':>6s} {'sessions':>8s} {'turns mean/med':>14s} {'cold tok':>8s} {'HIT CEILING':>11s} {'mean ISL':>8s}")
    for p in args:
        c = ceiling(p, limit)
        print(f"{p.split('/')[-1]:40s} {c['requests']:6d} {c['sessions']:8d} {c['turns_mean']:8.1f}/{c['turns_median']:<5d} {c['cold_token_share']:8.3f} {c['hit_ceiling']:11.3f} {c['mean_isl']:8.0f}")
