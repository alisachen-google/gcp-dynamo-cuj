#!/usr/bin/env python3
"""Convert the raw SemiAnalysis cc-traces-weka dataset (one JSON object per trace: root requests + nested subagent entries)
into the stream-level trace aiperf's agentic_replay actually replays, so the simulator can respect workload parity:
  trace -> {"id", "duration_s", "streams": [ {"kind": "root"|"subagent", "spawn_t": s, "turns": [ {"t": s, "in": tokens,
            "out": tokens, "hash_ids": [...], "api_time": s, "delay": end-to-start s from previous turn (None for first)} ]} ]}
Delay rule = aiperf weka_trace._end_to_start_delay_ms: max(0, t_k - (t_{k-1} + api_{k-1})).  Subagent inner request
timestamps are absolute (root-trace coordinates) unless they precede the spawn marker (then relative), as in the loader.
Hash ids are local to the trace (hash_id_scope=local); the simulator salts them per trace id and per play.
usage: agentx_stream_trace.py <traces.jsonl> <out.jsonl>"""
import sys, json, math
def delay(prev, cur):
    if prev is None: return None
    api = prev.get("api_time"); api = api if isinstance(api, (int, float)) and math.isfinite(api) else 0.0
    return max(0.0, cur["t"] - (prev["t"] + max(api, 0.0)))
n_tr = n_root = n_sub = n_turn = 0
with open(sys.argv[2], "w") as out:
    for line in open(sys.argv[1]):
        tr = json.loads(line); n_tr += 1
        root = []; subs = []
        for r in tr["requests"]:
            if r.get("type") == "subagent":
                inner = []
                for q in r.get("requests", []):
                    if q.get("type") == "subagent": continue
                    t = q["t"] + r["t"] if q["t"] + 1e-3 < r["t"] else q["t"]
                    inner.append({"t": t, "in": q["in"], "out": q["out"], "hash_ids": q["hash_ids"], "api_time": q.get("api_time")})
                inner.sort(key=lambda x: x["t"])
                if inner: subs.append({"kind": "subagent", "spawn_t": r["t"], "turns": inner})
            else:
                root.append({"t": r["t"], "in": r["in"], "out": r["out"], "hash_ids": r["hash_ids"], "api_time": r.get("api_time")})
        root.sort(key=lambda x: x["t"])
        streams = ([{"kind": "root", "spawn_t": 0.0, "turns": root}] if root else []) + subs
        for s in streams:
            prev = None
            for q in s["turns"]:
                q["delay"] = delay(prev, q); prev = q; n_turn += 1
        n_root += 1 if root else 0; n_sub += len(subs)
        dur = max([q["t"] + (q.get("api_time") or 0) for s in streams for q in s["turns"]] + [0.0])
        out.write(json.dumps({"id": tr["id"], "duration_s": dur, "block_size": tr.get("block_size", 64), "streams": streams}) + "\n")
print(f"traces={n_tr} root_streams={n_root} subagent_streams={n_sub} turns={n_turn}")
