#!/usr/bin/env python3
"""Pick the winning KV-router flag variant at one client count from MEASURED cells.
Winner = highest total tok/s/GPU among variants whose KNEE-CHECK is not POST-KNEE and whose TTFT p95 is not worse than
default KV's; it must beat default KV by > --min-gain (2 %), else prints 'kv' (no tuned winner).
Prints: <winner> <total/GPU> <ttft_p95_s>
usage: pick_agentx_flag_winner.py <jobprefix> <gpus> <clients> <variants csv> --logs <runner logs...>"""
import sys, csv, io, re, subprocess, argparse
ap = argparse.ArgumentParser(); ap.add_argument("jp"); ap.add_argument("gpus", type=int); ap.add_argument("clients", type=int); ap.add_argument("variants")
ap.add_argument("--logs", nargs="*", default=[]); ap.add_argument("--min-gain", type=float, default=0.02); a = ap.parse_args()
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout
knee = {}
for lg in a.logs:
    try:
        for l in open(lg):
            m = re.search(r"KNEE-CHECK (\S+)-agentx-(\w+)-c(\d+): (\S+)", l)
            if m: knee[(m.group(2), int(m.group(3)))] = m.group(4)
    except FileNotFoundError: pass
def cell(pol):
    d = sh(f"gcloud storage ls gs://alisachen-models/perf/ 2>/dev/null | grep '_alisachen-{a.jp}-agentx-{pol}-c{a.clients}/' | tail -1").strip()
    f = d and sh(f"gcloud storage ls -r '{d}' 2>/dev/null | grep profile_export_aiperf.csv | grep -v warmup | head -1").strip()
    if not f: return None
    m = {r[0]: r for r in csv.reader(io.StringIO(sh(f"gcloud storage cat '{f}' 2>/dev/null"))) if r}
    try: return {"pol": pol, "tot": float(m["Total Token Throughput (tokens/sec)"][1]) / a.gpus, "p95": float(m["Time to First Token (ms)"][12]) / 1000, "knee": knee.get((pol, a.clients), "?")}
    except (KeyError, ValueError, IndexError): return None
base = cell("kv"); cand = [c for c in (cell(v) for v in a.variants.split(",") if v) if c]
for c in [base] + cand:
    if c: print(f"# {c['pol']}: {c['tot']:,.0f} tok/s/GPU, TTFT p95 {c['p95']:.2f} s, {c['knee']}", file=sys.stderr)
good = [c for c in cand if not c["knee"].startswith("POST") and (not base or (c["tot"] > base["tot"] * (1 + a.min_gain) and c["p95"] <= base["p95"]))]
w = max(good, key=lambda c: c["tot"], default=None)
print(f"{w['pol']} {w['tot']:.0f} {w['p95']:.2f}" if w else f"kv {base['tot'] if base else 0:.0f} {base['p95'] if base else 0:.2f}")
