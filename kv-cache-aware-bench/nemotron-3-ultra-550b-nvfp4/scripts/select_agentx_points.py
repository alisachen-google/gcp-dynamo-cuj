#!/usr/bin/env python3
"""Pick the KV-vs-RR comparison cells from MEASURED AgentX ladders and emit the flag-sweep points.
Reads each finished cell's aiperf summary CSV from gs://alisachen-models/perf/*_alisachen-<jobprefix>-agentx-<pol>-c<N>/ and
the runner log's KNEE-CHECK verdicts, then applies the rules used throughout the study (TTFT standard = p95):
  same config : RR's throughput-peak cell (largest RR total tok/s/GPU among stationary-or-not RR cells) if KV ran it too,
                else the largest client count both policies ran
  same SLO    : per policy, the highest-throughput cell with TTFT p95 <= --ttft (20 s), P90 interactivity >= --p90 (20)
                and a stationary knee verdict
  flag sweep  : all variants at the KV same-SLO cell, plus the tuned router at the same-config cell
usage: select_agentx_points.py <jobprefix> <gpus> <kv clients csv> <rr clients csv> --logs <runner logs...> [--out-md f] [--out-points f]"""
import sys, csv, io, re, subprocess, argparse
ap = argparse.ArgumentParser(); ap.add_argument("jp"); ap.add_argument("gpus", type=int); ap.add_argument("kv"); ap.add_argument("rr")
ap.add_argument("--logs", nargs="*", default=[]); ap.add_argument("--ttft", type=float, default=20.0); ap.add_argument("--p90", type=float, default=20.0)
ap.add_argument("--out-json", default=""); ap.add_argument("--ttft-fallback", type=float, default=0.0); ap.add_argument("--out-md", default=""); ap.add_argument("--out-points", default=""); ap.add_argument("--variants", default="kvs3c08,kvs2c08,kvt05"); a = ap.parse_args()
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
knee = {}
for lg in a.logs:
    try:
        for l in open(lg):
            m = re.search(r"KNEE-CHECK (\S+)-agentx-(\w+)-c(\d+): (\S+)", l)
            if m: knee[(m.group(2), int(m.group(3)))] = m.group(4)
    except FileNotFoundError: pass
def cell(pol, c):
    d = sh(f"gcloud storage ls gs://alisachen-models/perf/ 2>/dev/null | grep '_alisachen-{a.jp}-agentx-{pol}-c{c}/' | tail -1").strip()
    if not d: return None
    f = sh(f"gcloud storage ls -r '{d}' 2>/dev/null | grep profile_export_aiperf.csv | grep -v warmup | head -1").strip()
    if not f: return None
    rows = list(csv.reader(io.StringIO(sh(f"gcloud storage cat '{f}' 2>/dev/null")))); hdr = None; m = {}
    for r in rows:
        if not r: continue
        if r[0] == "Metric" and len(r) > 2: hdr = r; continue
        if r[0] == "Metric": hdr = None; continue
        if hdr and len(r) == len(hdr): m[r[0]] = dict(zip(hdr[1:], r[1:]))
        elif len(r) == 2: m[r[0]] = {"Value": r[1]}
    g = lambda k, col: float(m.get(k, {}).get(col) or "nan")
    return {"pol": pol, "clients": c, "tot": g("Effective Total Throughput (tokens/sec)", "avg") / a.gpus, "out": g("Output Token Throughput (tokens/sec)", "Value") / a.gpus,
            "p50": g("Time to First Token (ms)", "p50") / 1000, "p95": g("Time to First Token (ms)", "p95") / 1000, "p90i": 1000 / g("Inter Token Latency (ms)", "p90"),
            "knee": knee.get((pol, c), "?"), "art": d.rstrip("/").split("/")[-1]}
cells = [x for x in ([cell("kv", int(c)) for c in a.kv.split(",") if c] + [cell("rr", int(c)) for c in a.rr.split(",") if c]) if x]
kv = {x["clients"]: x for x in cells if x["pol"] == "kv"}; rr = {x["clients"]: x for x in cells if x["pol"] == "rr"}
ok = lambda x: x["p95"] <= a.ttft and x["p90i"] >= a.p90 and not x["knee"].startswith("POST")
best = lambda d: max((x for x in d.values() if ok(x)), key=lambda x: x["tot"], default=None)
rr_peak = max(rr.values(), key=lambda x: x["tot"], default=None); both = sorted(set(kv) & set(rr))
same_cfg = rr_peak["clients"] if rr_peak and rr_peak["clients"] in kv else (both[-1] if both else None)
kv_slo, rr_slo = best(kv), best(rr); rr_slo_ttft = a.ttft
if rr_slo is None and a.ttft_fallback:   # no RR cell inside the SLO: relax RR's budget to the fallback (reported as such)
    okf = lambda x: x["p95"] <= a.ttft_fallback and x["p90i"] >= a.p90 and not x["knee"].startswith("POST")
    rr_slo = max((x for x in rr.values() if okf(x)), key=lambda x: x["tot"], default=None); rr_slo_ttft = a.ttft_fallback
pts = []
if kv_slo: pts += [f"{v}:{kv_slo['clients']}" for v in a.variants.split(",")]
if same_cfg and (not kv_slo or same_cfg != kv_slo["clients"]): pts.append(f"kvs3c08:{same_cfg}")
md = ["| policy | clients | total tok/s/GPU | output/GPU | TTFT p50 / p95 | P90 interactivity | knee | artifact |", "|---|---|---|---|---|---|---|---|"]
for x in sorted(cells, key=lambda x: (x["pol"], x["clients"])): md.append(f"| {x['pol']} | {x['clients']} | {x['tot']:,.0f} | {x['out']:.1f} | {x['p50']:.2f} / {x['p95']:.2f} s | {x['p90i']:.1f} | {x['knee']} | {x['art']} |")
md.append("")
if same_cfg: md.append(f"**Same config: {same_cfg} clients** — KV {kv[same_cfg]['tot']:,.0f} vs RR {rr[same_cfg]['tot']:,.0f} = {kv[same_cfg]['tot']/rr[same_cfg]['tot']:.2f}x; TTFT p95 {kv[same_cfg]['p95']:.1f} vs {rr[same_cfg]['p95']:.1f} s; P90 {kv[same_cfg]['p90i']:.0f} vs {rr[same_cfg]['p90i']:.0f}.")
if kv_slo and rr_slo: md.append(f"**Same SLO (KV TTFT p95 <= {a.ttft:g} s, RR <= {rr_slo_ttft:g} s, P90 >= {a.p90:g}, stationary): KV {kv_slo['clients']} -> {kv_slo['tot']:,.0f} vs RR {rr_slo['clients']} -> {rr_slo['tot']:,.0f} = {kv_slo['tot']/rr_slo['tot']:.2f}x.**")
md.append(f"Flag-sweep points: {' '.join(pts) or '(none: no KV cell met the SLO)'}")
out = "\n".join(md); print(out)
if a.out_md: open(a.out_md, "w").write(out + "\n")
if a.out_points: open(a.out_points, "w").write(" ".join(pts) + "\n")
if a.out_json:
    import json
    json.dump({"rr_slo_ttft": rr_slo_ttft, "rr_peak": rr_peak and rr_peak["clients"], "same_cfg": same_cfg, "kv_slo": kv_slo, "rr_slo": rr_slo, "kv": kv, "rr": rr}, open(a.out_json, "w"), indent=1)
