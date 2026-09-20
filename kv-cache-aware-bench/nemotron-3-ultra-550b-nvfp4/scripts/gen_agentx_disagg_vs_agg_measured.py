#!/usr/bin/env python3
"""Regenerate the MEASURED section (## 7) of AGENTX_DISAGG_VS_AGG.md from harvested AgentX cells.
disagg 8:8 (64 GPU): sim-results/measured_agentx_d88.json (written by gen_agentx_d88_report.py).
agg 6 x TP4 (24 GPU): every gs://alisachen-models/perf/*_alisachen-n3u-agg-ns{,2}-agentx-<pol>-c<N>/ cell, harvested on demand
(scripts/harvest_agentx_cell.sh); the latest run of a (policy, clients) pair wins.  disagg 12:6 (72 GPU): the five cells measured
on 2026-09-16 (AGENTX_D72_RESULTS.md).  Run after every harvest."""
import csv, glob, json, os, re, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = "/mnt/disks/scratch/agentx_recs"
ART = "https://console.cloud.google.com/storage/browser/alisachen-models/perf/"; SLO = 10.0
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout
NAME = {"kv": "default KV", "rr": "round-robin", "kvs3c08": "load scale 3 / credit 0.8", "kvs2c08": "load scale 2 / credit 0.8", "kvt05": "temperature 0.5",
        "kvt02": "temperature 0.2", "kvd05": "credit decay 0.5", "kvd10": "credit decay 1.0", "kvd20": "credit decay 2.0", "kvc15": "overlap credit 1.5",
        "kvc20": "overlap credit 2.0", "kvs3c08d05": "load scale 3 / credit 0.8 + decay 0.5", "kvs3c08d10": "load scale 3 / credit 0.8 + decay 1.0"}
# ---- agg cells
agg = {}
for d in sh("gsutil ls gs://alisachen-models/perf/").split():
    m = re.search(r"/(\d+)_alisachen-(n3u-agg-ns2?)-agentx-(\w+)-c(\d+)/$", d)
    if not m: continue
    ts, fleet, pol, c = m.group(1), m.group(2), m.group(3), int(m.group(4)); D = f"{H}/{fleet.split('-')[-1]}_{pol}_c{c}"
    have = os.path.exists(D + "/profile_export_aiperf.csv") and glob.glob(f"{H}/{ts}_alisachen-{fleet}-agentx-{pol}-c{c}_recs.jsonl")
    if not have: sh(f"{ROOT}/scripts/harvest_agentx_cell.sh {fleet} {pol} {c} 24")
    if not os.path.exists(D + "/profile_export_aiperf.csv"): continue
    rec = sorted(glob.glob(f"{H}/*_alisachen-{fleet}-agentx-{pol}-c{c}_recs.jsonl"))
    if not rec or not rec[-1].split("/")[-1].startswith(ts): continue          # the harvest dir holds the LATEST run of this cell only
    a = {r[0]: r for r in csv.reader(open(D + "/profile_export_aiperf.csv")) if r}
    try: tt, it = a["Time to First Token (ms)"], a["Inter Token Latency (ms)"]; tot = float(a["Total Token Throughput (tokens/sec)"][1]) / 24
    except (KeyError, ValueError): continue
    ck = ik = None
    for r in csv.reader(open(D + "/server_metrics_export.csv")):
        if len(r) > 6 and r[2] == "dynamo_frontend_cached_tokens": ck = float(r[6])
        if len(r) > 6 and r[2] == "dynamo_frontend_input_sequence_tokens": ik = float(r[6])
    old = agg.get((pol, c))
    if old and old["ts"] > ts: continue
    agg[(pol, c)] = dict(pol=pol, clients=c, ts=ts, art=f"{ts}_alisachen-{fleet}-agentx-{pol}-c{c}", tot=tot, p50=float(tt[9]) / 1e3, p95=float(tt[12]) / 1e3,
                         p90i=1000 / float(it[11]), hit=(ck / ik if ck and ik else float("nan")), gpus=24)
# post-knee agg cells (measured stationarity verdicts, AGENTX_AGG_RESULTS.md)
for k in (("kv", 384), ("rr", 384)):
    if k in agg: agg[k]["post"] = True
# ---- disagg 8:8
d88 = {(x["pol"], x["clients"]): dict(x, gpus=64, post=x["knee"].startswith("POST")) for x in json.load(open(f"{ROOT}/sim-results/measured_agentx_d88.json"))}
# ---- disagg 12:6 (72 GPU), measured 2026-09-16, default KV only
d126 = {("kv", c): dict(pol="kv", clients=c, tot=t, p50=p50, p95=p95, p90i=i, hit=h, gpus=72, art=a, ts=a.split("_")[0]) for c, t, p50, p95, i, h, a in (
    (96, 2471, 0.30, 1.37, 104, 0.94, "1789540700_alisachen-n3u-mnnvl-126-agentx-kv-c96"), (192, 4510, 0.36, 1.77, 89, 0.94, "1789555800_alisachen-n3u-mnnvl-126-agentx-kv-c192"),
    (384, 8637, 0.43, 2.58, 64, 0.92, "1789565000_alisachen-n3u-mnnvl-126-agentx-kv-c384"), (480, 10434, 0.50, 3.52, 57, 0.91, "1789574199_alisachen-n3u-mnnvl-126-agentx-kv-c480"),
    (768, 15004, 0.80, 7.00, 45, 0.88, "1789582655_alisachen-n3u-mnnvl-126-agentx-kv-c768"))}
for x in d126.values():
    real = sh(f"gsutil ls gs://alisachen-models/perf/ | grep 'mnnvl-126-agentx-kv-c{x['clients']}/' | tail -1").strip().rstrip("/").split("/")[-1]
    if real: x["art"] = real; x["ts"] = real.split("_")[0]
L = lambda x: f"[{x['ts']}]({ART}{x['art']})"
ok = lambda x: x["p95"] <= SLO and not x.get("post")
def best(cells, pols): return max((x for x in cells.values() if x["pol"] in pols and ok(x)), key=lambda x: x["tot"], default=None)
TUNED_AGG = [p for p in NAME if p not in ("kv", "rr")]; TUNED_D = TUNED_AGG
o = ["## 7. Measured comparison (regenerated from harvested cells by `scripts/gen_agentx_disagg_vs_agg_measured.py`)", "",
     f"All cells: AgentX scenario, 3,600 s window, total = (input + output) tok/s per GPU, TTFT standard = p95, SLO = TTFT p95 ≤ {SLO:g} s with a stationary",
     "knee check. Fleets: **agg** 6 × TP4 = 24 GPU; **disagg 8:8** 8 prefill + 8 decode TP4 = 64 GPU; **disagg 12:6** = 72 GPU (default KV only,",
     "measured 2026-09-16 before the pools were resized). Per-run logs: AGENTX_AGG_RESULTS.md, AGENTX_D88_RESULTS.md, AGENTX_D72_RESULTS.md, RUN_INDEX.md.", "",
     "### 7.1 Framing A — best throughput per GPU inside the same SLO (the number that ranks architectures)", "",
     "| fleet | policy | best cell inside the SLO | clients / GPU | total tok/s/GPU | TTFT p50 / p95 | P90 interactivity | hit rate | GPUs per 1,000 sessions | logs |", "|---|---|---|---|---|---|---|---|---|---|"]
rows = []
for fname, cells in (("agg 24 GPU", agg), ("disagg 8:8, 64 GPU", d88), ("disagg 12:6, 72 GPU", d126)):
    for label, pols in (("round-robin", ["rr"]), ("default KV", ["kv"]), ("tuned KV (best flag set)", TUNED_AGG)):
        b = best(cells, pols)
        if not b: continue
        lab = label if label != "tuned KV (best flag set)" else f"tuned KV ({NAME.get(b['pol'], b['pol'])})"
        cpg = b["clients"] / b["gpus"]; rows.append((fname, label, b))
        o.append(f"| {fname} | {lab} | {b['clients']} clients | {cpg:.1f} | **{b['tot']:,.0f}** | {b['p50']:.2f} / {b['p95']:.2f} s | {b['p90i']:.0f} | {b['hit']:.2f} | {1000 / cpg:,.0f} | {L(b)} |")
g = {(f, l): b for f, l, b in rows}
def ratio(a, b): return f"{a['tot'] / b['tot']:.2f}×" if a and b else "—"
o += ["", "Reading (same SLO):"]
for lab in ("default KV", "tuned KV (best flag set)", "round-robin"):
    A, Dd = g.get(("agg 24 GPU", lab)), g.get(("disagg 8:8, 64 GPU", lab))
    if A and Dd: o.append(f"- **{lab}**: disagg 8:8 {Dd['tot']:,.0f} vs agg {A['tot']:,.0f} tok/s/GPU = **{ratio(Dd, A)}** per GPU; clients per GPU {Dd['clients'] / 64:.1f} vs {A['clients'] / 24:.1f}"
                         f" → {1000 / (Dd['clients'] / 64):,.0f} vs {1000 / (A['clients'] / 24):,.0f} GPUs per 1,000 sessions; P90 interactivity {Dd['p90i']:.0f} vs {A['p90i']:.0f} tok/s/user.")
o += ["- The agg cells inside the SLO are the *measured* ones; a cell is only as close to the 10 s boundary as the ladder allows (see the TTFT p95 column — a value well",
      "  below 10 s means the true SLO point lies between that cell and the next one up, so the fleet's SLO throughput is a lower bound).", "",
      "### 7.2 Framing C — equal load per GPU (default KV)", "",
      "| clients / GPU | agg: clients → total/GPU · TTFT p95 · P90 | disagg 8:8: clients → total/GPU · TTFT p95 · P90 | disagg 12:6: clients → total/GPU · TTFT p95 · P90 | disagg 8:8 ÷ agg |", "|---|---|---|---|---|"]
def near(cells, cpg):
    c = [x for x in cells.values() if x["pol"] == "kv"]; return min(c, key=lambda x: abs(x["clients"] / x["gpus"] - cpg)) if c else None
fmt = lambda x, cpg: (f"{x['clients']} → {x['tot']:,.0f} · {x['p95']:.1f} s · {x['p90i']:.0f}" + (" (post-knee)" if x.get("post") else "")) if x and abs(x["clients"] / x["gpus"] - cpg) / cpg <= 0.15 else "—"
for cpg in (2, 3, 4, 6, 8, 10.5, 12, 16, 18):
    a_, d_, e_ = near(agg, cpg), near(d88, cpg), near(d126, cpg)
    fa, fd = fmt(a_, cpg), fmt(d_, cpg)
    r_ = ratio(d_, a_) if fa != '—' and fd != '—' and not a_.get('post') and not d_.get('post') else '—'
    o.append(f"| {cpg:g} | {fa} | {fd} | {fmt(e_, cpg)} | {r_} |")
ka = max((x for x in agg.values() if x["pol"] == "kv" and not x.get("post")), key=lambda x: x["tot"], default=None)
kd = max((x for x in d88.values() if x["pol"] == "kv" and not x.get("post")), key=lambda x: x["tot"], default=None)
o += ["", "Cells are matched to the nearest measured client count within ±15 % of the target load per GPU; “—” = no cell that close.", "",
      "### 7.3 Peak (stationary) throughput and where each fleet saturates", "",
      "| fleet | default KV peak (stationary) | clients / GPU at the peak | first post-knee cell | RR peak (stationary) |", "|---|---|---|---|---|"]
for fname, cells, gp in (("agg 24 GPU", agg, 24), ("disagg 8:8, 64 GPU", d88, 64)):
    pk = max((x for x in cells.values() if x["pol"] == "kv" and not x.get("post")), key=lambda x: x["tot"], default=None)
    po = min((x for x in cells.values() if x["pol"] == "kv" and x.get("post")), key=lambda x: x["clients"], default=None)
    rp = max((x for x in cells.values() if x["pol"] == "rr" and not x.get("post")), key=lambda x: x["tot"], default=None)
    o.append(f"| {fname} | {pk['tot']:,.0f} at {pk['clients']} clients (TTFT p95 {pk['p95']:.1f} s) | {pk['clients'] / gp:.1f} | {po['clients'] if po else '—'} clients ({po['clients'] / gp:.1f}/GPU) | {rp['tot']:,.0f} at {rp['clients']} clients (TTFT p95 {rp['p95']:.1f} s) |")
if ka and kd:
    o += ["", f"**Verdict from the measured cells.** Per GPU, disagg 8:8's stationary default-KV peak is {kd['tot']:,.0f} vs agg's {ka['tot']:,.0f} (**{kd['tot'] / ka['tot']:.2f}×**), reached at",
          f"{kd['clients'] / 64:.1f} vs {ka['clients'] / 24:.1f} clients per GPU. Agg is the better fleet only at light load per GPU, where its packed decode batches are fuller and no KV",
          "transfer is paid; from mid load upward disagg serves more tokens per GPU with a shorter TTFT tail *and* 2–3× the per-user interactivity, because decode runs on",
          "its own tier. The router matters as much as the architecture: inside the SLO round-robin gives up "
          + " and ".join(f"{(1 - g[(f, 'round-robin')]['tot'] / max(g[(f, l)]['tot'] for l in ('default KV', 'tuned KV (best flag set)') if (f, l) in g)) * 100:.0f} % on {f.split(',')[0]}" for f in ("agg 24 GPU", "disagg 8:8, 64 GPU") if (f, 'round-robin') in g) + " of the best KV configuration's throughput.",
          "The best flags differ by fleet — disagg wants *more* cache affinity (overlap credit 1.5; prefill is its bottleneck), agg wants *load awareness* (load scale 3 / credit 0.8;",
          "its problem is over-packed workers); credit decay and router temperature lose on both."]
s = open(f"{ROOT}/AGENTX_DISAGG_VS_AGG.md").read(); i = s.find("\n## 7. Measured comparison")
s = (s[:i] if i >= 0 else s.rstrip("\n")) + "\n\n" + "\n".join(o) + "\n"; open(f"{ROOT}/AGENTX_DISAGG_VS_AGG.md", "w").write(s)
print(f"AGENTX_DISAGG_VS_AGG.md section 7 regenerated: agg cells={len(agg)} disagg 8:8 cells={len(d88)}")
