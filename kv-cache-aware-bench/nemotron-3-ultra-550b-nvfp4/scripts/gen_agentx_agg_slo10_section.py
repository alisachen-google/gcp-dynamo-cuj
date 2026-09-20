#!/usr/bin/env python3
"""(Re)generate section 'vi' of AGENTX_AGG_RESULTS.md: agg cells that saturate the TTFT p95 <= 10 s SLO, the KV-vs-RR same-SLO comparison
and the KV flag sweep at the selected KV 10 s point.  Reads the cells harvested by scripts/harvest_agentx_cell.sh and the per-fleet
runner logs for the knee verdicts; also appends new runs to RUN_INDEX.md.  Run after every harvest."""
import csv, glob, json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); H = "/mnt/disks/scratch/agentx_recs"; G = 24; SLO = 10.0
ART = "https://console.cloud.google.com/storage/browser/alisachen-models/perf/"; GH = "https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench"
NAME = {"kv": "default KV", "rr": "round-robin", "kvs3c08": "load scale 3, credit 0.8", "kvs2c08": "load scale 2, credit 0.8", "kvt05": "router temperature 0.5",
        "kvd05": "credit decay 0.5", "kvd10": "credit decay 1.0", "kvs3c08d05": "load scale 3, credit 0.8 + decay 0.5"}
FLAGS = {"kv": "`--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs`", "rr": "`--router-mode round-robin`",
         "kvs3c08": "kv + `--router-prefill-load-scale 3.0 --router-kv-overlap-score-credit 0.8`", "kvs2c08": "kv + `--router-prefill-load-scale 2.0 --router-kv-overlap-score-credit 0.8`",
         "kvt05": "kv with `--router-temperature 0.5`", "kvd05": "kv + `--router-kv-overlap-score-credit-decay 0.5`", "kvd10": "kv + `--router-kv-overlap-score-credit-decay 1.0`",
         "kvs3c08d05": "kv + load scale 3.0, credit 0.8, decay 0.5"}
knee = {}
for f in glob.glob("/tmp/agentx_agg_n3u-agg-ns*.log"):
    for m in re.finditer(r"KNEE-CHECK (n3u-agg-ns2?)-agentx-(\w+)-c(\d+): (\S+) \|.*?q1=([\d.]+)s q4=([\d.]+)s", open(f, errors="ignore").read()):
        knee[(m.group(2), int(m.group(3)))] = (m.group(4), m.group(5), m.group(6))
def load(fleet, pol, c):
    D = f"{H}/{fleet.split('-')[-1]}_{pol}_c{c}"; rec = sorted(glob.glob(f"{H}/*_alisachen-{fleet}-agentx-{pol}-c{c}_recs.jsonl"))
    if not rec or not os.path.exists(D + "/profile_export_aiperf.csv"): return None
    art = os.path.basename(rec[-1]).replace("_recs.jsonl", ""); a = {r[0]: r for r in csv.reader(open(D + "/profile_export_aiperf.csv")) if r}
    tt, it = a["Time to First Token (ms)"], a["Inter Token Latency (ms)"]; ck = ik = None
    for r in csv.reader(open(D + "/server_metrics_export.csv")):
        if len(r) > 6 and r[2] == "dynamo_frontend_cached_tokens": ck = float(r[6])
        if len(r) > 6 and r[2] == "dynamo_frontend_input_sequence_tokens": ik = float(r[6])
    ev, wl, ws = [], [], []
    for l in open(rec[-1]):
        try: md = json.loads(l)["metadata"]
        except Exception: continue
        s, e = md.get("request_start_ns"), md.get("request_end_ns")
        if not s or not e: continue
        if md.get("benchmark_phase") == "warmup": ws += [s, e]
        elif md.get("benchmark_phase") == "profiling": ev += [(s, 1), (e, -1)]
    ev.sort(); cur = area = peak = 0; prev = ev[0][0]
    for t, d in ev: area += cur * (t - prev); prev = t; cur += d; peak = max(peak, cur)
    out = float(a["Output Token Throughput (tokens/sec)"][1]); k = knee.get((pol, c), ("?", "", ""))
    return dict(pol=pol, clients=c, fleet=fleet, art=art, ts=art.split("_")[0], tot=float(a["Total Token Throughput (tokens/sec)"][1]) / G, out=out, p50=float(tt[9]) / 1e3, p95=float(tt[12]) / 1e3,
                p99=float(tt[13]) / 1e3, itl50=float(it[9]), itl90=float(it[11]), p90i=1000 / float(it[11]), infl=area / (ev[-1][0] - ev[0][0]), peak=peak, hit=ck / ik,
                wu=(max(ws) - min(ws)) / 1e9 if ws else 0, knee=k[0], kq=f"q1 {k[1]} → q4 {k[2]} s" if k[1] else "")
cells = {}
for d in glob.glob(f"{H}/ns*_*_c*"):
    m = re.match(r"(ns2?)_(\w+)_c(\d+)$", os.path.basename(d)); 
    if not m: continue
    x = load("n3u-agg-" + m.group(1), m.group(2), int(m.group(3)))
    if x and (x["pol"], x["clients"]) not in cells or (x and x["ts"] > cells[(x["pol"], x["clients"])]["ts"]): cells[(x["pol"], x["clients"])] = x
L = lambda x: f"[{x['ts']}]({ART}{x['art']})"
def row(x, base=None):
    d = f" ({(x['tot'] / base['tot'] - 1) * 100:+.1f}%)" if base and x is not base else ""
    dt = f" ({(x['p95'] / base['p95'] - 1) * 100:+.0f}%)" if base and x is not base else ""
    st = "**POST-KNEE**" if x["knee"].startswith("POST") else (f"yes ({x['kq']})" if x["kq"] else "?")
    return f"| {x['tot']:,.0f}{d} | {x['out']:,.0f} ({x['out'] / G:.1f}) | {x['p50']:.2f} / {x['p95']:.2f}{dt} / {x['p99']:.1f} s | {x['itl50']:.1f} / {x['itl90']:.1f} → {x['p90i']:.1f} | {x['infl']:.1f} (peak {x['peak']}) | {x['hit']:.2f} | {x['wu']:,.0f} s | {st} | {L(x)} |"
kvpt = max((x for x in cells.values() if x["pol"] == "kv" and x["p95"] <= SLO and x["clients"] > 96), key=lambda x: x["clients"], default=None)
PT = kvpt["clients"] if kvpt else None
o = ["## vi. Cells that saturate the TTFT p95 ≤ 10 s SLO, and the KV flag sweep at that point (np-2 fleets, 2026-09-20)", "",
     "The §iii ladder brackets 10 s but has no cell near it (default KV: 96 → 5.4 s, 192 → 11.2–11.7 s; RR: 48 → 8.3 s, 96 → 12.6 s), so the same-SLO",
     "comparison under-sold every policy. These cells sit just inside the budget. Two agg fleets in parallel on np-2",
     f"([`run_agentx_agg_slo10_np2_v2.sh`]({GH}/nemotron-3-ultra-550b-nvfp4/scripts/run_agentx_agg_slo10_np2_v2.sh); manifests [`n3u-agg-newstack-np2.yaml`]({GH}/sglang/manifests/n3u-agg-newstack-np2.yaml) /",
     f"[`n3u-agg-newstack2-np2.yaml`]({GH}/sglang/manifests/n3u-agg-newstack2-np2.yaml); job template [`sgl-d72-agentx.yaml`]({GH}/manifests/perf/sgl-d72-agentx.yaml); runner + flag variants",
     f"[`agentx_runner_flags.sh`]({GH}/nemotron-3-ultra-550b-nvfp4/scripts/agentx_runner_flags.sh)); same AgentX scenario, 3,600 s window, 24 GPUs.", "",
     "### Near-saturated cells per policy", "",
     "| policy | clients | router flags | total tok/s/GPU | output tok/s (/GPU) | TTFT p50 / p95 / p99 | ITL p50 / p90 → P90 | in-flight | hit rate | warm-up wall | stationary | logs + artifacts |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
sel = [x for x in (cells.get(("rr", 64)), kvpt, cells.get(("kv", 176)), cells.get(("kv", 144)), cells.get(("kvs3c08", 256))) if x]
seen = set()
for x in sel:
    if id(x) in seen: continue
    seen.add(id(x)); o.append(f"| {NAME.get(x['pol'], x['pol'])} | {x['clients']} | {FLAGS.get(x['pol'], '')} " + row(x))
rr = max((x for x in cells.values() if x["pol"] == "rr" and x["p95"] <= SLO and not x["knee"].startswith("POST")), key=lambda x: x["tot"], default=None)
tn = max((x for x in cells.values() if x["pol"] not in ("kv", "rr") and x["p95"] <= SLO and not x["knee"].startswith("POST")), key=lambda x: x["tot"], default=None)
o += ["", f"### KV vs RR inside the same SLO (TTFT p95 ≤ {SLO:g} s, stationary)", "", "| policy | cell | total tok/s/GPU | vs RR | TTFT p50 / p95 | P90 interactivity | hit rate | logs |", "|---|---|---|---|---|---|---|---|"]
for lab, x in (("round-robin", rr), ("default KV", kvpt), (f"tuned KV ({NAME.get(tn['pol'], '')})" if tn else "tuned KV", tn)):
    if x: o.append(f"| {lab} | {x['clients']} clients | {x['tot']:,.0f} | {x['tot'] / rr['tot']:.2f}× | {x['p50']:.2f} / {x['p95']:.2f} s | {x['p90i']:.0f} | {x['hit']:.2f} | {L(x)} |" if rr else f"| {lab} | {x['clients']} | {x['tot']:,.0f} | — | {x['p50']:.2f} / {x['p95']:.2f} s | {x['p90i']:.0f} | {x['hit']:.2f} | {L(x)} |")
if PT:
    base = cells[("kv", PT)]; fl = sorted((x for x in cells.values() if x["clients"] == PT and x["pol"] not in ("kv", "rr")), key=lambda x: x["ts"])
    o += ["", f"### KV flag sweep at the KV 10 s point ({PT} clients; same variants as the 192-client sweep)", "",
          "| router flags | total tok/s/GPU | output tok/s (/GPU) | TTFT p50 / p95 / p99 | ITL p50 / p90 → P90 | in-flight | hit rate | warm-up wall | stationary | logs + artifacts |", "|---|---|---|---|---|---|---|---|---|---|",
          f"| {NAME['kv']} (baseline) " + row(base, base)]
    o += [f"| {NAME.get(x['pol'], x['pol'])}: {FLAGS.get(x['pol'], '')} " + row(x, base) for x in fl]
    if not fl: o.append("| (flag cells running) | | | | | | | | | |")
s = open(f"{ROOT}/AGENTX_AGG_RESULTS.md").read(); i = s.find("\n## vi. Cells that saturate")
if i >= 0:
    j = s.find("\n## ", i + 5); s = s[:i] + "\n" + "\n".join(o) + "\n" + (s[j:] if j >= 0 else "")
else:
    j = s.find("\n## iv. Simulation-vs-real gap"); s = s[:j] + "\n" + "\n".join(o) + "\n" + s[j:]
open(f"{ROOT}/AGENTX_AGG_RESULTS.md", "w").write(s)
ri = open(f"{ROOT}/RUN_INDEX.md").read(); added = 0
for x in sorted(cells.values(), key=lambda x: x["ts"]):
    if x["ts"] in ri or x["ts"] < "1789890000": continue
    r = f"| 2026-09-20 | agg 24 new stack on np-2{' (fleet 2)' if x['fleet'].endswith('ns2') else ''} — AgentX 10 s SLO cells / flag sweep | {x['pol']} | {x['clients']} clients | `{x['art'].split('alisachen-')[1]}` ({x['out']:,.0f} tok/s · {'POST-KNEE' if x['knee'].startswith('POST') else 'AT/PRE'} · TTFT p95 {x['p95']:.2f} s) | {L(x)} |\n"
    n = int(re.search(r"\n(\d+) jobs listed\.", ri).group(1)); k = ri.rfind("\n| 2026-"); e = ri.find("\n", k + 1) + 1
    ri = ri[:e] + r + ri[e:]; ri = ri.replace(f"\n{n} jobs listed.", f"\n{n + 1} jobs listed."); added += 1
open(f"{ROOT}/RUN_INDEX.md", "w").write(ri); print(f"agg section vi regenerated: point={PT}, cells={len(cells)}, run-index rows added={added}")
