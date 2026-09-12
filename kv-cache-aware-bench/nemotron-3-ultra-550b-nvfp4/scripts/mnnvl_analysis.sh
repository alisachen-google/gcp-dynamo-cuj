#!/bin/bash
# Post-MNNVL-resweep analysis (goal 2026-09-12): once the MNNVL disagg re-sweep
# completes, compare MNNVL disagg vs agg (is disagg better on NVLink?), vs the
# host-staged disagg set (transport gain), and vs DynoSim v1 (drift — expected
# to SHRINK vs host-staged, since the sim assumes ~free transfer ≈ MNNVL).
set -u
S=/tmp/claude-731364623/-home-alisachen-google-com/260418f9-118c-483a-a09a-45ff9cd547c7/scratchpad
LOG=/tmp/mnnvl_analysis.log
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
say "waiting for N3U MNNVL RESWEEP DONE"
until grep -q "N3U MNNVL RESWEEP DONE" /tmp/resweep_mnnvl.log 2>/dev/null; do
  grep -qE "VIOLATION|STACK TIMEOUT|SMOKE FAILED" /tmp/resweep_mnnvl.log 2>/dev/null && { say "resweep halted; no analysis"; exit 1; }
  sleep 300
done
say "resweep done; fetching MNNVL point summaries"
mkdir -p $S/mnnvl
for pt in kv-c12 rr-c12 kv-c24 rr-c24 kv-c48 rr-c48 kv-c96 rr-c96; do
  R=$(gcloud storage ls gs://alisachen-models/perf/ 2>/dev/null | grep "n3u-mnnvl-${pt}" | tail -1)
  RUN=$(gcloud storage ls -r "$R" 2>/dev/null | grep "trace_c.*/:$" | grep -v cache-warmup | head -1 | sed 's|/:$||')
  gcloud storage cp "${RUN}/profile_export_aiperf.json" $S/mnnvl/${pt}.json 2>/dev/null && say "fetched $pt"
done
python3 - <<'PYEOF' >> "$LOG" 2>&1
import json, csv, glob, os
S="/tmp/claude-731364623/-home-alisachen-google-com/260418f9-118c-483a-a09a-45ff9cd547c7/scratchpad"
def m(d,k,s):
    v=d.get(k,{}); return v.get(s) if isinstance(v,dict) else None
def pt(f):
    d=json.load(open(f)); return dict(thr=m(d,"output_token_throughput","avg"),
        p50=m(d,"time_to_first_token","p50")/1000, p95=m(d,"time_to_first_token","p95")/1000,
        rps=m(d,"request_throughput","avg"))
# MNNVL silicon
mn={}
for f in glob.glob(f"{S}/mnnvl/*.json"):
    n=os.path.basename(f)[:-5]; mn[n]=pt(f)
# host-staged disagg (certified rs612) for transport comparison
hs={}
for f in glob.glob(f"{S}/rs612/*.json"):
    n=os.path.basename(f)[:-5]; hs[n]=pt(f)
# DynoSim v1 disagg (zero-transfer ~ MNNVL) 6:12
sim={}
p=os.path.expanduser("~/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/dynosim_n3u_disagg72_v1.csv")
if os.path.exists(p):
    for r in csv.DictReader(open(p)):
        if r["pd"]=="6:12":
            sim[(r["policy"],int(r["conc"]))]=float(r["throughput_tok_s"])
AGG_BOUNDED=69.0  # tok/s/GPU, KV c32 bounded
print("=== MNNVL disagg (72 GPU) vs host-staged vs DynoSim ===")
print(f"{'pt':>8} | {'MNNVL tok/s':>11} {'/GPU':>5} {'p95':>6} | {'host-staged':>11} | {'MNNVL/HS':>8} | {'DynoSim v1':>10} | {'sim/MNNVL':>9}")
for c in (12,24,48,96):
    for pol in ("kv","rr"):
        k=f"{pol}-c{c}"; 
        if k not in mn: continue
        mv=mn[k]["thr"]; hv=hs.get(k,{}).get("thr")
        simpol="kv-nvda" if pol=="kv" else "rr"
        sv=sim.get((simpol,c))
        print(f"{k:>8} | {mv:11.0f} {mv/72:5.1f} {mn[k]['p95']:5.1f}s | {hv if hv else 0:11.0f} | "
              f"{(mv/hv if hv else 0):7.2f}x | {sv if sv else 0:10.0f} | {(sv/mv if sv and mv else 0):8.2f}x")
best=max((mn[k]["thr"]/72 for k in mn if k.startswith("kv")), default=0)
print(f"\nBest MNNVL disagg KV: {best:.1f} tok/s/GPU  vs  agg bounded {AGG_BOUNDED}/GPU  -> disagg/agg = {best/AGG_BOUNDED:.2f}x")
print("(>1 means disagg finally beats agg on NVLink; <1 means agg still wins)")
PYEOF
say "analysis table written to $LOG"
say "MNNVL ANALYSIS DONE"
