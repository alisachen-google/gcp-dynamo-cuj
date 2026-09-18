#!/bin/bash
# 8p/8d AgentX, remaining programme after the KV ladder (192/384/672/768/1152, knee located): RR first, then KV-flag tuning
# at the comparison cells, SLO = TTFT p95 <= 10 s (P90 interactivity >= 20, stationary).  One sequential process; the fleet
# stays up between cells (agentx_runner_keep.sh, KEEP_FLEET=1) so no other tenant can take the nodes between stages.
#   1  RR ladder 192 / 384 / 240 / 96 (blocking), then RR 672 / 768 (RR at KV's high-load cells; a failure there is not blocking)
#   2  KV bisection for the 10 s SLO boundary (default KV: 4.8 s at 384, 17.0 s at 672): KV 480, then 576 if 480 is inside else 432
#   3  selection from measured cells (select_agentx_points.py --ttft 10): same config = RR's throughput-peak cell (KV is run there
#      first if missing); same SLO = best stationary cell per policy inside the SLO
#   4  flag sweep phase A at the KV same-SLO cell, one factor at a time vs default KV: load scale 3 / credit 0.8, load scale 2 /
#      credit 0.8, queue policy WSPT, temperature 0.5
#   5  phase B with the phase-A winner (pick_agentx_flag_winner.py: best total tok/s/GPU, stationary, TTFT p95 not worse, > 2 % gain):
#      winner at the same-config cell, and winner at the next KV client count above the SLO cell (does tuning move the SLO cell up?)
#   6  paired KV cells still missing for the same-config table (240, 96), then scale the fleet down
# Any MNNVL transport violation stops everything.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
D=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts; S=$D/agentx_runner_keep.sh
M=$HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-88.yaml; W=n3u-mnnvl-88-prefill,n3u-mnnvl-88-decode
export GATE_STRICT=1 BENCH_POOL=np-2 MNNVL_GUARD=1 JOB_WAIT_ITERS=160 KEEP_FLEET=1
LOG=/tmp/agentx_88_rest.log; SLO=${SLO_TTFT:-10}; VARIANTS=${VARIANTS:-kvs3c08,kvs2c08,kvwspt,kvt05}
KVL="192,384,672,768,1152"; RRL=""
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
cellrun(){ # <pol:clients> -> 0 ok, 1 cell failed, exits on MNNVL violation
  bash "$S" n3u-mnnvl-88 "$M" "$1" "N3U AGENTX 88 KVX DONE" /tmp/agentx_88_kvx.log "N3U AGENTX 88 REST CELL $1 DONE" "$LOG" 0 np-2 n3u-mnnvl-88 "$W"; rc=$?
  grep -q "MNNVL TRANSPORT VIOLATION" "$LOG" && { say "MNNVL violation - programme stopped"; exit 2; }
  return $rc; }
p95(){ python3 - "$1" "$2" <<'PY'
import sys,subprocess,csv,io
sh=lambda c: subprocess.run(c,shell=True,capture_output=True,text=True).stdout
d=sh(f"gcloud storage ls gs://alisachen-models/perf/ | grep '_alisachen-n3u-mnnvl-88-agentx-{sys.argv[1]}-c{sys.argv[2]}/' | tail -1").strip()
f=sh(f"gcloud storage ls -r '{d}' | grep profile_export_aiperf.csv | grep -v warmup | head -1").strip()
m={r[0]:r for r in csv.reader(io.StringIO(sh(f"gcloud storage cat '{f}'"))) if r}
print(f"{float(m['Time to First Token (ms)'][12])/1000:.2f}")
PY
}
sel(){ python3 $D/select_agentx_points.py n3u-mnnvl-88 64 "$KVL" "$RRL" --ttft $SLO --variants "$VARIANTS" --logs /tmp/agentx_88.log "$LOG" --out-md /tmp/agentx_88_points.md --out-json /tmp/agentx_88_points.json > /tmp/agentx_88_select.log 2>&1; }
js(){ python3 -c "import json;d=json.load(open('/tmp/agentx_88_points.json'));v=$1;print(v if v is not None else '')"; }

say "=== stage 1: RR ladder"
for c in 192 384 240 96; do cellrun "rr:$c" || { say "RR c$c failed - HALTING programme"; exit 1; }; RRL="${RRL:+$RRL,}$c"; done
for c in 672 768; do cellrun "rr:$c" && RRL="$RRL,$c" || say "RR c$c did not complete (not blocking)"; done
say "N3U AGENTX 88 RR DONE ($RRL)"

say "=== stage 2: KV bisection for TTFT p95 <= $SLO s"
if cellrun "kv:480"; then KVL="$KVL,480"; P=$(p95 kv 480); say "KV c480 TTFT p95 = $P s"
  N=$(python3 -c "print(576 if float('$P') <= $SLO else 432)"); cellrun "kv:$N" && KVL="$KVL,$N" || say "KV c$N did not complete"
else say "KV c480 failed - selecting from existing cells"; fi

say "=== stage 3: selection"
sel; PK=$(js "d['rr_peak']")
if [ -n "$PK" ] && ! echo ",$KVL," | grep -q ",$PK,"; then say "RR peak c$PK has no KV cell - running it"; cellrun "kv:$PK" && KVL="$KVL,$PK"; sel; fi
CFG=$(js "d['same_cfg']"); KS=$(js "d['kv_slo'] and d['kv_slo']['clients']"); RS=$(js "d['rr_slo'] and d['rr_slo']['clients']")
say "N3U AGENTX 88 SELECT DONE same_config=c$CFG kv_slo=c$KS rr_slo=c$RS (SLO TTFT p95 <= $SLO s)"; cat /tmp/agentx_88_points.md >> "$LOG"
[ -z "$KS" ] && { say "no KV cell inside the SLO - flag sweep at c384"; KS=384; }

say "=== stage 4: flag sweep phase A at c$KS ($VARIANTS)"
for v in ${VARIANTS//,/ }; do cellrun "$v:$KS" || say "flag cell $v c$KS did not complete"; done
WIN=$(python3 $D/pick_agentx_flag_winner.py n3u-mnnvl-88 64 $KS "$VARIANTS" --logs "$LOG" 2>> "$LOG"); say "phase A winner at c$KS: $WIN"; WV=${WIN%% *}

say "=== stage 5: flag sweep phase B (winner $WV)"
if [ "$WV" != "kv" ]; then
  [ -n "$CFG" ] && [ "$CFG" != "$KS" ] && { cellrun "$WV:$CFG" || say "phase B $WV c$CFG did not complete"; }
  UP=$(python3 -c "k=sorted(int(x) for x in '$KVL'.split(',')); print(next((x for x in k if x > $KS), ''))")
  [ -n "$UP" ] && [ "$UP" != "$CFG" ] && { cellrun "$WV:$UP" || say "phase B $WV c$UP did not complete"; }
else say "no variant beat default KV by > 2 % - phase B skipped"; fi
say "N3U AGENTX 88 FLAGS DONE"

say "=== stage 6: remaining paired KV cells"
for c in 240 96; do echo ",$KVL," | grep -q ",$c," || { cellrun "kv:$c" && KVL="$KVL,$c"; }; done
sel
for d in n3u-mnnvl-88-prefill n3u-mnnvl-88-decode n3u-mnnvl-88-frontend; do kubectl scale deployment/$d -n dynamo-cloud --replicas=0 >> "$LOG" 2>&1; done
say "N3U AGENTX 88 REST DONE kv=$KVL rr=$RRL"
