#!/bin/bash
# Agg 24-GPU (6 x TP4) AgentX KV-flag tuning, credit decay instead of router temperature, at the 192-client comparison point.
# Two agg fleets run in parallel on np-2 (12 of its 17 nodes) once the 8p/8d programme has released them.
#   phase 1 (parallel)  fleet ns : credit decay 0.5      fleet ns2 : credit decay 1.0
#   health check        each new cell's warm-up wall time vs the 2026-09-16 agg cells at 192 clients (1,635-1,651 s); if a fleet is
#                       off by > 1.5 % the default-KV baseline is re-run on it so the comparison stays like for like
#   phase 2 (parallel)  if decay wins: ns = agg winner (load scale 3 / credit 0.8) + the better decay, ns2 = decay 2.0 (only if 1.0 beat 0.5)
#                       if decay does not win: ns = agg winner + decay 0.5 (does mild decay add anything?), ns2 idle
# Baselines already measured (2026-09-16, 192 clients): default KV 9,655 tok/s/GPU, TTFT p95 11.68 s; load scale 3 / credit 0.8 11,012, 6.33 s.
# Winner rule = pick_agentx_flag_winner.py (stationary; > 10 % TTFT p95 cut with throughput within 1 %, or > 2 % throughput with TTFT no worse).
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
D=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts; S=$D/agentx_runner_flags.sh; MD=$HOME/kv-cache-aware-bench/sglang/manifests
export GATE_STRICT=1 BENCH_POOL=np-2 MNNVL_GUARD=0 JOB_WAIT_ITERS=160 KEEP_FLEET=1
LOG=/tmp/agentx_agg_slo10.log; C=${AGG_CLIENTS:-192}; REF=1640
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
authwait(){ n=0; until kubectl get ns dynamo-cloud >/dev/null 2>&1; do [ $((n % 12)) = 0 ] && say "kubectl auth/API unavailable - waiting (ADC refresh needed?)"; n=$((n+1)); sleep 300; done; }
declare -A MAN=([n3u-agg-ns]=$MD/n3u-agg-newstack-np2.yaml [n3u-agg-ns2]=$MD/n3u-agg-newstack2-np2.yaml)
cell(){ # <fleet> <variant> <clients>: resumable - skips a measured cell, waits for a running one, else runs it; per-fleet log
  F=$1; V=$2; C=$3; J=alisachen-$F-agentx-$V-c$C; L=/tmp/agentx_agg_$F.log; authwait
  st=$(kubectl get jobs -n dynamo-cloud "$J" --no-headers 2>/dev/null | awk '{print $2}')
  if [ "$st" = "Complete" ] && grep -q "KNEE-CHECK $F-agentx-$V-c$C:" "$L" 2>/dev/null; then say "$F $V c$C already measured - skipped"; return 0; fi
  if [ "$st" = "Running" ]; then say "$F $V c$C is in flight - waiting for it"; for i in $(seq 1 320); do st=$(kubectl get jobs -n dynamo-cloud "$J" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Running" ] || [ -z "$st" ] || break; sleep 60; done
    [ "$st" = "Complete" ] && { python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "$F-agentx-$V-c$C" >> "$L" 2>&1 || true; say "$F AgentX $V c$C done (job=Complete) [adopted]"; return 0; }; return 1; fi
  say "=== $F AgentX $V c$C"
  bash "$S" $F "${MAN[$F]}" "$V:$C" "N3U AGENTX 88 KVX DONE" /tmp/agentx_88_kvx.log "AGG CELL $F $V DONE" "$L" 0 np-2 $F $F-worker; rc=$?
  if [ $rc -ne 0 ]; then authwait; for i in $(seq 1 320); do st=$(kubectl get jobs -n dynamo-cloud "$J" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Running" ] || break; sleep 60; done
    [ "$st" = "Complete" ] && { say "$F $V: job Complete although the runner gave up - recovering"; python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "$F-agentx-$V-c$C" >> "$L" 2>&1 || true; rc=0; }; fi
  [ $rc = 0 ] && say "$F AgentX $V c$C done" || say "$F AgentX $V c$C did not complete (rc=$rc)"; return $rc; }
p95of(){ # <fleet> <variant> <clients> -> harvests the cell (appends the summary to the log) and prints its TTFT p95 in seconds
  $D/harvest_agentx_cell.sh $1 $2 $3 24 2>/dev/null | tee -a "$LOG" | grep -o "TTFT p50/p90/p95=[0-9./]*s" | sed 's/.*\///; s/s$//'; }
down(){ for d in $1-worker $1-frontend; do kubectl scale deployment/$d -n dynamo-cloud --replicas=0 >> "$LOG" 2>&1; done; }
PICK="python3 $D/pick_agentx_flag_winner.py n3u-agg-ns2\\? 24 $C"; PLOGS="/tmp/agentx_agg_n3u-agg-ns.log /tmp/agentx_agg_n3u-agg-ns2.log"

# Agg 24-GPU AgentX: cells that SATURATE the TTFT p95 <= 10 s SLO (2026-09-20).  The measured agg ladder brackets 10 s but has no
# cell near it: default KV 96 -> 5.4 s / 192 -> 11.2-11.7 s; RR 48 -> 8.3 s / 96 -> 12.6 s; tuned KV (load scale 3 / credit 0.8)
# 192 -> 6.2-6.3 s / (default 384 is post-knee).  Two fleets in parallel on np-2 once the 8p/8d follow-up has released it:
#   fleet ns  : default KV at 160 clients (linear interpolation puts 10 s near 165), then ONE bisection cell if it is not within
#               8.5-10 s (176 if below 8.5 s, 144 if above 10 s)
#   then BOTH fleets tune the KV flags at the selected KV 10 s point with the same pattern as the 192-client sweep (load scale 3 /
#   credit 0.8, load scale 2 / credit 0.8, credit decay 0.5, temperature 0.5, credit decay 1.0, load scale 3 / credit 0.8 + decay 0.5),
#   pulling variants from one shared queue so neither fleet idles.  np-2 has 17 GPU nodes = two 6-node agg fleets (a third needs 18).
#   fleet ns2 : RR at 64 clients (interpolation: 10 s near 67), then tuned KV at 256 clients (does the tuned router hold 10 s there?)
say "waiting for the 8p/8d follow-up to release np-2 ('N3U AGENTX 88 FOLLOWUP DONE')"
until grep -q "N3U AGENTX 88 FOLLOWUP DONE" /tmp/agentx_88_rest.log 2>/dev/null; do sleep 60; done
say "=== agg cells that saturate the 10 s TTFT p95 SLO (two fleets in parallel)"
QF=/tmp/agentx_agg_slo_queue.txt; PF=/tmp/agentx_agg_slo_point.txt; rm -f $PF; [ -s $QF ] || printf 'kvs3c08\nkvs2c08\nkvd05\nkvt05\nkvd10\nkvs3c08d05\n' > $QF
pop(){ flock $QF.lock bash -c "v=\$(head -1 $QF); [ -n \"\$v\" ] && sed -i 1d $QF; echo \$v"; }
flags(){ # <fleet>: take variants off the shared queue until it is empty; every cell at the selected KV 10 s point
  until [ -s $PF ]; do sleep 30; done; PT=$(cat $PF)
  while :; do V=$(pop); [ -z "$V" ] && break; cell $1 $V $PT && say "agg $V c$PT: $($D/harvest_agentx_cell.sh $1 $V $PT 24 2>/dev/null | head -1 | grep -o 'total/GPU=.*hit=[0-9.]*')" || say "agg $V c$PT did not complete"; done; }
laneA(){ cell n3u-agg-ns kv 160 || return 1; P=$(p95of n3u-agg-ns kv 160); say "agg default KV c160 TTFT p95 = $P s"; PT=160
  N=$(python3 -c "p=float('${P:-0}'); print('' if 8.5 <= p <= 10 else (176 if p < 8.5 else 144))")
  if [ -n "$N" ]; then say "c160 is not within 8.5-10 s -> one bisection cell at c$N"
    if cell n3u-agg-ns kv $N; then P2=$(p95of n3u-agg-ns kv $N); say "agg default KV c$N TTFT p95 = $P2 s"
      PT=$(python3 -c "p1,p2=float('${P:-0}'),float('${P2:-99}'); ok=[c for c,p in ((160,p1),($N,p2)) if p<=10]; print(max(ok) if ok else 96)"); fi
  else say "c160 saturates the SLO - no bisection cell needed"; fi
  say "AGG KV 10 s POINT = c$PT (largest measured default-KV client count with TTFT p95 <= 10 s) - flag sweep runs there"; echo $PT > $PF; flags n3u-agg-ns; }
laneB(){ cell n3u-agg-ns2 rr 64 && say "agg RR c64 TTFT p95 = $(p95of n3u-agg-ns2 rr 64) s"
  cell n3u-agg-ns2 kvs3c08 256 && say "agg tuned KV (load scale 3 / credit 0.8) c256 TTFT p95 = $(p95of n3u-agg-ns2 kvs3c08 256) s"; flags n3u-agg-ns2; }
laneA & PA=$!; sleep 5; laneB & PB=$!; wait $PA; wait $PB
PT=$(cat $PF 2>/dev/null); C=$PT
WIN=$(python3 $D/pick_agentx_flag_winner.py 'n3u-agg-ns2\?' 24 $PT kvs3c08,kvs2c08,kvd05,kvt05,kvd10,kvs3c08d05 --logs $PLOGS 2>> "$LOG"); say "AGG flag sweep at the KV 10 s point c$PT - winner: $WIN"
down n3u-agg-ns; down n3u-agg-ns2; say "AGG SLO10 DONE"
