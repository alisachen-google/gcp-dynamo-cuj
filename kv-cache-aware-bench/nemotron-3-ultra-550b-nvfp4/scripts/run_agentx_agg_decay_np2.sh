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
LOG=/tmp/agentx_agg_decay.log; C=${AGG_CLIENTS:-192}; REF=1640
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
authwait(){ n=0; until kubectl get ns dynamo-cloud >/dev/null 2>&1; do [ $((n % 12)) = 0 ] && say "kubectl auth/API unavailable - waiting (ADC refresh needed?)"; n=$((n+1)); sleep 300; done; }
declare -A MAN=([n3u-agg-ns]=$MD/n3u-agg-newstack-np2.yaml [n3u-agg-ns2]=$MD/n3u-agg-newstack2-np2.yaml)
cell(){ # <fleet> <variant>: resumable - skips a measured cell, waits for a running one, else runs it; per-fleet log
  F=$1; V=$2; J=alisachen-$F-agentx-$V-c$C; L=/tmp/agentx_agg_$F.log; authwait
  st=$(kubectl get jobs -n dynamo-cloud "$J" --no-headers 2>/dev/null | awk '{print $2}')
  if [ "$st" = "Complete" ] && grep -q "KNEE-CHECK $F-agentx-$V-c$C:" "$L" 2>/dev/null; then say "$F $V c$C already measured - skipped"; return 0; fi
  if [ "$st" = "Running" ]; then say "$F $V c$C is in flight - waiting for it"; for i in $(seq 1 320); do st=$(kubectl get jobs -n dynamo-cloud "$J" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Running" ] || [ -z "$st" ] || break; sleep 60; done
    [ "$st" = "Complete" ] && { python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "$F-agentx-$V-c$C" >> "$L" 2>&1 || true; say "$F AgentX $V c$C done (job=Complete) [adopted]"; return 0; }; return 1; fi
  say "=== $F AgentX $V c$C"
  bash "$S" $F "${MAN[$F]}" "$V:$C" "N3U AGENTX 88 KVX DONE" /tmp/agentx_88_kvx.log "AGG CELL $F $V DONE" "$L" 0 np-2 $F $F-worker; rc=$?
  if [ $rc -ne 0 ]; then authwait; for i in $(seq 1 320); do st=$(kubectl get jobs -n dynamo-cloud "$J" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Running" ] || break; sleep 60; done
    [ "$st" = "Complete" ] && { say "$F $V: job Complete although the runner gave up - recovering"; python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "$F-agentx-$V-c$C" >> "$L" 2>&1 || true; rc=0; }; fi
  [ $rc = 0 ] && say "$F AgentX $V c$C done" || say "$F AgentX $V c$C did not complete (rc=$rc)"; return $rc; }
warm(){ # <fleet> <variant> -> warm-up wall seconds of that cell, from its harvested records
  $D/harvest_agentx_cell.sh $1 $2 $C 24 2>/dev/null | tee -a "$LOG" | grep -o "wall=[0-9]*s" | grep -o "[0-9]*"; }
down(){ for d in $1-worker $1-frontend; do kubectl scale deployment/$d -n dynamo-cloud --replicas=0 >> "$LOG" 2>&1; done; }
PICK="python3 $D/pick_agentx_flag_winner.py n3u-agg-ns2\\? 24 $C"; PLOGS="/tmp/agentx_agg_n3u-agg-ns.log /tmp/agentx_agg_n3u-agg-ns2.log"

say "waiting for the 8p/8d programme to release np-2 ('N3U AGENTX 88 REST DONE')"
until grep -q "N3U AGENTX 88 REST DONE" /tmp/agentx_88_rest.log 2>/dev/null; do sleep 60; done
say "=== agg credit-decay tuning at $C clients: phase 1 (two fleets in parallel)"
cell n3u-agg-ns kvd05 & P1=$!; sleep 90; cell n3u-agg-ns2 kvd10 & P2=$!; wait $P1; R1=$?; wait $P2; R2=$?

for fv in "n3u-agg-ns kvd05 $R1" "n3u-agg-ns2 kvd10 $R2"; do set -- $fv; [ "$3" = "0" ] || continue
  W=$(warm $1 $2); OK=$(python3 -c "print(1 if abs(${W:-0}-$REF)/$REF <= 0.015 else 0)")
  say "health check $1 $2: warm-up wall ${W:-?} s vs reference $REF s -> $([ "$OK" = 1 ] && echo comparable || echo 'NOT comparable: re-running the default-KV baseline on this fleet')"
  [ "$OK" = 1 ] || cell $1 kv; done

WD=$($PICK kvd05,kvd10 --logs $PLOGS 2>> "$LOG"); say "phase 1 verdict (decay vs default KV): $WD"; WV=${WD%% *}
say "=== phase 2"
if [ "$WV" != "kv" ]; then
  cell n3u-agg-ns kvs3c08${WV#kv} & P1=$!; P2=""
  [ "$WV" = "kvd10" ] && { sleep 90; cell n3u-agg-ns2 kvd20 & P2=$!; } || down n3u-agg-ns2
  wait $P1; [ -n "$P2" ] && wait $P2
else down n3u-agg-ns2; cell n3u-agg-ns kvs3c08d05; fi
for f in n3u-agg-ns n3u-agg-ns2; do for v in kvs3c08d05 kvs3c08d10 kvd20 kv; do kubectl get jobs -n dynamo-cloud alisachen-$f-agentx-$v-c$C --no-headers 2>/dev/null | grep -q Complete && $D/harvest_agentx_cell.sh $f $v $C 24 >> "$LOG" 2>&1; done; done
WIN=$($PICK kvd05,kvd10,kvd20,kvs3c08,kvs3c08d05,kvs3c08d10 --logs $PLOGS 2>> "$LOG"); say "AGG DECAY overall winner at c$C: $WIN"
down n3u-agg-ns; down n3u-agg-ns2; say "AGG DECAY DONE"
