#!/bin/bash
# Live KV-routing flag sweep at the selected agg throughput point (user spec
# 2026-09-11): objective = max output tok/s/GPU at sustained concurrency.
# Fixed: --router-temperature 0. Swept: credit-decay in the lower interval
# {0.5, 0.65, 0.85} under --router-queue-policy wspt, prefill-load-scale
# {1.0, 0.5}; defaults as control + defaults-repeat as drift control.
# Cell: N3U agg conc 48 (KV best bounded, throughput-max objective).
# Chains behind deep-conc (np-3). wspt acceptance is validated per variant —
# a frontend that rejects the flag fails its rollout and the variant is
# logged SKIPPED (not a campaign halt).
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
ARM=n3u-agg-kv
LOG=/tmp/agg_flag_sweep.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
C=48
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

declare -A V=(
  [control]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
  [wspt-s1-d050]="--router-mode kv --router-temperature 0.0 --router-queue-policy wspt --router-prefill-load-scale 1.0 --router-kv-overlap-score-credit 1.0 --router-kv-overlap-score-credit-decay 0.5"
  [wspt-s1-d065]="--router-mode kv --router-temperature 0.0 --router-queue-policy wspt --router-prefill-load-scale 1.0 --router-kv-overlap-score-credit 1.0 --router-kv-overlap-score-credit-decay 0.65"
  [wspt-s1-d085]="--router-mode kv --router-temperature 0.0 --router-queue-policy wspt --router-prefill-load-scale 1.0 --router-kv-overlap-score-credit 1.0 --router-kv-overlap-score-credit-decay 0.85"
  [wspt-s05-d065]="--router-mode kv --router-temperature 0.0 --router-queue-policy wspt --router-prefill-load-scale 0.5 --router-kv-overlap-score-credit 1.0 --router-kv-overlap-score-credit-decay 0.65"
  [control-repeat]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
)
ORDER="control wspt-s1-d050 wspt-s1-d065 wspt-s1-d085 wspt-s05-d065 control-repeat"

say "waiting for DEEP CONC DONE (np-3)"
until grep -q "DEEP CONC DONE" /tmp/deep_conc.log 2>/dev/null; do
  grep -qE "VIOLATION|STACK TIMEOUT" /tmp/deep_conc.log 2>/dev/null && { say "deep-conc halted; proceeding"; break; }
  sleep 600
done

say "=== deploying agg fleet (6xTP4, np-3)"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-agg-kv.yaml >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-worker -n $NS --replicas=6 >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-frontend -n $NS --replicas=1 >> "$LOG" 2>&1
for d in ${ARM}-worker ${ARM}-frontend; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; exit 1; }
done
say "fleet ready"

for v in $ORDER; do
  say "=== variant $v: ${V[$v]}"
  ARGS_JSON=$(python3 -c "import json;print(json.dumps(['-m','dynamo.frontend']+'''${V[$v]}'''.split()+['--request-plane','nats']))")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$ARGS_JSON}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1
  if ! kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=300s >> "$LOG" 2>&1; then
    say "variant $v SKIPPED: frontend rejected flags (rollout failed) — restoring control"
    ARGS_JSON=$(python3 -c "import json;print(json.dumps(['-m','dynamo.frontend']+'''${V[control]}'''.split()+['--request-plane','nats']))")
    kubectl patch deployment ${ARM}-frontend -n $NS --type=json \
      -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$ARGS_JSON}]" >> "$LOG" 2>&1
    kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=300s >> "$LOG" 2>&1
    continue
  fi
  sleep 300
  JOB=alisachen-n3u-aggfs-${v}-c${C}
  kubectl delete job -n $NS "$JOB" --ignore-not-found --wait=true >> "$LOG" 2>&1
  sed -e "s|/model-cache/alisachen/Kimi-K2.5-NVFP4|${N3U_DIR}|g" \
      -e "s|alisachen/Kimi-K2.5-NVFP4|${N3U_SERVED}|g" \
      -e "s|models--alisachen--Kimi-K2.5-NVFP4|models--alisachen--Nemotron-3-Ultra-550B-A55B-NVFP4|g" \
      -e "s/sgl-disagg72-kv/${ARM}/g" \
      -e "s/name: alisachen-sgl-d72-flagsweep/name: ${JOB}/" \
      -e "s/alisachen-sgl-d72-flagsweep/${JOB}/g" \
      -e "/name: CONCURRENCIES/{n;s/value: .*/value: \"${C}\"/}" \
      -e "/name: BENCHMARK_DURATION/{n;s/value: .*/value: \"1800\"/}" \
      "$TMPL" | kubectl apply -n $NS -f - >> "$LOG" 2>&1
  st=""
  for i in $(seq 1 60); do
    st=$(kubectl get jobs -n $NS "$JOB" --no-headers 2>/dev/null | awk '{print $2}')
    [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break
    sleep 120
  done
  say "variant $v done (job=$st)"
  [ "$st" != "Complete" ] && { say "BENCH VIOLATION on $v - HALTING"; exit 2; }
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "aggfs-${v}-c${C}" >> "$LOG" 2>&1 || true
done
kubectl scale deployment/${ARM}-worker deployment/${ARM}-frontend -n $NS --replicas=0 >> "$LOG" 2>&1
say "AGG FLAG SWEEP DONE"
