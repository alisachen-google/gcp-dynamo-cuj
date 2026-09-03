#!/bin/bash
# SLO-framing boundary points (fix TTFT p95 SLO, compare max compliant
# throughput): agg rr:8 + kv:48, then d72 kv:8 + rr:8. Same per-point protocol
# (fresh frontend, 300s settle, 900s warmup + 1800s measure). Knee checks are
# logged only (boundary points, mixed verdicts expected); job failure or (d72)
# transport-gate failure halts.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
LOG=/tmp/slo_points.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
declare -A ROUTER=(
  [kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
  [rr]="--router-mode round-robin"
)

run_point() { # ARM variant conc guard(yes/no)
  local ARM=$1 v=$2 C=$3 GUARD=$4
  say "=== $ARM point $v c$C"
  local ARGS_JSON=$(python3 -c "import json;print(json.dumps(['-m','dynamo.frontend']+'''${ROUTER[$v]}'''.split()+['--request-plane','nats']))")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$ARGS_JSON}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1
  kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=300s >> "$LOG" 2>&1
  sleep 300
  local JOB=alisachen-${ARM}-${v}-c${C}
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
  local st=""
  for i in $(seq 1 60); do
    st=$(kubectl get jobs -n $NS "$JOB" --no-headers 2>/dev/null | awk '{print $2}')
    [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break
    sleep 120
  done
  say "point $ARM $v c$C done (job=$st)"
  [ "$st" != "Complete" ] && { say "BENCH VIOLATION - HALTING"; exit 2; }
  if [ "$GUARD" = "yes" ]; then
    bash "$HOME/DynamoBench/common/kv-transport-guard.sh" gate "$ARM" >> "$LOG" 2>&1 \
      || { say "RDMA VIOLATION on $ARM $v c$C - HALTING"; exit 2; }
    say "gate PASS"
  fi
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "${ARM}-${v}-c${C}" >> "$LOG" 2>&1 || true
}

say "=== SLO points phase A: agg fleet"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-agg-kv.yaml >> "$LOG" 2>&1
kubectl scale deployment/n3u-agg-kv-worker -n $NS --replicas=6 >> "$LOG" 2>&1
kubectl scale deployment/n3u-agg-kv-frontend -n $NS --replicas=1 >> "$LOG" 2>&1
for d in n3u-agg-kv-worker n3u-agg-kv-frontend; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; exit 1; }
done
run_point n3u-agg-kv rr 8 no
run_point n3u-agg-kv kv 48 no

say "=== phase B: swap to d72 fleet"
kubectl scale deployment/n3u-agg-kv-worker deployment/n3u-agg-kv-frontend -n $NS --replicas=0 >> "$LOG" 2>&1
sleep 90
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-d72.yaml >> "$LOG" 2>&1
for d in n3u-d72-prefill n3u-d72-decode n3u-d72-frontend; do
  kubectl scale deployment/$d -n $NS --replicas=$( [ "$d" = "n3u-d72-prefill" ] && echo 6 || { [ "$d" = "n3u-d72-decode" ] && echo 12 || echo 1; } ) >> "$LOG" 2>&1
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; exit 1; }
done
run_point n3u-d72 kv 8 yes
run_point n3u-d72 rr 8 yes
kubectl scale deployment/n3u-d72-prefill deployment/n3u-d72-decode deployment/n3u-d72-frontend -n $NS --replicas=0 >> "$LOG" 2>&1
say "SLO POINTS DONE"
