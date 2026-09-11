#!/bin/bash
# Deep-concurrency disagg points (user directive 2026-09-11): kv/rr x conc
# 615/1024 on the 6:12 certified-transport fleet. Same per-point protocol;
# AIPERF_HTTP_CONNECTION_LIMIT raised 200 -> 1100 (else client caps effective
# concurrency); knee checks logged (post-knee expected); RDMA guard halts.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
ARM=n3u-d72
LOG=/tmp/deep_conc.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
declare -A ROUTER=(
  [kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
  [rr]="--router-mode round-robin"
)
POINTS="kv:615 rr:615 kv:1024 rr:1024"

say "=== deep-conc: deploying 6P+12D (certified transport)"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-d72.yaml >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-prefill -n $NS --replicas=6 >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-decode -n $NS --replicas=12 >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-frontend -n $NS --replicas=1 >> "$LOG" 2>&1
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; exit 1; }
done
for tier in prefill decode; do
  WP=$(kubectl get pods -n $NS -l app=${ARM}-${tier} -o name | head -1)
  EV=$(kubectl logs -n $NS "$WP" -c $tier 2>/dev/null | grep -c "rc_mlx5" || true)
  say "fleet-start $tier rc_mlx5 lines: $EV"
done
say "fleet ready"

for point in $POINTS; do
  v=${point%%:*}; C=${point##*:}
  say "=== point $v c$C"
  ARGS_JSON=$(python3 -c "import json;print(json.dumps(['-m','dynamo.frontend']+'''${ROUTER[$v]}'''.split()+['--request-plane','nats']))")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$ARGS_JSON}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1
  kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=300s >> "$LOG" 2>&1
  sleep 300
  JOB=alisachen-n3u-d72-deep-${v}-c${C}
  kubectl delete job -n $NS "$JOB" --ignore-not-found --wait=true >> "$LOG" 2>&1
  sed -e "s|/model-cache/alisachen/Kimi-K2.5-NVFP4|${N3U_DIR}|g" \
      -e "s|alisachen/Kimi-K2.5-NVFP4|${N3U_SERVED}|g" \
      -e "s|models--alisachen--Kimi-K2.5-NVFP4|models--alisachen--Nemotron-3-Ultra-550B-A55B-NVFP4|g" \
      -e "s/sgl-disagg72-kv/${ARM}/g" \
      -e "s/name: alisachen-sgl-d72-flagsweep/name: ${JOB}/" \
      -e "s/alisachen-sgl-d72-flagsweep/${JOB}/g" \
      -e "/name: CONCURRENCIES/{n;s/value: .*/value: \"${C}\"/}" \
      -e "/name: BENCHMARK_DURATION/{n;s/value: .*/value: \"1800\"/}" \
      -e "/name: AIPERF_HTTP_CONNECTION_LIMIT/{n;s/value: .*/value: \"1100\"/}" \
      "$TMPL" | kubectl apply -n $NS -f - >> "$LOG" 2>&1
  st=""
  for i in $(seq 1 70); do
    st=$(kubectl get jobs -n $NS "$JOB" --no-headers 2>/dev/null | awk '{print $2}')
    [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break
    sleep 120
  done
  say "point $v c$C done (job=$st)"
  [ "$st" != "Complete" ] && { say "BENCH VIOLATION on $v c$C - HALTING"; exit 2; }
  bash "$HOME/DynamoBench/common/kv-transport-guard.sh" gate "$ARM" >> "$LOG" 2>&1 \
    || { say "RDMA VIOLATION on $v c$C - HALTING"; exit 2; }
  say "gate PASS"
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "deep-${v}-c${C}" >> "$LOG" 2>&1 || true
done
kubectl scale deployment/${ARM}-prefill deployment/${ARM}-decode deployment/${ARM}-frontend -n $NS --replicas=0 >> "$LOG" 2>&1
say "DEEP CONC DONE"
