#!/bin/bash
# N3U 72-GPU disagg (6P+12D TP4, sim-best bounded split): single fleet on np-3,
# router swapped per point via frontend patch. Ladder anchors low (silicon runs
# ~2.5x below sim absolutes): kv/rr x conc 24/48/96/144. Per point: fresh
# frontend -> 300s settle -> 900s warmup + 1800s measure -> RDMA transport gate
# (stop policy) -> knee gate (halt only where expected bounded).
# This campaign doubles as the transfer-ceiling diagnostic: if kv:24 measures
# post-knee, the Kimi-era ~1.5 req/s disagg ceiling persists for 4x-lighter
# transfers -> halt and report.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
ARM=n3u-d72
LOG=/tmp/sweep_n3u_d72.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

declare -A ROUTER=(
  [kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
  [rr]="--router-mode round-robin"
)
POINTS="kv:24:bounded rr:24:bounded kv:48:bounded rr:48:post-ok kv:96:post-ok rr:96:post-ok kv:144:post-ok rr:144:post-ok"

say "=== n3u d72 campaign: deploying 6P+12D fleet"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-d72.yaml >> "$LOG" 2>&1
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 \
    || { say "STACK TIMEOUT: $d not ready"; exit 1; }
done
say "fleet ready"

for point in $POINTS; do
  v=$(echo $point | cut -d: -f1); C=$(echo $point | cut -d: -f2); EXPECT=$(echo $point | cut -d: -f3)
  say "=== point $v conc $C (expect $EXPECT): ${ROUTER[$v]}"
  ARGS_JSON=$(python3 -c "import json;print(json.dumps(['-m','dynamo.frontend']+'''${ROUTER[$v]}'''.split()+['--request-plane','nats']))")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$ARGS_JSON}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1
  kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=300s >> "$LOG" 2>&1
  say "fresh frontend up; settling 300s"
  sleep 300
  JOB=alisachen-n3u-d72-${v}-c${C}
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
    [ "$st" = "Complete" ] && break
    [ "$st" = "Failed" ] && break
    sleep 120
  done
  say "point $v c$C done (job=$st)"
  if [ "$st" != "Complete" ]; then
    say "RDMA/BENCH VIOLATION: $v c$C status=$st - HALTING campaign"
    exit 2
  fi
  # KV-RDMA stop policy: transport gate on every point (NIXL KV+state transfer)
  if ! bash "$HOME/DynamoBench/common/kv-transport-guard.sh" gate "$ARM" >> "$LOG" 2>&1; then
    say "RDMA/BENCH VIOLATION: transport gate FAIL on $v c$C - HALTING campaign"
    exit 2
  fi
  say "gate PASS"
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "n3u-d72-${v}-c${C}" >> "$LOG" 2>&1
  kc=$?
  if [ "$kc" -eq 2 ] && [ "$EXPECT" = "bounded" ]; then
    say "KNEE VIOLATION: $v c$C POST-KNEE but expected bounded - HALTING for user review"
    exit 2
  elif [ "$kc" -eq 2 ]; then
    say "knee: $v c$C post-knee (expected knee-location point, continuing)"
  fi
done
say "N3U D72 SWEEP DONE"
