#!/bin/bash
# N3U 24-GPU agg comparison (SELECTION.md ladder): single 6xTP4 fleet on np-3,
# router swapped per point via frontend patch. Points kv/rr x conc 16/32/64/128.
# Per point: fresh frontend -> 300s settle -> 900s trace warmup + 1800s measure
# (in-job) -> knee check. Halt on job failure or post-knee at a cell the sim
# predicts bounded; RR 64/128 are knee-location points (verdict logged only).
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
ARM=n3u-agg-kv          # single fleet; arm name kept for endpoint/labels
LOG=/tmp/sweep_n3u_agg.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

declare -A ROUTER=(
  [kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
  [rr]="--router-mode round-robin"
)
# point = variant:conc:expected(bounded|post-ok)
POINTS="kv:16:bounded rr:16:bounded kv:32:bounded rr:32:bounded kv:64:bounded rr:64:post-ok kv:128:bounded rr:128:post-ok"

say "=== n3u agg campaign: tearing down smoke, deploying 6xTP4 fleet"
kubectl delete deployment n3u-smoke-frontend n3u-smoke-worker -n $NS --ignore-not-found >> "$LOG" 2>&1
kubectl delete service n3u-smoke-frontend -n $NS --ignore-not-found >> "$LOG" 2>&1
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-agg-kv.yaml >> "$LOG" 2>&1
for d in ${ARM}-worker ${ARM}-frontend; do
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
  JOB=alisachen-n3u-agg-${v}-c${C}
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
    say "BENCH VIOLATION: $v c$C status=$st - HALTING campaign"
    exit 2
  fi
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "n3u-agg-${v}-c${C}" >> "$LOG" 2>&1
  kc=$?
  if [ "$kc" -eq 2 ] && [ "$EXPECT" = "bounded" ]; then
    say "KNEE VIOLATION: $v c$C POST-KNEE but sim predicted bounded - HALTING for user review"
    exit 2
  elif [ "$kc" -eq 2 ]; then
    say "knee: $v c$C post-knee (expected - knee-location point, continuing)"
  fi
done
say "N3U AGG SWEEP DONE"
