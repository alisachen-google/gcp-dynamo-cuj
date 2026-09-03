#!/bin/bash
# Completes the N3U disagg KV-vs-RR dataset (Kimi pattern) after slo_points.sh:
#   - kv:16  — KV knee refinement (bounded@12, post@24)
#   - rr:4   — adaptive RR floor probe, ONLY if slo run's rr:8 measured post-knee
#   - flag variants at the bounded c12 cell: scale2 / credit08 / temp05
# Waits for "SLO POINTS DONE", re-deploys the d72 fleet, runs, scales down.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
ARM=n3u-d72
LOG=/tmp/d72_complete.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
declare -A ROUTER=(
  [kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
  [rr]="--router-mode round-robin"
  [kv-scale2]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 2.0"
  [kv-credit08]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit 0.8"
  [kv-temp05]="--router-mode kv --router-temperature 0.5 --router-queue-policy fcfs"
)

say "waiting for SLO POINTS DONE"
until grep -q "SLO POINTS DONE" /tmp/slo_points.log 2>/dev/null; do
  grep -qE "VIOLATION|STACK TIMEOUT" /tmp/slo_points.log 2>/dev/null && { say "SLO RUN FAILED - aborting follow-on"; exit 1; }
  sleep 300
done
say "SLO points complete; deciding rr floor probe"
POINTS="kv:16 kv-scale2:12 kv-credit08:12 kv-temp05:12"
if grep -q "KNEE-CHECK n3u-d72-rr-c8: POST-KNEE" /tmp/slo_points.log; then
  POINTS="kv:16 rr:4 kv-scale2:12 kv-credit08:12 kv-temp05:12"
  say "rr:8 was post-knee -> including rr:4 floor probe"
fi
say "POINTS: $POINTS"

say "re-deploying d72 fleet"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-d72.yaml >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-prefill -n $NS --replicas=6 >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-decode -n $NS --replicas=12 >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-frontend -n $NS --replicas=1 >> "$LOG" 2>&1
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; exit 1; }
done
say "fleet ready"

for point in $POINTS; do
  v=${point%%:*}; C=${point##*:}
  say "=== point $v c$C: ${ROUTER[$v]}"
  ARGS_JSON=$(python3 -c "import json;print(json.dumps(['-m','dynamo.frontend']+'''${ROUTER[$v]}'''.split()+['--request-plane','nats']))")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$ARGS_JSON}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1
  kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=300s >> "$LOG" 2>&1
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
    [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break
    sleep 120
  done
  say "point $v c$C done (job=$st)"
  [ "$st" != "Complete" ] && { say "BENCH VIOLATION on $v c$C - HALTING"; exit 2; }
  bash "$HOME/DynamoBench/common/kv-transport-guard.sh" gate "$ARM" >> "$LOG" 2>&1 \
    || { say "RDMA VIOLATION on $v c$C - HALTING"; exit 2; }
  say "gate PASS"
  # explicit UCX transport evidence (user directive): RDMA (rc_mlx5) vs
  # MNNVL/NVLink (cuda_ipc) vs tcp, sampled from one prefill + one decode worker
  for tier in prefill decode; do
    WP=$(kubectl get pods -n $NS -l app=${ARM}-${tier} -o name 2>/dev/null | head -1)
    EV=$(kubectl logs -n $NS "$WP" -c $tier --tail=20000 2>/dev/null | \
         grep -oE "rc_mlx5|cuda_ipc|nvls|tcp/[a-z0-9]+" | sort | uniq -c | tr '\n' ' ')
    say "UCX transport evidence [$v c$C ${tier}]: ${EV:-none-found}"
  done
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "n3u-d72-${v}-c${C}" >> "$LOG" 2>&1 || true
done
kubectl scale deployment/${ARM}-prefill deployment/${ARM}-decode deployment/${ARM}-frontend -n $NS --replicas=0 >> "$LOG" 2>&1
say "D72 DATASET COMPLETE"
