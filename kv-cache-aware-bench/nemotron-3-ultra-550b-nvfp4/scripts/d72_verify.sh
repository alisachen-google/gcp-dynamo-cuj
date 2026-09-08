#!/bin/bash
# Knee/ceiling verification (user directive 2026-09-08):
#  Phase A (6:12): kv:144, rr:144, kv:192 — does KV's ceiling keep climbing
#    past c96 (3,412)? does RR stay pinned ~1,750?
#  Phase B (3:15 = decode-heaviest, disagg's best per-GPU case): kv:96, kv:144.
# Same protocol: fresh frontend, 300s settle, 900s warmup + 1800s measure,
# RDMA gate (halt on fail), knee logged, UCX transport evidence per point.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
ARM=n3u-d72
LOG=/tmp/d72_verify.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
declare -A ROUTER=(
  [kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
  [rr]="--router-mode round-robin"
)

deploy() { # n_prefill n_decode
  say "deploying fleet ${1}P+${2}D"
  kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-d72.yaml >> "$LOG" 2>&1
  kubectl scale deployment/${ARM}-prefill -n $NS --replicas=$1 >> "$LOG" 2>&1
  kubectl scale deployment/${ARM}-decode -n $NS --replicas=$2 >> "$LOG" 2>&1
  kubectl scale deployment/${ARM}-frontend -n $NS --replicas=1 >> "$LOG" 2>&1
  for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do
    kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; exit 1; }
  done
  say "fleet ready"
}

point() { # variant conc tag
  local v=$1 C=$2 TAG=$3
  say "=== [$TAG] point $v c$C"
  local ARGS_JSON=$(python3 -c "import json;print(json.dumps(['-m','dynamo.frontend']+'''${ROUTER[$v]}'''.split()+['--request-plane','nats']))")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$ARGS_JSON}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1
  kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=300s >> "$LOG" 2>&1
  sleep 300
  local JOB=alisachen-n3u-d72-${TAG}-${v}-c${C}
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
  say "point [$TAG] $v c$C done (job=$st)"
  [ "$st" != "Complete" ] && { say "BENCH VIOLATION - HALTING"; exit 2; }
  bash "$HOME/DynamoBench/common/kv-transport-guard.sh" gate "$ARM" >> "$LOG" 2>&1 \
    || { say "RDMA VIOLATION - HALTING"; exit 2; }
  say "gate PASS"
  for tier in prefill decode; do
    local WP=$(kubectl get pods -n $NS -l app=${ARM}-${tier} -o name 2>/dev/null | head -1)
    local EV=$(kubectl logs -n $NS "$WP" -c $tier --tail=20000 2>/dev/null | \
         grep -oE "rc_mlx5|cuda_ipc|tcp/[a-z0-9]+" | sort | uniq -c | tr '\n' ' ')
    say "UCX evidence [$TAG $v c$C ${tier}]: ${EV:-none-found}"
  done
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "n3u-d72-${TAG}-${v}-c${C}" >> "$LOG" 2>&1 || true
}

deploy 6 12
point kv 144 s612
point rr 144 s612
point kv 192 s612

say "=== reshaping to 3:15"
kubectl scale deployment/${ARM}-prefill -n $NS --replicas=3 >> "$LOG" 2>&1
sleep 120
kubectl scale deployment/${ARM}-decode -n $NS --replicas=15 >> "$LOG" 2>&1
for d in ${ARM}-prefill ${ARM}-decode; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; exit 1; }
done
point kv 96 s315
point kv 144 s315

kubectl scale deployment/${ARM}-prefill deployment/${ARM}-decode deployment/${ARM}-frontend -n $NS --replicas=0 >> "$LOG" 2>&1
say "D72 VERIFY DONE"
