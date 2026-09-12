#!/bin/bash
# Full N3U 72-GPU disagg re-sweep on MNNVL+mooncake (goal 2026-09-12 step C).
# GATED on the MNNVL smoke passing. 6P+12D (n3u-mnnvl-full), kv/rr x conc
# 12/24/48/96. Per point: fresh frontend -> 300s settle -> 900s warmup + 1800s
# measure -> MNNVL transport gate (cuda_ipc required, no tcp/rdma/fallback) ->
# knee check. Halts on bench failure or MNNVL-transport violation.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
ARM=n3u-mnnvl-full
LOG=/tmp/resweep_mnnvl.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
GUARD=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/mnnvl_transport_guard.sh
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
declare -A ROUTER=(
  [kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
  [rr]="--router-mode round-robin"
)
POINTS="kv:12 rr:12 kv:24 rr:24 kv:48 rr:48 kv:96 rr:96"

say "waiting for MNNVL smoke SUCCESS"
until grep -q "MNNVL SMOKE: SUCCESS" /tmp/mnnvl_smoke.log 2>/dev/null; do
  grep -qE "MNNVL SMOKE: FAIL|MNNVL SMOKE FAIL" /tmp/mnnvl_smoke.log 2>/dev/null && { say "SMOKE FAILED — aborting full re-run (transport unproven)"; exit 1; }
  sleep 300
done
say "smoke passed; tearing down smoke, deploying 6P+12D MNNVL fleet"
kubectl scale deployment/n3u-mnnvl-prefill n3u-mnnvl-decode n3u-mnnvl-frontend -n $NS --replicas=0 2>/dev/null
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-full.yaml >> "$LOG" 2>&1
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; exit 1; }
done
# fleet-start transport evidence
for tier in prefill decode; do
  WP=$(kubectl get pods -n $NS -l app=${ARM}-${tier} -o name | head -1)
  IPC=$(kubectl logs -n $NS "$WP" -c $tier 2>/dev/null | grep -icE "cuda_ipc|mnnvl" || true)
  say "fleet-start $tier cuda_ipc/mnnvl lines: $IPC"
done
say "fleet ready on MNNVL"

for point in $POINTS; do
  v=${point%%:*}; C=${point##*:}
  say "=== point $v c$C"
  ARGS_JSON=$(python3 -c "import json;print(json.dumps(['-m','dynamo.frontend']+'''${ROUTER[$v]}'''.split()+['--request-plane','nats']))")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$ARGS_JSON}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1
  kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=400s >> "$LOG" 2>&1
  sleep 300
  JOB=alisachen-n3u-mnnvl-${v}-c${C}
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
  if ! bash "$GUARD" "$ARM" >> "$LOG" 2>&1; then
    say "MNNVL TRANSPORT VIOLATION on $v c$C - HALTING (KV not on NVLink)"; exit 2
  fi
  say "MNNVL gate PASS"
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "n3u-mnnvl-${v}-c${C}" >> "$LOG" 2>&1 || true
done
kubectl scale deployment/${ARM}-prefill ${ARM}-decode ${ARM}-frontend -n $NS --replicas=0 >> "$LOG" 2>&1
say "N3U MNNVL RESWEEP DONE"
