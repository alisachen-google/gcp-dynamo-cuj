#!/bin/bash
# N3U topology verification: 9:9 split on MNNVL+mooncake, KV-only, c48/96/144.
# Tests whether NVLink transport shifts the optimal split away from 6:12.
# v2 (2026-09-13) after the first attempt timed out at 0/9 prefill pods:
#  - PINNED single-context kubeconfig (a background process flips the shared
#    file to cmcs-live; the old run likely deployed to the wrong cluster).
#  - stale ComputeDomain cleanup before deploy (a node belongs to ONE CD; a
#    leftover 6:12 CD would hold the IMEX channels and starve the 9:9 claim).
#  - domain-availability wait: needs >=18 np-3 nodes with 0 GPUs allocated.
#    Never touches other tenants' pods — waits for them.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned   # <cluster> only
NS=dynamo-cloud
ARM=n3u-mnnvl-99
LOG=/tmp/resweep_mnnvl_99.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
GUARD=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/mnnvl_transport_guard.sh
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
POINTS="48 96 144"
NEED_NODES=18
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }

say "ctx=$(kubectl config current-context) — v2 9:9 verification start"
[ "$(kubectl config current-context)" = "<cluster>" ] || { say "WRONG CONTEXT — abort"; exit 1; }

# ---- pre-flight: clear OUR stale fleets + ComputeDomains (never other tenants') ----
say "pre-flight: removing our stale fleets/CDs"
for d in n3u-mnnvl-full-prefill n3u-mnnvl-full-decode n3u-mnnvl-full-frontend \
         n3u-mnnvl-99-prefill n3u-mnnvl-99-decode n3u-mnnvl-99-frontend \
         n3u-mnnvl-prefill n3u-mnnvl-decode n3u-mnnvl-frontend n3u-agg-ns n3u-agg-ns-frontend; do
  kubectl delete deployment/$d -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1
done
for cd in n3u-mnnvl-full-cd n3u-mnnvl-99-cd n3u-mnnvl-cd; do
  kubectl delete computedomain/$cd -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1
done
sleep 90

# ---- domain-availability wait: >=18 np-3 nodes fully free ----
free_nodes() {
  local n=0
  for node in $(kubectl get nodes -l cloud.google.com/gke-nodepool=np-3 -o name 2>/dev/null); do
    node=${node#node/}
    used=$(kubectl describe node "$node" 2>/dev/null | awk '/Allocated resources/,0' | grep "nvidia.com/gpu" | awk '{print $2}')
    [ "${used:-0}" = "0" ] && n=$((n+1))
  done
  echo $n
}
say "waiting for >=${NEED_NODES} free np-3 nodes (polling every 5m; not touching other tenants)"
while :; do
  F=$(free_nodes)
  say "free np-3 nodes: $F / need ${NEED_NODES}"
  [ "$F" -ge "$NEED_NODES" ] && break
  sleep 300
done

# ---- deploy 9:9 ----
say "domain available; deploying 9:9 MNNVL fleet"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-99.yaml >> "$LOG" 2>&1
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || {
    say "STACK TIMEOUT: $d"
    kubectl get pods -n $NS -l app=$d -o wide >> "$LOG" 2>&1
    kubectl describe pods -n $NS -l app=$d 2>/dev/null | grep -A8 "^Events" | tail -25 >> "$LOG" 2>&1
    exit 1; }
done
for tier in prefill decode; do
  WP=$(kubectl get pods -n $NS -l app=${ARM}-${tier} -o name | head -1)
  IPC=$(kubectl logs -n $NS "$WP" -c $tier 2>/dev/null | grep -icE "cuda_ipc|mnnvl" || true)
  say "fleet-start $tier cuda_ipc/mnnvl lines: $IPC"
done
FEP=$(kubectl get pods -n $NS -l app=${ARM}-frontend -o name | head -1)
for r in $(seq 1 30); do kubectl exec -n $NS "$FEP" -c frontend -- curl -s -m 10 localhost:8000/v1/models 2>/dev/null | grep -q Nemotron && break; sleep 20; done
say "9:9 fleet ready on MNNVL, model registered"

# ---- KV ladder ----
for C in $POINTS; do
  say "=== 9:9 kv c$C"
  JOB=alisachen-n3u-mnnvl-99-kv-c${C}
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
  say "9:9 kv c$C done (job=$st)"
  [ "$st" != "Complete" ] && { say "BENCH VIOLATION on 9:9 kv c$C - HALTING"; exit 2; }
  if ! bash "$GUARD" "$ARM" >> "$LOG" 2>&1; then
    say "MNNVL TRANSPORT VIOLATION on 9:9 kv c$C - HALTING (KV not on NVLink)"; exit 2
  fi
  say "MNNVL gate PASS"
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "n3u-mnnvl-99-kv-c${C}" >> "$LOG" 2>&1 || true
done
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do
  kubectl scale deployment/$d -n $NS --replicas=0 >> "$LOG" 2>&1
done
say "N3U MNNVL 99 SWEEP DONE"
