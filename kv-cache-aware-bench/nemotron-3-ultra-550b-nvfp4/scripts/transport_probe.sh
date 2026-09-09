#!/bin/bash
# Transport certification probe (goal 2026-09-09): find the healthiest UCX
# config for disagg KV transfer, in order of preference:
#   A) UCX_IB_GPU_DIRECT_RDMA=y + UCX_TLS=cuda_copy,rc_x   (direct, no tcp)
#   B) UCX_IB_GPU_DIRECT_RDMA=n + UCX_TLS=cuda_copy,rc_x   (host-staged, no tcp)
#   C) current baseline (host-staged, tcp in TLS for wireup only; guard asserts
#      no tcp on the data path)
# Method: 1P+1D n3u fleet; per config: patch env -> restart workers -> wait
# ready -> 20 disagg requests through the frontend -> grep NIXL/UCX errors.
# First config with 20/20 clean completions and zero REMOTE_DISCONNECT wins.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
ARM=n3u-d72
LOG=/tmp/transport_probe.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

patch_env() { # GPU_DIRECT_VAL TLS_VAL
  for d in ${ARM}-prefill ${ARM}-decode; do
    kubectl set env deployment/$d -n $NS -c ${d##*-} \
      UCX_IB_GPU_DIRECT_RDMA="$1" UCX_TLS="$2" >> "$LOG" 2>&1
  done
}

wait_ready() {
  for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do
    kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || return 1
  done
  sleep 120   # registration settle
}

probe_traffic() { # tag -> returns 0 clean / 1 dirty
  local TAG=$1
  local FE=$(kubectl get pods -n $NS -l app=${ARM}-frontend -o name | head -1)
  local OK=0
  for i in $(seq 1 20); do
    R=$(kubectl exec -n $NS "$FE" -c frontend -- sh -c \
      'P=$(python3 -c "import random;print(\" \".join(str(random.random()) for _ in range(1500)))"); \
       curl -s -m 120 -X POST localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
       -d "{\"model\":\"alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":16}"' 2>/dev/null)
    echo "$R" | grep -q '"content"' && OK=$((OK+1))
  done
  say "[$TAG] completions: $OK/20"
  local ERR=0
  for tier in prefill decode; do
    local WP=$(kubectl get pods -n $NS -l app=${ARM}-${tier} -o name | head -1)
    local E=$(kubectl logs -n $NS "$WP" -c $tier --since=30m 2>/dev/null | \
              grep -cE "REMOTE_DISCONNECT|NIXL_ERR|Foreign traffic|UCX.*ERROR" || true)
    say "[$TAG] $tier error-lines: $E"
    [ "$E" -gt 0 ] && ERR=1
    # capture startup wireup evidence (fixes the tail-sampling gap)
    kubectl logs -n $NS "$WP" -c $tier 2>/dev/null | grep -E "rc_mlx5|cuda_ipc|tcp" | head -8 | \
      sed "s/^/[$TAG $tier wireup] /" >> "$LOG"
  done
  [ "$OK" -eq 20 ] && [ "$ERR" -eq 0 ]
}

say "=== transport probe: deploying 1P+1D"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-d72.yaml >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-prefill deployment/${ARM}-decode -n $NS --replicas=1 >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-frontend -n $NS --replicas=1 >> "$LOG" 2>&1

for CFG in "y:cuda_copy,rc_x:A-direct-notcp" "n:cuda_copy,rc_x:B-hoststaged-notcp" "n:cuda_copy,rc_x,tcp:C-baseline"; do
  GD=$(echo $CFG | cut -d: -f1); TLS=$(echo $CFG | cut -d: -f2); TAG=$(echo $CFG | cut -d: -f3)
  say "=== testing $TAG (GPU_DIRECT=$GD TLS=$TLS)"
  patch_env "$GD" "$TLS"
  wait_ready || { say "[$TAG] fleet failed to come up (wireup failure likely)"; continue; }
  if probe_traffic "$TAG"; then
    say "TRANSPORT VERDICT: $TAG HEALTHY (GPU_DIRECT=$GD TLS=$TLS)"
    kubectl scale deployment/${ARM}-prefill deployment/${ARM}-decode deployment/${ARM}-frontend -n $NS --replicas=0 >> "$LOG" 2>&1
    exit 0
  fi
  say "[$TAG] UNHEALTHY"
done
say "TRANSPORT VERDICT: NONE HEALTHY - manual investigation needed"
kubectl scale deployment/${ARM}-prefill deployment/${ARM}-decode deployment/${ARM}-frontend -n $NS --replicas=0 >> "$LOG" 2>&1
exit 1
