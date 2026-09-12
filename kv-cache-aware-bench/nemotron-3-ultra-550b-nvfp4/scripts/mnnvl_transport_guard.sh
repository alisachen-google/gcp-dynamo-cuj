#!/bin/bash
# MNNVL transport health gate (2026-09-12): for the MNNVL+mooncake KV path,
# HEALTHY = cuda_ipc/MNNVL evidence present, NO tcp on the CUDA data path, NO
# fallback/disconnect. (Inverse of the RDMA guard, which required rc_mlx5.)
# Usage: mnnvl_transport_guard.sh <pod-grep-pattern>   # exit 0 healthy / 1 violation
set -uo pipefail
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud; PAT=$1
viol=0; ipc_total=0
for pod in $(kubectl get pods -n $NS --no-headers 2>/dev/null | grep -E "$PAT" | grep -E "prefill|decode" | grep Running | awk '{print $1}'); do
  log=$(kubectl logs -n $NS "$pod" --tail=20000 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')
  # hard violations: transfer aborts / explicit fallback
  if echo "$log" | grep -qE "REMOTE_DISCONNECT|NIXL_ERR|transfer failed|falling back|fallback to tcp"; then
    echo "MNNVL-GUARD VIOLATION [$pod]: transfer error / fallback"; viol=1
  fi
  # tcp on a CUDA data-path proto row = host/tcp datapath (wireup tcp is benign)
  if echo "$log" | grep -E "cuda" | grep -qE "tcp/[a-z0-9]+.*(zero-copy|rndv|copy-in|multi-frag)|(zero-copy|rndv|copy-in|multi-frag).*tcp/[a-z0-9]+"; then
    echo "MNNVL-GUARD VIOLATION [$pod]: tcp on CUDA data path"; viol=1
  fi
  ipc=$(echo "$log" | grep -icE "cuda_ipc|mnnvl" || true)
  rdma=$(echo "$log" | grep -c "rc_mlx5" || true)
  ipc_total=$((ipc_total+ipc))
  # if the KV path went to rc_mlx5 with no cuda_ipc, it fell to (host-staged) RDMA
  if [ "$ipc" -eq 0 ] && [ "$rdma" -gt 0 ]; then
    echo "MNNVL-GUARD VIOLATION [$pod]: rc_mlx5 present, cuda_ipc absent — fell to RDMA not MNNVL"; viol=1
  fi
done
[ "$ipc_total" -eq 0 ] && { echo "MNNVL-GUARD: no cuda_ipc/MNNVL evidence — transport unproven"; viol=1; }
[ "$viol" -eq 0 ] && { echo "MNNVL-GUARD PASS (cuda_ipc evidence=$ipc_total, no tcp/rdma/fallback on data path)"; exit 0; } || exit 1
