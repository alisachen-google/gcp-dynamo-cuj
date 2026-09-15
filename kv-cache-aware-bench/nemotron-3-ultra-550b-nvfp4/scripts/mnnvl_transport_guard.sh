#!/bin/bash
# MNNVL transport health gate — v2 (2026-09-15). Runs AFTER each bench point.
# The KV path on MNNVL is MOONCAKE (--disaggregation-transfer-backend mooncake,
# MC_FORCE_MNNVL=1) — NOT UCX/NIXL (0 UCX lines in pods; UCX_* env is inert).
# v1's positive check counted "cuda_ipc|mnnvl" log lines, which matched the
# deployment NAME (n3u-mnnvl-*) in ordinary request logs -> always passed.
# v2 evidence (all valid post-run, no timing dependence):
#   NEG  no transfer failures / explicit fallback in the run window   (proven: caught rr:288)
#   POS1 mooncake Transfer Engine moved KV: max "Throughput: X MB/s" in the log window > 0
#   POS2 RDMA path physically absent in the pod: /dev/infiniband missing (no mrdma claim)
#   POS3 mooncake forced to MNNVL: MC_FORCE_MNNVL=1 in the pod environment
# For live, during-run physical proof (NVLink Rx counters up, NIC idle) use
# mnnvl_live_probe.sh — not part of the gate because links are idle after a run.
# Usage: mnnvl_transport_guard.sh <pod-grep-pattern>   # exit 0 healthy / 1 violation
set -uo pipefail
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
NS=dynamo-cloud; PAT=$1; viol=0; te_max_all=0; checked=0
for pod in $(kubectl get pods -n $NS --no-headers 2>/dev/null | grep -E "$PAT" | grep -E "prefill|decode" | grep Running | awk '{print $1}'); do
  ctr=prefill; echo "$pod" | grep -q decode && ctr=decode; checked=$((checked+1))
  log=$(kubectl logs -n $NS "$pod" -c $ctr --tail=40000 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')
  # NEG: real transfer failures / fallback (mooncake + sglang disagg + any UCX/NIXL remnants)
  # exact failure signatures only (generic "transfer failed"/"falling back" matched ~361 benign
  # startup lines on fresh prefill pods with 0 request errors -> false halts on 2026-09-15)
  neg=$(echo "$log" | grep -E "Decode transfer failed for request|Prefill transfer failed for request|KVTransferError|REMOTE_DISCONNECT|NIXL_ERR|fallback to tcp")
  if [ -n "$neg" ]; then
    n=$(echo "$neg" | wc -l)
    echo "MNNVL-GUARD VIOLATION [$pod]: $n transfer-failure lines; e.g.: $(echo "$neg" | head -2 | cut -c1-160 | tr '\n' '|')"; viol=1
  fi
  # POS1: mooncake actually moved bytes (prefill side sends; decode side may report 0)
  te=$(echo "$log" | grep -oE "Transfer Engine Stats.*Throughput: [0-9.]+ MB/s" | grep -oE "[0-9.]+ MB/s" | awk '{if($1>m)m=$1} END{print m+0}')
  te_max_all=$(awk -v a="$te" -v b="$te_max_all" 'BEGIN{print (a+0>b+0)?a+0:b+0}')
  # POS2: no RDMA device nodes in the pod -> RDMA/UCX KV path physically impossible
  if kubectl exec -n $NS "$pod" -c $ctr -- test -e /dev/infiniband 2>/dev/null; then
    echo "MNNVL-GUARD VIOLATION [$pod]: /dev/infiniband present — an RDMA path exists in this pod"; viol=1
  fi
  # POS3: mooncake pinned to MNNVL
  if ! kubectl exec -n $NS "$pod" -c $ctr -- sh -c 'env | grep -q "^MC_FORCE_MNNVL=1$"' 2>/dev/null; then
    echo "MNNVL-GUARD VIOLATION [$pod]: MC_FORCE_MNNVL=1 not set"; viol=1
  fi
done
[ "$checked" -eq 0 ] && { echo "MNNVL-GUARD: no running prefill/decode pods matched '$PAT'"; exit 1; }
[ "$(awk -v a="$te_max_all" 'BEGIN{print (a+0>0)?1:0}')" = "1" ] || { echo "MNNVL-GUARD VIOLATION: mooncake Transfer Engine reported no throughput in the log window — KV did not move via mooncake"; viol=1; }
[ "$viol" -eq 0 ] && { echo "MNNVL-GUARD PASS (pods=$checked; mooncake TE peak ${te_max_all} MB/s; no /dev/infiniband; MC_FORCE_MNNVL=1; no transfer failures)"; exit 0; } || exit 1
