#!/bin/bash
# dsr1-faithful GPUDirect smoke (user directive 2026-09-11): replicate the
# EXACT pre-rebuild dsr1 RDMA config — UCX_TLS=cuda_copy,rc_x,tcp and NO
# UCX_IB_GPU_DIRECT_RDMA variable (UCX default = GPUDirect enabled) — on a
# 1P+1D n3u fleet, and test whether healthy direct RDMA works now.
# Distinct from the 09-09 probe config A (which tested =y with no-tcp TLS).
# Chains behind the deep-conc campaign (shares np-3 + deployment names).
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
ARM=n3u-d72
LOG=/tmp/dsr1_gdr_smoke.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }

say "waiting for DEEP CONC DONE"
until grep -q "DEEP CONC DONE" /tmp/deep_conc.log 2>/dev/null; do
  grep -qE "VIOLATION|STACK TIMEOUT" /tmp/deep_conc.log 2>/dev/null && { say "deep-conc halted; proceeding to smoke anyway"; break; }
  sleep 600
done

say "=== deploying 1P+1D with dsr1-faithful env"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-d72.yaml >> "$LOG" 2>&1
# dsr1-faithful: TLS back to rc_x+tcp, GPU_DIRECT variable REMOVED (default=on)
for d in ${ARM}-prefill ${ARM}-decode; do
  kubectl set env deployment/$d -n $NS -c ${d##*-} \
    UCX_TLS="cuda_copy,rc_x,tcp" UCX_IB_GPU_DIRECT_RDMA- >> "$LOG" 2>&1
done
kubectl scale deployment/${ARM}-prefill deployment/${ARM}-decode -n $NS --replicas=1 >> "$LOG" 2>&1
kubectl scale deployment/${ARM}-frontend -n $NS --replicas=1 >> "$LOG" 2>&1
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; exit 1; }
done
sleep 120
# verify the env actually landed in pod specs (hard-won lesson)
for tier in prefill decode; do
  WP=$(kubectl get pods -n $NS -l app=${ARM}-${tier} -o name | head -1)
  kubectl get "$WP" -n $NS -o json | python3 -c "
import json,sys
p=json.load(sys.stdin)
c=[x for x in p['spec']['containers'] if x['name']=='${tier}'][0]
env={e['name']:e.get('value') for e in c['env']}
print('[env-check ${tier}] TLS=', env.get('UCX_TLS'), ' GPU_DIRECT=', env.get('UCX_IB_GPU_DIRECT_RDMA','ABSENT(default=on)'))" >> "$LOG" 2>&1
done

say "probe traffic: 20 requests"
FE=$(kubectl get pods -n $NS -l app=${ARM}-frontend -o name | head -1)
OK=0
for i in $(seq 1 20); do
  R=$(kubectl exec -n $NS "$FE" -c frontend -- sh -c \
    'P=$(python3 -c "import random;print(\" \".join(str(random.random()) for _ in range(1500)))"); \
     curl -s -m 120 -X POST localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
     -d "{\"model\":\"alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":16}"' 2>/dev/null)
  echo "$R" | grep -q '"content"' && OK=$((OK+1))
done
say "completions: $OK/20"
ERR=0
for tier in prefill decode; do
  WP=$(kubectl get pods -n $NS -l app=${ARM}-${tier} -o name | head -1)
  E=$(kubectl logs -n $NS "$WP" -c $tier --since=40m 2>/dev/null | grep -cE "REMOTE_DISCONNECT|NIXL_ERR|Foreign traffic|UCX.*ERROR" || true)
  say "$tier error-lines: $E"; [ "$E" -gt 0 ] && ERR=1
  # proto-table memtype evidence: GDR shows zero-copy rows on cuda memory
  kubectl logs -n $NS "$WP" -c $tier 2>/dev/null | grep -E "rc_mlx5|cuda" | grep -iE "zero-copy|rndv|copy-in" | head -6 | \
    sed "s/^/[proto $tier] /" >> "$LOG"
done
if [ "$OK" -eq 20 ] && [ "$ERR" -eq 0 ]; then
  say "GDR SMOKE VERDICT: HEALTHY — GPUDirect works with the dsr1-faithful config"
else
  say "GDR SMOKE VERDICT: STILL BROKEN ($OK/20, err=$ERR) — regression persists even with dsr1-exact config"
fi
kubectl scale deployment/${ARM}-prefill deployment/${ARM}-decode deployment/${ARM}-frontend -n $NS --replicas=0 >> "$LOG" 2>&1
# restore certified env on the deployments for whoever deploys next
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-d72.yaml >> "$LOG" 2>&1
say "DSR1 GDR SMOKE DONE"
