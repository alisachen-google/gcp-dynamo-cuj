#!/bin/bash
# Profiled agg-vs-disagg comparison at each arm's peak-bounded KV cell:
#   agg  n3u-agg-prof  (24 GPU)  KV c32   |   disagg n3u-mnnvl-prof 6:12 (72 GPU) KV c48
# Same stack (0.5.16/1.4.2/FI 0.6.18). Gated behind the agg re-sweep (shared GPUs).
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
NS=dynamo-cloud; LOG=/tmp/profiled_comparison.log; OUT=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/profiles
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
GUARD=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/mnnvl_transport_guard.sh
CAP=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/profile_capture.sh
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4; N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
bench(){ # $1=ARM $2=conc $3=jobname
  kubectl delete job -n $NS "$3" --ignore-not-found --wait=true >> "$LOG" 2>&1
  sed -e "s|/model-cache/alisachen/Kimi-K2.5-NVFP4|${N3U_DIR}|g" -e "s|alisachen/Kimi-K2.5-NVFP4|${N3U_SERVED}|g" \
      -e "s|models--alisachen--Kimi-K2.5-NVFP4|models--alisachen--Nemotron-3-Ultra-550B-A55B-NVFP4|g" \
      -e "s/sgl-disagg72-kv/$1/g" -e "s/name: alisachen-sgl-d72-flagsweep/name: $3/" -e "s/alisachen-sgl-d72-flagsweep/$3/g" \
      -e "/name: CONCURRENCIES/{n;s/value: .*/value: \"$2\"/}" -e "/name: BENCHMARK_DURATION/{n;s/value: .*/value: \"1800\"/}" \
      "$TMPL" | kubectl apply -n $NS -f - >> "$LOG" 2>&1; }
waitjob(){ st=""; for i in $(seq 1 75); do st=$(kubectl get jobs -n $NS "$1" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break; sleep 120; done; echo "$st"; }
register(){ FEP=$(kubectl get pods -n $NS -l app=$1-frontend -o name | head -1); for r in $(seq 1 30); do kubectl exec -n $NS "$FEP" -c frontend -- curl -s -m 10 localhost:8000/v1/models 2>/dev/null | grep -q Nemotron && return; sleep 20; done; }
tps(){ kubectl logs -n $NS -l job-name=$1 2>/dev/null | grep -iE "Output Token Throughput \(tokens/sec\)" | grep -oE "[0-9][0-9,]*\.[0-9]+" | head -1 | tr -d ,; }
free_nodes(){ n=0; for node in $(kubectl get nodes -l cloud.google.com/gke-nodepool=np-3 -o name); do node=${node#node/}; u=$(kubectl describe node "$node" | awk '/Allocated resources/,0' | grep "nvidia.com/gpu" | awk '{print $2}'); [ "${u:-0}" = "0" ] && n=$((n+1)); done; echo $n; }

say "waiting for agg re-sweep (N3U AGG NEWSTACK SWEEP DONE)"
until grep -q "N3U AGG NEWSTACK SWEEP DONE" /tmp/resweep_agg_newstack.log 2>/dev/null; do
  grep -qE "HALTING|STACK TIMEOUT|SMOKE FAIL" /tmp/resweep_agg_newstack.log 2>/dev/null && { say "agg re-sweep halted — not proceeding"; exit 1; }; sleep 300; done
kubectl scale deployment -n $NS -l 'app in (n3u-agg-ns,n3u-agg-ns-frontend)' --replicas=0 >> "$LOG" 2>&1; sleep 90

# ---- A: agg KV c32 ----
say "=== A: profiled AGG KV c32"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-agg-prof.yaml >> "$LOG" 2>&1
for d in n3u-agg-prof-frontend n3u-agg-prof; do kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT $d"; exit 1; }; done
register n3u-agg-prof; sleep 60
bench n3u-agg-prof 32 alisachen-n3u-agg-prof-kv-c32
bash "$CAP" n3u-agg-prof "$OUT/agg-kv-c32" 3300 >> "$LOG" 2>&1 &
st=$(waitjob alisachen-n3u-agg-prof-kv-c32); ATPS=$(tps alisachen-n3u-agg-prof-kv-c32); say "A done (job=$st) tok/s=$ATPS"; wait
[ "$st" != "Complete" ] && { say "A BENCH FAIL - HALTING"; exit 2; }
python3 $HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py n3u-agg-prof-kv-c32 >> "$LOG" 2>&1 || true
for d in n3u-agg-prof-frontend n3u-agg-prof; do kubectl delete deployment/$d -n $NS --wait=false >> "$LOG" 2>&1; done; sleep 120

# ---- B: disagg 6:12 KV c48 ----
say "=== B: profiled DISAGG 6:12 KV c48"
while :; do F=$(free_nodes); say "free np-3 nodes: $F / 18"; [ "$F" -ge 18 ] && break; sleep 300; done
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-prof.yaml >> "$LOG" 2>&1
for d in n3u-mnnvl-prof-prefill n3u-mnnvl-prof-decode n3u-mnnvl-prof-frontend; do kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT $d"; exit 1; }; done
register n3u-mnnvl-prof; sleep 60
bench n3u-mnnvl-prof 48 alisachen-n3u-mnnvl-prof-kv-c48
bash "$CAP" n3u-mnnvl-prof-prefill,n3u-mnnvl-prof-decode "$OUT/disagg-kv-c48" 3300 >> "$LOG" 2>&1 &
st=$(waitjob alisachen-n3u-mnnvl-prof-kv-c48); DTPS=$(tps alisachen-n3u-mnnvl-prof-kv-c48); say "B done (job=$st) tok/s=$DTPS"; wait
[ "$st" != "Complete" ] && { say "B BENCH FAIL - HALTING"; exit 2; }
bash "$GUARD" n3u-mnnvl-prof >> "$LOG" 2>&1 || { say "B MNNVL VIOLATION - HALTING"; exit 2; }
python3 $HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py n3u-mnnvl-prof-kv-c48 >> "$LOG" 2>&1 || true
for d in n3u-mnnvl-prof-prefill n3u-mnnvl-prof-decode n3u-mnnvl-prof-frontend; do kubectl delete deployment/$d -n $NS --wait=false >> "$LOG" 2>&1; done
kubectl delete computedomain/n3u-mnnvl-prof-cd -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1

python3 $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/analyze_profiles.py "$OUT/agg-kv-c32" "$OUT/disagg-kv-c48" --agg-tps "${ATPS:-0}" --disagg-tps "${DTPS:-0}" > "$OUT/GAP_ANALYSIS.md" 2>> "$LOG"
say "PROFILED COMPARISON DONE"
