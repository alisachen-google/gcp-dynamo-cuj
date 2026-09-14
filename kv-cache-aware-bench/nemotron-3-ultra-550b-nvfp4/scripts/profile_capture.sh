#!/bin/bash
# profile_capture.sh <app-label-csv> <outdir> <duration_s>
# For every pod under the given app labels: nvidia-smi util every 5s, :9090/metrics
# every 10s, then the worker log (SGLang per-step scheduler lines). Zero-overhead
# layers; the "why agg wins" evidence. Container name = label suffix (agg|prefill|decode).
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
NS=dynamo-cloud; LABELS=$1; OUT=$2; DUR=${3:-3300}; mkdir -p "$OUT"; PIDS=""
for L in ${LABELS//,/ }; do
  case "$L" in *-prefill) CTR=prefill;; *-decode) CTR=decode;; *) CTR=agg;; esac
  for P in $(kubectl get pods -n $NS -l app=$L -o name); do P=${P#pod/}
    kubectl exec -n $NS "$P" -c $CTR -- nvidia-smi --query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used --format=csv,noheader -l 5 > "$OUT/$P.smi.csv" 2>/dev/null & PIDS="$PIDS $!"
    ( while :; do echo "### $(date -u +%s)"; kubectl exec -n $NS "$P" -c $CTR -- curl -s -m 5 localhost:9090/metrics 2>/dev/null; sleep 10; done ) > "$OUT/$P.metrics" 2>/dev/null & PIDS="$PIDS $!"
    echo "$P $CTR" >> "$OUT/pods.txt"
  done
done
echo "capture started $(date -u +%s) pods=$(wc -l < "$OUT/pods.txt")" > "$OUT/START"
sleep "$DUR"
kill $PIDS 2>/dev/null; sleep 2
while read P CTR; do kubectl logs -n $NS "$P" -c $CTR --since=${DUR}s > "$OUT/$P.log" 2>/dev/null; done < "$OUT/pods.txt"
echo "capture done $(date -u +%s)" > "$OUT/DONE"
