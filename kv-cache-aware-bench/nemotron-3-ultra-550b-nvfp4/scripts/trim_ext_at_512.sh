#!/bin/bash
# User scope 2026-09-14: extended 6:12 sweep = c144..512 (kv+rr). Runner order is
# ...rr:512 -> kv:768; stop after rr:512's knee line, scale down, write DONE marker.
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
NS=dynamo-cloud; ARM=n3u-mnnvl-full; LOG=/tmp/resweep_mnnvl_ext.log
until grep -q "KNEE-CHECK n3u-mnnvl-rr-c512" "$LOG" 2>/dev/null; do
  grep -qE "VIOLATION|HALTING|STACK TIMEOUT|WRONG CONTEXT" "$LOG" 2>/dev/null && exit 1
  sleep 30; done
for p in $(pgrep -f "[r]esweep_mnnvl_d72_ext"); do kill $p; done
kubectl delete job -n $NS alisachen-n3u-mnnvl-kv-c768 --ignore-not-found --wait=false >/dev/null 2>&1
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do kubectl scale deployment/$d -n $NS --replicas=0 >/dev/null 2>&1; done
echo "[$(date -u +%F' '%H:%M:%S)] trimmed at rr:512 per user scope (c144-512); fleet scaled down" >> "$LOG"
echo "[$(date -u +%F' '%H:%M:%S)] N3U MNNVL EXT SWEEP DONE" >> "$LOG"
