#!/bin/bash
# Follows one MTP decode + prefill worker; saves the first traceback/scheduler-exception it sees to /tmp/mtp_first_crash.log.
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned; NS=dynamo-cloud; OUT=/tmp/mtp_first_crash.log; : > $OUT
until [ "$(kubectl get pods -n $NS -l app=n3u-mnnvl-99mtp-decode --no-headers 2>/dev/null | grep -c Running)" -ge 9 ]; do sleep 60; done
for tier in decode prefill; do
  P=$(kubectl get pods -n $NS -l app=n3u-mnnvl-99mtp-$tier -o name | head -1)
  ( kubectl logs -n $NS $P -c $tier -f --tail=0 2>/dev/null | sed -u 's/\x1b\[[0-9;]*m//g' | grep --line-buffered -iE -A80 "Traceback|Scheduler hit an exception|CUDA error|illegal memory|out of memory|Xid|sigquit|assert" | head -400 > $OUT.$tier ) &
done
wait
