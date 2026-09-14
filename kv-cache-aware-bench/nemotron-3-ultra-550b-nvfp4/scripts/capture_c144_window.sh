#!/bin/bash
# Read-only live capture during the c144 point's MEASURE window (900s warm + 1800s measure).
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned; NS=dynamo-cloud
J=alisachen-n3u-mnnvl-kv-c144
until kubectl get pods -n $NS -l job-name=$J --no-headers 2>/dev/null | grep -q Running; do sleep 30; done
sleep 1000
bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/profile_capture.sh n3u-mnnvl-full-prefill,n3u-mnnvl-full-decode \
  $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/profiles/live-disagg-kv-c144 900
