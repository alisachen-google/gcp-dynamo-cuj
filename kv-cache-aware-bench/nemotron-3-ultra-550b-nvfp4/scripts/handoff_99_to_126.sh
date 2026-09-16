#!/bin/bash
# Waits for the 9:9 AgentX c192 job, harvests it (guard + knee), then hands np-3 to the 12:6 AgentX ladders.
set -u; export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned; NS=dynamo-cloud; LOG=/tmp/agentx_99.log
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
J=alisachen-n3u-mnnvl-99-agentx-kv-c192; st=""
for i in $(seq 1 120); do st=$(kubectl get jobs -n $NS $J --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break; sleep 120; done
say "n3u-mnnvl-99 AgentX kv c192 done (job=$st) [handoff wrapper]"
if [ "$st" = "Complete" ]; then
  bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/mnnvl_transport_guard.sh n3u-mnnvl-99 >> "$LOG" 2>&1 && say "MNNVL gate PASS" || say "MNNVL TRANSPORT VIOLATION on 9:9 c192"
  python3 $HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py n3u-mnnvl-99-agentx-kv-c192 >> "$LOG" 2>&1 || true
fi
say "N3U AGENTX 99 DONE"   # gate for the 12:6 runners (which tear down 9:9 in their np-3 pre-flight)
