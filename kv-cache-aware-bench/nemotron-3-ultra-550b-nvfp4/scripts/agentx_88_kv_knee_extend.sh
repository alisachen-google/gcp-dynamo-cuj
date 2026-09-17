#!/bin/bash
# 8p/8d AgentX: extend the KV ladder until the knee is found.  After the planned KV ladder finishes, look at the KNEE-CHECK
# verdict of the largest KV cell run so far; while it is not POST-KNEE, run the next larger client count (one cell at a time).
# Writes "N3U AGENTX 88 KVX DONE" when the knee is found or the candidates are exhausted; a failed extension cell also ends
# the extension (the chain continues), but an MNNVL transport violation stops everything (no DONE marker).
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
D=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts; S=$D/agentx_runner_keep.sh
M=$HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-88.yaml; W=n3u-mnnvl-88-prefill,n3u-mnnvl-88-decode
export GATE_STRICT=1 BENCH_POOL=np-2 MNNVL_GUARD=1 JOB_WAIT_ITERS=160 KEEP_FLEET=1
LOG=/tmp/agentx_88_kvx.log; LAST=${LAST_PLANNED:-1152}; CAND=${KVX_CANDIDATES:-"1536 2048 2560"}
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
say "waiting for gate 'N3U AGENTX 88 KV DONE' in /tmp/agentx_88.log"
until grep -q "N3U AGENTX 88 KV DONE$" /tmp/agentx_88.log 2>/dev/null; do sleep 60; done
for C in $CAND; do
  V=$(cat /tmp/agentx_88.log "$LOG" 2>/dev/null | grep "KNEE-CHECK n3u-mnnvl-88-agentx-kv-c${LAST}:" | tail -1)
  say "largest KV cell c$LAST -> ${V:-no verdict}"
  echo "$V" | grep -q "POST-KNEE" && { say "KV knee found at or before c$LAST - no further extension"; break; }
  say "KV knee not found by c$LAST - extending to c$C"
  bash "$S" n3u-mnnvl-88 "$M" "kv:$C" "N3U AGENTX 88 KV DONE" /tmp/agentx_88.log "N3U AGENTX 88 KVX CELL c$C DONE" "$LOG" 0 np-2 n3u-mnnvl-88 "$W"; rc=$?
  grep -q "MNNVL TRANSPORT VIOLATION" "$LOG" && { say "MNNVL violation in extension - chain stopped"; exit 2; }
  [ $rc -ne 0 ] && { say "extension cell c$C did not complete (rc=$rc) - ending extension"; break; }
  LAST=$C
done
for d in n3u-mnnvl-88-prefill n3u-mnnvl-88-decode n3u-mnnvl-88-frontend; do kubectl scale deployment/$d -n dynamo-cloud --replicas=0 >> "$LOG" 2>&1; done
say "N3U AGENTX 88 KVX DONE"
