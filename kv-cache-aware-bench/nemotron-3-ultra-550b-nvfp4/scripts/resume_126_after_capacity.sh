#!/bin/bash
# Resume the 12:6 AgentX programme after the 2026-09-16 23:15 UTC node-pool resize.
# Chain (each gated on the previous DONE marker, GATE_STRICT so a HALTING never opens a gate):
#   1. KV 1440 (the interrupted knee cell)        -> "N3U AGENTX 126 RESUME DONE"  /tmp/agentx_126_resume.log
#   2. RR 192 / 96 / 384 (same-config + same-SLO)  -> "N3U AGENTX 126 RR DONE"      /tmp/agentx_126_rr.log
#   3. flag sweep kvs3c08/kvs2c08/kvt05 x 480,192  -> "N3U AGENTX 126 FLAGS DONE"   /tmp/agentx_126_flags.log
# Run this only once np-3 shows >= 18 Ready nodes (kubectl get nodes -l cloud.google.com/gke-nodepool=np-3): the 12:6
# deployments are still present and will grab the nodes as they appear, so need-free-nodes is 0 here and the runner goes
# straight to the manifest apply + rollout wait.
set -u
S="$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/agentx_runner.sh"
M="$HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-126.yaml"
W="n3u-mnnvl-126-prefill,n3u-mnnvl-126-decode"
echo "N3U AGENTX 126 RESUME GATE $(date -u '+%F %T')" > /tmp/agentx_126_resume_gate.log
export GATE_STRICT=1
nohup bash "$S" n3u-mnnvl-126 "$M" "kv:1440"                     "N3U AGENTX 126 RESUME GATE" /tmp/agentx_126_resume_gate.log "N3U AGENTX 126 RESUME DONE" /tmp/agentx_126_resume.log 0 np-3 n3u-mnnvl-126 "$W" > /tmp/agentx_126_resume.nohup 2>&1 &
echo "kv:1440 runner pid=$!"
nohup bash "$S" n3u-mnnvl-126 "$M" "rr:192 rr:96 rr:384"          "N3U AGENTX 126 RESUME DONE" /tmp/agentx_126_resume.log      "N3U AGENTX 126 RR DONE"     /tmp/agentx_126_rr.log     0 np-3 n3u-mnnvl-126 "$W" > /tmp/agentx_126_rr.nohup 2>&1 &
echo "rr runner pid=$!"
nohup bash "$S" n3u-mnnvl-126 "$M" "kvs3c08:480 kvs2c08:480 kvt05:480 kvs3c08:192 kvs2c08:192 kvt05:192" "N3U AGENTX 126 RR DONE" /tmp/agentx_126_rr.log "N3U AGENTX 126 FLAGS DONE" /tmp/agentx_126_flags.log 0 np-3 n3u-mnnvl-126 "$W" > /tmp/agentx_126_flags.nohup 2>&1 &
echo "flags runner pid=$!"
