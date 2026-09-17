#!/bin/bash
# 64-GPU disagg AgentX programme on np-2 (16 x a4x-maxgpu-4g = 8 prefill + 8 decode TP4 workers, one MNNVL ComputeDomain).
# Points come from the v5 stream-level simulation (sim-results/dynosim_n3u_agentx_d64_v5.csv, KNEE_ANALYSIS.md "64 GPUs"):
#   KV ladder 192 / 384 / 576 / 768 / 960   (same-config cell 384, same-SLO cell 768, 960 brackets the P90 >= 20 limit)
#   RR        192 / 384 / 96                (192 = RR's best cell inside TTFT p95 <= 20 s, 384 = RR throughput peak / same config)
#   flags     kvs3c08 / kvs2c08 / kvt05 at 768, kvs3c08 at 384
# PRECONDITION: np-2 must have 16 FREE nodes. They currently belong to other tenants; do not start this until they are released.
# The runner waits on need-free-nodes=16 and never deletes anything it did not create.
set -u
D=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts; S=$D/agentx_runner.sh
M=$HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-88.yaml; W=n3u-mnnvl-88-prefill,n3u-mnnvl-88-decode
export GATE_STRICT=1 BENCH_POOL=np-2 MNNVL_GUARD=1
echo "N3U AGENTX 88 GATE $(date -u '+%F %T')" > /tmp/agentx_88_gate.log
nohup bash "$S" n3u-mnnvl-88 "$M" "kv:192 kv:384 kv:576 kv:768 kv:960" "N3U AGENTX 88 GATE" /tmp/agentx_88_gate.log "N3U AGENTX 88 KV DONE" /tmp/agentx_88.log 16 np-2 n3u-mnnvl-88 "$W" > /tmp/agentx_88.nohup 2>&1 &
echo "kv ladder pid=$!"
nohup bash "$S" n3u-mnnvl-88 "$M" "rr:192 rr:384 rr:96" "N3U AGENTX 88 KV DONE" /tmp/agentx_88.log "N3U AGENTX 88 RR DONE" /tmp/agentx_88_rr.log 0 np-2 n3u-mnnvl-88 "$W" > /tmp/agentx_88_rr.nohup 2>&1 &
echo "rr pid=$!"
nohup bash "$S" n3u-mnnvl-88 "$M" "kvs3c08:768 kvs2c08:768 kvt05:768 kvs3c08:384" "N3U AGENTX 88 RR DONE" /tmp/agentx_88_rr.log "N3U AGENTX 88 FLAGS DONE" /tmp/agentx_88_flags.log 0 np-2 n3u-mnnvl-88 "$W" > /tmp/agentx_88_flags.nohup 2>&1 &
echo "flags pid=$!"
