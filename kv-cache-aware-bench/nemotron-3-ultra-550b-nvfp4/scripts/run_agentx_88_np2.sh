#!/bin/bash
# 64-GPU disagg AgentX programme on np-2 (16 x a4x-maxgpu-4g = 8 prefill + 8 decode TP4 workers, one MNNVL ComputeDomain).
#   stage 1  concurrency sweep, KV-aware : kv 192 / 384 / 672 / 768 / 1152  (first = smoke; 672 default-KV SLO cell, 768 tuned SLO cell, 1152 KV throughput knee)
#   stage 1b KV knee extension (agentx_88_kv_knee_extend.sh): while the largest KV cell's KNEE-CHECK is not POST-KNEE, run
#            kv 1536 / 2048 / 2560 one at a time, fleet kept up between cells -> "KVX DONE"
#   stage 2  concurrency sweep, RR       : rr 192 / 384 / 240 / 96          (192 last RR cell inside TTFT p95 <= 20 s, 240 first outside, 384 RR peak)
#            then kv 240 / 96 so EVERY RR cell has a KV cell at the same client count (same-config pairs: 96, 192, 240, 384)
#   stage 3  pick the KV-vs-RR comparison cells from the MEASURED ladders (select_agentx_points.py: same config = RR's peak
#            cell, same SLO = best cell per policy with TTFT p95 <= 20 s, P90 >= 20, stationary) -> /tmp/agentx_88_points.md
#   stage 4  KV-router flag sweep at the selected cells (scale 3 / scale 2 credit 0.8, temperature 0.5 at the KV same-SLO
#            cell; tuned router at the same-config cell).  Falls back to the sim's choice (768 / 384) if selection is empty.
#   stage 5  rr 672 / 768 : RR at the KV same-SLO client counts (default-KV and tuned-KV cells) (expected post-knee; last, so a failure cannot block stages 3-4)
# SKIP_STAGE1=1 relaunches stages 2-5 only (stage 1 already running).
# Ladder client counts come from the v5 stream-level simulation (KNEE_ANALYSIS.md "64-GPU disagg").
# PRECONDITION: stage 1 waits until 16 READY np-2 nodes carry no GPU requests.  Those nodes belong to other tenants today;
# this script never deletes or preempts anything it did not create.
set -u
D=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts; S=$D/agentx_runner.sh
M=$HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-88.yaml; W=n3u-mnnvl-88-prefill,n3u-mnnvl-88-decode
export GATE_STRICT=1 BENCH_POOL=np-2 MNNVL_GUARD=1 JOB_WAIT_ITERS=160
KV="192 384 672 768 1152"; RR="192 384 240 96"; KV2="240 96"; RRX="672 768"
if [ "${SKIP_STAGE1:-0}" != "1" ]; then
echo "N3U AGENTX 88 GATE $(date -u '+%F %T')" > /tmp/agentx_88_gate.log
nohup bash "$S" n3u-mnnvl-88 "$M" "$(for c in $KV; do printf 'kv:%s ' $c; done)" "N3U AGENTX 88 GATE" /tmp/agentx_88_gate.log "N3U AGENTX 88 KV DONE" /tmp/agentx_88.log 16 np-2 n3u-mnnvl-88 "$W" > /tmp/agentx_88.nohup 2>&1 &
echo "stage 1 (KV ladder) pid=$!"
fi
nohup bash "$D/agentx_88_kv_knee_extend.sh" > /tmp/agentx_88_kvx.nohup 2>&1 &
echo "stage 1b (KV knee extension) pid=$!"
nohup bash "$S" n3u-mnnvl-88 "$M" "$(for c in $RR; do printf 'rr:%s ' $c; done; for c in $KV2; do printf 'kv:%s ' $c; done)" "N3U AGENTX 88 KVX DONE" /tmp/agentx_88_kvx.log "N3U AGENTX 88 RR DONE" /tmp/agentx_88_rr.log 0 np-2 n3u-mnnvl-88 "$W" > /tmp/agentx_88_rr.nohup 2>&1 &
echo "stage 2 (RR ladder + paired KV cells) pid=$!"
nohup bash -c "
  until grep -q 'N3U AGENTX 88 RR DONE' /tmp/agentx_88_rr.log 2>/dev/null; do sleep 300; done
  X=\$(grep -o 'KVX CELL c[0-9]* DONE' /tmp/agentx_88_kvx.log 2>/dev/null | grep -o '[0-9][0-9]*' | sort -un | tr '\n' ',' | sed 's/,\$//')
  python3 $D/select_agentx_points.py n3u-mnnvl-88 64 \"${KV// /,},${KV2// /,}\${X:+,\$X}\" '${RR// /,}' --logs /tmp/agentx_88.log /tmp/agentx_88_kvx.log /tmp/agentx_88_rr.log --out-md /tmp/agentx_88_points.md --out-points /tmp/agentx_88_flag_points.txt > /tmp/agentx_88_select.log 2>&1
  PTS=\$(cat /tmp/agentx_88_flag_points.txt 2>/dev/null); [ -z \"\$PTS\" ] && PTS='kvs3c08:768 kvs2c08:768 kvt05:768 kvs3c08:384'
  echo \"[\$(date -u '+%F %T')] N3U AGENTX 88 SELECT DONE points: \$PTS\" >> /tmp/agentx_88_select.log
  exec bash $S n3u-mnnvl-88 $M \"\$PTS\" 'N3U AGENTX 88 SELECT DONE' /tmp/agentx_88_select.log 'N3U AGENTX 88 FLAGS DONE' /tmp/agentx_88_flags.log 0 np-2 n3u-mnnvl-88 $W
" > /tmp/agentx_88_flags.nohup 2>&1 &
echo "stage 3+4 (select, then flag sweep) pid=$!"
nohup bash "$S" n3u-mnnvl-88 "$M" "$(for c in $RRX; do printf 'rr:%s ' $c; done)" "N3U AGENTX 88 FLAGS DONE" /tmp/agentx_88_flags.log "N3U AGENTX 88 RRX DONE" /tmp/agentx_88_rrx.log 0 np-2 n3u-mnnvl-88 "$W" > /tmp/agentx_88_rrx.nohup 2>&1 &
echo "stage 5 (RR at the KV same-SLO client count) pid=$!"
