#!/bin/bash
set -e
trap 'echo "Cleaning up..."; kill 0 2>/dev/null || true' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"/tmp/n3u-fp4"}
export HF_TOKEN=${HF_TOKEN:-"None"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"/tmp/n3u-fp4"}
export HEAD_NODE_IP=${HEAD_NODE_IP:-"0.0.0.0"}
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

FRONTEND_SYSTEM_PORT=${FRONTEND_SYSTEM_PORT:-8080}
AGG_SYSTEM_PORT=${AGG_SYSTEM_PORT:-8081}
PREFILL_WORKERS=0
DECODE_WORKERS=1
PREFILL_SYSTEM_PORT_BASE=${PREFILL_SYSTEM_PORT_BASE:-8082}
DECODE_SYSTEM_PORT_BASE=${DECODE_SYSTEM_PORT_BASE:-$((PREFILL_SYSTEM_PORT_BASE + PREFILL_WORKERS))}

SGLANG_KV_EVENT_PORT_BASE=${SGLANG_KV_EVENT_PORT_BASE:-5557}


DECODE_GPU=8
DECODE_GPU_OFFSET=0
for ((w=0; w<DECODE_WORKERS; w++)); do
  BASE=$(( DECODE_GPU_OFFSET + w * DECODE_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+DECODE_GPU-1)))
  WORKER_IDX=$(( w + 1 ))
  SYSTEM_PORT=$(( DECODE_SYSTEM_PORT_BASE + w ))
  WORKER_NAME="dynamo-worker-decode"
  if (( DECODE_WORKERS > 1 )); then
    WORKER_NAME="${WORKER_NAME}-${WORKER_IDX}"
  fi
  EVENT_PORT=$(( SGLANG_KV_EVENT_PORT_BASE + PREFILL_WORKERS + w ))
  ( CUDA_VISIBLE_DEVICES=$GPU_LIST \
  OTEL_SERVICE_NAME="${WORKER_NAME}" \
  DYN_SYSTEM_PORT="${SYSTEM_PORT}" \
    python3 -m dynamo.sglang \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1 --kv-cache-dtype fp8_e4m3 --max-running-requests 512 --max-prefill-tokens 97500 --expert-parallel-size 1 --disaggregation-transfer-backend nixl --cuda-graph-bs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 16 20 24 28 32 32 40 48 56 64 64 80 96 112 128 128 160 192 224 256 256 320 384 448 512 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --moe-runner-backend deepep_moe --disaggregation-mode decode \
      --host "0.0.0.0" \
      --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${EVENT_PORT}\"}" \
      --enable-metrics 2>&1 | sed "s/^/[Decode $w] /" ) &
done
wait
