#!/bin/bash
set -e
trap 'echo "Cleaning up..."; kill 0 2>/dev/null || true' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"/tmp/n3u"}
export HF_TOKEN=${HF_TOKEN:-"None"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"/tmp/n3u"}
export HEAD_NODE_IP=${HEAD_NODE_IP:-"0.0.0.0"}
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

FRONTEND_SYSTEM_PORT=${FRONTEND_SYSTEM_PORT:-8080}
AGG_SYSTEM_PORT=${AGG_SYSTEM_PORT:-8081}
PREFILL_WORKERS=1
DECODE_WORKERS=0
PREFILL_SYSTEM_PORT_BASE=${PREFILL_SYSTEM_PORT_BASE:-8082}
DECODE_SYSTEM_PORT_BASE=${DECODE_SYSTEM_PORT_BASE:-$((PREFILL_SYSTEM_PORT_BASE + PREFILL_WORKERS))}

SGLANG_KV_EVENT_PORT_BASE=${SGLANG_KV_EVENT_PORT_BASE:-5557}
OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend --router-mode kv --http-port "8000" 2>&1 | sed "s/^/[Frontend] /" &

PREFILL_GPU=4
for ((w=0; w<PREFILL_WORKERS; w++)); do
  BASE=$(( w * PREFILL_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+PREFILL_GPU-1)))
  WORKER_IDX=$(( w + 1 ))
  SYSTEM_PORT=$(( PREFILL_SYSTEM_PORT_BASE + w ))
  WORKER_NAME="dynamo-worker-prefill"
  if (( PREFILL_WORKERS > 1 )); then
    WORKER_NAME="${WORKER_NAME}-${WORKER_IDX}"
  fi
  EVENT_PORT=$(( SGLANG_KV_EVENT_PORT_BASE + w ))
  (
  CUDA_VISIBLE_DEVICES=$GPU_LIST \
  OTEL_SERVICE_NAME="${WORKER_NAME}" \
  DYN_SYSTEM_PORT="${SYSTEM_PORT}" \
    python3 -m dynamo.sglang \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --tensor-parallel-size 4 --pipeline-parallel-size 1 --data-parallel-size 1 --kv-cache-dtype auto --max-running-requests 1 --max-prefill-tokens 97500 --expert-parallel-size 1 --disaggregation-transfer-backend nixl --moe-runner-backend deepep_moe --disaggregation-mode prefill \
      --host "0.0.0.0" \
      --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${EVENT_PORT}\"}" \
      --enable-metrics 2>&1 | sed "s/^/[Prefill $w] /" ) &
done

wait
