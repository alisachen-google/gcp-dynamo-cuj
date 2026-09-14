#!/bin/bash
# render_agg_template.sh kv|rr [out.yaml]  — render the all-in-one agg template.
# Override any var via env, e.g. IMAGE=... DYNAMO_VER=1.3.1 ./render_agg_template.sh rr
set -eu
MODE=${1:?kv|rr}; OUT=${2:-/dev/stdout}
case "$MODE" in
  kv) export ROUTER_FLAGS="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs";;
  rr) export ROUTER_FLAGS="--router-mode round-robin";;
  *) echo "mode must be kv|rr" >&2; exit 1;;
esac
export ARM=${ARM:-n3u-agg-$MODE} NS=${NS:-dynamo-cloud} POOL=${POOL:-np-3}
export IMAGE=${IMAGE:-lmsysorg/sglang:v0.5.19-cu130-runtime} DYNAMO_VER=${DYNAMO_VER:-1.4.2} FLASHINFER_VER=${FLASHINFER_VER:-0.6.18}
export MODEL_DIR=${MODEL_DIR:-/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4} SERVED_NAME=${SERVED_NAME:-alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4}
export WORKERS=${WORKERS:-6} TP=${TP:-4} EP=${EP:-4} MAX_RUNNING=${MAX_RUNNING:-16}
python3 - "$(dirname "$0")/../templates/n3u-agg-serving-template.yaml" > "$OUT" <<'PY2'
import os,re,sys
allowed="ARM NS POOL IMAGE DYNAMO_VER FLASHINFER_VER ROUTER_FLAGS MODEL_DIR SERVED_NAME WORKERS TP EP MAX_RUNNING".split()
t=open(sys.argv[1]).read()
print(re.sub(r"\$\{(\w+)\}", lambda m: os.environ[m[1]] if m[1] in allowed else m[0], t), end="")
PY2
