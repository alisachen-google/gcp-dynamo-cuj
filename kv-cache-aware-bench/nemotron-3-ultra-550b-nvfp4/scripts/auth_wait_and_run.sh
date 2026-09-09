#!/bin/bash
# Orchestrator: wait for the user's ADC handoff, then chain the whole goal:
# install creds -> verify cluster -> transport probe -> adopt winning UCX env
# into the generator (regenerate manifests) -> full d72 re-sweep.
set -u
LOG=/tmp/auth_chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" >> "$LOG"; }
ADC=~/.config/gcloud/application_default_credentials.json
BUCKET_OBJ=gs://alisachen-models/adc-transfer.json
GEN=$HOME/kv-cache-aware-bench/sglang/scripts/gen_sglang_arms.py
T0=$(date -u +%s)

say "waiting for fresh ADC at $BUCKET_OBJ (uploaded after $(date -u))"
for i in $(seq 1 288); do   # up to 24h, 5-min polls
  TS=$(gcloud storage objects describe "$BUCKET_OBJ" --format="value(updateTime)" 2>/dev/null)
  if [ -n "$TS" ]; then
    TSE=$(date -u -d "$TS" +%s 2>/dev/null || echo 0)
    if [ "$TSE" -gt "$T0" ]; then
      say "fresh ADC found (updated $TS); installing"
      gcloud storage cp "$BUCKET_OBJ" "$ADC" >> "$LOG" 2>&1
      gcloud storage rm "$BUCKET_OBJ" >> "$LOG" 2>&1
      say "bucket copy deleted"
      break
    fi
  fi
  sleep 300
done
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
if ! timeout 60 kubectl get nodes >/dev/null 2>&1; then
  say "AUTH CHAIN FAILED: cluster still unreachable after ADC wait"
  exit 1
fi
NP3=$(kubectl get nodes -l cloud.google.com/gke-nodepool=np-3 --no-headers 2>/dev/null | awk '$2=="Ready"' | wc -l)
say "cluster reachable; np-3 Ready nodes: $NP3"
[ "$NP3" -lt 18 ] && say "WARNING: np-3 has $NP3 Ready nodes (<18) — resweep may queue"

say "=== running transport probe"
rm -f /tmp/transport_probe.log
bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/transport_probe.sh
PR=$?
VERDICT=$(grep "TRANSPORT VERDICT" /tmp/transport_probe.log | tail -1)
say "probe exit=$PR: $VERDICT"
[ "$PR" -ne 0 ] && { say "AUTH CHAIN HALTED: no healthy transport config"; exit 2; }

GD=$(echo "$VERDICT" | grep -o "GPU_DIRECT=[yn]" | cut -d= -f2)
TLS=$(echo "$VERDICT" | grep -o "TLS=[a-z_,]*" | cut -d= -f2)
say "adopting into generator: GPU_DIRECT_RDMA=$GD UCX_TLS=$TLS"
python3 - "$GD" "$TLS" "$GEN" <<'EOF'
import sys
gd, tls, gen = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(gen).read()
src = src.replace('{"name": "UCX_TLS", "value": "cuda_copy,rc_x,tcp"}',
                  f'{{"name": "UCX_TLS", "value": "{tls}"}}')
src = src.replace('{"name": "UCX_IB_GPU_DIRECT_RDMA", "value": "n"}',
                  f'{{"name": "UCX_IB_GPU_DIRECT_RDMA", "value": "{gd}"}}')
open(gen, "w").write(src)
print("generator updated")
EOF
cd $HOME/kv-cache-aware-bench && python3 sglang/scripts/gen_sglang_arms.py np-3 >> "$LOG" 2>&1
say "manifests regenerated with certified transport"

say "=== launching re-sweep"
rm -f /tmp/resweep_n3u_d72.log
bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/resweep_n3u_d72.sh
RS=$?
say "resweep exit=$RS"
grep -E "DONE|VIOLATION|TIMEOUT" /tmp/resweep_n3u_d72.log | tail -2 >> "$LOG"
say "AUTH CHAIN COMPLETE (resweep exit=$RS)"
