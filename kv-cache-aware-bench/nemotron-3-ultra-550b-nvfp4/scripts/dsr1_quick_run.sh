#!/bin/bash
# Quick dsr1 job, August m2r recipe made-to-complete (2026-09-12):
# waits for ADC handoff -> isolated etcd -> m2r 1P+1D+FE with ALL learned fixes
# (host-staged transport since GPUDirect is broken, PVC frontend model volume,
# registration-gated) -> small completion battery -> report -> scale down.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
LOG=/tmp/dsr1_quick.log
MAN=$HOME/kv-cache-aware-bench/sglang/manifests/dsr1-m2r-verbatim.yaml
IMG=quay.io/coreos/etcd:v3.5.15
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
T0=$(date -u +%s)

say "waiting for fresh ADC"
for i in $(seq 1 288); do
  TS=$(gcloud storage objects describe gs://alisachen-models/adc-transfer.json --format="value(updateTime)" 2>/dev/null)
  if [ -n "$TS" ] && [ "$(date -u -d "$TS" +%s 2>/dev/null || echo 0)" -gt "$T0" ]; then
    gcloud storage cp gs://alisachen-models/adc-transfer.json ~/.config/gcloud/application_default_credentials.json >> "$LOG" 2>&1
    TOK=$(gcloud auth application-default print-access-token 2>/dev/null)
    curl -s -o /dev/null -X DELETE -H "Authorization: Bearer $TOK" "https://storage.googleapis.com/storage/v1/b/alisachen-models/o/adc-transfer.json"
    say "ADC installed, bucket copy deleted"; break
  fi
  sleep 300
done
timeout 60 kubectl get nodes >/dev/null 2>&1 || { say "AUTH FAILED"; exit 1; }

# isolated etcd (avoids 1.3.1 model-card contamination of the 0.8.1 frontend)
say "deploying isolated etcd-m2r"
cat <<EOF | kubectl apply -f - >> "$LOG" 2>&1
apiVersion: apps/v1
kind: Deployment
metadata: {name: etcd-m2r, namespace: ${NS}}
spec:
  replicas: 1
  selector: {matchLabels: {app: etcd-m2r}}
  template:
    metadata: {labels: {app: etcd-m2r}}
    spec:
      tolerations: [{operator: Exists}]
      containers:
      - {name: etcd, image: ${IMG}, command: [etcd, --advertise-client-urls, "http://0.0.0.0:2379", --listen-client-urls, "http://0.0.0.0:2379"], ports: [{containerPort: 2379}]}
---
apiVersion: v1
kind: Service
metadata: {name: etcd-m2r, namespace: ${NS}}
spec: {selector: {app: etcd-m2r}, ports: [{port: 2379, targetPort: 2379}]}
EOF
kubectl rollout status deployment/etcd-m2r -n $NS --timeout=120s >> "$LOG" 2>&1

say "applying m2r manifest + fixes"
kubectl apply -n $NS -f "$MAN" >> "$LOG" 2>&1
# fix 1: point all three at isolated etcd; fix 2: host-staged transport (GDR broken)
for d in dsr1-m2r-prefill dsr1-m2r-decode dsr1-m2r-frontend; do
  c=$(kubectl get deployment $d -n $NS -o jsonpath='{.spec.template.spec.containers[0].name}' 2>/dev/null)
  [ -z "$c" ] && continue
  kubectl set env deployment/$d -n $NS -c $c ETCD_ENDPOINTS=http://etcd-m2r.${NS}.svc.cluster.local:2379 >> "$LOG" 2>&1
done
for d in dsr1-m2r-prefill dsr1-m2r-decode; do
  kubectl set env deployment/$d -n $NS -c ${d##*-} UCX_IB_GPU_DIRECT_RDMA=n >> "$LOG" 2>&1
done
# fix 3: frontend model volume must be the PVC (not emptyDir) to resolve the path
kubectl patch deployment dsr1-m2r-frontend -n $NS --type=strategic \
  -p '{"spec":{"template":{"spec":{"volumes":[{"name":"model","emptyDir":null,"persistentVolumeClaim":{"claimName":"model-cache"}}]}}}}' >> "$LOG" 2>&1

for d in dsr1-m2r-prefill dsr1-m2r-decode dsr1-m2r-frontend; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; exit 1; }
done
say "fleet rolled out; waiting for model registration"

FE=$(kubectl get pods -n $NS -l app=dsr1-m2r-frontend -o name | head -1)
REG=0
for i in $(seq 1 40); do
  kubectl exec -n $NS "$FE" -c frontend -- curl -s -m 15 localhost:8000/v1/models 2>/dev/null | grep -q "DeepSeek" && { REG=1; break; }
  sleep 60
done
[ "$REG" = 1 ] || { say "MODEL NEVER REGISTERED"; exit 1; }
say "model registered; running 10-request completion battery"

OK=0; ERR=0
for i in $(seq 1 10); do
  R=$(timeout 150 kubectl exec -n $NS "$FE" -c frontend -- sh -c 'P=$(python3 -c "import random;print(\" \".join(str(random.random()) for _ in range(800)))"); curl -s -m 120 -X POST localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"deepseek-ai/DeepSeek-R1\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":32}"' 2>/dev/null)
  echo "$R" | grep -qE '"content"' && OK=$((OK+1))
done
say "completions: $OK/10"
for tier in prefill decode; do
  WP=$(kubectl get pods -n $NS -l app=dsr1-m2r-${tier} -o name | head -1)
  E=$(kubectl logs -n $NS "$WP" -c $tier --since=20m 2>/dev/null | grep -cE "REMOTE_DISCONNECT|NIXL_ERR|transfer failed" || true)
  say "$tier transfer-error lines: $E"
  kubectl logs -n $NS "$WP" -c $tier 2>/dev/null | grep "rc_mlx5" | grep -iE "zero-copy|rndv" | head -2 | sed "s/^/[proto $tier] /" | cut -c1-150 >> "$LOG"
done
[ "$OK" -ge 8 ] && say "DSR1 QUICK RUN: SUCCESS ($OK/10 completions on host-staged RDMA)" || say "DSR1 QUICK RUN: FAILED ($OK/10)"
kubectl scale deployment/dsr1-m2r-prefill deployment/dsr1-m2r-decode deployment/dsr1-m2r-frontend etcd-m2r -n $NS --replicas=0 >> "$LOG" 2>&1
say "DSR1 QUICK RUN DONE"
