#!/bin/bash
# GDR-fix watcher (goal step 3 automation): polls the node fleet for the
# nvidia_peermem fix; when it appears -> certifies the direct path with the
# transport probe -> flips the generator to GPUDirect (keeping certified
# no-tcp TLS) -> regenerates manifests -> re-runs the full disagg ladder.
# Run: nohup bash gdr_fix_watch.sh &   (safe to leave running for days)
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
LOG=/tmp/gdr_fix_watch.log
GEN=$HOME/kv-cache-aware-bench/sglang/scripts/gen_sglang_arms.py
say() { echo "[$(date -u +%F %H:%M:%S)] $*" >> "$LOG"; }

check_node_peermem() { # $1 = pool
  local NODE=$(kubectl get nodes -l cloud.google.com/gke-nodepool=$1 --no-headers 2>/dev/null | awk '$2=="Ready"{print $1; exit}')
  [ -z "$NODE" ] && return 1
  kubectl delete pod gdrchk --ignore-not-found -n $NS --wait=true >/dev/null 2>&1
  cat <<EOF | kubectl apply -f - >/dev/null 2>&1
apiVersion: v1
kind: Pod
metadata: {name: gdrchk, namespace: ${NS}}
spec:
  nodeName: ${NODE}
  hostPID: true
  restartPolicy: Never
  tolerations: [{operator: Exists}]
  containers:
  - name: c
    image: busybox
    securityContext: {privileged: true}
    command: ["sh","-c"]
    args: ["nsenter -t 1 -m -- sh -c 'lsmod | grep -q peermem && echo PEERMEM_LOADED || (find /lib/modules/\$(uname -r) -name \"*peermem*\" 2>/dev/null | grep -q . && echo PEERMEM_AVAILABLE || echo PEERMEM_ABSENT)'"]
EOF
  local i out=""
  for i in 1 2 3 4 5 6; do sleep 10; out=$(kubectl logs gdrchk -n $NS 2>/dev/null); [ -n "$out" ] && break; done
  kubectl delete pod gdrchk -n $NS --wait=false >/dev/null 2>&1
  say "pool $1 node $NODE: ${out:-no-result}"
  echo "$out" | grep -qE "PEERMEM_LOADED|PEERMEM_AVAILABLE"
}

say "=== GDR-fix watcher started (checks np-3 + np-1 every 6h)"
while true; do
  if check_node_peermem np-3 || check_node_peermem np-1; then
    say "PEERMEM DETECTED — certifying direct path"
    # switch probe order irrelevant: transport_probe tries GPU_DIRECT=y first
    rm -f /tmp/transport_probe.log
    bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/transport_probe.sh
    V=$(grep "TRANSPORT VERDICT" /tmp/transport_probe.log | tail -1)
    say "probe: $V"
    if echo "$V" | grep -q "A-direct-notcp HEALTHY"; then
      say "GPUDirect certified — adopting into generator + relaunching disagg ladder"
      python3 - <<'PYEOF'
gen = "/home/alisachen_google_com/kv-cache-aware-bench/sglang/scripts/gen_sglang_arms.py"
src = open(gen).read()
src = src.replace('{"name": "UCX_IB_GPU_DIRECT_RDMA", "value": "n"}',
                  '{"name": "UCX_IB_GPU_DIRECT_RDMA", "value": "y"}  # node fix landed: peermem restored, probe A certified')
open(gen, "w").write(src)
PYEOF
      (cd $HOME/kv-cache-aware-bench && python3 sglang/scripts/gen_sglang_arms.py np-3 >> "$LOG" 2>&1)
      rm -f /tmp/resweep_n3u_d72.log
      bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/resweep_n3u_d72.sh
      say "GDR RESWEEP EXIT=$? — see /tmp/resweep_n3u_d72.log"
      say "GDR FIX PIPELINE COMPLETE"
      exit 0
    else
      say "peermem present but probe not healthy yet — will re-check next cycle"
    fi
  fi
  sleep 21600
done
