#!/bin/bash
# Waits for a FULL 18-node NVL72 domain to free (any pool), regenerates the
# MNNVL full fleet onto that pool, then runs the disagg re-sweep on MNNVL+mooncake.
# Smoke already passed (transport verified cuda_ipc). Self-launching / unattended.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs
NS=dynamo-cloud
LOG=/tmp/mnnvl_domain_wait.log
GEN=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/gen_mnnvl_arms.py
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }

say "waiting for a full 18-node domain (any pool), polling 30m"
POOL=""
for i in $(seq 1 96); do   # up to 48h
  kubectl get pods -A -o wide --no-headers 2>/dev/null | awk '$4=="Running" && $1!~/^(kube-system|gmp|gke|nvidia-dra)/{print $8}' | sort -u > /tmp/occ_nodes.txt
  for p in 3 2 1 4; do
    tot=$(kubectl get nodes --no-headers 2>/dev/null | awk -v P="np-$p-" '$2=="Ready" && $1 ~ P' | wc -l)
    occ=$(grep -c "np-$p-" /tmp/occ_nodes.txt)
    free=$((tot-occ))
    if [ "$tot" -ge 18 ] && [ "$free" -ge 18 ]; then POOL="np-$p"; break; fi
  done
  [ -n "$POOL" ] && { say "domain free: $POOL (18 nodes)"; break; }
  sleep 1800
done
[ -z "$POOL" ] && { say "NO FULL DOMAIN after 48h — still capacity-blocked; not launching"; exit 1; }

say "regenerating MNNVL full fleet onto $POOL"
python3 - "$POOL" <<PYEOF
import sys,re
gen="$GEN"; pool=sys.argv[1]
s=open(gen).read()
s=re.sub(r'POOL="np-[0-9]"','POOL="%s"'%pool,s)
open(gen,'w').write(s)
PYEOF
cd $HOME/kv-cache-aware-bench && python3 sglang/scripts/../nemotron-3-ultra-550b-nvfp4/scripts/gen_mnnvl_arms.py 6 12 18 n3u-mnnvl-full >> "$LOG" 2>&1
say "launching MNNVL re-sweep on $POOL"
# smoke marker already SUCCESS; the re-sweep runner gates on it and proceeds
bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/resweep_mnnvl_d72.sh
say "MNNVL DOMAIN-WAIT RESWEEP COMPLETE (exit=$?)"
