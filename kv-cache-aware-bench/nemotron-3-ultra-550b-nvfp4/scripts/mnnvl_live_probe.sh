#!/bin/bash
# Live physical proof that KV rides NVLink (run DURING a bench point, 10 s window):
# decode GPU0 NVLink Rx/Tx deltas (Rx >> Tx = receiving KV), eth0 bytes (must be ~idle),
# /dev/infiniband absent, IMEX channel present. Usage: mnnvl_live_probe.sh <decode-app-label>
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned; NS=dynamo-cloud
P=$(kubectl get pods -n $NS -l app=${1:-n3u-mnnvl-full-decode} -o name | head -1); P=${P#pod/}
kubectl exec -n $NS $P -c decode -- sh -c '
r0=$(nvidia-smi nvlink -gt d -i 0 | grep -m1 "Data Rx" | grep -oE "[0-9]+"); t0=$(nvidia-smi nvlink -gt d -i 0 | grep -m1 "Data Tx" | grep -oE "[0-9]+"); e0=$(cat /sys/class/net/eth0/statistics/rx_bytes)
sleep 10
r1=$(nvidia-smi nvlink -gt d -i 0 | grep -m1 "Data Rx" | grep -oE "[0-9]+"); t1=$(nvidia-smi nvlink -gt d -i 0 | grep -m1 "Data Tx" | grep -oE "[0-9]+"); e1=$(cat /sys/class/net/eth0/statistics/rx_bytes)
echo "NVLink link0 (GPU0, 10s): Rx $(( (r1-r0)/1024 )) MiB  Tx $(( (t1-t0)/1024 )) MiB   | eth0 rx $(( (e1-e0)/1048576 )) MiB"
[ -e /dev/infiniband ] && echo "/dev/infiniband: PRESENT (RDMA possible)" || echo "/dev/infiniband: absent (RDMA impossible)"
ls /dev/nvidia-caps-imex-channels 2>/dev/null | sed "s/^/IMEX channel: /"'
