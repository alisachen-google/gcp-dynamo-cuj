#!/bin/bash
# N3U 6:12 MNNVL disagg — EXTENDED concurrency sweep (user priority 2026-09-14):
# completes the real curve on the full DynoSim grid beyond the banked c12-96:
# kv/rr x 144/192/288/384/512/768 -> ceiling, post-knee, saturation, point-for-
# point drift vs sim. Same fleet/flags as the c12-96 run (n3u-mnnvl-full), same
# per-point protocol (fresh frontend, 300s settle, 900s warm, 1800s measure),
# MNNVL transport gate + knee check per point. Pinned kubeconfig; CD pre-flight.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
NS=dynamo-cloud; ARM=n3u-mnnvl-full; LOG=/tmp/resweep_mnnvl_ext.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
GUARD=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/mnnvl_transport_guard.sh
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
declare -A ROUTER=([kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs" [rr]="--router-mode round-robin")
POINTS="kv:144 rr:144 kv:192 rr:192 kv:288 rr:288 kv:384 rr:384 kv:512 rr:512 kv:768 rr:768"

say "ctx=$(kubectl config current-context) — EXT sweep start"
[ "$(kubectl config current-context)" = "<cluster>" ] || { say "WRONG CONTEXT"; exit 1; }
# pre-flight: wait for any OTHER CD of ours to finish terminating (one CD per node)
for i in $(seq 1 40); do
  O=$(kubectl get computedomains -n $NS --no-headers 2>/dev/null | grep -vE "^${ARM}-cd " | wc -l)
  [ "$O" -eq 0 ] && break; say "waiting for $O stale CD(s) to terminate"; sleep 30
done
# domain wait: >=18 free np-3 nodes
free_nodes(){ n=0; for node in $(kubectl get nodes -l cloud.google.com/gke-nodepool=np-3 -o name); do node=${node#node/}
  u=$(kubectl describe node "$node" | awk '/Allocated resources/,0' | grep "nvidia.com/gpu" | awk '{print $2}'); [ "${u:-0}" = "0" ] && n=$((n+1)); done; echo $n; }
while :; do F=$(free_nodes); say "free np-3 nodes: $F / 18"; [ "$F" -ge 18 ] && break; sleep 300; done

say "deploying 6:12 MNNVL fleet"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-full.yaml >> "$LOG" 2>&1
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d"; kubectl describe pods -n $NS -l app=$d | grep -A8 "^Events" | tail -20 >> "$LOG"; exit 1; }
done
for tier in prefill decode; do WP=$(kubectl get pods -n $NS -l app=${ARM}-${tier} -o name | head -1)
  say "fleet-start $tier cuda_ipc/mnnvl lines: $(kubectl logs -n $NS "$WP" -c $tier 2>/dev/null | grep -icE 'cuda_ipc|mnnvl' || echo 0)"; done
say "fleet ready on MNNVL"

for point in $POINTS; do
  v=${point%%:*}; C=${point##*:}; say "=== point $v c$C"
  FE_CMD="pip install -q \"ai-dynamo==1.4.2\" && exec python3 -m dynamo.frontend ${ROUTER[$v]} --request-plane nats"
  J=$(python3 -c "import json,sys;print(json.dumps([sys.argv[1]]))" "$FE_CMD")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$J}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1
  kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=600s >> "$LOG" 2>&1; sleep 120
  FEP=$(kubectl get pods -n $NS -l app=${ARM}-frontend -o name | head -1)
  for r in $(seq 1 30); do kubectl exec -n $NS "$FEP" -c frontend -- curl -s -m 10 localhost:8000/v1/models 2>/dev/null | grep -q Nemotron && break; sleep 20; done; sleep 60
  JOB=alisachen-n3u-mnnvl-${v}-c${C}
  kubectl delete job -n $NS "$JOB" --ignore-not-found --wait=true >> "$LOG" 2>&1
  sed -e "s|/model-cache/alisachen/Kimi-K2.5-NVFP4|${N3U_DIR}|g" -e "s|alisachen/Kimi-K2.5-NVFP4|${N3U_SERVED}|g" \
      -e "s|models--alisachen--Kimi-K2.5-NVFP4|models--alisachen--Nemotron-3-Ultra-550B-A55B-NVFP4|g" \
      -e "s/sgl-disagg72-kv/${ARM}/g" -e "s/name: alisachen-sgl-d72-flagsweep/name: ${JOB}/" -e "s/alisachen-sgl-d72-flagsweep/${JOB}/g" \
      -e "/name: CONCURRENCIES/{n;s/value: .*/value: \"${C}\"/}" -e "/name: BENCHMARK_DURATION/{n;s/value: .*/value: \"1800\"/}" \
      "$TMPL" | kubectl apply -n $NS -f - >> "$LOG" 2>&1
  st=""; for i in $(seq 1 75); do st=$(kubectl get jobs -n $NS "$JOB" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break; sleep 120; done
  say "point $v c$C done (job=$st)"
  [ "$st" != "Complete" ] && { say "BENCH VIOLATION on $v c$C - HALTING"; exit 2; }
  bash "$GUARD" "$ARM" >> "$LOG" 2>&1 || { say "MNNVL TRANSPORT VIOLATION on $v c$C - HALTING"; exit 2; }
  say "MNNVL gate PASS"
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "n3u-mnnvl-${v}-c${C}" >> "$LOG" 2>&1 || true
done
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do kubectl scale deployment/$d -n $NS --replicas=0 >> "$LOG" 2>&1; done
say "N3U MNNVL EXT SWEEP DONE"
