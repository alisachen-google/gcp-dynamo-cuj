#!/bin/bash
# agentx_runner.sh <ARM> <manifest> "<points e.g. kv:48 kv:96>" <gate-marker> <gate-log> <done-marker> <log> <need-free-nodes> <pool> <jobprefix> <worker-labels-csv>
# AgentX-mode client (aiperf --scenario inferencex-agentx-mvp, template sgl-d72-agentx.yaml): concurrency = live session trees,
# end-to-start think-time replayed, per-play first_turn_prefix cache-bust, 3600 s per point. First point = smoke (halts on failure).
# env: GATE_STRICT=1 (gate opens only on the DONE marker), BENCH_POOL=<pool for the aiperf pod, default np-1>, MNNVL_GUARD=1 (run the transport guard on pools other than np-3), JOB_WAIT_ITERS=<n x 120 s per cell, default 100>. Only Ready nodes count as free.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
ARM=$1; MAN=$2; PTS=$3; GATE=$4; GLOG=$5; DONE=$6; LOG=$7; NEED=$8; POOL=$9; JP=${10}; WL=${11}
NS=dynamo-cloud; TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-agentx.yaml
GUARD=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/mnnvl_transport_guard_v3.sh
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4; N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
declare -A ROUTER=([kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs" [rr]="--router-mode round-robin" [kvs3c08]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 3.0 --router-kv-overlap-score-credit 0.8" [kvs2c08]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 2.0 --router-kv-overlap-score-credit 0.8" [kvt05]="--router-mode kv --router-temperature 0.5 --router-queue-policy fcfs" [kvwspt]="--router-mode kv --router-temperature 0.0 --router-queue-policy wspt" [kvd05]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit-decay 0.5" [kvd10]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit-decay 1.0" [kvd20]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit-decay 2.0" [kvs4c08]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 4.0 --router-kv-overlap-score-credit 0.8" [kvs3c08d05]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 3.0 --router-kv-overlap-score-credit 0.8 --router-kv-overlap-score-credit-decay 0.5" [kvs3c08d10]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 3.0 --router-kv-overlap-score-credit 0.8 --router-kv-overlap-score-credit-decay 1.0")
free_nodes(){ n=0; for node in $(kubectl get nodes -l cloud.google.com/gke-nodepool=$POOL --no-headers | awk '$2=="Ready"{print $1}'); do u=$(kubectl describe node "$node" | awk '/Allocated resources/,0' | grep "nvidia.com/gpu" | awk '{print $2}'); [ "${u:-0}" = "0" ] && n=$((n+1)); done; echo $n; }
say "waiting for gate '$GATE' in $GLOG"
until grep -q "$GATE" "$GLOG" 2>/dev/null; do [ "${GATE_STRICT:-0}" = "0" ] && grep -qE "HALTING|STACK TIMEOUT" "$GLOG" 2>/dev/null && { say "gate job halted — proceeding after teardown"; break; }; sleep 300; done
if [ "$POOL" = "np-3" ]; then
  for cd in n3u-mnnvl-full-cd n3u-mnnvl-99-cd n3u-mnnvl-cd n3u-mnnvl-126-cd n3u-mnnvl-315-cd n3u-mnnvl-99mtp-cd; do kubectl delete computedomain/$cd -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1; done
  for d in $(kubectl get deploy -n $NS -o name | grep -E "n3u-mnnvl-(full|99|126|315|99mtp)"); do kubectl delete $d -n $NS --wait=false >> "$LOG" 2>&1; done
  sleep 90
fi
[ "$NEED" = "0" ] || { while :; do F=$(free_nodes); say "free $POOL nodes: $F / need $NEED"; [ "$F" -ge "$NEED" ] && break; sleep 300; done; }
say "deploying $ARM"; kubectl apply -n $NS -f "$MAN" >> "$LOG" 2>&1
for d in $(echo "$WL" | tr ',' ' ') ${ARM}-frontend; do kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT $d — HALTING"; exit 1; }; done
first=1
for point in $PTS; do
  T0=$(date -u +%FT%TZ)
  v=${point%%:*}; C=${point##*:}; say "=== $ARM AgentX $v c$C $([ $first = 1 ] && echo '(smoke)')"
  FE="pip install -q \"ai-dynamo==1.4.2\" && exec python3 -m dynamo.frontend ${ROUTER[$v]} --request-plane nats"
  J=$(python3 -c "import json,sys;print(json.dumps([sys.argv[1]]))" "$FE")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$J}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1; kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=600s >> "$LOG" 2>&1; sleep 45
  FEP=$(kubectl get pods -n $NS -l app=${ARM}-frontend -o name | head -1)
  for r in $(seq 1 30); do kubectl exec -n $NS "$FEP" -c frontend -- curl -s -m 10 localhost:8000/v1/models 2>/dev/null | grep -q Nemotron && break; sleep 20; done; sleep 15
  JOB=alisachen-${JP}-agentx-${v}-c${C}; kubectl delete job -n $NS "$JOB" --ignore-not-found --wait=true >> "$LOG" 2>&1
  sed -e "s|/model-cache/alisachen/Kimi-K2.5-NVFP4|${N3U_DIR}|g" -e "s|alisachen/Kimi-K2.5-NVFP4|${N3U_SERVED}|g" \
      -e "s|models--alisachen--Kimi-K2.5-NVFP4|models--alisachen--Nemotron-3-Ultra-550B-A55B-NVFP4|g" \
      -e "s/sgl-disagg72-kv/${ARM}/g" -e "s/name: alisachen-sgl-d72-agentx/name: ${JOB}/" -e "s/alisachen-sgl-d72-agentx/${JOB}/g" \
      -e "s|cloud.google.com/gke-nodepool: np-1|cloud.google.com/gke-nodepool: ${BENCH_POOL:-np-1}|" \
      -e "/name: CONCURRENCIES/{n;s/value: .*/value: \"${C}\"/}" -e "/name: BENCHMARK_DURATION/{n;s/value: .*/value: \"3600\"/}" \
      "$TMPL" | kubectl apply -n $NS -f - >> "$LOG" 2>&1
  st=""; for i in $(seq 1 $(( ${JOB_WAIT_ITERS:-100} * 4 ))); do st=$(kubectl get jobs -n $NS "$JOB" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break; sleep 30; done
  say "$ARM AgentX $v c$C done (job=$st)"
  [ "$st" != "Complete" ] && { say "$([ $first = 1 ] && echo 'AGENTX SMOKE FAIL' || echo 'BENCH VIOLATION') on $v c$C - HALTING"; kubectl logs -n $NS -l job-name=$JOB --tail=30 2>/dev/null | grep -iE "error|scenario|unknown" | tail -8 >> "$LOG"; exit 2; }
  if [ "$POOL" = "np-3" ] || [ "${MNNVL_GUARD:-0}" = "1" ]; then GUARD_SINCE=$T0 bash "$GUARD" "$ARM" >> "$LOG" 2>&1; grc=$?; if [ $grc = 3 ]; then say "MNNVL gate: transport healthy; cell $v c$C OVERLOADED (requests hit the 300 s prefill-wait timeout)"; elif [ $grc != 0 ]; then say "MNNVL TRANSPORT VIOLATION - HALTING"; exit 2; else say "MNNVL gate PASS"; fi; fi
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "${JP}-agentx-${v}-c${C}" >> "$LOG" 2>&1 || true
  first=0
done
[ "${KEEP_FLEET:-0}" = "1" ] || for d in $(echo "$WL" | tr ',' ' ') ${ARM}-frontend; do kubectl scale deployment/$d -n $NS --replicas=0 >> "$LOG" 2>&1; done
say "$DONE"
