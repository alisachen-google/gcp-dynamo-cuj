#!/bin/bash
# Remaining disagg runs (2026-09-15): 6:12 re-verify rr:48 rr:96 -> 9:9 kv:96 kv:144.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
NS=dynamo-cloud; LOG=/tmp/pareto_verify.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
GUARD=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/mnnvl_transport_guard.sh
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4; N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
declare -A ROUTER=([kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs" [rr]="--router-mode round-robin")
free_nodes(){ n=0; for node in $(kubectl get nodes -l cloud.google.com/gke-nodepool=np-3 -o name); do node=${node#node/}; u=$(kubectl describe node "$node" | awk '/Allocated resources/,0' | grep "nvidia.com/gpu" | awk '{print $2}'); [ "${u:-0}" = "0" ] && n=$((n+1)); done; echo $n; }
ladder(){ # $1=ARM $2=manifest $3=points "kv:96 rr:48"  $4=jobprefix
  local ARM=$1 MAN=$2 PTS=$3 JP=$4
  local want=$(python3 -c "import yaml,sys;print(sum(d['spec']['replicas'] for d in yaml.safe_load_all(open(sys.argv[1])) if d and d.get('kind')=='Deployment' and 'frontend' not in d['metadata']['name']))" "$MAN")
  local have=$(kubectl get pods -n $NS -l "app in (${ARM}-prefill,${ARM}-decode)" --no-headers 2>/dev/null | grep -c Running)
  if [ "$have" -ge "$want" ] && kubectl get deployment/${ARM}-frontend -n $NS >/dev/null 2>&1; then
    say "[$ARM] fleet already up ($have/$want workers) — resuming without redeploy"
  else
  for cd in n3u-mnnvl-full-cd n3u-mnnvl-99-cd n3u-mnnvl-cd n3u-mnnvl-126-cd n3u-mnnvl-315-cd n3u-mnnvl-99mtp-cd; do kubectl delete computedomain/$cd -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1; done
  for d in n3u-mnnvl-full-prefill n3u-mnnvl-full-decode n3u-mnnvl-full-frontend n3u-mnnvl-99-prefill n3u-mnnvl-99-decode n3u-mnnvl-99-frontend n3u-mnnvl-126-prefill n3u-mnnvl-126-decode n3u-mnnvl-126-frontend n3u-mnnvl-315-prefill n3u-mnnvl-315-decode n3u-mnnvl-315-frontend; do kubectl delete deployment/$d -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1; done
  sleep 90
  while :; do F=$(free_nodes); say "[$ARM] free np-3 nodes: $F / 18"; [ "$F" -ge 18 ] && break; sleep 300; done
  say "[$ARM] deploying"; kubectl apply -n $NS -f "$MAN" >> "$LOG" 2>&1
  for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT $d"; exit 1; }; done
  fi
  for point in $PTS; do
    v=${point%%:*}; C=${point##*:}; say "=== $ARM $v c$C"
    FE="pip install -q \"ai-dynamo==1.4.2\" && exec python3 -m dynamo.frontend ${ROUTER[$v]} --request-plane nats"
    J=$(python3 -c "import json,sys;print(json.dumps([sys.argv[1]]))" "$FE")
    kubectl patch deployment ${ARM}-frontend -n $NS --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$J}]" >> "$LOG" 2>&1
    kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1; kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=600s >> "$LOG" 2>&1; sleep 120
    FEP=$(kubectl get pods -n $NS -l app=${ARM}-frontend -o name | head -1)
    for r in $(seq 1 30); do kubectl exec -n $NS "$FEP" -c frontend -- curl -s -m 10 localhost:8000/v1/models 2>/dev/null | grep -q Nemotron && break; sleep 20; done; sleep 60
    JOB=alisachen-${JP}-${v}-c${C}; kubectl delete job -n $NS "$JOB" --ignore-not-found --wait=true >> "$LOG" 2>&1
    sed -e "s|/model-cache/alisachen/Kimi-K2.5-NVFP4|${N3U_DIR}|g" -e "s|alisachen/Kimi-K2.5-NVFP4|${N3U_SERVED}|g" \
        -e "s|models--alisachen--Kimi-K2.5-NVFP4|models--alisachen--Nemotron-3-Ultra-550B-A55B-NVFP4|g" \
        -e "s/sgl-disagg72-kv/${ARM}/g" -e "s/name: alisachen-sgl-d72-flagsweep/name: ${JOB}/" -e "s/alisachen-sgl-d72-flagsweep/${JOB}/g" \
        -e "/name: CONCURRENCIES/{n;s/value: .*/value: \"${C}\"/}" -e "/name: BENCHMARK_DURATION/{n;s/value: .*/value: \"1800\"/}" \
        "$TMPL" | kubectl apply -n $NS -f - >> "$LOG" 2>&1
    st=""; for i in $(seq 1 75); do st=$(kubectl get jobs -n $NS "$JOB" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break; sleep 120; done
    say "$ARM $v c$C done (job=$st)"; [ "$st" != "Complete" ] && { say "BENCH VIOLATION $ARM $v c$C - HALTING"; exit 2; }
    bash "$GUARD" "$ARM" >> "$LOG" 2>&1 || { say "MNNVL TRANSPORT VIOLATION $ARM $v c$C - HALTING"; exit 2; }
    say "MNNVL gate PASS"; python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "${JP}-${v}-c${C}" >> "$LOG" 2>&1 || true
  done
  for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do kubectl scale deployment/$d -n $NS --replicas=0 >> "$LOG" 2>&1; done
}
say "waiting for AgentX 9:9 ladder (N3U AGENTX 99 DONE) to free np-3"
until grep -q "N3U AGENTX 99 DONE" /tmp/agentx_99.log 2>/dev/null; do grep -qE "HALTING|STACK TIMEOUT" /tmp/agentx_99.log 2>/dev/null && { say "MTP halted — proceeding after teardown"; break; }; sleep 300; done
for d in n3u-mnnvl-99mtp-prefill n3u-mnnvl-99mtp-decode n3u-mnnvl-99mtp-frontend; do kubectl delete deployment/$d -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1; done
kubectl delete computedomain/n3u-mnnvl-99mtp-cd -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1
ladder n3u-mnnvl-126 $HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-126.yaml "kv:48 kv:96" n3u-mnnvl-126
ladder n3u-mnnvl-315 $HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-315.yaml "kv:96" n3u-mnnvl-315
ladder n3u-mnnvl-99  $HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-99.yaml  "kv:120" n3u-mnnvl-99
say "N3U PARETO VERIFY DONE"
