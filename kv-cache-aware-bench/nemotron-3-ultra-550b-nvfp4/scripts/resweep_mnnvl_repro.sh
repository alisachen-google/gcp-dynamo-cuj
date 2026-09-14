#!/bin/bash
# Reproduction + completion on the LIVE 6:12 MNNVL fleet (2026-09-14 evening):
#  kv:96 kv:120 kv:144 — the c144 spike (6,221 tok/s, 86/GPU, AT/PRE) sits between a
#  POST-knee c96 (3,691) and a collapsed c192 (2,707); it flips the agg-vs-disagg
#  verdict if real, so it is reproduced before being banked. Then kv:384 kv:512 to
#  finish the KV ladder to 512. RR 384/512 skipped: RR already fails transfers at
#  c288 (83/5782 errors) — running deeper only reproduces failures.
#  On completion: scale down, mark EXT DONE, then chain 9:9 -> agg(to 512) -> profiling.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
NS=dynamo-cloud; ARM=n3u-mnnvl-full; LOG=/tmp/resweep_mnnvl_repro.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
GUARD=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/mnnvl_transport_guard.sh
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4; N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
declare -A ROUTER=([kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs" [rr]="--router-mode round-robin")
POINTS="kv:96 kv:120 kv:144 kv:384 kv:512"
say "ctx=$(kubectl config current-context) — repro start"
R=$(kubectl get pods -n $NS -l "app in (${ARM}-prefill,${ARM}-decode)" --no-headers | grep -c Running)
[ "$R" -ge 18 ] || { say "fleet not up ($R/18) — redeploying"; kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-full.yaml >> "$LOG" 2>&1
  for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT $d"; exit 1; }; done; }
say "fleet ready ($R/18 running)"
for point in $POINTS; do
  v=${point%%:*}; C=${point##*:}; say "=== point $v c$C"
  FE_CMD="pip install -q \"ai-dynamo==1.4.2\" && exec python3 -m dynamo.frontend ${ROUTER[$v]} --request-plane nats"
  J=$(python3 -c "import json,sys;print(json.dumps([sys.argv[1]]))" "$FE_CMD")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$J}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1; kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=600s >> "$LOG" 2>&1; sleep 120
  FEP=$(kubectl get pods -n $NS -l app=${ARM}-frontend -o name | head -1)
  for r in $(seq 1 30); do kubectl exec -n $NS "$FEP" -c frontend -- curl -s -m 10 localhost:8000/v1/models 2>/dev/null | grep -q Nemotron && break; sleep 20; done; sleep 60
  JOB=alisachen-n3u-mnnvl-${v}-c${C}; kubectl delete job -n $NS "$JOB" --ignore-not-found --wait=true >> "$LOG" 2>&1
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
say "N3U MNNVL REPRO DONE"; echo "[$(date -u +%F' '%H:%M:%S)] N3U MNNVL EXT SWEEP DONE" >> /tmp/resweep_mnnvl_ext.log
# ---- chain: 9:9 -> agg (to 512); the parked profiled comparison gates on agg DONE ----
: > /tmp/resweep_mnnvl_99.log; bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/resweep_mnnvl_99.sh
: > /tmp/resweep_agg_newstack.log; bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/resweep_agg_newstack.sh
