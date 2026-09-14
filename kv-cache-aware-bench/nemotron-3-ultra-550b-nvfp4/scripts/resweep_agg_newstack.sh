#!/bin/bash
# N3U 24-GPU AGG re-sweep on the latest+workable stack (SGLang 0.5.16 / Dynamo
# 1.4.2 / FlashInfer 0.6.18, base image v0.5.19) — for a version-consistent
# agg<->disagg comparison and old-stack (0.5.14/1.3.1) vs new-stack drift.
# GATED behind BOTH disagg sweeps finishing (shared GPUs). SMOKE-FIRST: the new
# stack is validated for disagg, not agg, so a c32 smoke + hybrid-reuse probe
# must pass before the full ladder commits ~6h. kv/rr x 16/32/64/128.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
NS=dynamo-cloud
ARM=n3u-agg-ns
LOG=/tmp/resweep_agg_newstack.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say() { echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
declare -A ROUTER=(
  [kv]="--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs"
  [rr]="--router-mode round-robin"
)
POINTS="kv:32 kv:16 rr:16 kv:64 rr:32 rr:64 kv:128 rr:128 kv:192 rr:192 kv:256 rr:256 kv:384 rr:384 kv:512 rr:512"  # smoke first; extended to 512 per user

swap_frontend() {  # $1=policy ; frontend is bash-c/pip -> must keep that form
  local FE_CMD="pip install -q \"ai-dynamo==1.4.2\" && exec python3 -m dynamo.frontend ${ROUTER[$1]} --request-plane nats"
  local J; J=$(python3 -c "import json,sys;print(json.dumps([sys.argv[1]]))" "$FE_CMD")
  kubectl patch deployment ${ARM}-frontend -n $NS --type=json \
    -p "[{\"op\":\"replace\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":$J}]" >> "$LOG" 2>&1
  kubectl rollout restart deployment/${ARM}-frontend -n $NS >> "$LOG" 2>&1
  kubectl rollout status deployment/${ARM}-frontend -n $NS --timeout=600s >> "$LOG" 2>&1
  sleep 120
  local FEP; FEP=$(kubectl get pods -n $NS -l app=${ARM}-frontend -o name | head -1)
  for r in $(seq 1 30); do kubectl exec -n $NS "$FEP" -c frontend -- curl -s -m 10 localhost:8000/v1/models 2>/dev/null | grep -q Nemotron && break; sleep 20; done
  sleep 60
}

run_point() {  # $1=policy $2=conc ; returns job status in $st
  local v=$1 C=$2 JOB=alisachen-${ARM}-${v}-c${C}
  kubectl delete job -n $NS "$JOB" --ignore-not-found --wait=true >> "$LOG" 2>&1
  sed -e "s|/model-cache/alisachen/Kimi-K2.5-NVFP4|${N3U_DIR}|g" \
      -e "s|alisachen/Kimi-K2.5-NVFP4|${N3U_SERVED}|g" \
      -e "s|models--alisachen--Kimi-K2.5-NVFP4|models--alisachen--Nemotron-3-Ultra-550B-A55B-NVFP4|g" \
      -e "s/sgl-disagg72-kv/${ARM}/g" \
      -e "s/name: alisachen-sgl-d72-flagsweep/name: ${JOB}/" \
      -e "s/alisachen-sgl-d72-flagsweep/${JOB}/g" \
      -e "/name: CONCURRENCIES/{n;s/value: .*/value: \"${C}\"/}" \
      -e "/name: BENCHMARK_DURATION/{n;s/value: .*/value: \"1800\"/}" \
      "$TMPL" | kubectl apply -n $NS -f - >> "$LOG" 2>&1
  st=""
  for i in $(seq 1 60); do
    st=$(kubectl get jobs -n $NS "$JOB" --no-headers 2>/dev/null | awk '{print $2}')
    [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break
    sleep 120
  done
}

say "waiting for both disagg sweeps to finish (N3U MNNVL 99 SWEEP DONE) to free GPUs"
until grep -q "N3U MNNVL 99 SWEEP DONE" /tmp/resweep_mnnvl_99.log 2>/dev/null; do
  grep -qE "VIOLATION|HALTING|STACK TIMEOUT" /tmp/resweep_mnnvl_99.log 2>/dev/null && { say "9:9 sweep HALTED — NOT proceeding (inspect first)"; exit 1; }
  sleep 300
done
say "tearing down any residual disagg fleets, deploying new-stack agg"
kubectl scale deployment -n $NS -l 'app in (n3u-mnnvl-full-prefill,n3u-mnnvl-full-decode,n3u-mnnvl-full-frontend,n3u-mnnvl-99-prefill,n3u-mnnvl-99-decode,n3u-mnnvl-99-frontend)' --replicas=0 >> "$LOG" 2>&1
sleep 120
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-agg-newstack.yaml >> "$LOG" 2>&1
for d in ${ARM}-frontend ${ARM}; do
  kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT: $d — new stack may not serve N3U agg; HALTING"; exit 1; }
done
# report effective engine version the new stack resolved to
WP=$(kubectl get pods -n $NS -l app=${ARM} -o name | head -1)
VER=$(kubectl logs -n $NS "$WP" -c agg 2>/dev/null | grep -iE "sglang.*version|__version__|Loaded model" | head -3)
say "new-stack agg fleet up; engine version lines: ${VER:-<none>}"

# ---- SMOKE: kv c32 + hybrid-reuse probe (cached_tokens>0) ----
say "=== SMOKE: kv c32 + hybrid radix-reuse probe"
swap_frontend kv
run_point kv 32
say "SMOKE kv c32 (job=$st)"
[ "$st" != "Complete" ] && { say "AGG SMOKE FAIL: c32 bench did not complete on new stack — HALTING"; exit 2; }
CT=$(kubectl logs -n $NS "$WP" -c agg 2>/dev/null | grep -icE "cached_tokens|prefix.*hit|reuse" || true)
say "hybrid-reuse evidence lines: $CT"
[ "${CT:-0}" -eq 0 ] && { say "AGG SMOKE WARN: no cached_tokens evidence — hybrid reuse may be off on 0.5.16; HALTING for inspection"; exit 3; }
say "AGG NEWSTACK SMOKE: SUCCESS"
python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "${ARM}-kv-c32" >> "$LOG" 2>&1 || true

# ---- FULL LADDER (kv:32 already banked as smoke) ----
for point in $POINTS; do
  v=${point%%:*}; C=${point##*:}
  [ "$v" = "kv" ] && [ "$C" = "32" ] && continue  # done in smoke
  say "=== point $v c$C"
  swap_frontend "$v"
  run_point "$v" "$C"
  say "point $v c$C done (job=$st)"
  [ "$st" != "Complete" ] && { say "BENCH VIOLATION on $v c$C - HALTING"; exit 2; }
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "${ARM}-${v}-c${C}" >> "$LOG" 2>&1 || true
done
kubectl scale deployment/${ARM}-frontend ${ARM} -n $NS --replicas=0 >> "$LOG" 2>&1
say "N3U AGG NEWSTACK SWEEP DONE"
