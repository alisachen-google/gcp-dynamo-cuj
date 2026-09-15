#!/bin/bash
# MTP (NEXTN speculative decoding, native Nemotron-3 MTP layer) on disagg 9:9 KV — three points
# c48/c96/c144 to compare against the MTP-off 9:9 runs (4,555 / 6,560 / pending). Gated behind
# the disagg remainder (np-3). c48 doubles as the smoke: halts if the spec-decode fleet fails.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
NS=dynamo-cloud; ARM=n3u-mnnvl-99mtp; LOG=/tmp/mtp_disagg.log
TMPL=$HOME/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml
GUARD=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/mnnvl_transport_guard.sh
N3U_DIR=/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4; N3U_SERVED=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
free_nodes(){ n=0; for node in $(kubectl get nodes -l cloud.google.com/gke-nodepool=np-3 -o name); do node=${node#node/}; u=$(kubectl describe node "$node" | awk '/Allocated resources/,0' | grep "nvidia.com/gpu" | awk '{print $2}'); [ "${u:-0}" = "0" ] && n=$((n+1)); done; echo $n; }
say "waiting for disagg remainder (N3U DISAGG REMAINDER DONE)"
until grep -q "N3U DISAGG REMAINDER DONE" /tmp/disagg_remainder.log 2>/dev/null; do grep -qE "VIOLATION|HALTING|STACK TIMEOUT" /tmp/disagg_remainder.log 2>/dev/null && { say "remainder halted — proceeding after teardown"; break; }; sleep 300; done
for cd in n3u-mnnvl-full-cd n3u-mnnvl-99-cd n3u-mnnvl-cd; do kubectl delete computedomain/$cd -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1; done
for d in n3u-mnnvl-full-prefill n3u-mnnvl-full-decode n3u-mnnvl-full-frontend n3u-mnnvl-99-prefill n3u-mnnvl-99-decode n3u-mnnvl-99-frontend; do kubectl delete deployment/$d -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1; done
sleep 90; while :; do F=$(free_nodes); say "free np-3 nodes: $F / 18"; [ "$F" -ge 18 ] && break; sleep 300; done
say "deploying 9:9 MTP fleet (NEXTN steps 3 / topk 1 / draft 4)"
kubectl apply -n $NS -f $HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-99mtp.yaml >> "$LOG" 2>&1
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do kubectl rollout status deployment/$d -n $NS --timeout=3600s >> "$LOG" 2>&1 || { say "STACK TIMEOUT $d — spec-decode fleet failed; HALTING"; kubectl logs -n $NS -l app=${ARM}-decode --tail=40 2>/dev/null | grep -iE "error|traceback|speculative|mtp" | tail -12 >> "$LOG"; exit 1; }; done
FEP=$(kubectl get pods -n $NS -l app=${ARM}-frontend -o name | head -1)
for r in $(seq 1 30); do kubectl exec -n $NS "$FEP" -c frontend -- curl -s -m 10 localhost:8000/v1/models 2>/dev/null | grep -q Nemotron && break; sleep 20; done
say "MTP fleet ready; spec evidence on decode: $(kubectl logs -n $NS $(kubectl get pods -n $NS -l app=${ARM}-decode -o name | head -1) -c decode 2>/dev/null | grep -ciE 'speculative|NEXTN|draft')"
for C in 48 96 144; do
  say "=== 9:9-MTP kv c$C"; JOB=alisachen-n3u-mnnvl-99mtp-kv-c${C}
  kubectl delete job -n $NS "$JOB" --ignore-not-found --wait=true >> "$LOG" 2>&1
  sed -e "s|/model-cache/alisachen/Kimi-K2.5-NVFP4|${N3U_DIR}|g" -e "s|alisachen/Kimi-K2.5-NVFP4|${N3U_SERVED}|g" \
      -e "s|models--alisachen--Kimi-K2.5-NVFP4|models--alisachen--Nemotron-3-Ultra-550B-A55B-NVFP4|g" \
      -e "s/sgl-disagg72-kv/${ARM}/g" -e "s/name: alisachen-sgl-d72-flagsweep/name: ${JOB}/" -e "s/alisachen-sgl-d72-flagsweep/${JOB}/g" \
      -e "/name: CONCURRENCIES/{n;s/value: .*/value: \"${C}\"/}" -e "/name: BENCHMARK_DURATION/{n;s/value: .*/value: \"1800\"/}" \
      "$TMPL" | kubectl apply -n $NS -f - >> "$LOG" 2>&1
  st=""; for i in $(seq 1 75); do st=$(kubectl get jobs -n $NS "$JOB" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break; sleep 120; done
  say "9:9-MTP kv c$C done (job=$st)"; [ "$st" != "Complete" ] && { say "BENCH VIOLATION - HALTING"; exit 2; }
  bash "$GUARD" "$ARM" >> "$LOG" 2>&1 || { say "MNNVL TRANSPORT VIOLATION - HALTING"; exit 2; }; say "MNNVL gate PASS"
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "n3u-mnnvl-99mtp-kv-c${C}" >> "$LOG" 2>&1 || true
  # spec-decode telemetry: mean accept length from decode logs (last 20k lines)
  for P in $(kubectl get pods -n $NS -l app=${ARM}-decode -o name | head -3); do
    kubectl logs -n $NS "$P" -c decode --tail=20000 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -oE "accept_length[=: ]+[0-9.]+" | awk -F'[=: ]+' '{s+=$2;n++} END{if(n) printf "'"$P"' accept_length mean=%.2f n=%d\n", s/n, n; else print "'"$P"' no accept_length lines"}' >> "$LOG"
  done
done
for d in ${ARM}-prefill ${ARM}-decode ${ARM}-frontend; do kubectl scale deployment/$d -n $NS --replicas=0 >> "$LOG" 2>&1; done
kubectl delete computedomain/${ARM}-cd -n $NS --ignore-not-found --wait=false >> "$LOG" 2>&1
say "N3U MTP DISAGG DONE"
