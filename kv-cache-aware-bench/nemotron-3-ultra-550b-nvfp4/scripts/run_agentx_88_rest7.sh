#!/bin/bash
# 8p/8d AgentX, remaining programme after the KV ladder (192/384/672/768/1152, knee located): RR first, then KV-flag tuning
# at the comparison cells, SLO = TTFT p95 <= 10 s (P90 interactivity >= 20, stationary).  One sequential process; the fleet
# stays up between cells (agentx_runner_keep.sh, KEEP_FLEET=1) so no other tenant can take the nodes between stages.
#   1  RR search (time-minimal, stops at the first POST-KNEE cell): 192, then up 240 / 288 / 384 (+1 midpoint) while inside 10 s,
#      or down 144 / 96 if 192 is outside 10 s; RR's SLO falls back to 20 s if nothing is inside 10 s; 384 kept as same-config pair
#      (ADOPT_RR=<clients> adopts an RR cell whose job is already running instead of restarting it)
#   2  KV bisection for the 10 s SLO boundary (default KV: 4.8 s at 384, 17.0 s at 672): KV 480, then 576 if 480 is inside else 432
#   3  selection from measured cells (select_agentx_points.py --ttft 10): same config = RR's throughput-peak cell (KV is run there
#      first if missing); same SLO = best stationary cell per policy inside the SLO
#   4  flag sweep, 6 runs in total. Phase A at the KV same-SLO cell vs default KV: load scale 3 / credit 0.8 (agg winner),
#      credit decay 0.5, credit decay 1.0, then run 4 chosen from those three (combine if both help, else push the one that helps)
#   5  phase B with the phase-A winner (pick_agentx_flag_winner.py: best total tok/s/GPU, stationary, TTFT p95 not worse, > 2 % gain):
#      winner at the same-config cell, and winner at the next KV client count above the SLO cell (does tuning move the SLO cell up?)
#   6  paired KV cells still missing for the same-config table (240, 96), then scale the fleet down
# Any MNNVL transport violation stops everything.
set -u
export KUBECONFIG=$HOME/kv-cache-aware-bench/.kubeconfig-cmcs-pinned
D=$HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts; S=$D/agentx_runner_flags.sh
M=$HOME/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-88.yaml; W=n3u-mnnvl-88-prefill,n3u-mnnvl-88-decode
export GATE_STRICT=1 BENCH_POOL=np-2 MNNVL_GUARD=1 JOB_WAIT_ITERS=160 KEEP_FLEET=1
LOG=/tmp/agentx_88_rest.log; SLO=${SLO_TTFT:-10}; VARIANTS=${VARIANTS:-kvs3c08,kvd05,kvd10}
KVL="192,384,480,672,768,1152"; RRL="96,144,192"
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" >> "$LOG"; }
authwait(){ n=0; until kubectl get ns dynamo-cloud >/dev/null 2>&1; do [ $((n % 12)) = 0 ] && say "kubectl auth/API unavailable - waiting (ADC refresh needed?)"; n=$((n+1)); sleep 300; done; }
post(){ # <pol> <clients>: guard + knee check for a cell whose job is Complete but whose runner did not get that far
  bash "$D/mnnvl_transport_guard.sh" n3u-mnnvl-88 >> "$LOG" 2>&1 || { say "MNNVL TRANSPORT VIOLATION - HALTING"; exit 2; }; say "MNNVL gate PASS"
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "n3u-mnnvl-88-agentx-$1-c$2" >> "$LOG" 2>&1 || true; }
unblock(){ # user-authorized 2026-09-18: if one of our workers is Pending for lack of GPUs, scale other tenants' np-2 workloads to 0 (spec backed up first)
  PP=$(kubectl get pods -n dynamo-cloud --no-headers 2>/dev/null | grep "n3u-mnnvl-88-" | awk '$3=="Pending"{print $1}' | head -1); [ -z "$PP" ] && return 0
  sleep 180; kubectl get pod $PP -n dynamo-cloud --no-headers 2>/dev/null | grep -q Pending || return 0
  kubectl describe pod $PP -n dynamo-cloud 2>/dev/null | grep -q "Insufficient nvidia.com/gpu" || { say "worker $PP is Pending but not for lack of GPUs - leaving other tenants alone"; return 0; }
  say "UNBLOCK: our worker $PP cannot schedule (Insufficient nvidia.com/gpu) - clearing other tenants' GPU pods on np-2 (user-authorized)"
  B=/mnt/disks/scratch/np2_backups/$(date -u +%Y%m%dT%H%M); mkdir -p $B
  for node in $(kubectl get nodes -l cloud.google.com/gke-nodepool=np-2 --no-headers | awk '{print $1}'); do
    kubectl get pods -A -o wide --field-selector spec.nodeName=$node --no-headers 2>/dev/null | awk '$1!="dynamo-cloud" && $1!="kube-system" && $1!~/^(gke-|gmp-|nvidia|gpu-operator)/ {print $1, $2}' | while read ns pod; do
      req=$(kubectl get pod $pod -n $ns -o jsonpath='{.spec.containers[*].resources.requests.nvidia\.com/gpu}{.spec.resourceClaims[*].name}' 2>/dev/null); [ -z "$req" ] && continue
      ok=$(kubectl get pod $pod -n $ns -o jsonpath='{.metadata.ownerReferences[0].kind}/{.metadata.ownerReferences[0].name}' 2>/dev/null)
      lws=$(kubectl get pod $pod -n $ns -o jsonpath='{.metadata.labels.leaderworkerset\.sigs\.k8s\.io/name}' 2>/dev/null)
      kubectl get pod $pod -n $ns -o yaml > $B/pod_${ns}_$pod.yaml 2>/dev/null
      if [ -n "$lws" ]; then kubectl get leaderworkerset $lws -n $ns -o yaml > $B/lws_${ns}_$lws.yaml; kubectl scale leaderworkerset $lws -n $ns --replicas=0 >> "$LOG" 2>&1; say "UNBLOCK: scaled leaderworkerset $ns/$lws to 0 (pod $pod on $node; backup $B)"
      elif [ "${ok%%/*}" = "StatefulSet" ]; then kubectl get statefulset ${ok#*/} -n $ns -o yaml > $B/sts_${ns}_${ok#*/}.yaml; kubectl scale statefulset ${ok#*/} -n $ns --replicas=0 >> "$LOG" 2>&1; say "UNBLOCK: scaled statefulset $ns/${ok#*/} to 0 (pod $pod on $node; backup $B)"
      else kubectl delete pod $pod -n $ns --wait=false >> "$LOG" 2>&1; say "UNBLOCK: deleted pod $ns/$pod on $node (owner ${ok:-none}; backup $B)"; fi
    done
  done; }
cellrun(){ # <pol:clients> -> 0 ok, 1 cell failed, exits on MNNVL violation
  authwait; kubectl apply -n dynamo-cloud -f "$M" >> "$LOG" 2>&1; sleep 60; unblock
  bash "$S" n3u-mnnvl-88 "$M" "$1" "N3U AGENTX 88 KVX DONE" /tmp/agentx_88_kvx.log "N3U AGENTX 88 REST CELL $1 DONE" "$LOG" 0 np-2 n3u-mnnvl-88 "$W"; rc=$?
  grep -q "MNNVL TRANSPORT VIOLATION" "$LOG" && { say "MNNVL violation - programme stopped"; exit 2; }
  if [ $rc -ne 0 ]; then authwait; J=alisachen-n3u-mnnvl-88-agentx-${1%%:*}-c${1##*:}
    for i in $(seq 1 $JOB_WAIT_ITERS); do st=$(kubectl get jobs -n dynamo-cloud "$J" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Running" ] || break; sleep 120; done
    [ "$st" = "Complete" ] && { say "cell $1: job is Complete although the runner gave up (rc=$rc) - recovering"; post ${1%%:*} ${1##*:}; rc=0; }; fi
  return $rc; }
p95(){ python3 - "$1" "$2" <<'PY'
import sys,subprocess,csv,io
sh=lambda c: subprocess.run(c,shell=True,capture_output=True,text=True).stdout
d=sh(f"gcloud storage ls gs://alisachen-models/perf/ | grep '_alisachen-n3u-mnnvl-88-agentx-{sys.argv[1]}-c{sys.argv[2]}/' | tail -1").strip()
f=sh(f"gcloud storage ls -r '{d}' | grep profile_export_aiperf.csv | grep -v warmup | head -1").strip()
m={r[0]:r for r in csv.reader(io.StringIO(sh(f"gcloud storage cat '{f}'"))) if r}
print(f"{float(m['Time to First Token (ms)'][12])/1000:.2f}")
PY
}
sel(){ python3 $D/select_agentx_points.py n3u-mnnvl-88 64 "$KVL" "$RRL" --ttft $SLO --ttft-fallback 20 --variants "$VARIANTS" --logs /tmp/agentx_88.log "$LOG" --out-md /tmp/agentx_88_points.md --out-json /tmp/agentx_88_points.json > /tmp/agentx_88_select.log 2>&1; }
js(){ python3 -c "import json;d=json.load(open('/tmp/agentx_88_points.json'));v=$1;print(v if v is not None else '')"; }

GUARD=$D/mnnvl_transport_guard.sh
adopt(){ # <pol> <clients>: a cell whose job is already running - wait for it, then guard + knee check exactly as the runner does
  J=alisachen-n3u-mnnvl-88-agentx-$1-c$2; st=""
  for i in $(seq 1 $JOB_WAIT_ITERS); do st=$(kubectl get jobs -n dynamo-cloud "$J" --no-headers 2>/dev/null | awk '{print $2}'); [ "$st" = "Complete" ] && break; [ "$st" = "Failed" ] && break; sleep 120; done
  say "n3u-mnnvl-88 AgentX $1 c$2 done (job=$st) [adopted]"; [ "$st" = "Complete" ] || return 1
  bash "$GUARD" n3u-mnnvl-88 >> "$LOG" 2>&1 || { say "MNNVL TRANSPORT VIOLATION - HALTING"; exit 2; }; say "MNNVL gate PASS"
  python3 "$HOME/kv-cache-aware-bench/sglang/scripts/knee_check.py" "n3u-mnnvl-88-agentx-$1-c$2" >> "$LOG" 2>&1 || true; }
postknee(){ grep "KNEE-CHECK n3u-mnnvl-88-agentx-$1-c$2:" "$LOG" | tail -1 | grep -q "POST-KNEE"; }
STOP=0; BEST10=""; HI=""

le(){ python3 -c "import sys; sys.exit(0 if float('$1') <= $2 else 1)"; }
docell(){ # <pol:clients>, resumable: skip a finished cell, adopt a running one, else run it
  pol=${1%%:*}; c=${1##*:}; J=alisachen-n3u-mnnvl-88-agentx-$pol-c$c; authwait
  st=$(kubectl get jobs -n dynamo-cloud "$J" --no-headers 2>/dev/null | awk '{print $2}')
  if [ "$st" = "Complete" ] && cat /tmp/agentx_88.log "$LOG" | grep -q "KNEE-CHECK n3u-mnnvl-88-agentx-$pol-c$c:"; then say "cell $1 already measured - skipped"; return 0; fi
  if [ "$st" = "Running" ]; then say "cell $1 is in flight - adopting it"; adopt $pol $c; return $?; fi
  if [ "$st" = "Complete" ]; then say "cell $1 finished without its checks - running them"; post $pol $c; return 0; fi
  cellrun "$1"; }
PICK="python3 $D/pick_agentx_flag_winner.py n3u-mnnvl-88 64"

say "=== v7 (resumable): in-flight flag run, RR ladder at KV's client counts up to RR's knee, selection, flag runs 2-6"
KS=480                                                   # KV same-SLO cell, measured: TTFT p95 7.12 s (SLO 10 s)
docell "kvs3c08:$KS" || say "flag cell kvs3c08 c$KS did not complete"

say "=== RR ladder at KV's client counts (ascending; stops at the first POST-KNEE cell)"
for c in 384 480 672 768 1152; do
  docell "rr:$c" || { say "RR c$c did not complete - RR ladder ends here"; break; }
  RRL="$RRL,$c"; say "RR c$c TTFT p95 = $(p95 rr $c) s"
  postknee rr $c && { say "RR knee found: c$c is POST-KNEE - no larger RR cells"; break; }
done
say "N3U AGENTX 88 RR DONE ($RRL)"

say "=== selection"
sel; CFG=$(js "d['same_cfg']"); RS=$(js "d['rr_slo'] and d['rr_slo']['clients']")
say "N3U AGENTX 88 SELECT DONE same_config=c$CFG kv_slo=c$KS rr_slo=c$RS (SLO TTFT p95 <= $SLO s, RR fallback 20 s)"; cat /tmp/agentx_88_points.md >> "$LOG"

say "=== flag sweep phase A at c$KS (runs 2-4 of 6)"
for v in kvd05 kvd10; do docell "$v:$KS" || say "flag cell $v c$KS did not complete"; done
WS=$($PICK $KS kvs3c08 --logs "$LOG" 2>> "$LOG"); WD=$($PICK $KS kvd05,kvd10 --logs "$LOG" 2>> "$LOG"); WS=${WS%% *}; WD=${WD%% *}
if   [ "$WS" != "kv" ] && [ "$WD" != "kv" ]; then R4=kvs3c08${WD#kv}
elif [ "$WS" != "kv" ]; then R4=kvs4c08
elif [ "$WD" != "kv" ]; then R4=kvd20
else R4=kvs2c08; fi
say "phase A: load-scale verdict=$WS decay verdict=$WD -> run 4 = $R4"; docell "$R4:$KS" || say "flag cell $R4 c$KS did not complete"
WIN=$($PICK $KS "kvs3c08,kvd05,kvd10,$R4" --logs "$LOG" 2>> "$LOG"); say "phase A winner at c$KS: $WIN"; WV=${WIN%% *}

say "=== flag sweep phase B (winner $WV; runs 5-6 of 6)"
if [ "$WV" != "kv" ]; then
  [ -n "$CFG" ] && [ "$CFG" != "$KS" ] && { docell "$WV:$CFG" || say "phase B $WV c$CFG did not complete"; }
  UP=$(python3 -c "k=sorted(int(x) for x in '$KVL'.split(',')); print(next((x for x in k if x > $KS), ''))")
  [ -n "$UP" ] && [ "$UP" != "$CFG" ] && { docell "$WV:$UP" || say "phase B $WV c$UP did not complete"; }
else say "no variant beat default KV by > 2 % - phase B skipped"; fi
say "N3U AGENTX 88 FLAGS DONE"
sel
for d in n3u-mnnvl-88-prefill n3u-mnnvl-88-decode n3u-mnnvl-88-frontend; do kubectl scale deployment/$d -n dynamo-cloud --replicas=0 >> "$LOG" 2>&1; done
say "N3U AGENTX 88 REST DONE kv=$KVL rr=$RRL"
