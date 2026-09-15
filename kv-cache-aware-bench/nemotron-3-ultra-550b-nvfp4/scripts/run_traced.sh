#!/bin/bash
# run_traced.sh <script> <deathlog> — logs why a background runner exits (signal/exit code/time).
S=$1; DL=$2; echo "[$(date -u +%F' '%H:%M:%S)] START $S pid=$$" >> "$DL"
trap 'echo "[$(date -u +%F" "%H:%M:%S)] SIGHUP received (ignored) pid=$$" >> "$DL"' HUP
trap 'echo "[$(date -u +%F" "%H:%M:%S)] SIGTERM received pid=$$ — killing child" >> "$DL"; kill $CH 2>/dev/null; exit 143' TERM
trap 'echo "[$(date -u +%F" "%H:%M:%S)] SIGINT received pid=$$" >> "$DL"; kill $CH 2>/dev/null; exit 130' INT
bash "$S" & CH=$!
wait $CH; RC=$?
echo "[$(date -u +%F' '%H:%M:%S)] EXIT $S rc=$RC (rc>128 => killed by signal $((RC-128)))" >> "$DL"
