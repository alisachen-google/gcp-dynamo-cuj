#!/bin/bash
# Requeue the preempted jobs behind the extended 6:12 sweep: 9:9 -> agg re-sweep.
until grep -q "N3U MNNVL EXT SWEEP DONE" /tmp/resweep_mnnvl_ext.log 2>/dev/null; do
  grep -qE "VIOLATION|HALTING|STACK TIMEOUT|WRONG CONTEXT" /tmp/resweep_mnnvl_ext.log 2>/dev/null && exit 1; sleep 300; done
: > /tmp/resweep_mnnvl_99.log
bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/resweep_mnnvl_99.sh      # 9:9 (has own pre-flight + domain wait)
: > /tmp/resweep_agg_newstack.log
bash $HOME/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/resweep_agg_newstack.sh  # gates on 99 DONE internally
