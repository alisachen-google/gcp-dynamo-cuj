#!/bin/bash
ps -eo pid,args | grep -E "trim_ext|capture_c144|resweep_mnnvl_d72_ext|chain_after_ext|run_profiled" | grep -v grep | grep -v pm_list
