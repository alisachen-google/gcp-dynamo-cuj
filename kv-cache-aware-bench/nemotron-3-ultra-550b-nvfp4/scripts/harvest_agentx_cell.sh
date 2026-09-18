#!/bin/bash
# harvest one AgentX cell: records + aiperf CSV + server metrics -> /mnt/disks/scratch/agentx_recs; prints the report numbers
# usage: harvest_agentx_cell.sh <jobprefix e.g. n3u-mnnvl-88> <pol> <clients> <gpus>
JP=$1; POL=$2; C=$3; G=$4; H=/mnt/disks/scratch/agentx_recs; D=$H/${JP##*-}_${POL}_c$C
A=$(gsutil ls gs://alisachen-models/perf/ | grep "_alisachen-${JP}-agentx-${POL}-c${C}/" | tail -1); A=${A%/}; B=$(basename $A)
R=$(gsutil ls $A/ | grep "trace_c${C}_" | head -1); mkdir -p $D
[ -s $H/${B}_recs.jsonl ] || gsutil -q cp ${R}profile_export.jsonl $H/${B}_recs.jsonl
gsutil -q cp ${R}profile_export_aiperf.csv ${R}server_metrics_export.csv $D/
python3 - "$D" "$H/${B}_recs.jsonl" "$G" "$B" <<'PY'
import csv,sys,json
D,recs,G,B=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
m={}
for r in csv.reader(open(D+"/profile_export_aiperf.csv")):
    if r: m[r[0]]=r
tt=m["Time to First Token (ms)"]; it=m["Inter Token Latency (ms)"]
tot=float(m["Total Token Throughput (tokens/sec)"][1]); out=float(m["Output Token Throughput (tokens/sec)"][1])
c=i=None
for r in csv.reader(open(D+"/server_metrics_export.csv")):
    if len(r)>6 and r[2]=="dynamo_frontend_cached_tokens": c=float(r[6])
    if len(r)>6 and r[2]=="dynamo_frontend_input_sequence_tokens": i=float(r[6])
ev=[];n=0
for l in open(recs):
    try: md=json.loads(l)["metadata"]
    except Exception: continue
    if md.get("benchmark_phase")!="profiling": continue
    s,e=md.get("request_start_ns"),md.get("request_end_ns")
    if s and e: ev+=[(s,1),(e,-1)]; n+=1
ev.sort(); cur=0; area=0; prev=ev[0][0]
for t,d in ev: area+=cur*(t-prev); prev=t; cur+=d
infl=area/(ev[-1][0]-ev[0][0])
print(f"{B}: total/GPU={tot/G:,.0f} out/GPU={out/G:.1f} out_tok_s={out:,.0f} req/s={float(m['Request Throughput (requests/sec)'][1]):.2f} n={n}"
      f" TTFT p50/p90/p95={float(tt[9])/1e3:.2f}/{float(tt[11])/1e3:.2f}/{float(tt[12])/1e3:.2f}s ITL p50/p90={float(it[9]):.1f}/{float(it[11]):.1f}ms P90={1000/float(it[11]):.0f}"
      f" inflight={infl:.1f} hit={c/i:.3f}")
PY
