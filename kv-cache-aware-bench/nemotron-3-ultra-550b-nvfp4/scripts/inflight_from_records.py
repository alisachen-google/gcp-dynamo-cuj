import json,sys
for f in sys.argv[1:]:
    ev=[]; n=0; lat=0.0; phases={}
    with open(f) as fh:
        for l in fh:
            try: r=json.loads(l)
            except: continue
            md=r.get("metadata",{}); ph=md.get("benchmark_phase"); phases[ph]=phases.get(ph,0)+1
            if ph!="profiling": continue
            s=md.get("request_start_ns"); e=md.get("request_end_ns")
            if not s or not e: continue
            ev.append((s,1)); ev.append((e,-1)); n+=1; lat+=(e-s)/1e9
    ev.sort(); cur=0; area=0.0; t0=ev[0][0]; t1=ev[-1][0]; prev=t0; peak=0
    for t,d in ev:
        area+=cur*(t-prev)/1e9; prev=t; cur+=d; peak=max(peak,cur)
    win=(t1-t0)/1e9
    print(f"{f.split('_alisachen-')[-1]}: profiling reqs={n} window={win:.0f}s mean_inflight={area/win:.1f} peak_inflight={peak} req/s={n/win:.2f} mean_latency={lat/n:.2f}s  (Little: {n/win*lat/n:.1f}) phases={phases}")
