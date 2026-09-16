"""DynoSim — AgentX concurrency semantics (aiperf --scenario inferencex-agentx-mvp).

Same engine model as dynosim_pd.simulate() (Dynamo KV-router worker_logit, FCFS prefill
tier, decode TPOT model, per-worker radix cache) but a different LOAD MODEL:
  * concurrency C = C live session lanes; a lane holds ONE session at a time and recycles
    to the next session when its last turn completes (session tree = linear here: the
    062126 corpus variant has no subagent fan-out, so in-flight <= C).
  * warm-up: each play starts at turn k ~ U(0.25, 0.75) x n_turns; turns 0..k-1 are issued
    back-to-back to prime the cache and are NOT measured.
  * think-time: turn i+1 is dispatched at max(prev_end, lane_t0 + (ts[i+1]-ts[k])) — the
    recorded start cadence, never before the previous response returns (approximation of
    AgentX's end-to-start replay: the corpus records starts, not response ends).
  * 10 s system idle cap: when nothing is in flight, all pending timers shift uniformly so
    the earliest fires within 10 s.
  * cache-bust first_turn_prefix: hash_ids are salted per play, so reuse survives only
    within a play.
  * metrics over a fixed profile window (default 3600 s) after warm-up.
Usage: dynosim_agentx.py <trace.jsonl> --splits 6:12,9:9,12:6 --clients 48,96,... --policies kv,rr --out csv
"""
import argparse, csv, heapq, json, random, importlib.util, pathlib, sys
from collections import defaultdict
spec=importlib.util.spec_from_file_location("dynosim_pd", pathlib.Path(__file__).with_name("dynosim_pd.py")); dp=importlib.util.module_from_spec(spec); spec.loader.exec_module(dp)

def load_sessions(path, limit=4000):
    rows=[json.loads(l) for l in open(path)][:limit]
    sess=defaultdict(list)
    for r in rows: sess[r["hash_ids"][0] if r["hash_ids"] else -1].append(r)
    out=[]
    for k,v in sess.items():
        v=sorted(v,key=lambda r:r["timestamp"])
        out.append(v)
    return out

class Engine:
    """disagg P:D or agg engine; serve(hash_ids, out_len, now) -> (ttft_s, tpot_s, done_t)"""
    def __init__(self, n_prefill, n_decode, policy, agg=False, router=None):
        self.agg=agg; self.policy=policy; self.router=router or dp.ROUTER
        self.P=[dp.PrefillWorker() for _ in range(n_prefill)]; self.D=[0]*(n_prefill if agg else n_decode)
        self.rr=0; self.hits=0; self.blocks=0
    def serve(self, hid, out_len, now):
        P=self.P; tb=len(hid)
        if self.policy=="rr": w=self.rr%len(P); self.rr+=1; ov=P[w].overlap_blocks(hid)
        else:
            best=None; w=0; ov=0
            for i,pw in enumerate(P):
                o=pw.overlap_blocks(hid); adj=max(0.0,tb-self.router["overlap_credit"]*o)
                sc=self.router["prefill_load_scale"]*adj+pw.load_blocks(now)
                if best is None or sc<best: best,w,ov=sc,i,o
        self.hits+=ov; self.blocks+=tb
        new=(tb-ov)*dp.BLOCK_TOKENS
        rate=dp.AGG_PREFILL_TOKRATE if (self.agg and hasattr(dp,"AGG_PREFILL_TOKRATE")) else dp.PREFILL_TOKRATE
        svc=max(0.005,new/rate); start=max(now,P[w].free_at); pf=start+svc
        P[w].free_at=pf; P[w].queued.append((pf,tb-ov)); P[w].insert(hid)
        if self.agg: d=w
        else: d=min(range(len(self.D)),key=lambda i:self.D[i])
        self.D[d]+=1
        tpot=(dp.agg_tpot_ms(self.D[d]) if self.agg and hasattr(dp,"agg_tpot_ms") else dp.TPOT_BASE_MS+dp.TPOT_SLOPE_MS*self.D[d])/1000.0
        done=pf+out_len*tpot
        return pf-now, tpot, done, d
    def release(self,d): self.D[d]-=1

def simulate_agentx(sessions, n_prefill, n_decode, policy, clients, window=3600.0, idle_cap=10.0, agg=False, seed=42, router=None):
    rnd=random.Random(seed); eng=Engine(n_prefill,n_decode,policy,agg,router)
    order=list(range(len(sessions))); rnd.shuffle(order); nxt=[0]
    def take():
        i=order[nxt[0]%len(order)]; nxt[0]+=1; return sessions[i]
    ev=[]  # (time, kind, lane, payload)   kinds: 'issue' (request ready), 'done' (decode complete, d)
    lanes=[]
    play=[0]
    def new_play(lane, now):
        s=take(); n=len(s); k=min(n-1, int(rnd.uniform(0.25,0.75)*n)) if n>1 else 0
        salt=play[0]; play[0]+=1
        lanes[lane]={"s":s,"k":k,"i":0,"salt":salt,"t0":None,"ts0":s[k]["timestamp"]}
        heapq.heappush(ev,(now,'issue',lane,None))
    for L in range(clients): lanes.append(None); new_play(L,0.0)
    inflight=0; T0=None; recs=[]; now=0.0
    while ev:
        t,kind,lane,pl=heapq.heappop(ev)
        # system idle cap: nothing in flight and the next issue is > idle_cap away -> shift timers
        if kind=='issue' and inflight==0 and t-now>idle_cap and now>0:
            shift=(t-now)-idle_cap; t-=shift
            ev=[(tt-shift if kk=='issue' else tt,kk,ll,pp) for (tt,kk,ll,pp) in ev]; heapq.heapify(ev)
        now=max(now,t)
        if kind=='done':
            eng.release(pl["d"]); inflight-=1
            st=lanes[lane]
            if pl["measured"]: recs.append(pl)
            st["i"]+=1
            if st["i"]>=len(st["s"]): new_play(lane,now); continue
            i=st["i"]; s=st["s"]
            if i<st["k"]: heapq.heappush(ev,(now,'issue',lane,None))          # warm-up: back-to-back
            else:
                if st["t0"] is None: st["t0"]=now; st["ts0"]=s[i]["timestamp"]
                target=st["t0"]+(s[i]["timestamp"]-st["ts0"])/1000.0          # recorded cadence (ms -> s), anchored at lane start
                heapq.heappush(ev,(max(now,target),'issue',lane,None))
            continue
        # issue
        st=lanes[lane]; i=st["i"]; r=st["s"][i]
        if i>=st["k"] and st["t0"] is None: st["t0"]=now; st["ts0"]=r["timestamp"]
        measured = i>=st["k"]
        if measured and T0 is None: T0=now
        if T0 is not None and now>T0+window: break
        hid=[(st["salt"],h) for h in r["hash_ids"]]
        ttft,tpot,done,d=eng.serve(hid,r["output_length"],now); inflight+=1
        heapq.heappush(ev,(done,'done',lane,{"ttft":ttft,"tpot":tpot,"out":r["output_length"],"inp":len(hid)*dp.BLOCK_TOKENS,"done":done,"start":now,"measured":measured,"d":d}))
    win=[x for x in recs if T0 is not None and T0<=x["done"]<=T0+window]
    if not win: return None
    dur=window; tt=sorted(x["ttft"] for x in win); out=sum(x["out"] for x in win)
    tp=sorted(x["tpot"] for x in win); inp=sum(x["inp"] for x in win)
    return {"throughput_tok_s":out/dur,"ttft_p50_s":tt[len(tt)//2],"ttft_p95_s":tt[int(len(tt)*.95)],"ttft_p99_s":tt[min(len(tt)-1,int(len(tt)*.99))],
            "tpot_mean_ms":sum(x["tpot"] for x in win)/len(win)*1000,"tpot_p50_ms":tp[len(tp)//2]*1000,"tpot_p90_ms":tp[min(len(tp)-1,int(len(tp)*.9))]*1000,
            "in_tok_per_req":inp/len(win),"total_tok_s":(inp+out)/dur,"hit_rate":eng.hits/max(1,eng.blocks),"req_per_s":len(win)/dur,"n":len(win)}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("--splits",default="6:12,9:9,12:6"); ap.add_argument("--clients",default="48,96,192,384,768,1536")
    ap.add_argument("--policies",default="kv,rr"); ap.add_argument("--agg",action="store_true",help="also simulate agg 6xTP4"); ap.add_argument("--window",type=float,default=3600); ap.add_argument("--out",default="dynosim_agentx.csv")
    a=ap.parse_args(); dp.apply_n3u_constants(); sessions=load_sessions(a.trace)
    print(f"sessions={len(sessions)} turns={sum(len(s) for s in sessions)}")
    rows=[]
    cfgs=[(tuple(map(int,s.split(":"))),False) for s in a.splits.split(",")]+([((6,0),True)] if a.agg else [])
    for (np_,nd),agg in cfgs:
        for c in map(int,a.clients.split(",")):
            for pol in a.policies.split(","):
                # policy tokens: kv | rr | kv-tuned (= kv-s3.0-c0.8) | kv-s<scale>-c<credit>
                base=pol; router=None
                if pol=="kv-tuned": base="kv"; router={"prefill_load_scale":3.0,"overlap_credit":0.8}
                elif pol.startswith("kv-s"):
                    import re as _re; mm=_re.match(r"kv-s([0-9.]+)-c([0-9.]+)",pol); base="kv"; router={"prefill_load_scale":float(mm.group(1)),"overlap_credit":float(mm.group(2))}
                m=simulate_agentx(sessions,np_,nd,base,c,a.window,agg=agg,router=router)
                if not m: print(f"{np_}:{nd} {pol} c{c}: no measured requests"); continue
                m.update(pd=("agg6" if agg else f"{np_}:{nd}"),clients=c,policy=pol); rows.append(m)
                print(f"{m['pd']:>5} {pol:>2} clients={c:>5} tok/s={m['throughput_tok_s']:7.0f} ({m['throughput_tok_s']/(24 if agg else 72):5.1f}/GPU) ttft p50={m['ttft_p50_s']:6.2f} p95={m['ttft_p95_s']:6.2f} tpot={m['tpot_mean_ms']:5.1f}ms hit={m['hit_rate']:.2f} req/s={m['req_per_s']:.2f} n={m['n']}")
    with open(a.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote",a.out)
