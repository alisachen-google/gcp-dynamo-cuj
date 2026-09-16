#!/usr/bin/env python3
"""Sim-vs-real comparison under the AgentX definition, apple to apple per arm: sim v3 (as published) vs measured,
plus (when sim-results/agentx_decomp.csv exists) the decomposition variants at each measured cell.
Panels: total tok/s per GPU, TTFT p95 (the standard), P90 interactivity — all vs clients; ratio table sim/real."""
import csv,json,pathlib,glob
R=pathlib.Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(R/"sim-results/dynosim_n3u_agentx_v3.csv")))
for f in glob.glob(str(R/"sim-results/agentx_v3/norm_part*.csv")): rows+=list(csv.DictReader(open(f)))
name={"6:12":"disagg 6:12","9:9":"disagg 9:9","12:6":"disagg 12:6","agg6":"agg 24-GPU"}; hue={"6:12":"--s1","9:9":"--s3","12:6":"--s4","agg6":"--s2"}
# measured points: [clients, total/GPU, ttft p95, P90 interactivity, output/GPU, knee]
MI={("9:9","kv",48):147.1,("9:9","kv",96):119.0,("9:9","kv",192):101.0,("agg6","kv",48):102.0,("agg6","rr",48):90.9,("agg6","kv",96):42.6,("12:6","kv",96):104.2,("agg6","rr",96):33.8,("agg6","kv",192):19.8,("12:6","kv",192):89.3,("agg6","rr",192):12.2}
meas=json.load(open(R/"sim-results/measured_agentx.json"))
M={}
for m in meas:
    M.setdefault((m["arm"],m["pol"]),[]).append([m["clients"],m["tot"],m.get("p95"),MI.get((m["arm"],m["pol"],m["clients"])),m["tpg"],m["knee"]])
S={}
for r in rows:
    if r["pd"] not in name or r["policy"] not in ("kv","rr"): continue
    g=24 if r["pd"]=="agg6" else 72
    S.setdefault((r["pd"],r["policy"]),{})[int(r["clients"])]=[int(r["clients"]),round(float(r["total_tok_s"])/g),round(float(r["ttft_p95_s"]),2),round(1000/float(r["tpot_p90_ms"]),1),round(float(r["throughput_tok_s"])/g,1)]
series=[]
for (pd,pol),d in sorted(S.items()):
    if (pd,pol) in M or pol=="kv":  # keep sim lines for arms with measurements (and KV everywhere for context)
        series.append({"name":f"{name[pd]} {pol.upper()} — sim","c":hue[pd],"pd":pd,"pol":pol,"kind":"sim","pts":[v for k,v in sorted(d.items())]})
for (pd,pol),pts in sorted(M.items()):
    series.append({"name":f"{name[pd]} {pol.upper()} — measured","c":hue[pd],"pd":pd,"pol":pol,"kind":"meas","pts":sorted(pts)})
decomp=[]
if (R/"sim-results/agentx_decomp.csv").exists(): decomp=list(csv.DictReader(open(R/"sim-results/agentx_decomp.csv")))
# ratio table rows: for each measured point find the sim cell
ratios=[]
for (pd,pol),pts in sorted(M.items()):
    for p in sorted(pts):
        sc=S.get((pd,pol),{}).get(p[0])
        if sc: ratios.append({"arm":name[pd],"pol":pol.upper(),"clients":p[0],"sim_tot":sc[1],"real_tot":p[1],"r_tot":round(sc[1]/p[1],2),"sim_p95":sc[2],"real_p95":p[2],"r_p95":(round(sc[2]/p[2],2) if p[2] else None),"sim_i90":sc[3],"real_i90":p[3],"r_i90":(round(sc[3]/p[3],2) if p[3] else None)})
head=(R/"reports/n3u-agentx-curve.html").read_text().split("<h1>")[0].replace("N3U AgentX-Concurrency Curve","N3U AgentX Sim vs Real")
html=head+'''<h1>Nemotron-3-Ultra 550B — simulation vs silicon under the AgentX concurrency definition (apple to apple)</h1>
<p class="sub">Dashed = DynoSim v3 as published (AIC-seeded constants, 4 k-request trace slice); solid with large markers = measured aiperf <code>--scenario inferencex-agentx-mvp</code> runs, same fleet manifests, same client counts. Three panels: total tokens (in+out) per second per GPU, <b>TTFT p95</b> (the comparison standard), and P90 interactivity (1000 / ITL p90, sim: 1000 / per-request TPOT p90). The ratio table gives sim ÷ real per measured cell; the step-by-step decomposition of the gap (output length, per-request hand-off constant, decode curve, trace representation) is AGENTX_D72_RESULTS.md §iv.</p>
<div class="row" role="group" aria-label="Chart controls"><span style="color:var(--ink2)">arm:</span><span class="seg" id="arms"><button data-a="agg6" aria-pressed="true">agg 24-GPU</button><button data-a="9:9" aria-pressed="true">disagg 9:9</button><button data-a="12:6" aria-pressed="true">disagg 12:6</button><button data-a="6:12" aria-pressed="false">disagg 6:12</button></span><label class="chk"><input type="checkbox" id="showkv" checked> KV</label><label class="chk"><input type="checkbox" id="showrr" checked> RR</label></div>
<h2 style="font-size:15px;font-weight:600;margin:14px 0 4px">Total throughput per GPU vs clients</h2><div class="chart" id="chart"></div>
<h2 style="font-size:15px;font-weight:600;margin:18px 0 4px">TTFT p95 vs clients (log scale)</h2><div class="chart" id="chart2"></div>
<h2 style="font-size:15px;font-weight:600;margin:18px 0 4px">P90 interactivity vs clients (log scale)</h2><div class="chart" id="chart3"></div>
<div class="legend" aria-label="Legend"><span><i class="ln d" style="border-color:var(--s2)"></i>agg 24-GPU</span><span><i class="ln d" style="border-color:var(--s3)"></i>disagg 9:9</span><span><i class="ln d" style="border-color:var(--s4)"></i>disagg 12:6</span><span><i class="ln d" style="border-color:var(--s1)"></i>disagg 6:12</span><span>dashed = sim · solid + large marker = measured &nbsp; ● KV ○ RR</span></div>
<h2 style="font-size:15px;font-weight:600;margin:18px 0 4px">Sim ÷ real at every measured cell</h2><div class="tw"><table id="tbl"></table></div>
<p class="fn">Measured totals = aiperf Effective Total Throughput ÷ GPUs; measured TTFT p95 and ITL p90 from the aiperf summary; sim totals count input tokens exactly per simulated request. Reading so far: the sim under-predicts total tokens (0.4–0.6×) because its trace slice carries fewer input tokens per request and no subagent fan-out; it over-predicts TTFT p95 (2–2.7× on disagg) because its per-worker FCFS prefill queue has a heavier tail than chunked, router-steered prefill; on output tokens and interactivity it is within 10–20% on disagg. On agg it is pessimistic on both throughput and interactivity at low load (0.35–0.44× on total tokens): its busy-stream decode refit has a cliff past batch 7 (28 + 5.7·bs ms) that the AgentX engine never hits (ITL p50 7–12 ms measured vs 27–40 ms simulated), and it over-packs KV-routed sessions onto one worker; removing the cliff recovers a third of the gap and the rest is the same trace-representation residual as disagg (AGENTX_AGG_RESULTS.md §iv). Rankings of arms and policies are the sim's reliable output; levels come from the measured markers.</p>
<script>
const S=__S__, RT=__RT__; const el=id=>document.getElementById(id); const ARMON={"agg6":true,"9:9":true,"12:6":true,"6:12":false}; let showKV=true, showRR=true;
const on=s=>ARMON[s.pd]&&((s.pol==="kv"&&showKV)||(s.pol==="rr"&&showRR));
function renderPanel(mode,cid){
 const W=1000,H=420,m={t:26,r:170,b:44,l:70}; const vis=S.filter(on);
 const xs=vis.flatMap(s=>s.pts.map(p=>p[0])); const xmin=Math.min(...xs,48), xmax=Math.max(...xs,192);
 const x=v=>m.l+(Math.log2(v)-Math.log2(xmin))/(Math.log2(xmax)-Math.log2(xmin))*(W-m.l-m.r);
 const val=p=>mode===1?p[1]:mode===2?p[2]:p[3]; const vals=vis.flatMap(s=>s.pts.map(val)).filter(v=>v!=null); const log=mode>1; const vmax=Math.max(...vals);
 let lo,hi; if(log){lo=mode===2?0.5:5; const e=Math.floor(Math.log10(vmax)); hi=[1,2,5,10].map(k=>k*Math.pow(10,e)).find(v=>v>=vmax)||Math.pow(10,e+1);} else {lo=0; hi=vmax*1.08;}
 const y=v=>log? m.t+(1-(Math.log10(Math.max(v,lo))-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo)))*(H-m.t-m.b) : m.t+(1-v/hi)*(H-m.t-m.b);
 let g=`<svg viewBox="0 0 ${W} ${H}" role="img"><title>${["","total tok/s per GPU","TTFT p95","P90 interactivity"][mode]} vs clients</title>`;
 if(log){const ticks=[]; for(let e=-1;e<=3;e++)[1,2,5].forEach(k=>{const v=k*Math.pow(10,e); if(v>=lo&&v<=hi)ticks.push(v)}); ticks.forEach(v=>{g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${m.l-8}" y="${y(v)+4}" font-size="11" fill="var(--ink2)" text-anchor="end">${v<1?v.toFixed(1):v.toLocaleString()}${mode===2?" s":""}</text>`})}
 else{const st=hi>2000?1000:10; for(let v=0;v<=hi;v+=st)g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${m.l-8}" y="${y(v)+4}" font-size="11" fill="var(--ink2)" text-anchor="end">${v.toLocaleString()}</text>`}
 [48,96,144,192,288,384,480,576,768,960,1152,1440,1920].filter(v=>v>=xmin&&v<=xmax).forEach(v=>{g+=`<text x="${x(v)}" y="${H-m.b+18}" font-size="11" fill="var(--ink2)" text-anchor="middle">${v}</text>`});
 g+=`<text x="${(m.l+W-m.r)/2}" y="${H-6}" font-size="11.5" fill="var(--ink2)" text-anchor="middle">AgentX concurrency — live session clients (log₂)</text><text transform="translate(14 ${(m.t+H-m.b)/2}) rotate(-90)" font-size="11.5" fill="var(--ink2)" text-anchor="middle">${["","total tokens / s / GPU","TTFT p95 (s)","P90 interactivity (tok/s/user)"][mode]}</text>`;
 const ends=[]; const hits=[];
 vis.forEach(s=>{const col=`var(${s.c})`; const pts=s.pts.filter(p=>val(p)!=null); if(!pts.length)return; const sim=s.kind==="sim";
  g+=`<path d="${pts.map((p,i)=>`${i?"L":"M"}${x(p[0])},${y(val(p))}`).join(" ")}" fill="none" stroke="${col}" stroke-width="${sim?1.8:2.4}" ${sim?'stroke-dasharray="6 5" opacity=".7"':''}/>`;
  pts.forEach(p=>{const cx=x(p[0]),cy=y(val(p)); const r=sim?3.5:6.5; g+=`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${s.pol==="kv"?col:'var(--bg)'}" stroke="${s.pol==="kv"?'var(--bg)':col}" stroke-width="${sim?1.4:2.4}" ${sim?'opacity=".8"':''}/>`; hits.push({cx,cy,t:`${s.name} · ${p[0]} clients`,v:`${p[1].toLocaleString()} total/GPU · TTFT p95 ${p[2]} s · P90 ${p[3]??"—"} · ${p[4]} out/GPU${s.kind==="meas"?" · "+p[5]:""}`})});
  const l=pts[pts.length-1]; ends.push({y:y(val(l)),x:x(l[0]),t:s.name});});
 ends.sort((a,b)=>a.y-b.y); for(let i=1;i<ends.length;i++)if(ends[i].y-ends[i-1].y<14)ends[i].y=ends[i-1].y+14; ends.forEach(e=>{g+=`<text x="${e.x+9}" y="${e.y+4}" font-size="10.5" fill="var(--ink)">${e.t}</text>`});
 g+=`<rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="transparent" id="hit"/></svg><div class="tip" role="status" aria-live="polite"></div>`;
 el(cid).innerHTML=g; const svg=el(cid).querySelector("svg"),tip=el(cid).querySelector(".tip");
 svg.querySelector("#hit").addEventListener("mousemove",ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width,py=(ev.clientY-r.top)*H/r.height; let b=null,bd=300; hits.forEach(h=>{const d=(h.cx-px)**2+(h.cy-py)**2; if(d<bd){bd=d;b=h}}); if(!b){tip.style.display="none";return} tip.innerHTML=`<b>${b.t}</b>${b.v}`; tip.style.display="block"; const cr=el(cid).getBoundingClientRect(); let tx=(ev.clientX-cr.left)+14; if(tx+tip.offsetWidth>cr.width)tx-=tip.offsetWidth+28; tip.style.left=tx+"px"; tip.style.top=Math.max(0,(ev.clientY-cr.top)-10)+"px";});
 svg.querySelector("#hit").addEventListener("mouseleave",()=>tip.style.display="none");
}
function table(){let t=`<tr><th>arm · policy</th><th>clients</th><th>sim total/GPU</th><th>real total/GPU</th><th>sim ÷ real</th><th>sim TTFT p95</th><th>real TTFT p95</th><th>sim ÷ real</th><th>sim P90</th><th>real P90</th><th>sim ÷ real</th></tr>`;
 RT.filter(r=>ARMON[Object.keys({"agg 24-GPU":"agg6","disagg 9:9":"9:9","disagg 12:6":"12:6","disagg 6:12":"6:12"}).length?({"agg 24-GPU":"agg6","disagg 9:9":"9:9","disagg 12:6":"12:6","disagg 6:12":"6:12"})[r.arm]:r.arm]).forEach(r=>{t+=`<tr><td>${r.arm} ${r.pol}</td><td>${r.clients}</td><td>${r.sim_tot.toLocaleString()}</td><td>${r.real_tot.toLocaleString()}</td><td><b>${r.r_tot}×</b></td><td>${r.sim_p95} s</td><td>${r.real_p95??"—"} s</td><td><b>${r.r_p95??"—"}×</b></td><td>${r.sim_i90}</td><td>${r.real_i90??"—"}</td><td><b>${r.r_i90??"—"}×</b></td></tr>`}); el("tbl").innerHTML=t;}
function render(){renderPanel(1,"chart");renderPanel(2,"chart2");renderPanel(3,"chart3");table();}
document.querySelectorAll("#arms button").forEach(b=>b.onclick=()=>{const a=b.dataset.a; ARMON[a]=!ARMON[a]; b.setAttribute("aria-pressed",ARMON[a]?"true":"false"); render()});
el("showkv").onchange=e=>{showKV=e.target.checked;render()}; el("showrr").onchange=e=>{showRR=e.target.checked;render()}; render();
</script>
'''
out=R/"reports/n3u-agentx-sim-vs-real.html"; out.write_text(html.replace("__S__",json.dumps(series)).replace("__RT__",json.dumps(ratios))); print("wrote",out,"series:",len(series),"ratio rows:",len(ratios))
for r in ratios: print(r)
