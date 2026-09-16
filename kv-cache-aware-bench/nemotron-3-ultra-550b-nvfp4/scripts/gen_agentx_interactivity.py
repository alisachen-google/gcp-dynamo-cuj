#!/usr/bin/env python3
"""P90 interactivity (tok/s/user = 1000 / per-request TPOT p90) vs total tok/s per GPU for the AgentX-semantics
simulation (sim v3, all splits × policies × client counts) with measured AgentX-mode points overlaid."""
import csv,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(R/"sim-results/dynosim_n3u_agentx_v3.csv")))
name={"3:15":"disagg 3:15","6:12":"disagg 6:12","9:9":"disagg 9:9","12:6":"disagg 12:6","15:3":"disagg 15:3","agg6":"agg 24-GPU"}
hue={"3:15":"--s5","6:12":"--s1","9:9":"--s3","12:6":"--s4","15:3":"--s6","agg6":"--s2"}
PN={"kv":"KV","rr":"RR","kv-tuned":"KV tuned"}
S={}
for r in rows:
    g=24 if r["pd"]=="agg6" else 72
    S.setdefault((r["pd"],r["policy"]),[]).append([int(r["clients"]),round(1000/float(r["tpot_p90_ms"]),1),round(float(r["total_tok_s"])/g),round(float(r["throughput_tok_s"])/g,1),round(float(r["ttft_p50_s"]),2)])
series=[{"name":f"{name[pd]} {PN.get(pol,pol)}","c":hue[pd],"pd":pd,"pol":pol,"pts":sorted(v)} for (pd,pol),v in sorted(S.items())]
# measured: [clients, P90 interactivity (1000/ITL p90 ms), total tok/s/GPU, output/GPU, ttft p50]
meas=json.load(open(R/"sim-results/measured_agentx.json")) if (R/"sim-results/measured_agentx.json").exists() else []
MI={("9:9",48):147.1,("9:9",96):119.0,("9:9",192):101.0,("agg6",48):102.0,("agg6-rr",48):90.9,("agg6",96):42.6,("12:6",96):104.2,("agg6-rr",96):33.8,("agg6",192):19.8,("12:6",192):89.3,("agg6-rr",192):12.2,("agg6",384):12.6}  # 1000/ITL p90
M=[{"arm":m["arm"],"pol":m["pol"],"clients":m["clients"],"inter":MI.get((m["arm"]+("-rr" if m["pol"]=="rr" else ""),m["clients"])),"tot":m["tot"],"tpg":m["tpg"],"p50":m["p50"]} for m in meas if (m["arm"]+("-rr" if m["pol"]=="rr" else ""),m["clients"]) in MI]
head=(R/"reports/n3u-agentx-curve.html").read_text().split("<h1>")[0].replace("N3U AgentX-Concurrency Curve","N3U AgentX Interactivity Frontier")
html=head+'''<h1>Nemotron-3-Ultra 550B — P90 interactivity vs total throughput per GPU, AgentX concurrency definition (simulated)</h1>
<p class="sub">Toggle P:D topologies (prefill:decode workers, 72 GPU) and the agg arm with the buttons; toggle routing policies with the checkboxes. Every point is one simulated cell (arm × routing policy × client count, labelled with the client count along each curve). x = P90 interactivity in tokens/s per user = 1000 / per-request TPOT p90 (the rate 90% of users exceed); y = <b>total</b> tokens (input + output) served per second per GPU, InferenceX's convention, with input tokens counted exactly per simulated request. Larger solid markers are measured AgentX-mode runs (x from aiperf ITL p90). Hue = arm; ● KV, ◆ tuned KV, ○ RR.</p>
<div class="row" role="group" aria-label="Chart controls"><label class="chk"><input type="checkbox" id="showkv" checked> KV-aware routing</label><label class="chk"><input type="checkbox" id="showkt" checked> tuned KV-aware</label><label class="chk"><input type="checkbox" id="showrr" checked> round-robin</label><span class="seg" id="arms" style="gap:0"><button data-a="3:15" aria-pressed="true">3:15</button><button data-a="6:12" aria-pressed="true">6:12</button><button data-a="9:9" aria-pressed="true">9:9</button><button data-a="12:6" aria-pressed="true">12:6</button><button data-a="15:3" aria-pressed="true">15:3</button><button data-a="agg6" aria-pressed="true">agg 24-GPU</button></span><label class="chk"><input type="checkbox" id="showlbl" checked> client labels</label></div>
<div class="chart" id="chart"></div>
<div class="legend" aria-label="Legend"><span><i class="ln" style="border-color:var(--s5)"></i>disagg 3:15</span><span><i class="ln" style="border-color:var(--s1)"></i>disagg 6:12</span><span><i class="ln" style="border-color:var(--s3)"></i>disagg 9:9</span><span><i class="ln" style="border-color:var(--s4)"></i>disagg 12:6</span><span><i class="ln" style="border-color:var(--s6)"></i>disagg 15:3</span><span><i class="ln" style="border-color:var(--s2)"></i>agg 24-GPU</span><span>● KV &nbsp; ◆ tuned KV &nbsp; ○ RR &nbsp; large = measured</span></div>
<div class="tiles" id="tiles"></div>
<details><summary>Table view</summary><div class="tw"><table id="tbl"></table></div></details>
<p class="fn">Sim = scripts/dynosim_agentx.py v3 (sim-results/dynosim_n3u_agentx_v3.csv): live-session-client load model, 1 h windows, per-request TPOT p90 across the window's requests. The sim's total tokens are under-predicted ~1.6–2× against silicon at the measured cells (it replays a 4 k-request slice with shorter inputs and no subagent fan-out; see AGENTX_D72_RESULTS.md §iv), so compare arms within the sim and use the measured markers for absolute level. Interactivity is high on disagg splits because decode batches stay small under replayed think-time; the busy-stream frontier (n3u-frontier.html) is the capacity view.</p>
<script>
const S=__S__, M=__M__; const el=id=>document.getElementById(id); let showKV=true,showKT=true,showRR=true,showLbl=true; const ARMON={"3:15":true,"6:12":true,"9:9":true,"12:6":true,"15:3":true,"agg6":true};
const on=s=>((s.pol==="kv"&&showKV)||(s.pol==="kv-tuned"&&showKT)||(s.pol==="rr"&&showRR))&&ARMON[s.pd];
function render(){
 const W=1000,H=500,m={t:26,r:40,b:50,l:70}; const vis=S.filter(on); const mv=M.filter(p=>on({pol:p.pol,pd:p.arm}));
 const xs=[...vis.flatMap(s=>s.pts.map(p=>p[1])),...mv.map(p=>p.inter)], ys=[...vis.flatMap(s=>s.pts.map(p=>p[2])),...mv.map(p=>p.tot)];
 const xmax=Math.ceil(Math.max(...xs,10)/20)*20+10, ymax=Math.max(...ys,100)*1.08;
 const x=v=>m.l+v/xmax*(W-m.l-m.r), y=v=>m.t+(1-v/ymax)*(H-m.t-m.b);
 let g=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="ct"><title id="ct">P90 interactivity versus total throughput per GPU</title>`;
 for(let v=0;v<=ymax;v+=1000)g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${m.l-8}" y="${y(v)+4}" font-size="11" fill="var(--ink2)" text-anchor="end">${v.toLocaleString()}</text>`;
 for(let v=0;v<=xmax;v+=20)g+=`<line y1="${m.t}" y2="${H-m.b}" x1="${x(v)}" x2="${x(v)}" stroke="var(--grid)"/><text x="${x(v)}" y="${H-m.b+18}" font-size="11" fill="var(--ink2)" text-anchor="middle">${v}</text>`;
 [20,30].forEach(f=>{g+=`<line y1="${m.t}" y2="${H-m.b}" x1="${x(f)}" x2="${x(f)}" stroke="var(--rule,#8a8983)" stroke-width="1.5" stroke-dasharray="3 5"/><text x="${x(f)+5}" y="${m.t+12}" font-size="11" fill="var(--ink2)">≥ ${f} tok/s/user</text>`});
 g+=`<text x="${(m.l+W-m.r)/2}" y="${H-8}" font-size="11.5" fill="var(--ink2)" text-anchor="middle">P90 interactivity — tokens/s per user (1000 / TPOT p90) → better</text><text transform="translate(16 ${(m.t+H-m.b)/2}) rotate(-90)" font-size="11.5" fill="var(--ink2)" text-anchor="middle">total tokens (in+out) / s / GPU ↑ better</text>`;
 const hits=[];
 vis.forEach(s=>{const col=`var(${s.c})`; const dash=s.pol==="rr"?'stroke-dasharray="6 5"':s.pol==="kv-tuned"?'stroke-dasharray="2 4"':'';
  g+=`<path d="${s.pts.map((p,i)=>`${i?"L":"M"}${x(p[1])},${y(p[2])}`).join(" ")}" fill="none" stroke="${col}" stroke-width="1.8" ${dash} opacity=".8"/>`;
  s.pts.forEach(p=>{const cx=x(p[1]),cy=y(p[2]);
   if(s.pol==="kv-tuned")g+=`<path d="M${cx},${cy-4.5}L${cx+4.5},${cy}L${cx},${cy+4.5}L${cx-4.5},${cy}Z" fill="${col}" stroke="var(--bg)" stroke-width="1.2"/>`;
   else g+=`<circle cx="${cx}" cy="${cy}" r="3.6" fill="${s.pol==="kv"?col:'var(--bg)'}" stroke="${s.pol==="kv"?'var(--bg)':col}" stroke-width="1.6"/>`;
   if(showLbl&&s.pol==="kv")g+=`<text x="${cx+5}" y="${cy-5}" font-size="9.5" fill="var(--ink2)">${p[0]}</text>`;
   hits.push({cx,cy,t:`${s.name} · ${p[0]} clients (sim)`,v:`${p[2].toLocaleString()} total/GPU · ${p[3]} output/GPU · P90 interactivity ${p[1]} · TTFT p50 ${p[4]} s`});});});
 mv.forEach(p=>{const col=`var(${({"3:15":"--s5","6:12":"--s1","9:9":"--s3","12:6":"--s4","15:3":"--s6","agg6":"--s2"})[p.arm]})`; const cx=x(p.inter),cy=y(p.tot);
  g+=`<circle cx="${cx}" cy="${cy}" r="7" fill="${col}" stroke="var(--bg)" stroke-width="2.4"/><text x="${cx+9}" y="${cy-8}" font-size="10.5" fill="var(--ink)">measured ${p.arm}·${p.clients}</text>`;
  hits.push({cx,cy,t:`measured ${p.arm} ${p.pol} · ${p.clients} clients`,v:`${p.tot.toLocaleString()} total/GPU · ${p.tpg} output/GPU · P90 interactivity ${p.inter} · TTFT p50 ${p.p50} s`});});
 g+=`<rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="transparent" id="hit"/></svg><div class="tip" id="tip" role="status" aria-live="polite"></div>`;
 el("chart").innerHTML=g; const svg=el("chart").querySelector("svg"),tip=el("tip");
 svg.querySelector("#hit").addEventListener("mousemove",ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width,py=(ev.clientY-r.top)*H/r.height; let b=null,bd=300; hits.forEach(h=>{const d=(h.cx-px)**2+(h.cy-py)**2; if(d<bd){bd=d;b=h}});
  if(!b){tip.style.display="none";return} tip.innerHTML=`<b>${b.t}</b>${b.v}`; tip.style.display="block"; const cr=el("chart").getBoundingClientRect(); let tx=(ev.clientX-cr.left)+14; if(tx+tip.offsetWidth>cr.width)tx-=tip.offsetWidth+28; tip.style.left=tx+"px"; tip.style.top=Math.max(0,(ev.clientY-cr.top)-10)+"px";});
 svg.querySelector("#hit").addEventListener("mouseleave",()=>tip.style.display="none");
 let tiles=""; [["9:9","kv"],["12:6","kv"],["agg6","kv"]].forEach(([pd,pol])=>{const s=S.find(q=>q.pd===pd&&q.pol===pol); if(!s||!on(s))return; const b20=s.pts.filter(p=>p[1]>=20).reduce((a,b)=>b[2]>a[2]?b:a,s.pts[0]); tiles+=`<div class="tile"><div class="k">${s.name} · best total ≥ 20 tok/s/user</div><div class="v">${b20[2].toLocaleString()}</div><div class="n">${b20[0]} clients · P90 ${b20[1]} · TTFT p50 ${b20[4]} s · ${b20[3]} output/GPU</div></div>`}); el("tiles").innerHTML=tiles;
 let t=`<tr><th>arm · policy</th><th>clients</th><th>P90 interactivity</th><th>total tok/s/GPU</th><th>output tok/s/GPU</th><th>TTFT p50 s</th></tr>`; vis.forEach(s=>s.pts.forEach(p=>{t+=`<tr><td>${s.name}</td><td>${p[0]}</td><td>${p[1]}</td><td>${p[2].toLocaleString()}</td><td>${p[3]}</td><td>${p[4]}</td></tr>`})); mv.forEach(p=>{t+=`<tr><td><b>measured ${p.arm} ${p.pol}</b></td><td>${p.clients}</td><td>${p.inter}</td><td>${p.tot.toLocaleString()}</td><td>${p.tpg}</td><td>${p.p50}</td></tr>`}); el("tbl").innerHTML=t;
}
["showkv","showkt","showrr","showlbl"].forEach(id=>el(id).onchange=e=>{({showkv:()=>showKV=e.target.checked,showkt:()=>showKT=e.target.checked,showrr:()=>showRR=e.target.checked,showlbl:()=>showLbl=e.target.checked})[id](); render()});
document.querySelectorAll("#arms button").forEach(b=>b.onclick=()=>{const a=b.dataset.a; ARMON[a]=!ARMON[a]; b.setAttribute("aria-pressed",ARMON[a]?"true":"false"); render()}); render();
</script>
'''
out=R/"reports/n3u-agentx-interactivity.html"; out.write_text(html.replace("__S__",json.dumps(series)).replace("__M__",json.dumps(M))); print("wrote",out,"series:",len(series),"measured:",len(M))
