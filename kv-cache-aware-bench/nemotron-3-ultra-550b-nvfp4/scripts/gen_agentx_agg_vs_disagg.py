#!/usr/bin/env python3
"""Agg (24 GPU) vs disagg (72 GPU, selectable P:D split, default 12:6 = sim-optimal on total tokens) under the
AgentX concurrency definition. Two panels (throughput per GPU, TTFT) with an x-axis toggle: clients, or clients per GPU
(load-normalised: agg /24, disagg /72). Sim v3 (+ load-normalised cells) with measured points overlaid."""
import csv,json,pathlib,glob
R=pathlib.Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(R/"sim-results/dynosim_n3u_agentx_v3.csv")))
for f in glob.glob(str(R/"sim-results/agentx_v3/norm_part*.csv")): rows+=list(csv.DictReader(open(f)))
name={"6:12":"disagg 6:12","9:9":"disagg 9:9","12:6":"disagg 12:6","agg6":"agg 24-GPU"}; hue={"6:12":"--s1","9:9":"--s3","12:6":"--s4","agg6":"--s2"}
PN={"kv":"KV","rr":"RR","kv-tuned":"KV tuned"}
S={}
for r in rows:
    if r["pd"] not in name: continue
    g=24 if r["pd"]=="agg6" else 72
    S.setdefault((r["pd"],r["policy"]),{})[int(r["clients"])]=[int(r["clients"]),round(float(r["total_tok_s"])/g),round(float(r["throughput_tok_s"])/g,1),round(float(r["ttft_p50_s"]),2),round(float(r["ttft_p95_s"]),2),round(1000/float(r["tpot_p90_ms"]),1),g]
series=[{"name":f"{name[pd]} {PN.get(pol,pol)}","c":hue[pd],"pd":pd,"pol":pol,"g":(24 if pd=="agg6" else 72),"pts":[v for k,v in sorted(d.items())]} for (pd,pol),d in sorted(S.items())]
meas=json.load(open(R/"sim-results/measured_agentx.json")) if (R/"sim-results/measured_agentx.json").exists() else []
M=[{"arm":m["arm"],"pol":m["pol"],"clients":m["clients"],"tot":m["tot"],"tpg":m["tpg"],"p50":m["p50"],"p95":m.get("p95"),"g":(24 if m["arm"]=="agg6" else 72)} for m in meas if m["arm"] in name]
head=(R/"reports/n3u-agentx-curve.html").read_text().split("<h1>")[0].replace("N3U AgentX-Concurrency Curve","N3U AgentX Agg vs Disagg")
html=head+'''<h1>Nemotron-3-Ultra 550B — aggregated (24 GPU) vs disaggregated (72 GPU) under the AgentX concurrency definition</h1>
<p class="sub">Simulated (sim v3, dashed) with measured AgentX-mode points (solid). Disagg defaults to the <b>12:6</b> split, the sim's optimum on total tokens per GPU; switch to 9:9 (optimum on output tokens and interactivity) or 6:12. The x-axis toggle matters: at equal <i>clients</i> the 72-GPU disagg fleet carries the same offered load as the 24-GPU agg fleet, so its per-GPU numbers are diluted 3× until it saturates; <i>clients per GPU</i> normalises the load (agg ÷ 24, disagg ÷ 72), which is how InferenceX scales concurrency with deployment size (~20 clients/GPU). y = total tokens (in+out) per second per GPU; toggle to output tokens.</p>
<div class="row" role="group" aria-label="Chart controls"><span style="color:var(--ink2)">x:</span><div class="seg" id="xm"><button data-x="c" aria-pressed="true">clients</button><button data-x="g" aria-pressed="false">clients per GPU</button></div><span style="color:var(--ink2)">disagg split:</span><div class="seg" id="sp"><button data-s="12:6" aria-pressed="true">12:6</button><button data-s="9:9" aria-pressed="false">9:9</button><button data-s="6:12" aria-pressed="false">6:12</button></div><span style="color:var(--ink2)">throughput:</span><div class="seg" id="ym"><button data-y="1" aria-pressed="true">total tok/s/GPU</button><button data-y="2" aria-pressed="false">output tok/s/GPU</button></div><span style="color:var(--ink2)">TTFT:</span><div class="seg" id="yt"><button data-y="3" aria-pressed="true">p50</button><button data-y="4" aria-pressed="false">p95</button></div><label class="chk"><input type="checkbox" id="showkv" checked> KV</label><label class="chk"><input type="checkbox" id="showkt"> tuned KV</label><label class="chk"><input type="checkbox" id="showrr"> RR</label></div>
<h2 style="font-size:15px;font-weight:600;margin:14px 0 4px">Throughput per GPU</h2><div class="chart" id="chart"></div>
<h2 style="font-size:15px;font-weight:600;margin:18px 0 4px">TTFT (log scale)</h2><div class="chart" id="chart2"></div>
<h2 style="font-size:15px;font-weight:600;margin:18px 0 4px">P90 interactivity (tok/s per user, log scale)</h2><div class="chart" id="chart3"></div>
<div class="legend" aria-label="Legend"><span><i class="ln d" style="border-color:var(--s2)"></i>agg 24-GPU</span><span><i class="ln d" style="border-color:var(--s4)"></i>disagg 12:6</span><span><i class="ln d" style="border-color:var(--s3)"></i>disagg 9:9</span><span><i class="ln d" style="border-color:var(--s1)"></i>disagg 6:12</span><span>● KV &nbsp; ◆ tuned KV &nbsp; ○ RR &nbsp; dashed = sim · solid = measured</span></div>
<div class="tiles" id="tiles"></div>
<details><summary>Table view</summary><div class="tw"><table id="tbl"></table></div></details>
<p class="fn">Sim = scripts/dynosim_agentx.py v3 with the load-normalised cells (144/288/576/1152 clients for the disagg splits). Analysis: AGENTX_DISAGG_VS_AGG.md. The sim under-predicts absolute totals ~1.6–2× at the measured cells (trace representation, AGENTX_D72_RESULTS.md §iv); compare arms within the sim, use measured markers for level.</p>
<script>
const S=__S__, M=__M__; const el=id=>document.getElementById(id); let XM="c", SPLIT="12:6", Y=1, YT=3, showKV=true, showKT=false, showRR=false;
const on=s=>(s.pd==="agg6"||s.pd===SPLIT)&&((s.pol==="kv"&&showKV)||(s.pol==="kv-tuned"&&showKT)||(s.pol==="rr"&&showRR));
const xv=(c,g)=>XM==="c"?c:c/g;
function renderPanel(mode,cid){
 const W=1000,H=420,m={t:26,r:150,b:44,l:70}; const vis=S.filter(on); const mv=M.filter(p=>on({pd:p.arm,pol:p.pol}));
 const xs=[...vis.flatMap(s=>s.pts.map(p=>xv(p[0],s.g))),...mv.map(p=>xv(p.clients,p.g))]; const xmin=Math.min(...xs), xmax=Math.max(...xs);
 const x=v=>m.l+(Math.log2(v)-Math.log2(xmin))/(Math.log2(xmax)-Math.log2(xmin))*(W-m.l-m.r);
 const val=(p,g)=>mode===1?p[1]:mode===2?p[2]:mode===3?p[3]:mode===4?p[4]:p[5]; const mval=p=>mode===1?p.tot:mode===2?p.tpg:mode===3?p.p50:mode===4?(p.p95||p.p50):null;
 const vals=[...vis.flatMap(s=>s.pts.map(p=>val(p,s.g))),...mv.map(mval).filter(v=>v!=null)]; const log=mode>=3; const vmax=Math.max(...vals);
 let lo,hi; if(log){lo=mode===5?1:0.1; const e=Math.floor(Math.log10(vmax)); hi=[1,2,5,10].map(k=>k*Math.pow(10,e)).find(v=>v>=vmax)||Math.pow(10,e+1);} else {lo=0; hi=vmax*1.08;}
 const y=v=>log? m.t+(1-(Math.log10(Math.max(v,lo))-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo)))*(H-m.t-m.b) : m.t+(1-v/hi)*(H-m.t-m.b);
 let g=`<svg viewBox="0 0 ${W} ${H}" role="img"><title>${["","total tok/s per GPU","output tok/s per GPU","TTFT p50","TTFT p95","P90 interactivity"][mode]} versus ${XM==="c"?"clients":"clients per GPU"}</title>`;
 if(log){const ticks=[]; for(let e=-1;e<=3;e++)[1,2,5].forEach(k=>{const v=k*Math.pow(10,e); if(v>=lo&&v<=hi)ticks.push(v)}); ticks.forEach(v=>{const major=(v/Math.pow(10,Math.floor(Math.log10(v)+1e-9)))===1; g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)" ${major?"":'stroke-dasharray="2 4"'}/><text x="${m.l-8}" y="${y(v)+4}" font-size="11" fill="var(--ink2)" text-anchor="end">${v<1?v.toFixed(1):v.toLocaleString()}${mode===5?"":" s"}</text>`})}
 else{const st=hi>2000?1000:10; for(let v=0;v<=hi;v+=st)g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${m.l-8}" y="${y(v)+4}" font-size="11" fill="var(--ink2)" text-anchor="end">${v.toLocaleString()}</text>`}
 const XT=XM==="c"?[48,96,192,384,480,768,960,1440,1920]:[1,2,4,8,16,20,32,40,64,80]; XT.filter(v=>v>=xmin&&v<=xmax).forEach(v=>{g+=`<text x="${x(v)}" y="${H-m.b+18}" font-size="11" fill="var(--ink2)" text-anchor="middle">${v}</text>`});
 g+=`<text x="${(m.l+W-m.r)/2}" y="${H-6}" font-size="11.5" fill="var(--ink2)" text-anchor="middle">${XM==="c"?"AgentX concurrency — live session clients (log₂)":"clients per GPU (agg ÷ 24, disagg ÷ 72; log₂)"}</text><text transform="translate(14 ${(m.t+H-m.b)/2}) rotate(-90)" font-size="11.5" fill="var(--ink2)" text-anchor="middle">${["","total tokens / s / GPU","output tokens / s / GPU","TTFT p50 (s)","TTFT p95 (s)","P90 interactivity (tok/s/user)"][mode]}</text>`;
 const ends=[]; const hits=[];
 vis.forEach(s=>{const col=`var(${s.c})`; const dash=s.pol==="rr"?'stroke-dasharray="6 5"':s.pol==="kv-tuned"?'stroke-dasharray="2 4"':'stroke-dasharray="6 5"';
  g+=`<path d="${s.pts.map((p,i)=>`${i?"L":"M"}${x(xv(p[0],s.g))},${y(val(p,s.g))}`).join(" ")}" fill="none" stroke="${col}" stroke-width="2" ${dash} opacity=".85"/>`;
  s.pts.forEach(p=>{const cx=x(xv(p[0],s.g)),cy=y(val(p,s.g)); if(s.pol==="kv-tuned")g+=`<path d="M${cx},${cy-4.5}L${cx+4.5},${cy}L${cx},${cy+4.5}L${cx-4.5},${cy}Z" fill="${col}" stroke="var(--bg)" stroke-width="1.2"/>`; else g+=`<circle cx="${cx}" cy="${cy}" r="4" fill="${s.pol==="kv"?col:'var(--bg)'}" stroke="${s.pol==="kv"?'var(--bg)':col}" stroke-width="1.8"/>`; hits.push({cx,cy,t:`${s.name} · ${p[0]} clients (${(p[0]/s.g).toFixed(1)}/GPU) sim`,v:`${p[1].toLocaleString()} total/GPU · ${p[2]} out/GPU · TTFT p50 ${p[3]} s · p95 ${p[4]} s · P90 ${p[5]}`})});
  const l=s.pts[s.pts.length-1]; ends.push({y:y(val(l,s.g)),x:x(xv(l[0],s.g)),t:s.name});});
 mv.forEach(p=>{const v=mval(p); if(v==null)return; const col=`var(${({"6:12":"--s1","9:9":"--s3","12:6":"--s4","agg6":"--s2"})[p.arm]})`; const cx=x(xv(p.clients,p.g)),cy=y(v); g+=`<circle cx="${cx}" cy="${cy}" r="6.5" fill="${col}" stroke="var(--bg)" stroke-width="2.4"/><text x="${cx+8}" y="${cy-7}" font-size="10.5" fill="var(--ink)">measured ${p.arm}·${p.clients}</text>`; hits.push({cx,cy,t:`measured ${p.arm} ${p.pol} · ${p.clients} clients`,v:`${p.tot.toLocaleString()} total/GPU · ${p.tpg} out/GPU · TTFT p50 ${p.p50} s`});});
 ends.sort((a,b)=>a.y-b.y); for(let i=1;i<ends.length;i++)if(ends[i].y-ends[i-1].y<14)ends[i].y=ends[i-1].y+14; ends.forEach(e=>{g+=`<text x="${e.x+9}" y="${e.y+4}" font-size="10.5" fill="var(--ink)">${e.t}</text>`});
 g+=`<rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="transparent" id="hit"/></svg><div class="tip" role="status" aria-live="polite"></div>`;
 el(cid).innerHTML=g; const svg=el(cid).querySelector("svg"),tip=el(cid).querySelector(".tip");
 svg.querySelector("#hit").addEventListener("mousemove",ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width,py=(ev.clientY-r.top)*H/r.height; let b=null,bd=300; hits.forEach(h=>{const d=(h.cx-px)**2+(h.cy-py)**2; if(d<bd){bd=d;b=h}}); if(!b){tip.style.display="none";return} tip.innerHTML=`<b>${b.t}</b>${b.v}`; tip.style.display="block"; const cr=el(cid).getBoundingClientRect(); let tx=(ev.clientX-cr.left)+14; if(tx+tip.offsetWidth>cr.width)tx-=tip.offsetWidth+28; tip.style.left=tx+"px"; tip.style.top=Math.max(0,(ev.clientY-cr.top)-10)+"px";});
 svg.querySelector("#hit").addEventListener("mouseleave",()=>tip.style.display="none");
 if(cid!=="chart")return;
 let tiles=""; vis.filter(s=>s.pol==="kv").forEach(s=>{const b=s.pts.reduce((a,p)=>p[1]>a[1]?p:a); tiles+=`<div class="tile"><div class="k">${s.name} · sim peak total</div><div class="v">${b[1].toLocaleString()}/GPU</div><div class="n">${b[0]} clients (${(b[0]/s.g).toFixed(0)}/GPU) · ${b[2]} out/GPU · TTFT p50 ${b[3]} s · P90 ${b[5]}</div></div>`}); el("tiles").innerHTML=tiles;
 let t=`<tr><th>arm · policy</th><th>clients</th><th>clients/GPU</th><th>total tok/s/GPU</th><th>output/GPU</th><th>TTFT p50</th><th>TTFT p95</th><th>P90 interactivity</th></tr>`; vis.forEach(s=>s.pts.forEach(p=>{t+=`<tr><td>${s.name}</td><td>${p[0]}</td><td>${(p[0]/s.g).toFixed(1)}</td><td>${p[1].toLocaleString()}</td><td>${p[2]}</td><td>${p[3]} s</td><td>${p[4]} s</td><td>${p[5]}</td></tr>`})); mv.forEach(p=>{t+=`<tr><td><b>measured ${p.arm} ${p.pol}</b></td><td>${p.clients}</td><td>${(p.clients/p.g).toFixed(1)}</td><td>${p.tot.toLocaleString()}</td><td>${p.tpg}</td><td>${p.p50} s</td><td>${p.p95||""}</td><td></td></tr>`}); el("tbl").innerHTML=t;
}
function render(){renderPanel(Y,"chart");renderPanel(YT,"chart2");renderPanel(5,"chart3");}
document.querySelectorAll("#xm button").forEach(b=>b.onclick=()=>{XM=b.dataset.x;document.querySelectorAll("#xm button").forEach(x=>x.setAttribute("aria-pressed",x===b?"true":"false"));render()});
document.querySelectorAll("#sp button").forEach(b=>b.onclick=()=>{SPLIT=b.dataset.s;document.querySelectorAll("#sp button").forEach(x=>x.setAttribute("aria-pressed",x===b?"true":"false"));render()});
document.querySelectorAll("#ym button").forEach(b=>b.onclick=()=>{Y=+b.dataset.y;document.querySelectorAll("#ym button").forEach(x=>x.setAttribute("aria-pressed",x===b?"true":"false"));render()});
document.querySelectorAll("#yt button").forEach(b=>b.onclick=()=>{YT=+b.dataset.y;document.querySelectorAll("#yt button").forEach(x=>x.setAttribute("aria-pressed",x===b?"true":"false"));render()});
["showkv","showkt","showrr"].forEach(id=>el(id).onchange=e=>{({showkv:()=>showKV=e.target.checked,showkt:()=>showKT=e.target.checked,showrr:()=>showRR=e.target.checked})[id]();render()}); render();
</script>
'''
out=R/"reports/n3u-agentx-agg-vs-disagg.html"; out.write_text(html.replace("__S__",json.dumps(series)).replace("__M__",json.dumps(M))); print("wrote",out,"series:",len(series),"measured:",len(M))
