#!/usr/bin/env python3
"""Regenerates reports/n3u-disagg-curve.html (throughput vs concurrency, 72-GPU disagg N3U).
y defaults to TOTAL tokens (input+output) per second per GPU (InferenceX convention); toggles: output/GPU, fleet output.
Measured totals = aiperf Effective Total Throughput / 72 (latest artifact per job; RR c48/c96 = fleet instance 2).
Sim total = req/s x (mean ISL 85,056 + mean OSL 1,454 of the trace) / 72 -- see footnote on the OSL mismatch."""
import json,pathlib
R=pathlib.Path(__file__).resolve().parents[1]
# [conc, output tok/s (fleet), knee, total tok/s per GPU]
S={
 "mkv":{"name":"MNNVL KV-aware 6:12","c":"--s1","pts":[[12,1734,"AT/PRE",4458],[24,2672,"AT/PRE",6441],[48,4684,"AT/PRE",9914],[96,4807,"AT/PRE",9359],[120,4878,"AT/PRE",9337],[144,4562,"POST",8379],[192,2707,"POST",4413],[288,2489,"POST",3619],[384,3082,"POST",4602],[512,3495,"POST",5075]]},
 "m99":{"name":"MNNVL KV 9:9 split","c":"--s3","pts":[[48,4555,"AT/PRE",9571],[96,6560,"AT/PRE",13729],[120,7089,"AT/PRE",14682],[144,7549,"AT/PRE",15578]]},
 "m126":{"name":"MNNVL KV 12:6 split","c":"--s4","pts":[[48,3797,"AT/PRE",8230],[96,5667,"AT/PRE",11979]]},
 "mkv1":{"name":"MNNVL KV 6:12 — fleet instance 1 (superseded)","c":"--ink3","sup":True,"pts":[[48,3573,"AT/PRE",None],[96,3691,"POST",None],[144,6221,"AT/PRE",None]]},
 "mrr":{"name":"MNNVL round-robin 6:12","c":"--s2","pts":[[12,1381,"AT/PRE",3280],[24,1980,"AT/PRE",4751],[48,2321,"POST",5033],[96,2484,"POST",4335],[144,2078,"POST",3214],[192,1931,"POST",2836]]},
 "skv":{"name":"DynoSim v1 KV 6:12","c":"--s1","sim":True,"pts":[[12,1751,"",1446],[24,3026,"",2500],[48,4725,"",3903],[96,5813,"",4802],[120,5873,"",4852],[144,5920,"",4891],[192,5613,"",4638],[288,4947,"",4087],[384,4397,"",3632],[512,3703,"",3059],[768,2841,"",2347]]},
 "srr":{"name":"DynoSim v1 RR 6:12","c":"--s2","sim":True,"pts":[[12,1505,"",1243],[24,2433,"",2010],[48,3196,"",2641],[96,3595,"",2970],[120,3580,"",2958],[144,3501,"",2893],[192,3363,"",2778],[288,3065,"",2532],[384,2841,"",2347],[512,2571,"",2124],[768,2130,"",1759]]},
}
head=(R/"reports/n3u-disagg-curve.html").read_text().split("<h1>")[0]
html=head+'''<h1>Nemotron-3-Ultra 550B — disaggregated, measured total throughput per GPU vs concurrency</h1>
<p class="sub">72 GPU, TP4/EP4 — the 6:12, 9:9 and 12:6 prefill:decode splits, GB300 NVL72, Weka 256K agentic trace replay, KV over NVLink (MNNVL + mooncake). y defaults to <b>total tokens (input + output) served per second per GPU</b>, InferenceX's convention; toggle to output tokens. Solid = silicon; dashed = DynoSim v1 in the same hue. Marker shape = knee verdict.</p>

<div class="row" role="group" aria-label="Chart controls">
  <div class="seg" id="unit"><button id="u-tot" aria-pressed="true">Total tok/s per GPU (in+out)</button><button id="u-gpu" aria-pressed="false">Output tok/s per GPU</button><button id="u-fleet" aria-pressed="false">Output tok/s (72 GPU)</button></div>
  <label class="chk"><input type="checkbox" id="showsim" checked> show DynoSim v1</label>
</div>

<div class="chart" id="chart"><div class="tip" id="tip" role="status" aria-live="polite"></div></div>

<div class="legend" id="legend" aria-label="Legend">
  <span><i class="ln" style="border-color:var(--s1)"></i>KV-aware 6:12</span>
  <span><i class="ln" style="border-color:var(--s2)"></i>round-robin 6:12</span>
  <span><i class="ln" style="border-color:var(--s3)"></i>KV 9:9 split</span>
  <span><i class="ln" style="border-color:var(--s4)"></i>KV 12:6 split</span>
  <span><i class="ln d" style="border-color:var(--ink3)"></i>DynoSim v1 (dashed, same hue)</span>
  <span>● at / pre-knee &nbsp; ○ post-knee &nbsp; ▢ fleet instance 1, superseded (output units only)</span>
</div>

<div class="tiles">
  <div class="tile"><div class="k">Peak bounded, total tok/s per GPU</div><div class="v">15,578</div><div class="n">9:9 KV · c144 · 104.8 output/GPU · TTFT p50 2.2 s · queue stationary · guard PASS</div></div>
  <div class="tile"><div class="k">Aggregated reference (new stack, KV c32)</div><div class="v">13,320 / GPU</div><div class="n">24-GPU agg bounded peak, 77.3 output/GPU · 9:9 c144 = <b>1.17× total, 1.36× output</b>; 6:12 c48 = 0.74× total</div></div>
  <div class="tile"><div class="k">6:12 peak bounded</div><div class="v">9,914 / GPU</div><div class="n">KV c48 · 65.1 output/GPU · p50 1.0 s (c120: 9,337 · 67.8 output)</div></div>
</div>

<details><summary>Table view (all series)</summary><div class="tw"><table id="tbl"></table></div></details>

<p class="fn">Every point passed the transport gate (mooncake on MNNVL, transfer-engine peak up to 2.0 GB/s per prefill worker, no RDMA device in the pod, no transfer failures). Knee verdicts from per-request timestamp stationarity; no latency SLO gate. Measured totals are aiperf <i>Effective Total Throughput</i> (input tokens counted at full length, i.e. including the ~89% cached prefix, exactly as InferenceX counts them) divided by 72. KV c48–c512, the 9:9 and 12:6 series, and RR c48/c96 are fleet-instance-2 measurements (2026-09-15); KV c12/c24 and RR c12/c24/c144/c192 are instance-1. RR c288 is omitted (1.4% transfer failures). <b>Sim total is under-predicted ~2.4×</b> even where sim output matches silicon: DynoSim replays the trace's recorded output lengths (mean 1,454 tokens/request) while the engine stops at EOS after ~590 on average, so for the same output tok/s the real fleet serves ~2.5× more requests — and therefore ~2.5× more input tokens — per second. Sim total = req/s × (85,056 + 1,454). Stack: SGLang 0.5.16 · Dynamo 1.4.2 · FlashInfer 0.6.18.</p>

<script>
const S=__S__;
const GPUS=72, AGG_TOT=13320, AGG_OUT=77.3, XT=[12,24,48,96,120,144,192,288,384,512,768];
const el=id=>document.getElementById(id);
let unit="tot", showSim=true;
function val(p){return unit==="tot"?p[3]:unit==="gpu"?p[1]/GPUS:p[1]}
function fmt(v){return unit==="gpu"?v.toFixed(1):Math.round(v).toLocaleString()}
function render(){
 const W=1000,H=440,m={t:26,r:170,b:44,l:70};
 const x=c=>m.l+(Math.log2(c)-Math.log2(12))/(Math.log2(768)-Math.log2(12))*(W-m.l-m.r);
 const vis=Object.entries(S).filter(([k,s])=>(!s.sim||showSim)).map(([k,s])=>[k,{...s,pts:s.pts.filter(p=>val(p)!=null)}]).filter(([k,s])=>s.pts.length);
 const ref=unit==="tot"?AGG_TOT:unit==="gpu"?AGG_OUT:AGG_OUT*GPUS;
 let ymax=ref; vis.forEach(([k,s])=>s.pts.forEach(p=>ymax=Math.max(ymax,val(p)))); ymax*=1.08;
 const y=v=>m.t+(1-v/ymax)*(H-m.t-m.b);
 const step=unit==="tot"?2000:unit==="gpu"?10:1000; const ticks=[]; for(let v=0;v<=ymax;v+=step)ticks.push(v);
 let g=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="ct"><title id="ct">Throughput versus concurrency, disaggregated Nemotron-3-Ultra</title>`;
 ticks.forEach(v=>{g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${m.l-8}" y="${y(v)+4}" font-size="11" fill="var(--ink2)" text-anchor="end">${unit==="gpu"?v:v.toLocaleString()}</text>`});
 XT.forEach(c=>{g+=`<text x="${x(c)}" y="${H-m.b+18}" font-size="11" fill="var(--ink2)" text-anchor="middle">${c}</text>`});
 g+=`<text x="${(m.l+W-m.r)/2}" y="${H-6}" font-size="11.5" fill="var(--ink2)" text-anchor="middle">concurrency — always-busy request streams (log₂ scale)</text>`;
 g+=`<text transform="translate(14 ${(m.t+H-m.b)/2}) rotate(-90)" font-size="11.5" fill="var(--ink2)" text-anchor="middle">${unit==="tot"?"total tokens (in+out) / s / GPU":unit==="gpu"?"output tokens / s / GPU":"output tokens / s (72 GPU)"}</text>`;
 const ay=y(ref);
 g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${ay}" y2="${ay}" stroke="var(--rule)" stroke-width="1.5" stroke-dasharray="3 5"/>`;
 g+=`<text x="${W-m.r+8}" y="${ay+4}" font-size="11" fill="var(--ink2)">agg 24-GPU bounded${unit==="fleet"?" (×72 equiv)":""}: ${fmt(ref)}</text>`;
 const ends=[];
 vis.forEach(([k,s])=>{const col=`var(${s.c})`;const d=s.pts.map((p,i)=>`${i?"L":"M"}${x(p[0])},${y(val(p))}`).join(" ");
  g+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="${s.sup?1.5:2}" ${s.sim?'stroke-dasharray="6 5" opacity=".8"':(s.sup?'stroke-dasharray="2 4"':'')} stroke-linejoin="round"/>`;
  if(!s.sim){s.pts.forEach(p=>{const cx=x(p[0]),cy=y(val(p));
   if(s.sup)g+=`<rect x="${cx-4}" y="${cy-4}" width="8" height="8" fill="var(--bg)" stroke="${col}" stroke-width="2"/>`;
   else if(p[2]==="POST")g+=`<circle cx="${cx}" cy="${cy}" r="5" fill="var(--bg)" stroke="${col}" stroke-width="2.2"/>`;
   else g+=`<circle cx="${cx}" cy="${cy}" r="5" fill="${col}" stroke="var(--bg)" stroke-width="2"/>`;});
   if(!s.sup){const last=s.pts[s.pts.length-1];ends.push({y:y(val(last)),x:x(last[0]),t:s.name});}}});
 ends.sort((a,b)=>a.y-b.y);for(let i=1;i<ends.length;i++)if(ends[i].y-ends[i-1].y<14)ends[i].y=ends[i-1].y+14;
 ends.forEach(e=>{g+=`<text x="${e.x+9}" y="${e.y+4}" font-size="11" fill="var(--ink)">${e.t}</text>`});
 g+=`<line id="xh" x1="0" x2="0" y1="${m.t}" y2="${H-m.b}" stroke="var(--ink3)" stroke-dasharray="2 3" style="display:none"/>`;
 g+=`<rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="transparent" id="hit"/></svg>`;
 el("chart").innerHTML=g+`<div class="tip" id="tip" role="status" aria-live="polite"></div>`;
 hover(x,vis,W,H,m);table(vis);
}
function hover(x,vis,W,H,m){
 const svg=el("chart").querySelector("svg"),tip=el("tip"),xh=svg.querySelector("#xh"),hit=svg.querySelector("#hit");
 const concs=XT.filter(c=>vis.some(([k,s])=>s.pts.some(p=>p[0]===c)));
 function mv(ev){const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width;
  let best=concs[0],bd=1e9;concs.forEach(c=>{const d=Math.abs(x(c)-px);if(d<bd){bd=d;best=c}});
  xh.setAttribute("x1",x(best));xh.setAttribute("x2",x(best));xh.style.display="";
  let h=`<b>concurrency ${best}</b>`;vis.forEach(([k,s])=>{const p=s.pts.find(q=>q[0]===best);if(!p)return;
   h+=`<div class="r"><i class="sw" style="background:var(${s.c});${s.sim?'opacity:.6':''}"></i>${s.name}: ${fmt(val(p))}${p[3]!=null&&unit!=="tot"?` <span style="color:var(--ink2)">(${p[3].toLocaleString()} total/GPU)</span>`:''}${unit==="tot"?` <span style="color:var(--ink2)">(${(p[1]/GPUS).toFixed(1)} out/GPU)</span>`:''}${p[2]?` <span style="color:var(--ink2)">${p[2]}</span>`:''}</div>`});
  tip.innerHTML=h;tip.style.display="block";const cr=el("chart").getBoundingClientRect();
  let tx=(ev.clientX-cr.left)+14,ty=(ev.clientY-cr.top)-10;if(tx+tip.offsetWidth>cr.width)tx-=tip.offsetWidth+28;tip.style.left=tx+"px";tip.style.top=Math.max(0,ty)+"px";}
 hit.addEventListener("mousemove",mv);hit.addEventListener("mouseleave",()=>{tip.style.display="none";xh.style.display="none"});
}
function table(vis){let h=`<tr><th>conc</th>${vis.map(([k,s])=>`<th>${s.name}</th>`).join("")}</tr>`;
 XT.forEach(c=>{if(!vis.some(([k,s])=>s.pts.some(p=>p[0]===c)))return;h+=`<tr><td>${c}</td>${vis.map(([k,s])=>{const p=s.pts.find(q=>q[0]===c);return `<td>${p?fmt(val(p))+(p[2]?" "+(p[2]==="POST"?"○":"●"):""):"—"}</td>`}).join("")}</tr>`});
 el("tbl").innerHTML=h;}
[["u-tot","tot"],["u-gpu","gpu"],["u-fleet","fleet"]].forEach(([id,u])=>{el(id).onclick=()=>{unit=u;["u-tot","u-gpu","u-fleet"].forEach(i=>el(i).setAttribute("aria-pressed",i===id?"true":"false"));render()}});
el("showsim").onchange=e=>{showSim=e.target.checked;render()};
render();
</script>
'''
out=R/"reports/n3u-disagg-curve.html"; out.write_text(html.replace("__S__",json.dumps(S))); print("wrote",out,len(html))
