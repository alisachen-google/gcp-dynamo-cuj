"""Build reports/n3u-agentx-curve.html: simulated throughput/GPU and TTFT vs AgentX concurrency (live session
clients) for disagg 6:12 / 9:9 (KV, RR) and agg 6xTP4; measured AgentX-mode points overlaid when present
(sim-results/measured_agentx.json: [{arm,pol,clients,tpg,p50,knee}])."""
import csv,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1]
ARMS=sys.argv[1].split(",") if len(sys.argv)>1 and sys.argv[1]!="all" else None
OUT=sys.argv[2] if len(sys.argv)>2 else "reports/n3u-agentx-curve.html"
TITLE=sys.argv[3] if len(sys.argv)>3 else ""
SIM=R/"sim-results/dynosim_n3u_agentx_v3.csv"
if not SIM.exists(): SIM=R/"sim-results/dynosim_n3u_agentx_v2.csv"
rows=[r for r in csv.DictReader(open(SIM)) if ARMS is None or r["pd"] in ARMS]
S={}
for r in rows:
    g=72 if r["pd"]!="agg6" else 24
    key=(r["pd"],r["policy"]); S.setdefault(key,[]).append([int(r["clients"]),round(float(r["throughput_tok_s"])/g,1),round(float(r["ttft_p50_s"]),2),round(float(r["ttft_p95_s"]),2),round(float(r["hit_rate"]),2),round(float(r["req_per_s"])*86510/g,0)])

for v in S.values(): v.sort()
meas=[m for m in (json.load(open(R/"sim-results/measured_agentx.json")) if (R/"sim-results/measured_agentx.json").exists() else []) if ARMS is None or m["arm"] in ARMS]
series=[]
name={"3:15":"disagg 3:15","6:12":"disagg 6:12","9:9":"disagg 9:9","12:6":"disagg 12:6","15:3":"disagg 15:3","agg6":"agg 24-GPU"}; hue={"3:15":"--s5","6:12":"--s1","9:9":"--s3","12:6":"--s4","15:3":"--s6","agg6":"--s2"}
PN={"kv":"KV","rr":"RR","kv-tuned":"KV tuned (scale 3, credit 0.8)"}
for (pd,pol),pts in sorted(S.items()): series.append({"name":f"{name[pd]} {PN.get(pol,pol.upper())} — sim","c":hue[pd],"pol":pol,"pts":pts,"sim":True})
html='''<title>N3U AgentX-Concurrency Curve__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{color-scheme:light;--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8983;--grid:#e6e5e0;--tile:#f5f4ef;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#8e5cd9;--s6:#d63b8f;--focus:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--ink3:#8c8b84;--grid:#2e2e2c;--tile:#222221;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#9d74e0;--s6:#e05aa5;--focus:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--ink3:#8c8b84;--grid:#2e2e2c;--tile:#222221;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#9d74e0;--s6:#e05aa5;--focus:#3987e5}
body{background:var(--bg);color:var(--ink);font:14px/1.5 "IBM Plex Sans",system-ui,sans-serif;padding-block:28px 40px;padding-inline:clamp(16px,4vw,40px);max-width:1080px;margin:0 auto}
h1{font-size:22px;font-weight:600;margin:0 0 4px;text-wrap:balance}.sub{color:var(--ink2);margin:0 0 18px;max-width:74ch}
.row{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin-bottom:10px}.seg{display:inline-flex;border:1px solid var(--grid);border-radius:6px;overflow:hidden}.seg button{background:none;border:0;color:var(--ink2);font:inherit;padding:5px 12px;cursor:pointer}.seg button[aria-pressed="true"]{background:var(--tile);color:var(--ink);font-weight:500}.seg button:focus-visible,label:focus-within{outline:2px solid var(--focus);outline-offset:1px}
label.chk{display:inline-flex;gap:6px;align-items:center;color:var(--ink2);cursor:pointer}
.chart{position:relative;width:100%}svg{width:100%;height:auto;display:block;font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.tip{position:absolute;pointer-events:none;background:var(--bg);border:1px solid var(--grid);border-radius:6px;padding:8px 10px;font:12px/1.45 "IBM Plex Mono",ui-monospace,monospace;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;min-width:230px}.tip b{display:block;margin-bottom:3px;font-weight:500}.tip .r{display:flex;gap:8px;align-items:center;white-space:nowrap}.tip .sw{width:10px;height:3px;border-radius:2px;flex:none}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0 22px;color:var(--ink2);font-size:13px}.legend span{display:inline-flex;align-items:center;gap:7px}.legend .ln{width:22px;height:0;border-top:2px solid;flex:none}.legend .ln.d{border-top-style:dashed}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:22px}.tile{background:var(--tile);border-radius:8px;padding:12px 14px}.tile .k{font-size:12px;color:var(--ink2);letter-spacing:.02em;text-transform:uppercase}.tile .v{font:500 24px/1.2 "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;margin:4px 0 2px}.tile .n{font-size:12px;color:var(--ink2)}
details{margin-top:6px}summary{cursor:pointer;color:var(--ink2)}.tw{overflow-x:auto;margin-top:10px}table{border-collapse:collapse;font:13px "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;min-width:640px}th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:500}
.fn{color:var(--ink2);font-size:12.5px;max-width:78ch;margin-top:18px}
</style>
<h1>Nemotron-3-Ultra 550B__TITLE__ — total throughput per GPU vs AgentX concurrency (live session clients)</h1>
<p class="sub">Concurrency here is the SemiAnalysis AgentX definition: the number of <b>live session clients</b> replaying the Weka 256K trace with its recorded think-time (10 s whole-system idle cap, per-play cache-bust), not always-busy request streams. y defaults to <b>total</b> tokens (input + output) served per second per GPU, InferenceX's convention (sim total = req/s × mean ISL 85,056 + OSL 1,454; measured = aiperf Active Total Throughput); toggle to output tokens. Dashed = DynoSim under those semantics (1 h windows); solid markers = measured AgentX-mode runs when available. Hue = arm; KV solid-marker, RR hollow.</p>
<div class="row" role="group" aria-label="Chart controls"><span style="color:var(--ink2)">throughput panel:</span><div class="seg" id="ym"><button data-y="5" aria-pressed="true">total tok/s per GPU (in+out)</button><button data-y="1" aria-pressed="false">output tok/s per GPU</button></div><span style="color:var(--ink2)">TTFT panel:</span><div class="seg" id="yt"><button data-y="2" aria-pressed="true">TTFT p50 (s)</button><button data-y="3" aria-pressed="false">TTFT p95 (s)</button></div><label class="chk"><input type="checkbox" id="showkv" checked> KV-aware routing</label><label class="chk"><input type="checkbox" id="showkt" checked> tuned KV-aware (scale 3, credit 0.8)</label><label class="chk"><input type="checkbox" id="showrr" checked> round-robin</label></div>
<h2 style="font-size:15px;font-weight:600;margin:14px 0 4px">Throughput per GPU vs AgentX concurrency</h2><div class="chart" id="chart"></div><h2 style="font-size:15px;font-weight:600;margin:18px 0 4px">TTFT vs AgentX concurrency (log scale)</h2><div class="chart" id="chart2"></div>
<div class="legend" aria-label="Legend"><span><i class="ln d" style="border-color:var(--s5)"></i>disagg 3:15</span><span><i class="ln d" style="border-color:var(--s1)"></i>disagg 6:12</span><span><i class="ln d" style="border-color:var(--s3)"></i>disagg 9:9</span><span><i class="ln d" style="border-color:var(--s4)"></i>disagg 12:6</span><span><i class="ln d" style="border-color:var(--s6)"></i>disagg 15:3</span><span><i class="ln d" style="border-color:var(--s2)"></i>agg 24-GPU</span><span>● KV &nbsp; ◆ tuned KV &nbsp; ○ RR &nbsp; dashed = simulated · solid = measured</span></div>
<div class="tiles" id="tiles"></div>
<details><summary>Table view</summary><div class="tw"><table id="tbl"></table></div></details>
<p class="fn">Client load ≠ stream load: at 48 clients the fleet serves ~1/4–1/5 of what 48 busy streams demand, so the knee moves from ~120 streams to hundreds of clients — the reason AgentX sweeps 480–1,920. Sim = scripts/dynosim_agentx.py (same engine model as the busy-stream sim; end-to-start delays approximated by the recorded start cadence anchored per lane). Measured AgentX-mode points come from manifests/perf/sgl-d72-agentx.yaml runs (aiperf --scenario inferencex-agentx-mvp).</p>
<script>
const S=__S__, M=__M__; const el=id=>document.getElementById(id); let Y=5, YT=2, showRR=true, showKV=true, showKT=true; const XT=[48,96,192,384,480,768,960,1440,1536,1920];
function renderPanel(Y,cid){
 const W=1000,H=440,m={t:26,r:160,b:44,l:66}; const x=c=>m.l+(Math.log2(c)-Math.log2(48))/(Math.log2(1920)-Math.log2(48))*(W-m.l-m.r);
 const on=pol=>(pol==="kv"?showKV:pol==="kv-tuned"?showKT:showRR); const vis=S.filter(s=>on(s.pol)); const mv=M.filter(p=>on(p.pol));
 const vals=[...vis.flatMap(s=>s.pts.map(p=>p[Y])),...mv.map(p=>Y===5?p.tot:Y===1?p.tpg:Y===2?p.p50:p.p95)]; const log=(Y===2||Y===3); const lo=log?0.05:0, hi=log?Math.pow(10,Math.ceil(Math.log10(Math.max(...vals)))):Math.max(...vals)*1.08;
 const y=v=>log? m.t+(1-(Math.log10(Math.max(v,lo))-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo)))*(H-m.t-m.b) : m.t+(1-v/hi)*(H-m.t-m.b);
 let g=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="ct"><title id="ct">Throughput or TTFT versus AgentX client concurrency</title>`;
 if(log){for(let d=Math.log10(lo);d<=Math.log10(hi);d++){const v=Math.pow(10,d);g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${m.l-8}" y="${y(v)+4}" font-size="11" fill="var(--ink2)" text-anchor="end">${v>=1?v:v} s</text>`}}
 else{const st=hi>2000?2000:hi>60?20:10; for(let v=0;v<=hi;v+=st)g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--grid)"/><text x="${m.l-8}" y="${y(v)+4}" font-size="11" fill="var(--ink2)" text-anchor="end">${v}</text>`}
 XT.forEach(c=>{g+=`<text x="${x(c)}" y="${H-m.b+18}" font-size="11" fill="var(--ink2)" text-anchor="middle">${c}</text>`});
 g+=`<text x="${(m.l+W-m.r)/2}" y="${H-6}" font-size="11.5" fill="var(--ink2)" text-anchor="middle">AgentX concurrency — live session clients (log₂)</text><text transform="translate(14 ${(m.t+H-m.b)/2}) rotate(-90)" font-size="11.5" fill="var(--ink2)" text-anchor="middle">${["","output tok/s per GPU","TTFT p50 (s)","TTFT p95 (s)","","total tok/s per GPU (input + output)"][Y]}</text>`;
 const ends=[];
 vis.forEach(s=>{const col=`var(${s.c})`; g+=`<path d="${s.pts.map((p,i)=>`${i?"L":"M"}${x(p[0])},${y(p[Y])}`).join(" ")}" fill="none" stroke="${col}" stroke-width="2" stroke-dasharray="${s.pol==="kv-tuned"?"2 4":"6 5"}" opacity=".85"/>`;
  s.pts.forEach(p=>{const cx=x(p[0]),cy=y(p[Y]); if(s.pol==="kv-tuned")g+=`<path d="M${cx},${cy-5}L${cx+5},${cy}L${cx},${cy+5}L${cx-5},${cy}Z" fill="${col}" stroke="var(--bg)" stroke-width="1.5" opacity=".9"/>`; else g+=`<circle cx="${cx}" cy="${cy}" r="4" fill="${s.pol==="kv"?col:'var(--bg)'}" stroke="${s.pol==="kv"?'var(--bg)':col}" stroke-width="1.8" opacity=".9"/>`}); const l=s.pts[s.pts.length-1]; ends.push({y:y(l[Y]),x:x(l[0]),t:s.name});});
 mv.forEach(p=>{const col=`var(${({"3:15":"--s5","6:12":"--s1","9:9":"--s3","12:6":"--s4","15:3":"--s6","agg6":"--s2"})[p.arm]})`; const v=Y===5?p.tot:Y===1?p.tpg:Y===2?p.p50:p.p95; g+=`<circle cx="${x(p.clients)}" cy="${y(v)}" r="6" fill="${p.pol==="kv"?col:'var(--bg)'}" stroke="${p.pol==="kv"?'var(--bg)':col}" stroke-width="2.4"/><text x="${x(p.clients)+8}" y="${y(v)-6}" font-size="10.5" fill="var(--ink)">${p.arm}·${p.clients}${p.knee==="POST"?" ○":""}</text>`});
 ends.sort((a,b)=>a.y-b.y); for(let i=1;i<ends.length;i++)if(ends[i].y-ends[i-1].y<14)ends[i].y=ends[i-1].y+14; ends.forEach(e=>{g+=`<text x="${e.x+9}" y="${e.y+4}" font-size="10.5" fill="var(--ink)">${e.t}</text>`});
 g+=`<line id="xh" x1="0" x2="0" y1="${m.t}" y2="${H-m.b}" stroke="var(--ink3)" stroke-dasharray="2 3" style="display:none"/><rect x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="transparent" id="hit"/></svg><div class="tip" role="status" aria-live="polite"></div>`;
 el(cid).innerHTML=g; const svg=el(cid).querySelector("svg"),tip=el(cid).querySelector(".tip"),xh=svg.querySelector("#xh");
 svg.querySelector("#hit").addEventListener("mousemove",ev=>{const r=svg.getBoundingClientRect(),px=(ev.clientX-r.left)*W/r.width; let best=XT[0],bd=1e9; XT.forEach(c=>{const d=Math.abs(x(c)-px); if(d<bd){bd=d;best=c}}); xh.setAttribute("x1",x(best));xh.setAttribute("x2",x(best));xh.style.display="";
  let h=`<b>${best} clients</b>`; vis.forEach(s=>{const p=s.pts.find(q=>q[0]===best); if(p)h+=`<div class="r"><i class="sw" style="background:var(${s.c});opacity:.6"></i>${s.name}: ${p[5].toLocaleString()} total/GPU · ${p[1]} out/GPU · p50 ${p[2]} s · p95 ${p[3]} s · hit ${p[4]}</div>`}); mv.filter(p=>p.clients===best).forEach(p=>{h+=`<div class="r"><i class="sw" style="background:var(${({"3:15":"--s5","6:12":"--s1","9:9":"--s3","12:6":"--s4","15:3":"--s6","agg6":"--s2"})[p.arm]})"></i>measured ${p.arm} ${p.pol}: ${p.tot.toLocaleString()} total/GPU · ${p.tpg} out/GPU · p50 ${p.p50} s · ${p.knee}</div>`});
  tip.innerHTML=h; tip.style.display="block"; const cr=el(cid).getBoundingClientRect(); let tx=(ev.clientX-cr.left)+14; if(tx+tip.offsetWidth>cr.width)tx-=tip.offsetWidth+28; tip.style.left=tx+"px"; tip.style.top=Math.max(0,(ev.clientY-cr.top)-10)+"px";});
 svg.querySelector("#hit").addEventListener("mouseleave",()=>{tip.style.display="none";xh.style.display="none"});
 if(cid!=="chart")return;
 let t=`<tr><th>clients</th>${vis.map(s=>`<th>${s.name}</th>`).join("")}</tr>`; XT.forEach(c=>{t+=`<tr><td>${c}</td>${vis.map(s=>{const p=s.pts.find(q=>q[0]===c); return `<td>${p?p[5].toLocaleString()+" total · "+p[1]+" out /GPU · "+p[2]+" s":"—"}</td>`}).join("")}</tr>`}); el("tbl").innerHTML=t;
 const kv=S.filter(s=>on(s.pol)&&(s.pol==="kv"||s.pol==="kv-tuned")).concat((showKV||showKT)?[]:S.filter(s=>s.pol==="rr")); let tiles=""; kv.forEach(s=>{const best=s.pts.reduce((a,b)=>b[5]>a[5]?b:a); tiles+=`<div class="tile"><div class="k">${s.name.replace(" — sim","")} · sim peak (total)</div><div class="v">${best[5].toLocaleString()}/GPU</div><div class="n">${best[0]} clients · ${best[1]} output/GPU · TTFT p50 ${best[2]} s · p95 ${best[3]} s</div></div>`}); el("tiles").innerHTML=tiles;
}
function render(){renderPanel(Y,"chart");renderPanel(YT,"chart2");}
document.querySelectorAll("#ym button").forEach(b=>b.onclick=()=>{Y=+b.dataset.y;document.querySelectorAll("#ym button").forEach(x=>x.setAttribute("aria-pressed",x===b?"true":"false"));render()});
document.querySelectorAll("#yt button").forEach(b=>b.onclick=()=>{YT=+b.dataset.y;document.querySelectorAll("#yt button").forEach(x=>x.setAttribute("aria-pressed",x===b?"true":"false"));render()}); el("showrr").onchange=e=>{showRR=e.target.checked;render()}; el("showkv").onchange=e=>{showKV=e.target.checked;render()}; el("showkt").onchange=e=>{showKT=e.target.checked;render()}; render();
</script>
'''
out=R/OUT; out.write_text(html.replace("__S__",json.dumps(series)).replace("__M__",json.dumps(meas)).replace("__TITLE__",TITLE)); print("wrote",out,"series:",len(series),"measured:",len(meas))
