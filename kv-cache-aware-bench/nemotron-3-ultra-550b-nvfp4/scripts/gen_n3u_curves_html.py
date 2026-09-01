"""Nemotron-3-Ultra 550B NVFP4 — DynoSim pareto/curves page (Kimi-style panels).

Reuses the sglang curves template machinery; adds cross-MODEL panels (Kimi vs
N3U, same simulator + trace, model constants swapped).
Usage: gen_n3u_curves_html.py <out.html>
"""
import csv, importlib.util, json, sys
from pathlib import Path

BASE = Path.home() / "kv-cache-aware-bench"
NR = BASE / "nemotron-3-ultra-550b-nvfp4/sim-results"
SR = BASE / "sglang/results"

spec = importlib.util.spec_from_file_location(
    "kimi_curves", BASE / "sglang/scripts/gen_sglang_curves_html.py")
kc = importlib.util.module_from_spec(spec); spec.loader.exec_module(kc)
load_one, series, cloud_of = kc.load_one, kc.series, kc.cloud_of

def main():
    out = sys.argv[1]
    n_agg = load_one(NR / "dynosim_n3u_agg_v1.csv")
    n_d72 = load_one(NR / "dynosim_n3u_disagg72_v1.csv")
    k_agg = load_one(SR / "dynosim_sgl_agg_v2.csv")
    k_d72 = load_one(SR / "dynosim_sgl_disagg72_v1.csv")
    data = {
        "aggttft": series(n_agg, "agg6", "ttft_p95_s"),
        "aggtput": series(n_agg, "agg6", "throughput_tok_s"),
        "aggcloud": cloud_of(n_agg, "agg6", 24),
        "d72pd": {pd: series(n_d72, pd, "throughput_tok_s")
                  for pd in ("3:15", "6:12", "9:9", "12:6", "15:3")},
        "d72ttft": series(n_d72, "6:12", "ttft_p95_s"),
        "d72cloud": cloud_of(n_d72, "6:12", 72),
        "xagg": {"kimi": cloud_of(k_agg, "agg6", 24), "n3u": cloud_of(n_agg, "agg6", 24)},
        "xd72": {"kimi": cloud_of(k_d72, "6:12", 72), "n3u": cloud_of(n_d72, "6:12", 72)},
    }
    tpl = kc.TEMPLATE
    # retitle + swap panel wiring for the n3u set
    tpl = tpl.replace("SGLang — DynoSim pareto curves (Kimi-K2.5, GB300)",
                      "Nemotron-3-Ultra — DynoSim pareto curves (GB300, sglang)")
    tpl = tpl.replace("SGLang backend — DynoSim pareto curves",
                      "Nemotron-3-Ultra 550B-A55B NVFP4 — DynoSim pareto curves")
    tpl = tpl.replace(
        "Kimi-K2.5-NVFP4 · GB300 · sgl-sim v1 seed (AIC SILICON ratio transfer onto live-calibrated\ntrtllm constants) · session-interleaved Weka 256k trace · selected cells ringed (agg conc 16 · 72-GPU 6:12/384) ·\nhover marks for detail · silicon overlays land after the live sglang ladders",
        "hybrid Mamba-2/LatentMoE · n3u-sim v1 (AIC 0.11.0 NVFP4 SILICON fit: prefill 19.7k/TP4-worker · "
        "TPOT 5.26+0.277bs no-cliff · cache unbounded) · same session-interleaved Weka 256k trace as Kimi · "
        "selection pending SELECTION.md · silicon overlays after smokes")
    # replace the panel wiring block (between the first sectionHeader and the tooltip code)
    start = tpl.index("sectionHeader('Disaggregated 24 GPU")
    end = tpl.index("const tip=document.getElementById('tip');")
    wiring = """
sectionHeader('Aggregated 24 GPU (6 \\u00d7 TP4) \\u2014 hybrid-Mamba agg');
lineChart('p1','1 \\u00b7 TTFT p95 vs concurrency','no-SLO study: 5 s line is reference only',D.aggttft,v=>v>=10?v.toFixed(0)+'s':v.toFixed(1)+'s','TTFT p95 (s, log)',{logy:true,ref5:true});
lineChart('p2','2 \\u00b7 Throughput vs concurrency','no TPOT batch cliff \\u2014 curves keep climbing where Kimi peaked at conc 16',D.aggtput,v=>v.toFixed(0),'output tok/s',{});
hullPanel('p3','3 \\u00b7 Efficiency frontier (agg, hull)','Hollow = TTFT p95 above 5 s (reference)',D.aggcloud,false);
hullPanel('p4','4 \\u00b7 E2E-normalized frontier (agg)','TTFT included in per-user rate',D.aggcloud,true);

sectionHeader('Disaggregated 72 GPU (18 \\u00d7 TP4)');
lineChart('p5','5 \\u00b7 TTFT p95 vs concurrency (6:12)','',D.d72ttft,v=>v>=10?v.toFixed(0)+'s':v.toFixed(1)+'s','TTFT p95 (s, log)',{logy:true,ref5:true});
panel('p6','6 \\u00b7 Throughput vs concurrency by P:D split','Ordinal blues = prefill workers 3\\u219215; dashed gray = RR at 6:12',()=>{
  const RAMP={'3:15':'o0','6:12':'o1','9:9':'o2','12:6':'o3','15:3':'o4'};
  const all=Object.values(D.d72pd).flatMap(s=>Object.values(s).flat());
  if(!all.length)return '';
  const concs=[...new Set(all.map(p=>p.conc))].sort((a,b)=>a-b);
  const xmin=Math.log(concs[0]),xmax=Math.log(concs[concs.length-1]);
  const X=c=>M.l+((Math.log(c)-xmin)/(xmax-xmin))*(PW-M.l-M.r);
  const ymax=Math.max(...all.map(p=>p.v))*1.1;
  const Y=v=>PH-M.b-(v/ymax)*(PH-M.t-M.b);
  let g=axes(X,Y,concs,[0,.25,.5,.75,1].map(f=>Math.round(ymax*f)),'concurrency (log)','output tok/s (system)');
  for(const pd of Object.keys(RAMP)){
   const pts=(D.d72pd[pd]&&(D.d72pd[pd]['kv-t']||D.d72pd[pd]['kv-nvda']))||[];if(!pts.length)continue;
   g+=`<path d="${pts.map((p,i)=>`${i?'L':'M'}${X(p.conc)} ${Y(p.v)}`).join('')}" fill="none" stroke="var(--${RAMP[pd]})" stroke-width="2"/>`;
   for(const p of pts) g+=mark(X(p.conc),Y(p.v),RAMP[pd],JSON.stringify({s:'KV '+pd,conc:p.conc,v:p.v.toFixed(0)+' tok/s',ttft95:p.ttft95}),false);
  }
  const rr=(D.d72pd['6:12']&&D.d72pd['6:12']['rr'])||[];
  g+=`<path d="${rr.map((p,i)=>`${i?'L':'M'}${X(p.conc)} ${Y(p.v)}`).join('')}" fill="none" stroke="var(--ref)" stroke-width="2" stroke-dasharray="5 4"/>`;
  for(const p of rr) g+=mark(X(p.conc),Y(p.v),'ref',JSON.stringify({s:'RR 6:12',conc:p.conc,v:p.v.toFixed(0)+' tok/s',ttft95:p.ttft95}),false);
  return g;
 },[['KV 3:15','o0'],['KV 6:12','o1'],['KV 9:9','o2'],['KV 12:6','o3'],['KV 15:3','o4'],['RR 6:12 (dashed)','ref']]);
hullPanel('p7','7 \\u00b7 Efficiency frontier (72 GPU, 6:12, hull)','Hollow = TTFT p95 above 5 s (reference)',D.d72cloud,false);

sectionHeader('Cross-model \\u2014 Kimi-K2.5 (attention-heavy) vs Nemotron-3-Ultra (hybrid Mamba), same trace + simulator');
for(const [pid,key,title,gpus] of [['p8','xagg','8 \\u00b7 Aggregated 24-GPU frontier',24],['p9','xd72','9 \\u00b7 Disagg 72-GPU frontier (6:12)',72]]){
 panel(pid,title,'Does the KV-vs-RR gap survive cheap recompute + abundant cache? Solid=KV, dashed=RR',()=>{
  const cl=D[key];
  const sets=[['Kimi KV','ref',frontier((cl.kimi['kv-t']||cl.kimi['kv-nvda']||[]))],['Kimi RR','o1',frontier(cl.kimi['rr']||[])],
              ['N3U KV','s3',frontier((cl.n3u['kv-t']||cl.n3u['kv-nvda']||[]))],['N3U RR','s1',frontier(cl.n3u['rr']||[])]];
  const all=sets.flatMap(s=>s[2]);if(!all.length)return '';
  const xmax=Math.max(...all.map(p=>p.x))*1.08,ymax=Math.max(...all.map(p=>p.y))*1.1;
  const X=v=>M.l+(v/xmax)*(PW-M.l-M.r),Y=v=>PH-M.b-(v/ymax)*(PH-M.t-M.b);
  let g=axes(X,Y,[0,.25,.5,.75,1].map(f=>Math.round(xmax*f)),[0,.25,.5,.75,1].map(f=>Math.round(ymax*f)),'tok/s per user','tok/s per GPU',v=>v);
  for(const [name,c,fr] of sets){if(!fr.length)continue;
   const dash=name.includes('RR')?' stroke-dasharray="5 4"':'';
   g+=`<path d="${fr.map((p,i)=>`${i?'L':'M'}${X(p.x)} ${Y(p.y)}`).join('')}" fill="none" stroke="var(--${c})" stroke-width="2"${dash}/>`;
   for(const p of fr) g+=mark(X(p.x),Y(p.y),c,JSON.stringify({s:name,conc:p.conc,v:p.y.toFixed(1)+' tok/s/GPU',ttft95:p.ttft95}),false);}
  return g;
 },[['N3U KV','s3'],['N3U RR (dashed)','s1'],['Kimi KV','ref'],['Kimi RR (dashed)','o1']]);
}

"""
    tpl = tpl[:start] + wiring + tpl[end:]
    tpl = tpl.replace("Engine pin: sglang 0.5.14 + dynamo 1.3.1 (v0.5.17 rejected — glue incompatibility).",
                      "Engine pin (planned): sglang 0.5.14 + dynamo 1.3.1, pending NemotronH smoke; aiconfigurator 0.11.0 + plotext 5.3.2.")
    Path(out).write_text(tpl.replace("__DATA__", json.dumps(data)))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
