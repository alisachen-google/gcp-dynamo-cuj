"""Derive PROFILED variants of the two comparison arms (same stack, same engine
flags) for the agg-vs-disagg gap analysis:
  n3u-agg-newstack.yaml  -> n3u-agg-prof.yaml   (agg KV c32 cell, 24 GPU)
  n3u-mnnvl-full.yaml    -> n3u-mnnvl-prof.yaml (disagg 6:12 KV c48 cell, 72 GPU)
Adds ONLY: --enable-metrics (engine Prometheus stats), SGLANG_TORCH_PROFILER_DIR
+ an emptyDir for best-effort torch traces. Throughput-affecting flags untouched.
"""
import yaml, pathlib, sys
def _s(d,x):
    if x.lower() in ("y","n","yes","no","true","false","on","off"):
        return d.represent_scalar("tag:yaml.org,2002:str",x,style="'")
    return d.represent_scalar("tag:yaml.org,2002:str",x)
yaml.add_representer(str,_s)
M=pathlib.Path.home()/"kv-cache-aware-bench/sglang/manifests"
JOBS=[("n3u-agg-newstack.yaml","n3u-agg-prof.yaml","n3u-agg-ns","n3u-agg-prof"),
      ("n3u-mnnvl-full.yaml","n3u-mnnvl-prof.yaml","n3u-mnnvl-full","n3u-mnnvl-prof")]
def ren(o,a,b):
    if isinstance(o,dict): return {k:ren(v,a,b) for k,v in o.items()}
    if isinstance(o,list): return [ren(v,a,b) for v in o]
    return o.replace(a,b) if isinstance(o,str) else o
for src,dst,old,new in JOBS:
    docs=[ren(d,old,new) for d in yaml.safe_load_all((M/src).read_text()) if d]
    for d in docs:
        if d.get("kind")!="Deployment": continue
        spec=d["spec"]["template"]["spec"]; c=spec["containers"][0]
        if c["name"]=="frontend": continue
        # worker: enable engine metrics + torch profiler dir
        c["args"]=[a.replace("-m dynamo.sglang","-m dynamo.sglang --enable-metrics",1) for a in c["args"]]
        c.setdefault("env",[]).append({"name":"SGLANG_TORCH_PROFILER_DIR","value":"/tmp/prof"})
        c.setdefault("volumeMounts",[]).append({"mountPath":"/tmp/prof","name":"prof"})
        spec.setdefault("volumes",[]).append({"name":"prof","emptyDir":{"sizeLimit":"20Gi"}})
    (M/dst).write_text(yaml.dump_all(docs,sort_keys=False,default_flow_style=False))
    print("wrote",M/dst)
