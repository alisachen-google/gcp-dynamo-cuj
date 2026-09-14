"""Transform the proven n3u-agg-kv.yaml onto the latest+workable stack.

Reuses the agg manifest's already-correct structure (6 x TP4/EP4 workers +
1 KV frontend, nodeSelector/tolerations/volumes/probes/envs) and swaps ONLY
the three stack-specific pieces to the disagg-validated combination:
  base image  lmsysorg/sglang:v0.5.14  -> v0.5.19-cu130-runtime
  dynamo glue ai-dynamo[sglang]==1.3.1 -> ==1.4.2  (+ flashinfer 0.6.18 force pin)
  frontend    tensorrtllm-runtime:1.3.1 (python3) -> sglang image + bash-c/pip
Effective runtime after 1.4.2 resolves: SGLang 0.5.16 + Dynamo 1.4.2 + FI 0.6.18.
Output: sglang/manifests/n3u-agg-newstack.yaml (deployments renamed n3u-agg-ns).
"""
import yaml, pathlib

def _str_pres(d,data):
    if data.lower() in ("y","n","yes","no","true","false","on","off"):
        return d.represent_scalar("tag:yaml.org,2002:str",data,style="'")
    return d.represent_scalar("tag:yaml.org,2002:str",data)
yaml.add_representer(str,_str_pres)

SRC = pathlib.Path.home()/"kv-cache-aware-bench/sglang/manifests/n3u-agg-kv.yaml"
OUT = pathlib.Path.home()/"kv-cache-aware-bench/sglang/manifests/n3u-agg-newstack.yaml"
IMG = "lmsysorg/sglang:v0.5.19-cu130-runtime"
OLD, NEW = "n3u-agg-kv", "n3u-agg-ns"
WORKER_PIP = ('pip install -q "ai-dynamo[sglang]==1.4.2" && '
              'pip install -q --force-reinstall --no-deps flashinfer-python==0.6.18 && ')
FE_ARGS = ('pip install -q "ai-dynamo==1.4.2" && exec python3 -m dynamo.frontend '
           '--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs '
           '--request-plane nats')

def rename(o):
    if isinstance(o, dict):
        return {k: rename(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rename(v) for v in o]
    if isinstance(o, str):
        return o.replace(OLD, NEW)
    return o

docs = [rename(d) for d in yaml.safe_load_all(SRC.read_text()) if d]
for d in docs:
    if d.get("kind") != "Deployment":
        continue
    c = d["spec"]["template"]["spec"]["containers"][0]
    c["image"] = IMG
    if c["name"] == "frontend":
        c["command"] = ["bash", "-c"]
        c["args"] = [FE_ARGS]
    else:  # the agg worker: keep every engine flag, only reforge the pip prefix
        old = " ".join(c["args"]) if isinstance(c["args"], list) else c["args"]
        # strip the old 'pip install ... && python3 -c "..." && exec python3 -m dynamo.sglang'
        tail = old.split("exec python3", 1)[1]  # everything after 'exec python3'
        c["args"] = [WORKER_PIP + "exec python3" + tail]

OUT.write_text(yaml.dump_all(docs, sort_keys=False, default_flow_style=False))
print("wrote", OUT)
