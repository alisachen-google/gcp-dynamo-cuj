import yaml
NS="dynamo-cloud"; POOL="np-3"; import sys
ARM=sys.argv[4] if len(sys.argv)>4 else "n3u-mnnvl"
NP=int(sys.argv[1]) if len(sys.argv)>1 else 1
ND=int(sys.argv[2]) if len(sys.argv)>2 else 1
NNODES=int(sys.argv[3]) if len(sys.argv)>3 else 2
MODEL="/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4"
SERVED="alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4"
IMG="lmsysorg/sglang:v0.5.19-cu130-runtime"
PIP=('pip install -q "ai-dynamo[sglang]==1.4.2" && '
     'pip install -q --force-reinstall --no-deps flashinfer-python==0.6.18 && ')
ETCD="http://dynamo-platform-etcd.dynamo-cloud.svc.cluster.local:2379"
NATS="nats://dynamo-platform-nats.dynamo-cloud.svc.cluster.local:4222"
KVEV='{"publisher":"zmq","endpoint":"tcp://*:5557","replay_endpoint":"tcp://*:5558"}'

def q(s): return s
def str_pres(d,data):
    if data.lower() in ("y","n","yes","no","true","false","on","off"):
        return d.represent_scalar("tag:yaml.org,2002:str",data,style="'")
    return d.represent_scalar("tag:yaml.org,2002:str",data)
yaml.add_representer(str,str_pres)

# MNNVL env (dsv4 recipe): NVLink transport, NO NET_DEVICES, NO GPU_DIRECT, NO mrdma
MNNVL_ENV=[
 {"name":"UCX_TLS","value":"cuda_copy,cuda_ipc,tcp"},
 {"name":"UCX_CUDA_IPC_ENABLE_MNNVL","value":"y"},
 {"name":"UCX_MEMTYPE_CACHE","value":"n"},
 {"name":"UCX_MEMTYPE_REG_WHOLE","value":"n"},
 {"name":"UCX_PROTO_INFO","value":"y"},
 {"name":"UCX_LOG_LEVEL","value":"info"},
 {"name":"MC_FORCE_MNNVL","value":"1"},
 {"name":"MC_TE_METRIC","value":"true"},
 {"name":"NCCL_MNNVL_ENABLE","value":"1"},
 {"name":"NCCL_CUMEM_ENABLE","value":"1"},
 {"name":"NCCL_SOCKET_IFNAME","value":"eth0"},
 {"name":"GLOO_SOCKET_IFNAME","value":"eth0"},
 {"name":"TP_SOCKET_IFNAME","value":"eth0"},
 {"name":"PYTORCH_CUDA_ALLOC_CONF","value":"expandable_segments:True"},
 {"name":"SGLANG_MOONCAKE_CUSTOM_MEM_POOL","value":"True"},
 {"name":"SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN","value":"1"},
 {"name":"SGLANG_DISABLE_REQUEST_LOGGING","value":"1"},
 {"name":"PYTHONUNBUFFERED","value":"1"},
]
def base_env():
    return [{"name":"ETCD_ENDPOINTS","value":ETCD},{"name":"NATS_SERVER","value":NATS},
            {"name":"DYN_NAMESPACE","value":ARM},
            {"name":"DYN_SYSTEM_PORT","value":"9090"},
            {"name":"HF_HOME","value":"/model-cache"},{"name":"HF_HUB_OFFLINE","value":"1"},
            {"name":"HF_MODULES_CACHE","value":"/tmp/hf_modules"}]

def wargs(mode):
    a=["-m","dynamo.sglang","--model-path",MODEL,"--served-model-name",SERVED,
       "--skip-tokenizer-init","--tp-size","4","--ep-size","4","--quantization","modelopt_fp4",
       "--request-plane","nats","--host","0.0.0.0","--trust-remote-code",
       "--mem-fraction-static","0.85","--context-length","262144","--page-size","64",
       "--watchdog-timeout","1000000","--kv-events-config",KVEV,
       "--disaggregation-mode",mode,"--disaggregation-transfer-backend","mooncake",
       "--disaggregation-bootstrap-port","30001"]
    if mode=="prefill": a+=["--chunked-prefill-size","16384","--max-running-requests","8"]
    else: a+=["--max-running-requests","64"]
    import shlex
    return PIP+"exec python3 "+" ".join(shlex.quote(x) for x in a)

def worker(mode,name):
    return {"apiVersion":"apps/v1","kind":"Deployment",
      "metadata":{"name":name,"namespace":NS,"labels":{"app":name}},
      "spec":{"replicas":1,"strategy":{"type":"Recreate"},"progressDeadlineSeconds":3600,
        "selector":{"matchLabels":{"app":name}},
        "template":{"metadata":{"labels":{"app":name},
           "annotations":{"gke-gcsfuse/volumes":"true","gke-gcsfuse/memory-limit":"4Gi","gke-gcsfuse/ephemeral-storage-limit":"1200Gi"}},
          "spec":{"nodeSelector":{"cloud.google.com/gke-nodepool":POOL,"kubernetes.io/arch":"arm64"},
            "tolerations":[{"key":"nvidia.com/gpu","operator":"Exists","effect":"NoSchedule"},
                           {"key":"kubernetes.io/arch","operator":"Equal","value":"arm64","effect":"NoSchedule"}],
            "resourceClaims":[{"name":"compute-domain-channel","resourceClaimTemplateName":ARM+"-cd-channel"}],
            "containers":[{"name":mode,"image":IMG,"command":["bash","-c"],"args":[wargs(mode)],
              "env":base_env()+MNNVL_ENV,
              "resources":{"limits":{"nvidia.com/gpu":"4"},"claims":[{"name":"compute-domain-channel"}]},
              "securityContext":{"runAsUser":0,"capabilities":{"add":["IPC_LOCK"]}},
              "startupProbe":{"failureThreshold":240,"httpGet":{"path":"/live","port":9090},"periodSeconds":60,"timeoutSeconds":20},
              "volumeMounts":[{"mountPath":"/model-cache","name":"model-cache","readOnly":True},
                              {"mountPath":"/dev/shm","name":"shm"}]}],
            "volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache","readOnly":True}},
                       {"name":"shm","emptyDir":{"medium":"Memory","sizeLimit":"250Gi"}}]}}}}

def frontend():
    args=('pip install -q "ai-dynamo==1.4.2" && exec python3 -m dynamo.frontend '
          '--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --request-plane nats')
    dep={"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":ARM+"-frontend","namespace":NS,"labels":{"app":ARM+"-frontend"}},
      "spec":{"replicas":1,"selector":{"matchLabels":{"app":ARM+"-frontend"}},
        "template":{"metadata":{"labels":{"app":ARM+"-frontend"},"annotations":{"gke-gcsfuse/volumes":"true","gke-gcsfuse/memory-limit":"4Gi"}},
          "spec":{"nodeSelector":{"cloud.google.com/gke-nodepool":POOL},
            "tolerations":[{"operator":"Exists"}],
            "containers":[{"name":"frontend","image":IMG,"command":["bash","-c"],"args":[args],
              "env":base_env(),
              "volumeMounts":[{"mountPath":"/model-cache","name":"model-cache","readOnly":True}]}],
            "volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache","readOnly":True}}]}}}}
    svc={"apiVersion":"v1","kind":"Service","metadata":{"name":ARM+"-frontend","namespace":NS},
         "spec":{"selector":{"app":ARM+"-frontend"},"ports":[{"name":"http","port":8000,"targetPort":8000}]}}
    return [dep,svc]

cd={"apiVersion":"resource.nvidia.com/v1beta1","kind":"ComputeDomain",
    "metadata":{"name":ARM+"-cd","namespace":NS},
    "spec":{"numNodes":NNODES,"channel":{"resourceClaimTemplate":{"name":ARM+"-cd-channel"}}}}

docs=[cd]+frontend()+[*[dict(worker("prefill",ARM+"-prefill"),**{"spec":dict(worker("prefill",ARM+"-prefill")["spec"],replicas=NP)})],*[dict(worker("decode",ARM+"-decode"),**{"spec":dict(worker("decode",ARM+"-decode")["spec"],replicas=ND)})]]
import pathlib
out=pathlib.Path.home()/("kv-cache-aware-bench/sglang/manifests/"+ARM+".yaml")
out.write_text(yaml.dump_all(docs,sort_keys=False,default_flow_style=False))
print("wrote",out)
