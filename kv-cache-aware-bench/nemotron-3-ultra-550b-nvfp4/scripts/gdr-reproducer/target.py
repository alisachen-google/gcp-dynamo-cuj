import torch, time, pickle
from nixl._api import nixl_agent, nixl_agent_config
agent = nixl_agent("target", nixl_agent_config(True, True, 7777))
buf = torch.full((256*1024*1024//2,), 7, dtype=torch.int16, device="cuda:0")
reg = agent.register_memory([buf])
xd = agent.get_xfer_descs([buf])
blob = pickle.dumps({"meta": agent.get_agent_metadata(), "descs": agent.get_serialized_descs(xd)})
open("/tmp/meta.bin","wb").write(blob)
print("TARGET_READY_V2", flush=True)
time.sleep(1800)
