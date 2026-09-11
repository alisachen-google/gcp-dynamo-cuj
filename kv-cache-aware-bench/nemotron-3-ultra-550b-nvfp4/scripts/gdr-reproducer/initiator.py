import torch, time, pickle, sys
from nixl._api import nixl_agent, nixl_agent_config
d = pickle.load(open("/tmp/meta.bin","rb"))
agent = nixl_agent("initiator", nixl_agent_config(False, True, 7778))
local = torch.zeros((256*1024*1024//2,), dtype=torch.int16, device="cuda:0")
agent.register_memory([local])
remote_name = agent.add_remote_agent(d["meta"])
rdescs = agent.deserialize_descs(d["descs"])
ldescs = agent.get_xfer_descs([local])
t0=time.time()
h = agent.initialize_xfer("READ", ldescs, rdescs, remote_name, b"uid1")
state = agent.transfer(h)
while True:
    state = agent.check_xfer_state(h)
    if state == "DONE": break
    if state == "ERR": print("XFER_ERR", flush=True); sys.exit(2)
    time.sleep(0.01)
dt=time.time()-t0
ok = bool((local[:1000]==7).all().item()) and bool((local[-1000:]==7).all().item())
print(f"XFER_DONE bytes=268435456 secs={dt:.3f} GBps={0.25/dt:.2f} verified={ok}", flush=True)
