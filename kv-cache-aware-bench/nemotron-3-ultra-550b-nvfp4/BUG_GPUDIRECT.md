# BUG: GPUDirect RDMA (NIC↔GPU peer DMA) broken on rebuilt GB300 node image

**Severity:** High — disables disaggregated KV-cache transfer over RDMA;
forces host-staged fallback at ~50–100× per-transfer cost.
**Component:** GKE GB300 node image / driver / IOMMU config.
**Status:** reproducible fleet-wide (np-1, np-2, np-3); regression vs pre-rebuild.

---

## Environment (fill cluster/project with real names when filing)

| | |
|---|---|
| Cluster / project | `<cluster>` / `<project>` (GKE, GB300 NVL72 pools np-1..np-4) |
| GPU | NVIDIA GB300 |
| GPU driver | **580.126.20** |
| Kernel | **6.12.85+** |
| Node OS | Container-Optimized OS (Google) |
| Container runtime | containerd 2.1.7 |
| NIC | ConnectX-8 (8× mlx5, RoCE v2, GID index 5) |
| Transport stack | NIXL + UCX 1.20.0 (bundled in nixl_cu13 wheel) / sglang disagg |

---

## Summary

Registering CUDA (GPU) memory for RDMA **succeeds via dmabuf**, but the actual
NIC→GPU peer-DMA transfer **aborts the RC queue-pair mid-flight**
(`NIXL_ERR_REMOTE_DISCONNECT`), with the remote process verified still alive.
The `nvidia_peermem` kernel module is **absent from the node image** entirely.
Host-staged RDMA (data copied GPU→host→NIC) works fine, proving the RDMA
fabric and NICs are healthy — the fault is specifically GPU-memory peer DMA.

This fleet ran GPUDirect KV transfer **successfully before the mid-August 2026
node rebuild**; it has failed identically on every probe since.

---

## Impact

- All disaggregated serving falls back to host-staged transport
  (`UCX_IB_GPU_DIRECT_RDMA=n`): measured 0.34 GB/s vs the multi-GB/s a direct
  path should give.
- Per-request KV transfer floor ~2.4 s (0.8 GB payload) that even fully
  cache-hit requests pay; prefill-tier effective throughput ~halved.
- Downstream: disagg bounded-concurrency knees compressed ~9×; a 3.4 GB/request
  model (Kimi-K2.5) becomes SLO-unservable disaggregated at any threshold.

---

## Reproduction

### A. Fastest signal — kernel module check (any node, ~1 min)
```bash
# privileged pod pinned to a GPU node, nsenter to host:
kubectl run gdrchk -n <ns> --restart=Never --image=busybox \
  --overrides='{"spec":{"nodeName":"<gpu-node>","hostPID":true,
   "tolerations":[{"operator":"Exists"}],
   "containers":[{"name":"c","image":"busybox","securityContext":{"privileged":true},
   "command":["sh","-c"],"args":["nsenter -t 1 -m -- sh -c '\''lsmod|grep peermem; modprobe nvidia_peermem 2>&1; find /lib/modules/$(uname -r) -iname \"*peermem*\"'\''"]}]}}'
kubectl logs gdrchk -n <ns>
```
**Observed:** `modprobe: FATAL: Module nvidia_peermem not found in directory
/lib/modules/6.12.85+` — and no `*peermem*` file exists on the node.

### B. Definitive A/B — 30-line NIXL transfer (2 GPU pods, ~2 min after pods up)
Reproducer committed at `scripts/gdr-reproducer/{target.py,initiator.py}`.
Two pods with 8× mrdma DRA claims, one GPU each, anti-affinity to separate
nodes; register a 256 MB CUDA buffer on the target, READ it from the initiator:
```bash
# target pod (B):     python3 target.py       # registers cuda buffer, dumps metadata
# initiator pod (A):  python3 initiator.py    # READs it, times + verifies
# flip one env var and re-run:
UCX_IB_GPU_DIRECT_RDMA=y  → NIXL_ERR_REMOTE_DISCONNECT (target stays alive)
UCX_IB_GPU_DIRECT_RDMA=n  → XFER_DONE, verified=True, 0.34 GB/s
```
Env on both pods: `UCX_TLS=cuda_copy,rc_x,tcp`, `UCX_NET_DEVICES=mlx5_0:1..mlx5_7:1`,
`UCX_IB_GID_INDEX=5`, `UCX_IB_ROCE_LOCAL_SUBNET=y`, `UCX_LOG_LEVEL=debug`,
`UCX_PROTO_INFO=y`.

---

## Evidence (verbatim UCX debug, initiator, GPU_DIRECT=y)

GDR detection probes — all fail because the module is absent:
```
ib_md.c: mlx5_0: cuda GPUDirect RDMA is not detected by checking /sys/kernel/mm/memory_peers/nv_mem/version
ib_md.c: mlx5_0: cuda GPUDirect RDMA is not detected by checking /sys/module/nvidia_peermem/version
ib_md.c: mlx5_0: cuda GPUDirect RDMA is not detected by checking /sys/module/nv_peer_mem/version
```
…but dmabuf registration succeeds:
```
cuda_copy_md.c: dmabuf is supported on cuda device 0
ib_md.c: mlx5_0: dmabuf is supported
```
Transfer then aborts:
```
nixl_cu13._bindings.nixlRemoteDisconnectError: NIXL_ERR_REMOTE_DISCONNECT
```
(target process confirmed alive afterward → transport-level QP abort, not a peer crash.)

Control, same pods/buffers, `GPU_DIRECT=n`:
```
XFER_DONE bytes=268435456 secs=0.743 GBps=0.34 verified=True
```

---

## Root-cause hypothesis

dmabuf registration path works, but the **NIC↔GPU peer-DMA datapath faults**.
Prime suspects on the rebuilt image:
1. **IOMMU / PCIe ACS** blocking peer-to-peer DMA (pre-rebuild image allowed it);
2. driver/firmware dmabuf mapping fault at kernel 6.12.85+ / driver 580.126.20;
3. missing `nvidia_peermem` (the classic GDR registration path) — but note
   shipping peermem alone will still fail if (1)/(2) is the cause; the
   **datapath** is the thing to fix.

## Regression boundary

Pre-mid-August-2026 node image on this same fleet ran GPUDirect KV transfer
successfully (rc_mlx5 CUDA zero-copy proto rows, no bypass flag). Failure
began after the node rebuild to the environment above.

## Verification after fix

Re-run reproducer B with `UCX_IB_GPU_DIRECT_RDMA=y`: PASS = `XFER_DONE
verified=True` at multi-GB/s, zero `REMOTE_DISCONNECT`, and UCX proto tables
showing single-stage `zero-copy … rc_mlx5 … cuda/dev[0]` rows (no `cuda_copy`
staging stage).
