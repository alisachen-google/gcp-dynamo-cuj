# GPUDirect RDMA Regression on <cluster> — Root Cause & Escalation

**One line**: the rebuilt GB300 node image (kernel `6.12.85+`, driver
`580.126.20`) ships **without the `nvidia_peermem` kernel module**, disabling
GPUDirect RDMA fleet-wide; all disaggregated KV transfer falls back to
host-staged RDMA at ~50–100× per-transfer cost.

## Root cause (established 2026-09-11)

1. **UCX diagnoses it explicitly** when registering CUDA memory for RDMA
   (any pod, any pool):
   `gdaki.c:661 UCX DIAG GDAKI not supported, please load Nvidia peermem
   driver by running "modprobe nvidia_peermem"`
2. **The module cannot be loaded — it is absent from the image**
   (privileged probe on `…np-1…56m1`):
   `modprobe: FATAL: Module nvidia_peermem not found in directory
   /lib/modules/6.12.85+` — and no `*peermem*` file exists under
   `/lib/modules` or `/home/kubernetes`.
3. **It worked before the rebuild**: the dsr1 sweeps (early Aug) ran KV over
   GPUDirect RDMA on this fleet — dedicated A/B report, rc_mlx5 CUDA
   zero-copy evidence, no bypass flag anywhere in those manifests. The
   mid-Aug node rebuild introduced the regression.

## Hypotheses eliminated by experiment (so the fix isn't redirected at us)

| hypothesis | test | result |
|---|---|---|
| Missing gib userspace | N3U smoke with full nccl-gib install + LD_LIBRARY_PATH | still fails (0/20, 112 UCX errs) |
| UCX TLS / env config | dsr1-verbatim env (tcp TLS, GPU_DIRECT unset) | still fails |
| pip-wheel UCX vintage | verbatim dsr1 stack (sglang 0.5.8 + dynamo 0.8.1) | UCX itself prints the peermem diagnosis |
| pool-specific fault | reproduced on np-1, np-2, np-3 | fleet-wide |

## Measured cost of the fallback (host-staged bypass, `UCX_IB_GPU_DIRECT_RDMA=n`)

- Per-request transfer floor ~2.4 s (0.8 GB at ~0.35 GB/s effective) that even
  fully-cached requests pay; prefill-tier effective throughput halved.
- Disagg bounded knees compressed ~9× (N3U: sim 144 → silicon 16).
- Kimi-K2.5 disagg (3.4 GB/request): hard ceiling ~1.6 req/s, **no
  SLO-servable operating point at any threshold** — topology verdicts for
  attention-heavy models are being decided by this fault.
- Full quantification: `D72_RESULTS.md` §drift; cross-model ceiling scaling
  (Kimi 1.6 vs N3U 3.3+ req/s, inverse to transfer volume) fingerprints the
  transfer path as the binding constraint.

## Minimal reproducer + definitive A/B (2026-09-11)

30-line raw NIXL transfer (`scripts/gdr-reproducer/`): register a 256 MB CUDA
buffer on pod B, READ it from pod A. Same pods, same buffers, one env var:

| `UCX_IB_GPU_DIRECT_RDMA` | result |
|---|---|
| `y` | `NIXL_ERR_REMOTE_DISCONNECT` mid-transfer; **target process verified alive after** (transport-level QP abort) |
| `n` | `XFER_DONE`, data verified, 0.34 GB/s (matches production drift-model inference of ~0.35 GB/s) |

UCX debug refines the mechanism: peermem probes all fail (module absent), but
**dmabuf is supported and registration SUCCEEDS** ("dmabuf is supported on
cuda device 0"; "mlx5_0: dmabuf is supported") — the failure is in the
**RDMA datapath to GPU memory** (NIC↔GPU peer DMA), which aborts the RC
connection. Prime suspects on the rebuilt image: IOMMU / PCIe ACS
configuration blocking peer DMA, or a kernel-6.12.85+/driver-580.126.20
dmabuf mapping fault.

## Ask

Fix the NIC↔GPU peer-DMA datapath on the rebuilt node image — most likely
IOMMU/PCIe-ACS settings (pre-rebuild image allowed peer DMA), or the
driver/firmware dmabuf mapping. Shipping `nvidia_peermem` is the alternative
registration path but will hit the same datapath fault if IOMMU/ACS is the
cause — the datapath is the thing to fix. The 30-line reproducer runs in any
2 GPU pods with mrdma claims in ~2 minutes.

## 20-minute verification once fixed

`nemotron-3-ultra-550b-nvfp4/scripts/transport_probe.sh` (config A:
GPU_DIRECT=y + no-tcp TLS): PASS = 20/20 completions, zero
REMOTE_DISCONNECT, proto tables showing single-stage `zero-copy … rc_mlx5 …
cuda` rows. We then re-benchmark disagg on the direct path (runners ready).
