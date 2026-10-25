# gpu-recipes export: DeepSeek-V4-Pro and DeepSeek-R1 Pareto-sweep recipes (A4X-MAX GKE)

This directory is a verbatim copy of the recipe files contributed to
[AI-Hypercomputer/gpu-recipes](https://github.com/AI-Hypercomputer/gpu-recipes) from the sweeps in
`../dsv4-sweep/`, `../dsr1-sweep/` and `../dsr1-fp8-sweep/`, laid out with the same paths as in that
repository so the README's `$REPO_ROOT/...` references resolve:

| Path | Content |
|---|---|
| `inference/a4xmax/disaggregated-serving/dynamo/pareto-sweep/README.md` | Setup guide: architecture, per-model Pareto maps (low latency / mid-curve / max throughput), deploy and benchmark steps |
| `inference/a4xmax/disaggregated-serving/dynamo/pareto-sweep/*.yaml` | 14 Helm values files, one per operating-point topology (DSv4 p1 to p8, DSR1 NVFP4 x3, DSR1 FP8 x3) |
| `src/frameworks/a4xmax/dynamo-configs/ds{v4,r1}-*-{prefill,decode}.yaml` | 28 SGLang serving configs, flag-for-flag from the srt-slurm recipes of each point |
| `src/docker/dynamo-sglang-deepseek/` | Dockerfiles for the SGLang + Dynamo stacks the points were measured with |

The recipe deliberately omits performance figures; the measured results live in `../reports/`
(`pareto-curves.md` and the per-model benchmark reports).

Upstream pull request: see the `gpu-recipes` repository, branch `dsv4-dsr1-pareto-a4xmax-dynamo`
on `alisachen-google/gpu-recipes`.
