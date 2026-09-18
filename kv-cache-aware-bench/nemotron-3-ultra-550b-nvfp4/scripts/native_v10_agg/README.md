# Frozen Native DynoSim V10 agg replay

These launch adapters are preserved from the completed `agg-kv-c192-calibrated-v10`
run. The controller adds `--router-args-file`, a JSON array containing the exact
router arguments from each hardware recipe, to replay the tuned KV settings.
The model, scheduler, timing coefficients and clock are unchanged.

The server must use Dynamo 1.4.2 with the local V10 patches and native core SHA-256
`5e298de426f7b27f1cc3d0647ac8eefec49d0abd989a01ac93e673e5bedd0d02`,
with AIConfigurator 0.11.0. The engine configuration's canonical SHA-256
(`json.dumps(config, sort_keys=True)`, UTF-8) is
`2f03549ffb13a3ad6bea8aa1717c32fcd03ce3ede0ea08e013f83d0ec5bfb276`.
Its preserved file bytes have SHA-256
`8a9fe3be2d6b3d3f344ab8a7ab68cd5057e880fb4626fac40e00c3d5db154858`.
The report's `native-v10/v10_patch_reconstruction.json` identifies the patches.

`run_path1_agentx.py --help` lists local endpoint, tokenizer, cache and interpreter
arguments. Each run reads the saved hardware AIPerf configuration and benchmark
ID, uses six TP4/EP4 worker models and a 3,600-second profiling window, and writes
commands, environment, build identity, metrics, raw records and completion status
to a new directory. Local NATS and etcd must already be available. Use unique
namespaces and ports for concurrent runs.

The controller owns only the processes it starts. It does not deploy to hardware.
Its historical `all_targets_pass` field checks both throughput and TTFT; the
original V10 calibration matrix accepted throughput only. Replay validity,
prediction error and the report's chosen latency SLO are distinct checks.

After the eight missing points complete, `../collect_agg_v10_holdouts.py` verifies
the frozen build/engine, hardware configuration, router arguments and warmup
identities, then preserves the compact report inputs. It retains errors against
hardware; it does not refit the model.
