# SGLang DiffusionGemma DG-R5 Full-NVFP4 Packet

Date: 2026-06-12 JST

Scope: Spark live-serving packet for the DG-R5 gate in
`docs/SGLANG_DIFFUSIONGEMMA_RUNTIME_LADDER_EPOCH2.md`.

This is a full NVFP4 K+V smoke/quality/routing row through SGLang's
`fp4_e2m1` KV pool with `SGLANG_FP4_KV_MIXED_KV=0`. It is not a capacity row;
capacity requires a separate matched denominator run after quality is green.

## Purpose

DG-R3 proved DiffusionGemma 26B-A4B text-only serving through the experimental
FlashInfer VO-split route for D=512 global layers. DG-R4 mixed-KV is now
deferred by the split-dtype de-scope decision in `mail/0077`. DG-R5 therefore
tests the headline path directly: store committed prefix KV as full NVFP4 K+V
while preserving the same revised DG-R2 text-only quality gate and D=512
VO-split routing proof.

## Preconditions

- Spark marker absent: `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` must not
  exist.
- `docker ps` empty.
- GB10 memory guardrails: one server, `--memory=100g --memory-swap=100g`, no
  concurrent comparator.
- Repo lives on B-backed Spark storage:
  `/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live`.
- Model weights are already cached locally; launch uses
  `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`.

## Command

From the Spark host:

```bash
cd /home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live
REPO_ROOT=/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live \
SOURCE_BRANCH=epoch2 \
IMAGE=sglang-source-stack-dgemma-024-0705924c-f99323bd:latest \
SGLANG_COMMIT=98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968 \
FLASHINFER_COMMIT=f99323bd7d1c \
MEM_FRACTION_STATIC=0.55 \
PORT=30125 \
bash scripts/run_sglang_dgemma_dgr5_fullnvfp4_smoke.sh
```

The script records artifacts under:

```text
results/sglang_dgemma_dgr5_fullnvfp4_smoke_<timestamp>/
```

## Launch Shape

The server launch is intentionally narrow:

```bash
python3 -m sglang.launch_server \
  --model-path google/diffusiongemma-26B-A4B-it \
  --dllm-algorithm Gemma4Renoise \
  --dllm-algorithm-config results/<run_id>/dllm_config.yaml \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --kv-cache-dtype fp4_e2m1 \
  --context-length 8192 \
  --page-size 256 \
  --mem-fraction-static 0.55 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

Environment:

- `SGLANG_FLASHINFER_VOSPLIT=1`
- `SGLANG_FP4_KV_MIXED_KV=0`
- `SGLANG_FP4_KV_TRACE_MODULE=1`
- `SGLANG_GEMMA4_TRACE_GEOMETRY=1`
- `SGLANG_GEMMA_KV_GEOMETRY=1`
- `FLASHINFER_PREFILL_DEBUG_ONCE=1`
- `SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src`
- `PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path`

The packet writes the same deterministic `Gemma4Renoise` config used by the
DG-R2 revised green row and DG-R3:

```yaml
max_denoising_steps: 48
seed: 1234
sampler_config:
  entropy_bound: 0.1
temperature_schedule:
  t_min: 0.4
  t_max: 0.8
stopping_config:
  confidence_threshold: 0.005
  stability_threshold: 1
```

## Green Gate

The row is green only if all of these pass:

- Server reaches readiness.
- Revised DG-R2 text-only quality client passes:
  `scripts/diffusion_gemma_dgr2_revised_text_quality_client.py`.
- Server log contains the explicit policy warning:
  `DiffusionGemma is using the experimental FlashInfer VO-split path`.
- Server log does not contain the mixed-KV warning:
  `SGLang FP4 KV mixed mode enabled`.
- Server args prove `kv_cache_dtype='fp4_e2m1'`.
- Pool configurator reports `mixed_kv=False`.
- Hybrid SWA pool proof shows both subpools are `MHATokenToKVPoolFP4`.
- Server log contains NVFP4 calibration or FlashInfer FP4 module/backend trace
  evidence.
- Geometry trace contains at least one D=512 FlashInfer attention line with a
  VO-split pass label, for example `extend_paged_vosplit0` or
  `extend_paged_vosplit1`.
- The same D=512 route exposes `head_dim_vo=256`.

If text quality passes but any KV proof is absent, the row is RED. That case
would only prove the BF16/no-KV-quant DG-R3 row still answers prompts.

## Stop-On-Red Order

1. Readiness failure: preserve `server.log`, `preflight.log`, `checkout.log`,
   and `docker_ps_after.txt`. Do not retry with higher memory until the log
   proves memory pressure rather than routing/import/KV-pool failure.
2. Full-NVFP4 proof missing: check `SGLANG_FP4_KV_MIXED_KV=0`,
   `--kv-cache-dtype fp4_e2m1`, and `SGLANG_GEMMA_KV_GEOMETRY=1` before
   changing model flags.
3. D=512/VO-split trace missing: do not call DG-R5 green. Inspect wrapper
   construction before trying capacity claims.
4. Quality gate failure: preserve `revised_text_quality.json`; compare against
   DG-R3 before changing prompts.

## Expected Artifact Set

- `preflight.log`
- `checkout.log`
- `server.log`
- `revised_text_quality.json`
- `quality_client.stdout`
- `quality_client.stderr`
- `quality_status.txt`
- `summary.md`
- `docker_ps_after.txt`
- `free_after.txt`

## Interpretation

GREEN means SGLang DiffusionGemma can serve the revised text-only row with full
NVFP4 K+V KV storage and the experimental FlashInfer D=512 VO-split route
enabled. It does not validate image prompts, CUDA graphs, long-context quality,
or a matched capacity denominator.
