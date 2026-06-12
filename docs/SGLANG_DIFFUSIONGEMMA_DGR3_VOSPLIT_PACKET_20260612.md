# SGLang DiffusionGemma DG-R3 VO-Split Packet

Date: 2026-06-12 JST

Scope: Spark live-serving packet for the DG-R3 gate in
`docs/SGLANG_DIFFUSIONGEMMA_RUNTIME_LADDER_EPOCH2.md`.

This is a BF16/no-KV-quant routing and text-coherence row. It is not an NVFP4
KV row and not a capacity row.

## Purpose

DG-R1/DG-R2 proved the upstream `Gemma4Renoise` DiffusionGemma path works under
the stock correctness-first policy: Triton attention, eager/no CUDA graphs, and
unchunked prefill. DG-R3 tests only one added complication: opt the same model
into SGLang's experimental FlashInfer D=512 VO-split route while preserving the
revised DG-R2 text-only quality gate.

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
SGLANG_COMMIT=dec4c040a8ede4561c1f26cccc599286643b49fd \
FLASHINFER_COMMIT=f99323bd7d1c \
MEM_FRACTION_STATIC=0.55 \
PORT=30125 \
bash scripts/run_sglang_dgemma_dgr3_vosplit_smoke.sh
```

The script records artifacts under:

```text
results/sglang_dgemma_dgr3_vosplit_smoke_<timestamp>/
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
  --context-length 8192 \
  --page-size 256 \
  --mem-fraction-static 0.55 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

Environment:

- `SGLANG_FLASHINFER_VOSPLIT=1`
- `SGLANG_GEMMA4_TRACE_GEOMETRY=1`
- `FLASHINFER_PREFILL_DEBUG_ONCE=1`
- `SPARK_FLASHINFER_SOURCE_ROOT=/flashinfer-src`
- `PYTHONPATH=/work/python_sitecustomize:/work/third_party/sglang/python:/tmp/flashinfer-python-path`

The packet writes the same deterministic `Gemma4Renoise` config used by the
DG-R2 revised green row:

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
- Geometry trace contains at least one D=512 FlashInfer attention line with a
  VO-split pass label, for example `extend_paged_vosplit0` or
  `extend_paged_vosplit1`.
- The same D=512 route exposes `head_dim_vo=256`.

If text quality passes but routing proof is absent, the row is RED. That case
would only prove the old stock path still answers prompts; it would not prove
DG-R3.

## Stop-On-Red Order

1. Readiness failure: preserve `server.log`, `preflight.log`, `checkout.log`,
   and `docker_ps_after.txt`. Do not retry with higher memory until the log
   proves memory pressure rather than routing/import failure.
2. Policy warning missing: treat as source-overlay/provenance failure. Check the
   SGLang checkout proof lines before changing launch flags.
3. D=512/VO-split trace missing: do not call DG-R3 green. Inspect the wrapper
   construction path before trying NVFP4 or capacity work.
4. Quality gate failure: preserve `revised_text_quality.json`; compare against
   the DG-R2 revised stock row before changing prompts.

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

GREEN means SGLang DiffusionGemma can serve the revised text-only BF16 row with
the experimental FlashInfer D=512 VO-split route enabled. It unlocks DG-R4
mixed-KV planning but does not itself validate mixed KV, full NVFP4 KV, CUDA
graphs, image prompts, or capacity.
