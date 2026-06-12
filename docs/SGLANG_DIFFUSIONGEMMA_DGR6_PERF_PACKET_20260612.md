# SGLang DiffusionGemma DG-R6 Performance Packet

Date: 2026-06-12 JST

Scope: Spark live-serving performance row for the DiffusionGemma ladder. This
is a before/after stack comparison:

- before: stock DiffusionGemma SGLang policy, Triton attention, BF16/auto KV
- after: explicit GB10 path, FlashInfer VO-split, full NVFP4 K+V

This packet does not claim image quality, CUDA graph safety, long-context
quality, or isolated kernel speed attribution.

## Preconditions

- `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` absent.
- `docker ps` empty.
- One server at a time under `--memory=100g --memory-swap=100g`.
- Model weights already cached under the Spark HF cache; launch uses
  `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`.
- Repo on B-backed Spark storage:
  `/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live`.

## Command

```bash
cd /home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live
REPO_ROOT=/home/jethac/spark_tmp/dgx-spark-hijinks-sglang-live \
SOURCE_BRANCH=epoch2 \
IMAGE=sglang-source-stack-dgemma-024-0705924c-f99323bd:latest \
SGLANG_COMMIT=98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968 \
FLASHINFER_COMMIT=f99323bd7d1c \
MEM_FRACTION_STATIC=0.55 \
PORT=30125 \
bash scripts/run_sglang_dgemma_dgr6_perf_pair.sh
```

Artifacts land under:

```text
results/sglang_dgemma_dgr6_perf_pair_<timestamp>/
```

## Measurement Shape

Both rows use:

- `--dllm-algorithm Gemma4Renoise`
- the deterministic DG-R2/R3/R5 `dllm_config.yaml`
- `--dtype bfloat16`
- `--context-length 8192`
- `--mem-fraction-static 0.55`
- `--disable-cuda-graph`
- `--disable-piecewise-cuda-graph`

The before row omits the experimental FlashInfer opt-in, so the stock
DiffusionGemma policy must force Triton attention.

The after row adds:

- `--attention-backend flashinfer`
- `--kv-cache-dtype fp4_e2m1`
- `--page-size 256`
- `SGLANG_FLASHINFER_VOSPLIT=1`
- `SGLANG_FP4_KV_MIXED_KV=0`

## Gates

Each row must pass:

- server readiness
- revised DG-R2 text-only quality gate
- OpenAI-compatible benchmark cases:
  - `short_decode`
  - `medium_decode`
  - `long_prefill`
  - `natural_long_prefill`

The before row must prove the stock policy line:

```text
Attention backend forced to triton for DiffusionGemma
```

The after row must prove:

- explicit FlashInfer VO-split policy warning
- full NVFP4 K+V pool evidence (`mixed_kv=False`, FP4 K/V pools)

## Stop-On-Red Order

1. Readiness failure: preserve `server.log`, `preflight.log`,
   `checkout.log`, and Docker/free-memory artifacts. Do not raise memory
   fraction without evidence.
2. Quality failure: preserve `revised_text_quality.json`; compare against
   DG-R2/DG-R5 before changing prompts.
3. Benchmark failure: preserve `openai_benchmark.json`, stdout, and stderr.
   Treat API/schema failures separately from slow or incoherent serving.
4. Route-proof failure: do not quote the row. Fix evidence capture or launch
   flags first.

## Interpretation

GREEN means the stock and tuned SGLang DiffusionGemma text-only paths both
serve the revised quality gate and the standard OpenAI benchmark prompts, and
the artifact summary reports measured TTFT/total-token throughput for the same
prompt set.

The row is intentionally a combined stack comparison: stock Triton/BF16/auto-KV
versus FlashInfer VO-split/full-NVFP4. DG-R5 remains the separate quality and
capacity source for the full-NVFP4 path.
