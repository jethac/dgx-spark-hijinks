# SGLang FP4-KV Deswizzle Flag Cache Check

Date: 2026-06-10 JST

Purpose: check Claude's suspected failure mode before further SGLang tracing: whether a
vLLM-style `FLASHINFER_PAGED_V_SF_DESWIZZLE` build flag leaked into SGLang's generated
FlashInfer paged-prefill modules. That would make FlashInfer de-swizzle SGLang's linear
V scale-factor layout and could corrupt the cached-prefix prefill path while leaving decode
clean.

## Checks

- Host cache check on the Spark:
  - searched `/home/jethac/.cache/flashinfer`
  - searched `/tmp`
  - searched the live repo checkout at
    `/home/jethac/spark_tmp/dgx-spark-hijinks-live-4df2367`
- Grep patterns:
  - `FLASHINFER_PAGED_V_SF_DESWIZZLE`
  - `batch_prefill`
  - `vllm_batch_prefill`
  - `nvfp4`
  - `extra_cuda`
  - `swa_1`

## Finding

No surviving host-side SGLang generated paged-prefill module or JIT log showed
`FLASHINFER_PAGED_V_SF_DESWIZZLE`.

The surviving FlashInfer host cache only had:

- `/home/jethac/.cache/flashinfer/0.6.11.post2/121a/flashinfer_jit.log`
- cached ops for `sampling`
- cached ops for `fp4_quantization_120f`

That JIT log did not contain the deswizzle macro, NVFP4 paged-prefill module names, or
extra CUDA flag lines.

## Caveat

This is an inconclusive negative, not proof that the failing SGLang container never compiled
with the macro. The model-serving containers from the failing rows were removed, and their
container-local `/root/.cache/flashinfer/...` generated modules are not available from the
host cache. Old repo artifacts and vLLM logs do show deliberate vLLM/probe use of
`-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`, but those are expected and do not prove leakage into
the SGLang run.

## Next Check

Run the standalone FlashInfer matrix:

- page size: `1`, `16`
- deswizzle macro: off, on
- operation: paged prefill, paged decode
- V scale layout: linear, matching SGLang

The matrix is implemented in:

- `scripts/run_flashinfer_nvfp4_page_deswizzle_matrix.sh`
