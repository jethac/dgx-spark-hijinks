# vLLM Gemma 3 FlashInfer Prefill Debug - 2026-06-09T143948JST

Status: completed red.

## Run
- Host: GB10 over Tailnet (`thinkstationpgx-00b4`, tailnet SSH reachable during this run).
- Campaign commit: `0713537` in the fresh live checkout.
- vLLM submodule: `jethac/vllm@1fabc6649`.
- FlashInfer submodule: `jethac/flashinfer@96be2fa8`.
- Image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`.
- Runner: `scripts/run_vllm_gemma3_contigout_probe_container.sh`.
- Key env:
  - `FLASHINFER_PREFILL_DEBUG_ONCE=1`
  - `FLASHINFER_CLEAR_PREFILL_CACHE=1`
  - `FLASHINFER_EXTRA_CUDAFLAGS="-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1 -gencode=arch=compute_121a,code=sm_121a"`
  - `TORCH_CUDA_ARCH_LIST=12.1a`

## Artifacts
- Server log: `results/vllm_gemma3_flashinfer_prefill_debug_20260609T143948JST_nvfp4_kv_flashinfer_eager_server.log`
- Import probe: `results/vllm_gemma3_flashinfer_prefill_debug_20260609T143948JST_nvfp4_kv_flashinfer_eager_import_probe.txt`
- First-token probe: `results/vllm_gemma3_flashinfer_prefill_debug_20260609T143948JST_nvfp4_kv_flashinfer_eager_first_token.json`
- FlashInfer prefill audit: `results/vllm_gemma3_flashinfer_prefill_debug_20260609T143948JST_nvfp4_kv_flashinfer_eager_flashinfer_prefill_debug_audit.json`

The full active-page tensor dump remains on the live host under the fresh checkout results
directory and was not added to git.

## Result
The Gemma 3 NVFP4-KV first-token corruption reproduced:

- `exact_spark_ok` first token: ` Reigns`
- `simple_math` first token: Gujarati text
- `short_decode` first token: `ioane`

The corrected audit parsed `4` FlashInfer paged-prefill identity lines and `4` matching
tensor lines. It found both SWA (`window_left=1023`) and global (`window_left=-1`) paged
prefill calls, but every generated module was compiled as raw byte KV rather than FP4 KV:

- `dtype_kv=uint8_t`
- `require_fp4_kv=0`
- `is_kv_fp4x2=0`
- `additional_tensors=` empty
- no `maybe_k_cache_sf` / `maybe_v_cache_sf` tensors

The live C++ tensor views still show packed byte carriers:

- `paged_k_cache` shape `[129777,16,16,64]`, dtype bits `8`
- `paged_v_cache` shape `[129777,16,16,64]`, dtype bits `8`

## Interpretation
This is not yet evidence of a low-level FA2 prefill math bug. The live vLLM -> FlashInfer
generated paged-prefill binding is failing to specialize the module as FP4 KV and failing
to pass the FP8 scale tensors. The next vLLM fix should trace and repair the Python/JIT
binding path that decides `require_fp4_kv`, `dtype_kv`, `additional_tensors`, and
`additional_tensor_dtypes` for Gemma 3 paged prefill.

The previous byte-like output observations remain real, but this run narrows the active
stop point: the generated prefill module is reading packed KV bytes without FP4 scale
metadata.
