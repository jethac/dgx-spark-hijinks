# vLLM Gemma 3 27B Active-Page Dump, 2026-06-09

## Purpose

Capture the exact active FlashInfer paged-prefill pages for the Gemma 3 27B
NVFP4-KV first-token failure after the wrapper-boundary trace localized corruption to
`BatchPrefillWithPagedKVCacheWrapper.run(...)`.

This is a diagnostic-only eager row, not a benchmark.

## Stack

- Model: `google/gemma-3-27b-it`
- Served name: `gemma3-27b-it`
- vLLM fork: `jethac/vllm@13da71884640567682cd3ddd4650d2ba3ecb5543`
  (`spark/hijinks-021-gemma3-tensor-trace`)
- Base image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- Source overlay: `/vllm-src` plus local FlashInfer source overlay
- CUDA target env: `TORCH_CUDA_ARCH_LIST=12.1a`
- FlashInfer env: `FLASHINFER_EXTRA_CUDAFLAGS=-DFLASHINFER_PAGED_V_SF_DESWIZZLE=1`
- vLLM flags: `--attention-backend flashinfer --kv-cache-dtype nvfp4 --dtype bfloat16
  --max-model-len 131072 --gpu-memory-utilization 0.85
  --max-num-batched-tokens 4096 --enforce-eager`

Import probe:

- device: `NVIDIA GB10`, capability `[12, 1]`
- torch: `2.12.0.dev20260408+cu130`, CUDA `13.0`
- FlashInfer: `0.6.9rc1`
- vLLM: `0.1.dev1+g13da71884`

Server log proof:

- vLLM selected `kv_cache_dtype=nvfp4`.
- Running model geometry logged `head_dim=128`, `num_heads=32`, `num_kv_heads=16`.
- Layer 5 is full/global (`sliding_window=None`).
- KV cache size: `1,717,939` tokens; max concurrency at 131,072 tokens: `13.11x`.
- vLLM logged: `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM
  V-scale-factor deswizzle enabled.`

## First-Token Result

The diagnostic reproduces the bad NVFP4-KV first-token signature:

| case | prompt tokens | first token |
|---|---:|---|
| `exact_spark_ok` | 18 | ` Reigns` |
| `simple_math` | 23 | Gujarati text |
| `short_decode` | 24 | `ioane` |

## Active-Page Finding

Three active-page dumps were emitted for layer 5. The first is a warmup/padded zero row.
The two request rows are the useful evidence:

| dump | prefill tokens | active pages | last page len | `out_after` byte-like | `out_after` mean | `out_after` max |
|---|---:|---|---:|---|---:|---:|
| `0002.pt` | 18 | `[13, 14]` | 2 | yes | 128.5285 | 255.0 |
| `0003.pt` | 23 | `[21, 22]` | 7 | yes | 129.3899 | 255.0 |

For both request rows, the first 16 `out_after` BF16 values exactly match the first 16
active V data bytes from `kv_data_pages[1]`:

```text
out_after head:       [240.0, 1.0, 226.0, 137.0, 145.0, 20.0, 186.0, 185.0, 33.0, 65.0, 47.0, 233.0, 91.0, 34.0, 145.0, 25.0]
active V data head:   [240,   1,   226,   137,   145,   20,   186,   185,   33,   65,   47,   233,   91,   34,   145,   25]
```

The matching V scale pages are nonzero FP8 values, so this is not a zero-scale or
missing-scale row. Example `0003.pt` active V scale head:

```text
[0.8125, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.375,
 0.34375, 0.28125, 0.28125, 0.75, 0.15625, 0.4375, 0.4375, 1.0]
```

## Conclusion

Gemma 3 27B NVFP4-KV remains red, but the failure is now narrowed again:

- vLLM selects the intended FlashInfer FA2 NVFP4-KV route on SM12x.
- The failing prompts are short enough that layer-5 full attention does not involve SWA
  eviction or sliding-window rotation.
- Prior traces showed sampled write/read packed K/V data and FP8 scale bytes match.
- The real FlashInfer paged prefill wrapper returns byte-like BF16 output whose head is
  identical to the active packed V payload bytes.

The next test should replay the dumped active paged-prefill call against a dequantized
reference and then inspect the FlashInfer paged-prefill specialization for a wrong V view
or packed-carrier interpretation bug.

## Artifacts

- `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_nvfp4_kv_flashinfer_eager_first_token.json`
- `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_nvfp4_kv_flashinfer_eager_tensor_trace.jsonl`
- `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_nvfp4_kv_flashinfer_eager_kv_trace.jsonl`
- `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_nvfp4_kv_flashinfer_eager_server.log`
- `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_nvfp4_kv_flashinfer_eager_editable_install.log`
- `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_nvfp4_kv_flashinfer_eager_import_probe.txt`
- `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_nvfp4_kv_flashinfer_eager_active_page_dump/`
- `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_summary.json`
- `results/vllm_gemma3_27b_active_page_dump_20260609T0216JST_dump_summary.md`
