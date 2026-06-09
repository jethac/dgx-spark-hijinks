# SGLang Qwen FP4-KV Dense-vs-Cached Trace

Run ID: `sglang_qwen_fp4kv_dense_cache_c3dae30f_e631a13fd_20260609T102017Z`

Date: 2026-06-09 JST

## Stack

- Image: `sglang-source-stack-c3dae30f-e631a13fd`
- Base image: `nvcr.io/nvidia/sglang:26.05-py3`
- Device: `NVIDIA GB10`, capability `[12, 1]`
- Torch: `2.12.0a0+5aff3928d8.nv26.05`, CUDA `13.2`
- FlashInfer: `0.6.13` from editable `third_party/flashinfer` (`c3dae30f`)
- SGLang: `0.5.12.post2.dev1022+ge631a13fd` from editable `third_party/sglang` (`e631a13fd`)
- SGLang kernel: `0.4.3`, source-built in the container
- Loaded extension: `/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so`

## Result

Current-head source-stack SGLang still fails the Qwen FP4-KV cached-prefix row.

| Case | Cache behavior | First token | Logprob |
| --- | --- | --- | --- |
| OpenAI first | `cached_tokens=0` | `**` | `-0.7235294580459595` |
| Native second | `cached_tokens=55` | `ark` / `838` | `-0.5874708890914917` |
| Native first | `cached_tokens=0` | `**` / `334` | `-0.7235294580459595` |
| OpenAI second | `cached_tokens=55` | `ark` | `-0.5874708890914917` |
| Flush between | `cached_tokens=0` for both | `**` | `-0.7235294580459595` |
| Namespace isolation | `cached_tokens=0` for both | `**` | `-0.7235294580459595` |

Interpretation: the failure still follows FP4 cached-prefix reuse, independent of endpoint order. Flush and namespace isolation avoid reuse and remain clean.

## Trace Status

Post-hoc parser update: the dense-cache trace comparator is now a clean request-bound
localization artifact for this run. It ignores warmup and health-check forwards that do not
belong to the request-order probe, while keeping request-bound trace schema checks strict:

- `trace_count`: `733`
- `compare_trace_count`: `432`
- `ignored_trace_count`: `301`
- request-bound event counts: `324` dense, `108` cached, `0` unknown
- matched tensor comparisons: `576`
- matched comparisons with `metric_ok=false`: `0`
- event schema issues: `0`
- comparator findings: none

First localized request-bound divergence:

- kind: `qwen2`
- label: `decoder_after_self_attn`
- layer: `0`
- field: `hidden_rows`
- dense rid: `openai-first`
- cached rid: `native-second`
- cosine: `0.6753943297351783`
- max abs: `0.6484375`
- rms: `0.16307947834888886`

Interpretation: the artifact quality is green, but behavior is still red. The same current-head
stack still diverges on FP4 cached-prefix reuse after the FlashInfer paged-prefill fix that
closed the vLLM Gemma 3 short gate.

## SM12x Build Notes

The source build succeeded, but the build log emitted repeated SM12x performance warnings:

- `242` `.multicast::cluster` / `cp.async.bulk{.tensor}` advisories.
- `109` references each to `compute_120a` and `compute_121a`.
- `74` `setmaxnreg` compatibility warnings.

Affected kernels included FP8 blockwise, int8 GEMM, FP8 GEMM, W4A8 grouped MoE, and FP8 blockwise MoE paths. These warnings are not the FP4-KV cache correctness bug, but they are a separate SGLang SM12x performance-portability issue to track.

## Artifacts

- `results/sglang_qwen_fp4kv_dense_cache_c3dae30f_e631a13fd_20260609T102017Z_summary.json`
- `results/sglang_qwen_fp4kv_dense_cache_c3dae30f_e631a13fd_20260609T102017Z_default.json`
- `results/sglang_qwen_fp4kv_dense_cache_c3dae30f_e631a13fd_20260609T102017Z_default_dense_cache_compare.json`
- `results/sglang_qwen_fp4kv_dense_cache_c3dae30f_e631a13fd_20260609T102017Z_dense_cache_trace_summary_audit.json`
- `results/sglang_qwen_fp4kv_dense_cache_c3dae30f_e631a13fd_20260609T102017Z_source_stack_summary.json`
