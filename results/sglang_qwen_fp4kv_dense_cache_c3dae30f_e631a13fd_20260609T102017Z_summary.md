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

The dense-cache trace comparator is not yet a clean localization oracle for this run:

- `trace_count`: `733`
- matched tensor comparisons: `576`
- matched comparisons with `metric_ok=false`: `0`
- event schema issues: `20`
- comparator findings: `25 trace event(s) failed schema checks`; `301 trace event(s) could not be matched to request rids`

The schema failures are missing `forward_pass_id` and `rids` on prefill/no-prefix trace events. Update the trace schema or attach request IDs to those events before using this artifact to claim a layer-local tensor divergence.

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
