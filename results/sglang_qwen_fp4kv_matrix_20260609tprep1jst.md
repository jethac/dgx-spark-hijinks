# SGLang Qwen FP4-KV Source-Stack Matrix, 2026-06-09

Purpose: rerun the four-row cached-prefix diagnostic matrix after clearing the
`sglang-kernel` ABI blocker with a source-built stack.

## Stack

- Base image: `nvcr.io/nvidia/sglang:26.05-py3`
- Prepared image: `sglang-source-stack-20260609tprep1jst`
- Device: `NVIDIA GB10`, compute capability `[12, 1]`
- Torch: `2.12.0a0+5aff3928d8.nv26.05`, CUDA `13.2`
- FlashInfer: editable `/work/third_party/flashinfer`, `flashinfer_python 0.6.13`
- FlashInfer commit in the checkout: `jethac/flashinfer@4c3c0d99`
- SGLang: editable `jethac/sglang@d96869237`, installed as
  `0.5.12.post2.dev1019+gd96869237`
- `sglang-kernel`: source-built `0.4.3`
- Loaded kernel extension:
  `/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so`

Source-stack artifacts:

- `results/sglang_source_stack_20260609tprep1jst_summary.json`
- `results/sglang_source_stack_20260609tprep1jst.log`

Matrix artifacts:

- `results/sglang_qwen_fp4kv_matrix_20260609tprep1jst_summary.json`
- `results/sglang_qwen_fp4kv_matrix_20260609tprep1jst_default.json`
- `results/sglang_qwen_fp4kv_matrix_20260609tprep1jst_full_paged.json`
- `results/sglang_qwen_fp4kv_matrix_20260609tprep1jst_force_miss.json`
- `results/sglang_qwen_fp4kv_matrix_20260609tprep1jst_force_miss_full_paged.json`

## Result

The source-stack image clears the runtime ABI blocker. The matrix rows all run to
completion and reproduce the functional localization:

| row | cached prefix? | first token result | interpretation |
|---|---:|---|---|
| default | yes, `55` tokens on second request | fresh rows `**`; cached rows `ark` | bad |
| full paged | yes, `55` tokens on second request | token stays newline, but cached logprob differs from fresh | cached reuse still changes distribution |
| force miss | no, `0` cached tokens | all rows `**`, identical logprob | clean recompute |
| force miss + full paged | no, `0` cached tokens | all rows newline, identical logprob | clean full-paged recompute |

The failure follows FP4 cached-prefix reuse. Forcing a radix miss makes the FP4
rows internally consistent; forcing full-paged attention without forcing a miss
does not make cached-prefix output equivalent to fresh output.

## Decision

SGLang FP4 KV remains red. The next fix/probe should compare full dense prefill
against FP4 cached-prefix attention/logits, with the reused-prefix path as the
distinguishing variable.

`SGLANG_RADIX_FORCE_MISS=1`, namespace isolation, and `--disable-radix-cache`
are correctness workarounds or diagnostic controls only. They are not acceptable
as the blessed FP4-KV capacity path because they avoid the prefix reuse behavior
that production serving needs.
