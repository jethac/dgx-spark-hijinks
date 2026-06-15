# 26B-A4B Block Attribution Run Stopped

Date: 2026-06-15 UTC / 2026-06-16 JST

Status: stopped by operator request via `$codex-autoresearch stop`.

## Scope

- Model: `google/gemma-4-26B-A4B-it`
- Host: Vast instance `41100426`, RTX PRO 6000 Blackwell Max-Q Workstation Edition, sm120
- Image: `nvidia/cuda:13.0.1-devel-ubuntu22.04`
- vLLM: `0.1.dev1+g1c9686c61.sm120a`
- Torch: `2.12.0+cu130`
- CUDA: `13.0`
- FlashInfer source: `/root/flashinfer`
- Script: `docs/vast_anchor/run_26b_block_attribution.sh`

## What Ran

The block-attribution ladder started the first row:

- `bf16` baseline
- `ctx=8185`
- `prefix=4096`
- `max_model_len=8192`
- `max_num_batched_tokens=4096`
- `gpu_memory_utilization=0.82`
- `MAX_JOBS=4`

The run was interrupted during baseline startup after model weight load and encoder profiling had begun. It was still in cold FlashInfer MoE JIT/startup and did not produce a scored row.

## Artifacts

- `dx26_block_attr_20260615T2050Z/RUN_INFO.txt`
- `dx26_block_attr_20260615T2050Z/run.log`
- `dx26_block_attr_20260615T2050Z/rows/bf16.log`

No `*.json`, `summary.tsv`, or `delta_report.tsv` was produced. This is an operator-stopped infrastructure artifact, not a quality result.

## Cleanup

The remote run processes were terminated, the partial artifact bundle was pulled locally, and Vast instance `41100426` was destroyed. After cleanup, only the pre-existing unlabeled Vast instance `41040954` remained active.
