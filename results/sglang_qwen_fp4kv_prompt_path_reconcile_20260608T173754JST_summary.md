# SGLang Qwen FP4-KV Prompt Path Reconciliation

Date: 2026-06-08 JST

Run id: `sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST`

Purpose: determine whether the SGLang Qwen FP4-KV quality failure is caused by OpenAI
Chat Completions prompt serialization, or by the native FP4 KV serving path after prompt
tokens are fixed.

## Setup

- image: `nvcr.io/nvidia/sglang:26.05-py3`
- source overlay: `/home/jethac/spark_tmp/sglang_matched_d7d931f_20260608T1545JST/sglang`
- source commit: `jethac/sglang@d7d931f530160ba86a2d55b4636d64baaeda3bec`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- fp8 comparator: port `30012`, `--kv-cache-dtype fp8_e4m3`
- FP4 candidate: port `30013`, `--kv-cache-dtype fp4_e2m1`
- shared flags: `--tp 1 --dtype bfloat16 --attention-backend flashinfer --page-size 1 --mem-fraction-static 0.40 --disable-cuda-graph --disable-piecewise-cuda-graph`
- FP4 env: `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`, `SGLANG_FP4_KV_TRACE_BACKEND=1`, `SGLANG_FP4_KV_AUTOCALIB=1`, `SGLANG_FP4_KV_ENABLE_CUDA_GRAPH=0`

Both containers were stopped after artifact capture.

## Artifacts

- client JSON: `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST.json`
- fp8 server log: `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_fp8_server.log`
- FP4 server log: `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_fp4_server.log`
- fp8 trace excerpt: `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_fp8_trace_excerpt.txt`
- FP4 trace excerpt: `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_fp4_trace_excerpt.txt`
- container inspect files:
  - `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_fp8_container_inspect.json`
  - `results/sglang_qwen_fp4kv_prompt_path_reconcile_20260608T173754JST_fp4_container_inspect.json`

## Result

Prompt serialization is **not** the cause.

Both endpoints return the same OpenAI prompt token IDs as the local Qwen chat-template
render:

| row | prompt token count | prompt SHA-256 | first prompt diff |
|---|---:|---|---|
| fp8 | `56` | `5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517` | `None` |
| FP4 | `56` | `5a5d4572e0e3d940a909b85dc4a00350094cbd1d55333c3d4f0a7974a91ee517` | `None` |

Native replay from either the exact OpenAI prompt IDs or the locally rendered prompt text
diverges immediately:

| path | fp8 first token | FP4 first token | divergence index |
|---|---|---|---:|
| native `/generate` from OpenAI prompt IDs | `**` (`334`, logprob `-0.7641`) | `ark` (`838`, logprob `-0.5875`) | `0` |
| native `/generate` from local rendered text | `**` (`334`, logprob `-0.7641`) | `ark` (`838`, logprob `-0.5875`) | `0` |

OpenAI Chat Completions behaves differently from native `/generate` for the FP4 row: the
FP4 OpenAI path still starts with `**`, `Engineering`, ` Note`, `:`, then chooses
` Validate` at token index 4 where fp8 chooses ` Valid`. That matches the earlier
token-index-4 rank-reversal class. The native `/generate` path is worse in this run and is
wrong from token 0 despite identical prompt IDs.

## Backend Trace

The FP4 server log confirms the intended native FP4 KV path was active:

- `KV Cache is allocated. dtype: torch.float4_e2m1fn_x2, #tokens: 3202080`
- `NVFP4 KV cache calibrated 28 layers from 4096 eager prefill tokens`
- backend traces include all-layer `decode` calls and all-layer `extend_merge_paged` calls
- K/V cache tensors are packed `torch.uint8`, shape `(3202081, 2, 64)`, stride `(128, 64, 1)`
- K/V scale tensors are `torch.float8_e4m3fn`, shape `(3202081, 1, 2, 8)`, stride `(16, 16, 8, 1)`

## Interpretation

This retires the OpenAI prompt-serialization hypothesis. The FP4 degradation is not caused
by a different chat template or prompt tokenization.

The remaining issue is endpoint/path-specific FP4 KV serving numerics or metadata:

- OpenAI Chat Completions with FP4 KV starts plausibly and diverges at token index 4.
- Native `/generate` with the same prompt IDs diverges at token index 0.
- Both use the same calibrated FP4 KV pool and FlashInfer backend trace surface.

The next useful SGLang work is to compare native `/generate` versus OpenAI Chat
Completions request metadata and prefill/decode state for FP4, then trace logits/hidden
state before sampling for the first generated token. Do not start Gemma or run more
capacity rows until this endpoint/path split is understood.
