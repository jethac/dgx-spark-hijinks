# jethac/vLLM Qwen3.6 Clean FA2 SM121a Row

Date: 2026-06-08 JST

## Target

- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-patchedfa2-cutlass`
- runtime ref: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2 + jethac/vllm@a919d635d + jethac/flash-attention@7d53245 + clean FA2 sm_121a`
- target model: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`
- drafter: `z-lab/Qwen3.6-35B-A3B-DFlash`
- served model: `qwen36-fast`
- Qwen chat setting: `chat_template_kwargs={"enable_thinking": false}`
- run id: `jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST`

## Result

The clean FA2 image serves Qwen3.6 NVFP4+DFlash successfully.

- row manifest: `ok=true`
- smoke: `spark-ok`
- reasoning fields: empty; normal OpenAI `message.content` is used
- CUDA graph mode: enabled
- DFlash speculative decode: enabled

## Benchmark

| case | prompt tokens | generated tokens | TTFT seconds | decode tok/s |
|---|---:|---:|---:|---:|
| `short_decode` | 27 | 64 | 0.094 | 61.07 |
| `medium_decode` | 39 | 192 | 0.120 | 56.97 |
| `long_prefill` | 2271 | 64 | 0.392 | 60.10 |

## Backend Evidence

Server log: `results/jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST_server.log`.

Key lines:

- vLLM version: `0.1.dev1+ga919d635d`
- resolved target: `Qwen3_5MoeForConditionalGeneration`
- resolved drafter: `DFlashDraftModel`
- selected `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM
- selected `'MARLIN' NvFp4 MoE backend`
- selected FlashAttention 2
- FlashInfer FP4 GEMM autotuner ran during startup
- captured CUDA graphs
- `GPU KV cache size: 1,241,920 tokens`
- `Maximum concurrency for 262,144 tokens per request: 4.74x`
- `Not enough SMs to use max_autotune_gemm mode`

Startup timings:

- target+drafters model loading: `22.84 GiB` and `156.82 s`
- backbone `torch.compile`: `97.39 s`
- EAGLE head `torch.compile`: `8.55 s`
- initial profiling/warmup: `108.79 s`
- CUDA graph capture: `13 s`, `1.62 GiB`
- engine init total: `320.27 s`, including `105.94 s` compilation

## Native Target Evidence

This serving row uses the clean image whose separate in-container audit proves the patched vLLM FlashAttention extension contains `sm_121a` cubins:

- `results/jethac_vllm_qwen_cleanfa2_patchedfa2_cutlass_audit_20260608T2355JST_incontainer_target_audit.md`

The serving log itself still does not include accepted CUDA build-target strings, so `results/jethac_qwen36_dflash_cleanfa2_sm121a_nothink_record2_20260608T2359JST_build_target_audit.json` reports no accepted log-level target evidence. Treat the native-target proof as image/binary evidence plus FA2 backend selection, not as server-log build-target evidence.

The server still warns that the selected weight path does not treat the GPU as native FP4 compute and therefore uses weight-only FP4 through Marlin. This row does not prove native FP4 weight GEMM/MoE on GB10.

## Failed First Recorder Pass

The first run id, `jethac_qwen36_dflash_cleanfa2_sm121a_nothink_20260608T2359JST`, started and served successfully, but the recorder failed because the PowerShell-to-SSH command mangled the JSON argument:

```text
{\ enable_thinking\: false}
```

That produced a `JSONDecodeError` in the smoke and benchmark commands. The failure is an operator quoting issue, not a model/server/runtime failure.

## Interpretation

This closes the clean vLLM packaging gap for the Qwen3.6 NVFP4+DFlash row: the workload now runs on a `jethac/vllm` image that skips AEON's FA2 binary and uses a patched, ABI-matched FA2 extension with `sm_121a` cubins.

Do not count this as a speedup claim. Compared with the earlier AEON-FA2-derived fork row (`47.22`, `58.88`, `61.62 tok/s`), this row is mixed: short decode is faster, medium and long-prefill decode are similar to slightly lower. The useful claim is clean packaging plus native FA2 binary proof with serving parity.
