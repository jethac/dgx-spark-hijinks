# jethac/vLLM Qwen3.6 DFlash Fork Row

Date: 2026-06-08 JST

Goal: run the AEON Qwen3.6 NVFP4+DFlash row through the `jethac/vllm` fork branch after the first derived image stopped on dependency drift.

## Target

- base image: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2`
- derived image: `jethac-vllm-aeon-q36:6804e1b81-ct017-humming-aeonfa2`
- fork: `jethac/vllm@6804e1b81e6ea2ca53bb5021151bdad0f201b11d3`
- target model: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4`
- drafter: `z-lab/Qwen3.6-35B-A3B-DFlash`
- run id: `jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST`
- Qwen chat setting: `chat_template_kwargs={"enable_thinking": false}`

## Image Progression

The first derived image, `jethac-vllm-aeon-q36:6804e1b81`, built and imported the forked vLLM package but exited before health on:

```text
ModuleNotFoundError: No module named 'compressed_tensors.compressors.pack_quantized'
```

The passing image required three narrow layers over that build:

1. `compressed-tensors==0.17.0 --no-deps`, matching the fork's `requirements/common.txt`.
2. `humming-kernels[cu13]==0.1.4 --no-deps` plus `pyelftools`, resolving the next import-time dependency.
3. AEON's original `_vllm_fa2_C.abi3.so`, copied back over the fork install after the precompiled vLLM FA2 extension failed with a PyTorch ABI mismatch.

Build logs:

- `results/jethac_vllm_aeon_q36_6804e1b81_ct017_image_build_20260608T0859JST.log`
- `results/jethac_vllm_aeon_q36_6804e1b81_ct017_humming_image_build_20260608T0902JST.log`
- `results/jethac_vllm_aeon_q36_6804e1b81_ct017_humming_aeonfa2_image_build_20260608T0908JST.log`

## Backend Evidence

Server log: `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_server.log`.

Key lines:

- vLLM version: `0.1.dev1+g6804e1b81`
- resolved target: `Qwen3_5MoeForConditionalGeneration`
- resolved drafter: `DFlashDraftModel`
- selected `FlashInferCutlassNvFp4LinearKernel` for NVFP4 GEMM
- selected `'MARLIN' NvFp4 MoE backend`
- selected FlashAttention 2
- captured CUDA graphs
- `GPU KV cache size: 1,251,446 tokens`
- `Maximum concurrency for 262,144 tokens per request: 4.77x`

The server also warned that this path does not treat the GPU as having native FP4 compute support and therefore uses weight-only FP4 with Marlin. That warning keeps this row out of the native `sm_121a` proof bucket.

## Benchmark

Artifacts:

- row manifest: `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_row_manifest.json`, `ok=true`
- chat smoke: `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_chat_smoke.json`, returned `spark-ok`
- compact benchmark: `results/jethac_qwen36_dflash_aeonfa2_nothink_20260608T0908JST_openai_benchmark.json`

| case | prompt tokens | generated tokens | TTFT seconds | decode tok/s |
|---|---:|---:|---:|---:|
| `short_decode` | 27 | 64 | 0.124 | 47.22 |
| `medium_decode` | 39 | 192 | 0.093 | 58.88 |
| `long_prefill` | 2271 | 64 | 0.448 | 61.62 |

## Interpretation

This is a passing `jethac/vllm` Qwen3.6 NVFP4+DFlash fork row, and it clears the earlier `compressed_tensors` stop point.

It is not a clean upstreamable install story yet. The row still depends on AEON's image, AEON's FA2 binary, and package-version surgery inside the container. It also does not prove native `sm_121` or `sm_121a` build targets; the host-side `.so` audit could not import container-local `vllm`/`flashinfer`, and the build-target audit found no accepted target evidence in the server log. The next proof is an in-container binary/JIT audit plus a clean wheel/container build path.
