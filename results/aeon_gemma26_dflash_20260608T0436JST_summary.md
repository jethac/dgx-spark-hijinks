# AEON Gemma 4 26B NVFP4+DFlash vLLM Row

Run ID: `aeon_gemma26_dflash_20260608T0436JST`

Status: local GB10 serving proof passed.

## Stack

- image: `ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:v2`
- vLLM: `0.20.1`
- PyTorch: `2.11.0+cu130`
- CUDA: `13.0`
- FlashInfer: `0.6.8.post1`
- Transformers: `5.7.0`
- Triton: `3.6.0`
- model: `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4`
- drafter: `z-lab/gemma-4-26B-A4B-it-DFlash`
- served aliases: `gemma4-aeon-uncensored`, `gemma4-fast`, `gemma4-deep`
- hardware key: `NVIDIA_GB10:sm_121:sms_48`

## Backend Evidence

Key server-log evidence:

- Gemma 4 target model resolved as `Gemma4ForConditionalGeneration`.
- DFlash drafter resolved as `DFlashDraftModel`.
- Gemma 4 heterogeneous head dimensions force `TRITON_ATTN`.
- linear layers use `FlashInferCutlassNvFp4LinearKernel`.
- MoE uses `VLLM_CUTLASS` NvFp4 backend.
- drafter attention uses `FLASH_ATTN`.
- model loading took `17.17 GiB` memory and `127.259276` seconds.
- CUDA graph capture finished and used `1.64 GiB`.
- KV cache size is `519,920` tokens at `--max-model-len 262144`.
- maximum concurrency at that context is `1.98x`.

## Benchmark

First compact row:

| case | prompt tokens | generated tokens | TTFT seconds | decode tok/s |
|---|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 0.109 | 47.99 |
| `medium_decode` | 40 | 192 | 0.083 | 51.01 |
| `long_prefill` | 2270 | 64 | 0.494 | 91.38 |

Warmed second row:

| case | prompt tokens | generated tokens | TTFT seconds | decode tok/s |
|---|---:|---:|---:|---:|
| `short_decode` | 28 | 64 | 0.098 | 47.91 |
| `medium_decode` | 40 | 192 | 0.087 | 53.60 |
| `long_prefill` | 2270 | 64 | 0.118 | 98.38 |

## Interpretation

This is the first local vLLM Gemma 4 26B row that materially beats the earlier BF16/unquantized vLLM row. The prior compact vLLM Gemma 4 26B A4B row was about `24 tok/s`; this AEON NVFP4+DFlash path is about `48-54 tok/s` on short/medium decode and near `98 tok/s` on the long-prefill benchmark shape.

Do not attribute this gain to the local FlashInfer `b12x` predicate patch. This run used AEON's container and checkpoint, not a `jethac/vllm` or `jethac/flashinfer` build. The win is the combination of a working NVFP4 checkpoint, vLLM 0.20.1, FlashInfer CUTLASS NVFP4 linear kernels, vLLM CUTLASS NvFp4 MoE, Triton target attention for Gemma 4, CUDA graphs, and DFlash.

## Caveats

- The container's `torch.cuda.get_arch_list()` reports `sm_120` but not explicit `sm_121`; runtime device capability is still `[12, 1]`.
- `cuda_build_target_audit.py` did not find explicit architecture target strings in the server log.
- The first row manifest was captured with host `python3`, which lacked `torch`; the warmed second benchmark was captured with the benchmark venv and includes `NVIDIA_GB10:sm_121:sms_48`.
- Server log warns that some NVFP4 global scales differ across fused parallel layers and says model accuracy should be verified.
- Server log warns: `Not enough SMs to use max_autotune_gemm mode`.
- This is serving throughput and smoke evidence, not paper-comparable accuracy.
- This does not prove SGLang, llama.cpp native FP4, or NVFP4 KV cache.

## Artifacts

- chat smoke: `results/aeon_gemma26_dflash_20260608T0436JST_chat_smoke.json`
- first benchmark: `results/aeon_gemma26_dflash_20260608T0436JST_openai_benchmark.json`
- warmed benchmark: `results/aeon_gemma26_dflash_20260608T0436JST_warm2_openai_benchmark.json`
- server log: `results/aeon_gemma26_dflash_20260608T0436JST_server.log`
- key log lines: `results/aeon_gemma26_dflash_20260608T0436JST_key_log_lines.txt`
- row manifest: `results/aeon_gemma26_dflash_20260608T0436JST_row_manifest.json`
- runtime probe: `results/aeon_gemma26_dflash_20260608T0436JST_runtime_probe.json`
- container versions: `results/aeon_gemma26_dflash_20260608T0436JST_container_versions.json`
- `spark_doctor`: `results/spark_doctor_aeon_gemma26_dflash_20260608T0436JST.md`
