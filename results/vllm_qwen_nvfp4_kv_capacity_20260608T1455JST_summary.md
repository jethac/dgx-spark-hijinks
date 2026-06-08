# vLLM Qwen NVFP4-KV Capacity Row

Date: 2026-06-08 JST

Scope: Qwen3.6 35B-A3B NVFP4 weights on a single GB10 system, no DFlash, matched fp8 KV versus NVFP4 KV serving through vLLM + FlashInfer.

This is a capacity/concurrency proof, not a decode-speed win and not a Gemma proof.

## Runtime

- image: `jethac-vllm-aeon-q36:a919d635d-cleanfa2-flashinfer-e152cf4d-nvfp4kv`
- runtime ref: `ghcr.io/aeon-7/vllm-spark-omni-q36:v2 + jethac/vllm@a919d635d + jethac/flashinfer@e152cf4d`
- model path: `/home/jethac/models/aeon/qwen36-nvfp4`
- served model: `qwen36-fast`
- vLLM: `0.1.dev1+ga919d635d`
- FlashInfer: `0.6.13` from `/opt/jethac-flashinfer`
- attention backend argument: `--attention-backend flashinfer`
- max model length: `262144`
- GPU memory utilization: `0.85`
- Qwen thinking disabled with `chat_template_kwargs={"enable_thinking": false}`

## Artifacts

- fp8 comparator manifest: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1442JST_fp8_flashinfer_row_manifest.json`
- fp8 comparator benchmark: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1442JST_fp8_flashinfer_openai_benchmark.json`
- fp8 comparator server log: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1442JST_fp8_flashinfer_server.log`
- NVFP4-KV manifest: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_nvfp4_kv_flashinfer_row_manifest.json`
- NVFP4-KV benchmark: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_nvfp4_kv_flashinfer_openai_benchmark.json`
- NVFP4-KV server log: `results/vllm_qwen_nvfp4_kv_capacity_20260608T1455JST_nvfp4_kv_flashinfer_server.log`
- image build log: `results/vllm_qwen_nvfp4kv_image_build_20260608T1420JST_retry2.log`

## Capacity Result

| KV dtype | attention block size | GPU KV cache tokens | max concurrency at 262k context |
|---|---:|---:|---:|
| `fp8` | `2096` | `6,364,935` | `24.28x` |
| `nvfp4` | `3728` | `11,146,226` | `42.52x` |

NVFP4-KV increased the vLLM-reported KV pool by `1.751x` and the max-concurrency estimate by `1.751x` at the same `262144` context and `0.85` memory fraction.

## Serving Result

| case | fp8 decode tok/s | NVFP4-KV decode tok/s | interpretation |
|---|---:|---:|---|
| `short_decode` | `43.001` | `43.014` | parity |
| `medium_decode` | `42.512` | `42.615` | parity |
| `long_prefill` | `42.684` | `42.898` | parity |

Both rows returned normal `message.content` with `reasoning_chars=0`. This is not a speedup claim.

## Backend Evidence

The fp8 row logs:

- `kv_cache_dtype='fp8'`
- `Using fp8 data type to store kv cache`
- `Using AttentionBackendEnum.FLASHINFER backend`
- `GPU KV cache size: 6,364,935 tokens`

The NVFP4-KV row logs:

- `kv_cache_dtype='nvfp4'`
- `Using nvfp4 data type to store kv cache`
- `Using AttentionBackendEnum.FLASHINFER backend`
- `Using FlashInfer FA2 backend for NVFP4 KV cache on SM12x with vLLM V-scale-factor deswizzle enabled.`
- `GPU KV cache size: 11,146,226 tokens`

## Caveats

- This was built as a derived proof image before `docs/CODEX_DIRECTION_VLLM_GEMMA_NVFP4_KV.md` reoriented the vLLM dev loop. Future Gemma iteration should use source overlay and standalone FlashInfer harnesses until the Gemma row passes.
- The Docker container env did not carry a raw `FLASHINFER_EXTRA_CUDAFLAGS` value. The deswizzle proof comes from the patched vLLM runtime log and FlashInfer source in the image, not from a container env claim.
- The row uses Qwen, not Gemma. It proves the Qwen/standard-attention NVFP4-KV path and package integration, while Gemma remains blocked by heterogeneous attention dims plus the FlashInfer FA2 `D=512` global-attention failure.
- The row does not prove native FP4 weight/MoE MMA. The server still selects `MARLIN` NvFp4 MoE for weights.
- The server-log build-target audit still cannot infer accepted native CUDA targets from these logs. Use image-build and binary/JIT evidence separately for native-target claims.
