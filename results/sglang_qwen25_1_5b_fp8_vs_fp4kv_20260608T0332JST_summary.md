# SGLang Qwen2.5 1.5B fp8-vs-fp4 KV Probe

Date: 2026-06-08 JST

Image: `nvcr.io/nvidia/sglang:26.05-py3`

Model: `Qwen/Qwen2.5-1.5B-Instruct`

Hardware key: `NVIDIA_GB10:sm_121:sms_48`

## fp8 KV Before Row

Command shape:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.40
```

Artifacts:

- `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_chat_smoke.json`
- `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_openai_benchmark.json`
- `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_server.log`
- `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_python_versions.txt`
- `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_cuda_so_audit.json`
- `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_build_target_audit.json`
- `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_image_inspect.json`
- `results/sglang_qwen25_1_5b_fp8kv_20260608T0332JST_container_inspect.json`

Result:

- chat smoke: passed, returned `spark-ok`
- attention backend: `flashinfer`
- KV cache dtype: `torch.float8_e4m3fn`
- KV pool: `3,113,713` tokens, K `20.79 GB`, V `20.79 GB`
- CUDA graphs: enabled; full and piecewise graph capture completed
- decode:
  - short: `59.09 tok/s`, TTFT `0.043 s`
  - medium: `58.43 tok/s`, TTFT `0.035 s`
  - long-prefill: `58.22 tok/s`, TTFT `0.036 s`, prompt tokens `2369`
- caveat: log warns no KV scaling factors were provided, so fp8 scales defaulted to `1.0`
- build-target audit: server log did not expose explicit CUDA architecture build targets
- CUDA shared-object audit: no explicit `sm_121` SASS claim should be inferred from this row

## BF16/Auto KV Comparator

Command shape:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --page-size 1 \
  --mem-fraction-static 0.40
```

Artifacts:

- `results/sglang_qwen25_1_5b_bf16auto_040mem_20260608T0409JST_chat_smoke.json`
- `results/sglang_qwen25_1_5b_bf16auto_040mem_20260608T0409JST_openai_benchmark.json`
- `results/sglang_qwen25_1_5b_bf16auto_040mem_20260608T0409JST_server.log`
- `results/sglang_qwen25_1_5b_bf16auto_040mem_20260608T0409JST_container_inspect.json`

Result:

- chat smoke: passed, returned `spark-ok`
- attention backend: `flashinfer`
- KV cache dtype: `torch.bfloat16`
- KV pool: `1,557,709` tokens, K `20.80 GB`, V `20.80 GB`
- CUDA graphs: enabled; full and piecewise graph capture completed
- decode:
  - short: `58.89 tok/s`, TTFT `0.042 s`
  - medium: `58.59 tok/s`, TTFT `0.035 s`
  - long-prefill: `57.73 tok/s`, TTFT `0.136 s`, prompt tokens `2369`

## Stock fp4_e2m1 KV Failures

FlashInfer attention attempt:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 \
  --kv-cache-dtype fp4_e2m1 \
  --attention-backend flashinfer \
  --page-size 1 \
  --mem-fraction-static 0.40
```

Artifact:

- `results/sglang_qwen25_1_5b_fp4kv_20260608T0336JST_startup.log`

Result: failed before server health. SGLang rejected the configuration in `KV4Compatibility`:

```text
KV4 MHA expects attention_backend to be one of ['triton', 'torch_native', 'flex_attention', 'trtllm_mha'], but got flashinfer
```

Triton attention attempt:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bfloat16 \
  --kv-cache-dtype fp4_e2m1 \
  --attention-backend triton \
  --page-size 1 \
  --mem-fraction-static 0.40
```

Artifact:

- `results/sglang_qwen25_1_5b_fp4kv_triton_20260608T0338JST_startup.log`

Result: allocated a larger FP4 KV pool, then failed before server health:

- KV cache dtype: `torch.float4_e2m1fn_x2`
- KV pool: `5,534,509` tokens, K `18.47 GB`, V `18.47 GB`
- capacity ratio versus fp8 row: about `1.78x`
- failure: `ImportError: cannot import name 'KVFP4QuantizeUtil' from 'sglang.srt.layers.quantization.kvfp4_tensor'`

## Patched Overlay fp4_e2m1 KV Attempts

Overlay source:

- SGLang fork branch: `jethac/sglang@spark/hijinks-018-fp4-e2m1-kv-sm121-serving`
- fork commit: `98ad46961`
- overlay helper: `scripts/patch_sglang_fp4_kv_site.py`

The overlay applied three Python-level fixes inside the NVIDIA 26.05 image:

- `server_args_fa4_gate=True`
- `server_args_mha_gate=True`
- `kvfp4_alias=True`

FlashInfer attention artifact:

- `results/sglang_qwen25_1_5b_fp4kv_patched_flashinfer_20260608T0349JST_startup.log`
- `results/sglang_qwen25_1_5b_fp4kv_patched_flashinfer_20260608T0349JST_container_inspect.json`

FlashInfer attention result:

- moved past stock `KV4Compatibility` and missing-alias failures
- allocated `5,539,718` FP4 KV tokens
- FlashInfer JIT targeted `compute_121a,code=sm_121a`
- failed compiling the FlashInfer FP4 decode kernel: `vec_dtypes.cuh(117): no suitable conversion function from const Params::DTypeKV to float`

Triton attention artifacts:

- `results/sglang_qwen25_1_5b_fp4kv_patched_triton_20260608T0352JST_startup.log`
- `results/sglang_qwen25_1_5b_fp4kv_patched_triton_nograph_20260608T0400JST_startup.log`
- `results/sglang_qwen25_1_5b_fp4kv_patched_triton_nographs_20260608T0404JST_chat_smoke.json`
- `results/sglang_qwen25_1_5b_fp4kv_patched_triton_nographs_20260608T0404JST_openai_benchmark.json`
- `results/sglang_qwen25_1_5b_fp4kv_patched_triton_nographs_20260608T0404JST_server.log`

Triton attention result:

- normal CUDA graph capture reached FP4 KV allocation, then stalled at the first graph-capture batch
- `--disable-cuda-graph` alone still entered piecewise CUDA graph capture and stalled after two dynamic-shape compiles
- disabling both standard and piecewise CUDA graphs reached serving and returned `spark-ok`
- no-graphs FP4 KV pool: `5,541,103` tokens, about `3.56x` BF16/auto capacity and `1.78x` fp8 capacity
- no-graphs short decode: `0.276 tok/s`, TTFT `5.812 s`, total `238.044 s` for 64 completion tokens
- output quality was visibly poor and repetitive on the short benchmark

## Interpretation

This is a useful SGLang before-state for issue #20 and #18.

The stock NVIDIA SGLang 26.05 image can serve the cached Qwen2.5 1.5B model with BF16/auto and fp8 KV on GB10 at about `58-59 tok/s` decode. fp8 provides about `2.0x` the BF16/auto KV pool at matched memory fraction with similar decode speed.

The patched overlay proves the first SGLang-side FP4 KV blockers are fixable and that the expected larger FP4 KV pool is real. It does not prove a usable FP4 KV serving stack. FlashInfer attention now fails inside FlashInfer FP4 E2M1 decode compilation, while Triton attention serves only with CUDA graphs disabled and is about two orders of magnitude slower than BF16/fp8.

Do not call SGLang FP4 KV blessed from this evidence. The next meaningful after-row is a clean `jethac/sglang` plus dependency build with graph-compatible FP4 KV, quality comparison against BF16/fp8, and no site-package overlay.
