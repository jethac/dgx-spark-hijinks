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

## Interpretation

This is a useful SGLang before-state for issue #20 and #18.

The stock NVIDIA SGLang 26.05 image can serve the cached Qwen2.5 1.5B model with fp8 KV on GB10 at about `58-59 tok/s` decode and a `3.11M` token KV pool. Stock `fp4_e2m1` does not currently produce a serving row on the same image. The FlashInfer route is blocked by an attention-backend compatibility gate; the Triton route reaches FP4 KV allocation and shows the expected `~1.78x` capacity increase, but then fails on missing `KVFP4QuantizeUtil`.

Do not call SGLang FP4 KV blessed from this evidence. The next meaningful after-row is the `jethac/sglang` fork with the same model and prompts, proving whether the compatibility gate and missing quantization utility path are fixed and whether output quality is acceptable.
