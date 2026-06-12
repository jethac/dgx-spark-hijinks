# SGLang DiffusionGemma DG-R6 Performance Pair

Status: GREEN

## Scope

Text-only serving performance comparison on GB10. The before row is the stock DiffusionGemma policy path: Triton attention with BF16/auto KV. The after row is the GB10-tuned path: FlashInfer VO-split with full NVFP4 K+V. Quality and capacity claims remain separate from this speed row.

## Provenance

- Pair run: `sglang_dgemma_dgr6_perf_pair_20260612T152803JST`
- Model: `google/diffusiongemma-26B-A4B-it`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- FlashInfer: `f99323bd7d1c`
- Common launch: `--dllm-algorithm Gemma4Renoise --dtype bfloat16 --context-length 8192 --mem-fraction-static 0.55 --disable-cuda-graph --disable-piecewise-cuda-graph`
- Prompt set: `short_decode`, `medium_decode`, `long_prefill`, `natural_long_prefill` from `scripts/openai_serving_benchmark.py`

## Gates

- `before_stock_triton_bf16` quality gate: PASS
- `before_stock_triton_bf16` OpenAI benchmark: PASS
- `after_flashinfer_fullnvfp4` quality gate: PASS
- `after_flashinfer_fullnvfp4` OpenAI benchmark: PASS
- before stock Triton policy proof: PASS
- after FlashInfer VO-split policy proof: PASS
- after full-NVFP4 K+V pool proof: PASS

## Capacity Context

| Row | Full-layer tokens | SWA tokens |
|---|---:|---:|
| stock Triton, BF16/auto KV | 67584 | 54016 |
| FlashInfer VO-split, full NVFP4 K+V | 237568 | 189952 |

Capacity ratio in this performance pair: `237568 / 67584 = 3.5152x` full-layer tokens.
SWA token ratio: `189952 / 54016 = 3.5166x`.

## Throughput

DiffusionGemma's OpenAI streaming path emits each measured completion as effectively one stream event, so `ttft_s` is nearly equal to `total_s` and the synthetic `decode_tok_s` field is not meaningful. Use total completion-token throughput for this row.

| Case | Before TTFT s | Before total s | Before total tok/s | After TTFT s | After total s | After total tok/s | After/Before total tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `long_prefill` | 4.245 | 4.245 | 15.076 | 4.074 | 4.075 | 15.707 | 1.0419 |
| `medium_decode` | 4.767 | 4.767 | 40.274 | 5.287 | 5.287 | 36.315 | 0.9017 |
| `natural_long_prefill` | 5.848 | 5.848 | 21.886 | 4.049 | 4.049 | 31.612 | 1.4444 |
| `short_decode` | 3.733 | 3.733 | 17.144 | 3.676 | 3.676 | 17.412 | 1.0157 |

## Non-Claims

- No image/multimodal quality claim.
- No CUDA graph safety claim.
- No long-context quality/PPL claim.
- The speed comparison is a combined stack comparison, not an isolated kernel-only attribution.

## Artifacts

- `before_stock_triton_bf16/server.log`
- `before_stock_triton_bf16/revised_text_quality.json`
- `before_stock_triton_bf16/openai_benchmark.json`
- `after_flashinfer_fullnvfp4/server.log`
- `after_flashinfer_fullnvfp4/revised_text_quality.json`
- `after_flashinfer_fullnvfp4/openai_benchmark.json`
