# SGLang DiffusionGemma DG-R5 Capacity Pair

Status: GREEN

## Scope

Matched allocator-capacity denominator for SGLang DiffusionGemma 26B-A4B on
GB10. This compares BF16/auto KV against full NVFP4 K+V at the same model,
image, SGLang commit, FlashInfer commit, page size, memory fraction, graph
policy, and deterministic `Gemma4Renoise` config.

This is a capacity row. The quality/routing gates are inherited from the two
source smoke rows and remain separately documented.

## Provenance

- Model: `google/diffusiongemma-26B-A4B-it`
- Image: `sglang-source-stack-dgemma-024-0705924c-f99323bd:latest`
- SGLang: `98bf8f129d701d2829f2d1a82c4ce6a8b2f5a968`
- FlashInfer: `f99323bd7d1c`
- Common launch: `--dllm-algorithm Gemma4Renoise --dllm-algorithm-config dllm_config.yaml --attention-backend flashinfer --dtype bfloat16 --page-size 256 --mem-fraction-static 0.55 --disable-cuda-graph --disable-piecewise-cuda-graph`
- BF16 comparator run: `results/sglang_dgemma_dgr5_bf16_capacity_baseline_20260612T150750JST/summary.md`
- Full-NVFP4 run: `results/sglang_dgemma_dgr5_fullnvfp4_smoke_20260612T145433JST/summary.md`

## Capacity

| KV mode | full-layer tokens | SWA tokens | cell size bytes | pool dtype |
|---|---:|---:|---:|---|
| BF16 / auto KV | 66,560 | 53,248 | 184,320 | `torch.bfloat16` |
| full NVFP4 K+V | 237,312 | 189,696 | 51,840 | `torch.float4_e2m1fn_x2` |

Capacity ratios:

- Full-layer token ratio: `237312 / 66560 = 3.5654x`.
- SWA token ratio: `189696 / 53248 = 3.5625x`.
- Cell-size byte ratio: `184320 / 51840 = 3.5556x`.

Quote this row as approximately **3.56x KV token capacity versus BF16/auto KV**
for this DiffusionGemma 26B-A4B SGLang/FlashInfer/VO-split launch envelope.

## Quality And Routing Side Conditions

Both source rows passed the revised DG-R2 text-only quality gate:

- `capital_japan_direct`
- `arithmetic_2_plus_2_direct`
- `dgx_spark_use`

The BF16 comparator proves the FlashInfer VO-split route at current SGLang
`98bf8f129d`; the full-NVFP4 row additionally proves:

- `SGLANG_FP4_KV_MIXED_KV=0`
- `mixed_kv=False`
- FP4 K/V pools allocated
- hybrid full/SWA subpools are `MHATokenToKVPoolFP4`
- D=512 global layers route through `extend_paged_vosplit*`
- `head_dim_vo=256`
- `deswizzle_macro_active=False`

## Non-Claims

- No fp8 denominator is included in this pair.
- No image/multimodal quality claim.
- No CUDA graph safety claim.
- No long-context quality/PPL claim.
- No throughput claim.

## Cleanup

After both sequential runs, `/home/jethac/spark_tmp/CLAUDE_WINDOW_OPEN` was
absent and `docker ps` was empty.
