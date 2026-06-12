# SGLang Gemma 4 AR Config / KV Denominator Audit

Date: 2026-06-12 JST

Status: GREEN for static config and denominator math only. This is not a model
load, serving, quality, or capacity row.

## Inputs

The `config.json` files were copied from the existing Spark Hugging Face cache
without downloading model payloads:

- `google/gemma-4-12B-it`
- `google/gemma-4-26B-A4B-it`
- `google/gemma-4-31B-it`
- `google/gemma-4-E4B-it` as the already-green SGLang reference point

The audit cross-checks these configs against the current SGLang source:

- `third_party/sglang/python/sglang/srt/utils/hf_transformers/config.py`
- `third_party/sglang/python/sglang/srt/configs/model_config.py`
- `third_party/sglang/python/sglang/srt/model_executor/pool_configurator.py`

## Geometry

SGLang's config loader intentionally inverts Gemma 4 fields for runtime use:
base `head_dim` / `num_key_value_heads` become full-attention/global geometry,
and `swa_*` preserves the sliding-window geometry. The source does this by
copying the checkpoint's original base fields to `swa_head_dim`,
`swa_v_head_dim`, and `swa_num_key_value_heads`, then replacing base
`head_dim` / `num_key_value_heads` with `global_head_dim` /
`num_global_key_value_heads`.

| model | full layers | SWA layers | full KV heads × D | SWA KV heads × D |
|---|---:|---:|---:|---:|
| Gemma 4 12B IT | 8 | 40 | 1 × 512 | 8 × 256 |
| Gemma 4 26B-A4B IT | 5 | 25 | 2 × 512 | 8 × 256 |
| Gemma 4 31B IT | 10 | 50 | 4 × 512 | 16 × 256 |
| Gemma 4 E4B IT | 7 | 35 | 2 × 512 | 2 × 256 |

## Denominator

For full NVFP4 K+V, SGLang's helper computes per layer:

```text
KV head bytes = K_data(D/2) + V_data(D/2) + K_scale(D/16) + V_scale(D/16)
              = 9 * D / 16 bytes for K plus 9 * D / 16 bytes for V combined
```

Equivalently, the packed cache uses `9*D/16` bytes per K/V tensor per head,
matching the vLLM `nvfp4_kv_cache_full_dim` reference convention. With BF16 K+V
at `4*D` bytes per head, the theoretical BF16/full-NVFP4 token ratio is
`4D / (9D/8) = 32/9 = 3.5556x`.

Using the SGLang hybrid-SWA default ratio observed in prior rows
(`swa_full_tokens_ratio=0.8`), the static cell sizes are:

| model | full-NVFP4 cell bytes | BF16 cell bytes | BF16 / NVFP4 |
|---|---:|---:|---:|
| Gemma 4 12B IT | 78,336 | 278,528 | 3.5556x |
| Gemma 4 26B-A4B IT | 51,840 | 184,320 | 3.5556x |
| Gemma 4 31B IT | 207,360 | 737,280 | 3.5556x |
| Gemma 4 E4B IT | 24,192 | 86,016 | 3.5556x |

## Interpretation

The current SGLang source matches the vLLM NVFP4 page-byte denominator for
Gemma 4 AR hybrid-SWA geometry. If 26B-A4B or 31B still reports negative token
capacity at live launch, the next suspect is not this K/V byte formula. It is
the live available-memory accounting after weights/graphs/runtime residency, or
a model-loading path that bypasses the transformed global/SWA geometry.

The live ladder packet must still record `SGLANG_GEMMA_KV_POOL_CONFIG` from the
server logs for each model; this static audit only predicts the denominator that
those logs should show.
