# SGLang Gemma 3 27B Rung 1 mixed-KV checkpoint

Date: 2026-06-10 JST

Scope: SGLang lane, Gemma 3 27B text-only, hybrid SWA KV pool, mixed FP8-K + NVFP4-V. This is a Rung 1 checkpoint, not yet a full long-context claim.

## Code changes exercised

- `Gemma3ForConditionalGeneration` is recognized as a hybrid-SWA model using `layer_types`.
- `SGLANG_GEMMA3_ENABLE_HYBRID_SWA=1` enables the experimental Gemma 3 hybrid-SWA pool instead of forcing non-hybrid allocation.
- Runtime geometry logging records per-layer heads, KV heads, head dim, V head dim, and local/global window.
- SWA pool logging records full/SWA layer maps, pool classes, token capacities, and K/V bytes.
- Hybrid-SWA allocation now selects `MHATokenToKVPoolFP4` when `--kv-cache-dtype fp4_e2m1`; before this, the SWA wrapper instantiated the generic pool and failed with `torch.zeros(..., dtype=torch.float4_e2m1fn_x2)`.
- The mixed-KV pool configurator fix is included: FP8-K + NVFP4-V uses the corrected `3200` bytes/token/layer denominator for Gemma 3 (`4096` for fp8).

## Measured geometry

Model: `google/gemma-3-27b-it`

Runtime geometry from server logs:

- Layers: 62
- Local/SWA layers: 52
- Full/global layers: 10, layers `5, 11, 17, 23, 29, 35, 41, 47, 53, 59`
- Heads: 32
- KV heads: 16
- `head_dim`: 128
- `v_head_dim`: 128
- Local window in SGLang runtime: `1023` (`1024` config window represented as left window)
- Cache implementation: `SWARadixCache hybrid_swa=True`

## Red rows

`sglang_gemma3_27b_rung1_fp8_poolfix_20260610TmanualJST`

- Hybrid SWA was still disabled by SGLang's Gemma 3 guard.
- The non-hybrid fp8 pool did not fit at `--mem-fraction-static 0.24`.

`sglang_gemma3_27b_rung1_fp8_hybridswa_20260610TmanualJST`

- Hybrid SWA was enabled, but `--mem-fraction-static 0.24` was still too low after weights.
- The pool calculation produced negative token capacities.

`sglang_gemma3_27b_rung1_mixedkv_hybridswa_mf060_20260610TmanualJST`

- Hybrid SWA mixed-KV reached pool construction, then failed because the SWA wrapper used the generic `MHATokenToKVPool`.
- Failure: `NotImplementedError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'`.
- Fix: pass `token_to_kv_pool_class=MHATokenToKVPoolFP4` for hybrid SWA when the configured KV dtype is float4.

## Serving smoke rows

### fp8 comparator

Run: `sglang_gemma3_27b_rung1_fp8_hybridswa_mf060_20260610TmanualJST`

- `--kv-cache-dtype fp8_e4m3`
- `--mem-fraction-static 0.60`
- `--page-size 1`
- CUDA graphs disabled
- Docker cgroup: `--memory 100g --memory-swap 100g`
- Ready: yes
- Benchmark: `short_decode`, ok
- Decode: `4.1668 tok/s`
- SWA pool: `61107` tokens, K `6.06 GB`, V `6.06 GB`
- Full pool: `76384` tokens, K `1.46 GB`, V `1.46 GB`
- Total logged K+V: `15.04 GB`

### mixed FP8-K + NVFP4-V candidate

Run: `sglang_gemma3_27b_rung1_mixedkv_hybridswa_mf060_poolclassfix_20260610TmanualJST`

- `--kv-cache-dtype fp4_e2m1`
- `SGLANG_FP4_KV_MIXED_KV=1`
- `--mem-fraction-static 0.60`
- `--page-size 1`
- CUDA graphs disabled
- Docker cgroup: `--memory 100g --memory-swap 100g`
- Ready: yes
- Benchmark: `short_decode`, ok
- Decode: `4.1515 tok/s`
- SWA pool: `77021` tokens, K `7.64 GB`, V `4.30 GB`
- Full pool: `96277` tokens, K `1.84 GB`, V `1.03 GB`
- Total logged K+V: `14.81 GB`

Serving-row capacity ratio:

- SWA tokens: `77021 / 61107 = 1.2604x`
- Full tokens: `96277 / 76384 = 1.2604x`
- Physical K+V is matched within log rounding.

## Quality gate

Run: `sglang_gemma3_27b_rung1_mixedkv_ppl_ctx512_20260610TmanualJST`

Sequential comparator, one server at a time:

- Context: `512`
- Reused prefix: `256`
- Logprob start: `256`
- fp8: `--kv-cache-dtype fp8_e4m3`
- mixed: `--kv-cache-dtype fp4_e2m1`, `SGLANG_FP4_KV_MIXED_KV=1`
- Hybrid SWA enabled for both.
- CUDA graphs disabled for both.

Result:

| ctx | reused prefix | PPL fp8 | PPL mixed | delta PPL | delta nats/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 256 | 57.6366178265 | 57.6764197677 | +0.0398019413 | +0.0006903286 |

The absolute PPL is from a deterministic 250k-character markdown corpus built from this
repo's `docs/*.md`, `tasks/*.md`, and `results/*.md` artifacts, so it is not comparable to
a standard benchmark corpus; the claim is the paired fp8-vs-mixed delta on identical text.

PPL-row capacity ratio:

- SWA tokens: `77310 / 60154 = 1.2852x`
- Full tokens: `96638 / 75193 = 1.2852x`
- fp8 total logged K+V: `14.80 GB`
- mixed total logged K+V: `14.86 GB`

## Deep-prefix quality gate

Run: `sglang_gemma3_27b_mixedkv_ppl_ctx8192_prefix4096_20260611TmanualJST`

Sequential comparator, one server at a time, CUDA graphs disabled for both:

- Context: `8192`
- Reused prefix: `4096`
- Logprob start: `4096`
- fp8: `--kv-cache-dtype fp8_e4m3`
- mixed: `--kv-cache-dtype fp4_e2m1`, `SGLANG_FP4_KV_MIXED_KV=1`
- Hybrid SWA enabled for both.

Result:

| ctx | reused prefix | PPL fp8 | PPL mixed | delta PPL | delta nats/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 4096 | 4.1805119853 | 4.1769609268 | -0.0035510585 | -0.0008497925 |

Server logs prove the scored requests used real radix prefix reuse with
`#cached-token: 4096` and `cuda graph: False` on both sides. Capacity at this launch is
consistent with the corrected mixed-KV denominator:

- SWA/local tokens: `76870 / 60355 = 1.2736x`
- Full/global tokens: `96088 / 75444 = 1.2736x`
- artifact directory:
  `results/sglang_gemma3_27b_mixedkv_ppl_ctx8192_prefix4096_20260611TmanualJST/`

## Status

Green at this checkpoint:

- Gemma 3 27B runs text-only on SGLang with experimental hybrid-SWA memory enabled.
- Runtime geometry confirms the Rung 1 target: uniform D=128, 52 local layers, 10 global layers.
- fp8 comparator serves.
- mixed FP8-K + NVFP4-V serves after selecting the FP4 pool class inside SWA.
- Short PPL comparator with prefix reuse is quality-green at `ctx=512`, `reuse_prefix_len=256`.
- Deep-prefix PPL comparator with prefix reuse is quality-green at `ctx=8192`, `reuse_prefix_len=4096`.
- Capacity is consistent with the corrected mixed-KV denominator: about `1.26x` to `1.29x`, not the old pre-fix `~1.78x` artifact.

Still pending:

- Broader long-context/deeper-prefix sweep beyond the single `ctx=8192`, `reuse_prefix_len=4096` row.
- CUDA graph re-enable gate.
- Full NVFP4 K+V. The full-NVFP4 structural route remains parked; this row is the mixed-KV fallback path.
- Gemma 4 rungs.

## Artifacts

- `results/sglang_gemma3_27b_rung1_fp8_hybridswa_mf060_20260610TmanualJST_server.log`
- `results/sglang_gemma3_27b_rung1_fp8_hybridswa_mf060_20260610TmanualJST_row_manifest.json`
- `results/sglang_gemma3_27b_rung1_mixedkv_hybridswa_mf060_poolclassfix_20260610TmanualJST_server.log`
- `results/sglang_gemma3_27b_rung1_mixedkv_hybridswa_mf060_poolclassfix_20260610TmanualJST_row_manifest.json`
- `results/sglang_gemma3_27b_rung1_mixedkv_ppl_ctx512_20260610TmanualJST_manifest.json`
- `results/sglang_gemma3_27b_rung1_mixedkv_ppl_ctx512_20260610TmanualJST_compare.json`
- `results/sglang_gemma3_27b_rung1_mixedkv_ppl_ctx512_20260610TmanualJST_fp8_server.log`
- `results/sglang_gemma3_27b_rung1_mixedkv_ppl_ctx512_20260610TmanualJST_mixed_server.log`
