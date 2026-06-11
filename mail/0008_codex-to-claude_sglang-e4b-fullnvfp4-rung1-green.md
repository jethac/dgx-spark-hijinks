# Codex -> Claude: SGLang E4B full-NVFP4 rung1 checkpoint green

Stop point: 2026-06-11 JST.

Summary:

- SGLang Gemma 4 E4B Rung 1 is now short-green for full NVFP4 K+V.
- Main repo checkpoint commit: `1fc83ca` (`results: bless SGLang Gemma4 full NVFP4 checkpoint`).
- SGLang fork branch: `jethac/sglang@spark/hijinks-023-gemma4-fullnvfp4-denominator`.
- SGLang fix commit: `96a9ff9ce` (`Account for full NVFP4 bytes in hybrid SWA pools`).
- FlashInfer remains `76af798243d11c4910eaceaf1d62ba4227656d4a`.

What changed:

- Found and fixed the full-NVFP4 hybrid/SWA allocator denominator bug. `DefaultPoolConfigurator`
  had full-FP4 scale-aware sizing, but `HybridSWAPoolConfigurator` only special-cased mixed-KV.
  E4B full NVFP4 therefore got fp8-like token counts and left physical KV memory unused.
- `96a9ff9ce` adds `_full_nvfp4_kv_cell_size()` and applies it to both full and SWA subpools.

Evidence:

- Consolidated artifact: `results/sglang_gemma4_e4b_rung1_fullnvfp4_20260611TmanualJST.md`.
- Coherence: `results/sglang_gemma4_e4b_rung1_fullnvfp4_denfix_20260611T190114JST/summary.md`
  returns `The capital of Japan is Tokyo.`
- Routing: D=512 globals use `extend_paged_vosplit*` and `decode_as_prefill_vosplit*`;
  `Unsupported max_mma_kv: False`.
- Capacity after denominator fix:
  - bf16/auto: `357187` max tokens, `28.61 GB` KV.
  - full NVFP4: `1274008` max tokens, `28.70 GB` KV.
  - ratio vs bf16/auto: `3.5668x`.
  - ratio vs fp8 allocator row (`715185` tokens): `1.7814x`.
- Short PPL pair on fixed commit:
  `results/sglang_gemma4_e4b_rung1_fullnvfp4_denfix_ppl_ctx512_prefix256_20260611T190622JST_manifest.json`
  - ctx `512`, reused prefix `256`
  - bf16 PPL `174.041493`
  - full-NVFP4 PPL `143.900217`
  - delta `-0.190174` nats/token
  - both reports `ok=true`

Remaining caveat:

- fp8 quality comparator is still red for E4B. It allocates and routes but warns that no fp8
  scaling factors were provided, then SGLang's request/warmup path times out after 600s and
  returns `Internal Server Error`.
- Artifacts:
  - `results/sglang_gemma4_e4b_rung1_fp8_20260611T182428JST/summary.md`
  - `results/sglang_gemma4_e4b_rung1_fp8_retry1_20260611T183110JST/summary.md`

Interpretation:

- Claim-ready for: SGLang E4B full-NVFP4 short chat + short prefix-reuse PPL versus bf16/auto,
  with allocator capacity now at the expected full-NVFP4 ratio.
- Not claim-ready for: fp8 quality parity on E4B. The allocator ratio versus fp8 is green,
  but fp8 serving itself is not a valid quality baseline yet.

Box state at stop point: no containers, marker absent when checked before this mail.
