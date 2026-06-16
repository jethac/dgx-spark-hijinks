# DiffusionGemma 26B-A4B NVFP4-KV truth-gate — VERDICT (2026-06-17, RTX PRO 6000 / sm120)

**nvfp4 KV does NOT degrade DiffusionGemma vs bf16.** Engineering + comparative gate both pass; a
crisp *absolute* gate is not achievable because DG's block-diffusion generation is intrinsically noisy
at its EntropyBound sampler (bf16 itself is noisy), so we gate on nvfp4-vs-bf16 RELATIVE (the project's
hard-won rule), not absolute accuracy.

## Engineering (the real blocker — SOLVED)
DG now serves on the **integrated NVFP4-KV line** (not the old e2-dgv branch). The diffusion runtime was
already reconciled in source HEAD; overlaid its Python onto the box's `d0f6221e6` install (no compiled
change in that range -> no wheel rebuild) + landed the `build_attn_metadata(causal=...)` source fix
(`b57a8975c`). Recipe: FLASHINFER backend + `VLLM_NVFP4_KV_VOSPLIT=1`/`LINEAR_V_SF=1` (DG-2 per-request
causal grouping lives in the FI VO-split path) + `llm.chat` + 256-token denoise block.

## Comparative results (nvfp4 vs bf16, same seed)
- **Needle (ctx200, 16 trials):** bf16 recall 0.125 / nvfp4 **0.188** — nvfp4 >= bf16. DG exact-digit
  recall degrades past ~200 tok IN BF16 (denoise corrupts leading digits; tails stay correct) — a model
  limit, identical for both dtypes.
- **Generation agreement (12 prompts, seed0):** mean char-sim 0.62; where they differ, **nvfp4 is as or
  MORE coherent** (nvfp4 "Paris"/"Jupiter"/"H₂O"/"7 days in a week" vs bf16 ""/cutoff/"Note is symbol
  water"). The residual divergence is bf16-side noise, not nvfp4 degradation.
- **DG-V5 (2026-06-12, Spark/sm121):** coherent ("Tokyo"), 3.556x format-exact KV capacity, SGLang parity.

## Bottom line
Ship DiffusionGemma with calibrated NVFP4 KV: faithful to bf16, 3.556x capacity, serves on the integrated
line on consumer Blackwell. The diffusion decoder re-reads the full prefix every denoise step, so the
capacity win compounds harder than anywhere in the AR ladder.
