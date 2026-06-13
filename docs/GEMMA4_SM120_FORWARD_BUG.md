# Gemma-4 forward produces degenerate output on sm_120 (consumer Blackwell)

**Status:** OPEN — investigating · **Opened:** 2026-06-13 · **Severity:** blocks the
vLLM Gemma-4 anchor ladder on x86 sm_120 (vast.ai), and any sm_120 Gemma-4 serving.

## Symptom

Gemma-4 generates **degenerate output** on sm_120 GPUs:
`GEN("The capital of France is") -> '111.1...'`. Prefill logits are degenerate
(top-1 predictions are `'1'`/`'.'`/`'-'` everywhere); wikitext-2 mean NLL ≈ 8.0
(PPL ≈ 3039) vs an expected ≈ 2–3. Greedy generation is garbage.

## What works vs breaks (the discriminating matrix)

All rows below are on a **clean RTX PRO 6000 / RTX 5090, cc (12,0) = sm_120**, x86_64,
HF eager (`transformers`, no vLLM/flashinfer unless noted).

| model | arch | sm_120 result |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct | non-Gemma | ✅ `' Paris. It was founded in 7'` |
| google/gemma-3-1b-it | Gemma-3 | ✅ `' Paris. The largest city in France'` |
| **google/gemma-4-12B-it** | Gemma-4 dense | ❌ `'111.1...'` |

So the breakage is **Gemma-4-specific** — generic Torch/cuBLAS on sm_120 is fine
(Qwen), the whole Gemma-3 family is fine, only Gemma-4 modeling code fails.

## Ruled out (Gemma-4-12B still breaks across all of these)

| axis | values tested | result |
| --- | --- | --- |
| KV / attention backend | Triton vs FlashInfer (vLLM) | bit-identical garbage (NLL 8.0195) → upstream of attention |
| framework | vLLM (custom_ops all & none) **and** pure HF eager | both broken (Codex 0113) → not vLLM/flashinfer |
| torch | 2.11.0+cu128 **and** 2.12.0+cu130 | both broken |
| transformers | 5.10.0.dev0 (`effde209`), 5.11.0, 5.12.0 | all broken (Codex 0113 + Claude) |
| dtype | bfloat16 **and** float16 | both broken |
| wheel `_C` cubins | cuobjdump confirms native sm_120a SASS present | not a missing-cubin issue |
| RMSNorm impl | native vs `_C` priority | broken either way (Codex 0113) |

→ **This is not a version regression in torch/transformers/the wheel, and not a dtype
or backend issue. It is Gemma-4 modeling × sm_120 hardware specifically.**

## Known-good contrast: sm_121 (Spark)

The campaign's coherent Gemma-4 serving (DG-V5, 31B TTFT, the whole ladder) is all on
**Spark, GB10 = sm_121**. So Gemma-4 works on sm_121 but not sm_120. The two are both
"consumer Blackwell" but distinct compute capabilities. **Open control:** run this exact
HF-eager test (Qwen / Gemma-3 / Gemma-4) on Spark sm_121 to confirm Gemma-4 HF-eager is
coherent there (Codex's box). If yes → confirmed sm_120-vs-sm_121 divergence in a
Gemma-4-specific op.

## Current hypothesis

A **Gemma-4-specific op** (something Gemma-4 uses that Gemma-3 does not — candidates:
the unified/multimodal text path, a new attention scaling / QK-norm, logit or attention
softcapping, or a specific GEMM shape) hits a **broken or mis-dispatched kernel on
sm_120** (cc 12.0) that is correct on sm_121 (cc 12.1). Degenerate uniform logits is the
signature of a collapsed/zeroed intermediate (e.g., an embedding-scale or norm whose
kernel returns garbage on this arch).

## The "where did it come in" question

The data says it is **not** a software-version regression on sm_120 (torch, transformers
all broken). The most likely truth: Gemma-4 was **validated only on sm_121 (Spark)**;
sm_120 was never exercised end-to-end for the forward, so it has been broken on sm_120
since Gemma-4 support landed. "Where it came in" therefore points to the **arch axis**,
not a commit — confirm with the sm_121 control, then localize the diverging op.

## Family to test (the "every Gemma-4 model" matrix — in progress)

Text models (it variants): **12B** (❌ confirmed), **31B**, **26B-A4B** (MoE), **E2B**
(MoE), **E4B** (MoE). Goal: does every Gemma-4 break on sm_120, or only some configs?
This isolates whether the broken op is universal to Gemma-4 or tied to a sub-family
(dense vs MoE, head_dim 256 vs 512).

## Next steps

1. **Model matrix** (vast.ai sm_120, HF eager): E2B, E4B (small, on a 32GB box); 31B,
   26B-A4B (on a 96GB box). Record GEN output per model.
2. **sm_121 control** (Codex, Spark): same HF-eager Qwen/Gemma-3/Gemma-4 test.
3. **Op localization**: once a small broken Gemma-4 (E2B) is in hand, compare per-layer
   hidden states sm_120-GPU vs CPU (fp32) to find the first diverging op; that op's kernel
   is the bug.
4. Fix path: either a transformers/torch-level kernel fix for sm_120, or a documented
   "Gemma-4 needs sm_121" constraint + route x86 anchor runs accordingly.

## Artifacts / refs
- Codex 0113 (vast sm_120 custom-ops isolation), `results/vast_sm120_d4f0_custom_ops_...`
- Claude HF-eager discriminator (this doc, 2026-06-13), scripts in `docs/vast_anchor/`
- vast box runbook: `docs/vast_anchor/SM120_NUMERICS_PLAN.md §5`
